"""Pull a TINY balanced slice of the REAL moderation dataset into data/real/.

Public repo, toxic domain: raw comment text lands ONLY under data/real/
(git-ignored) and is NEVER committed - only derived labels/metrics are published
(see docs/dataset.md). This streams google/civil_comments (CC0, no login needed),
binarizes the continuous `toxicity` score to the single safe/unsafe label at a
chosen threshold TAU, and writes a class-balanced {text, label, toxicity} jsonl
small enough to eyeball every row before we scale.

`toxicity` in the source is the FRACTION of human raters who flagged the comment,
so TAU=0.5 means "unsafe if at least a majority of raters flagged it". TAU is a
conceptual call: print the slice, look at the boundary, then decide.

REPRODUCIBILITY / DRIFT: `seed` fixes OUR sampling, but not the upstream data. So
every pull PINS a dataset revision (a HuggingFace commit SHA) and writes a small
committable MANIFEST (SHA + a content hash of the rows + n/tau/seed/date). The
manifest is derived metadata, NOT raw text, so it is safe to commit - it is the
receipt that a later pull is the same upstream data. A re-pull with identical
params but a different SHA or content hash prints a loud drift warning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ft_cm.config import REAL_DATA_DIR
from ft_cm.taxonomy import LABELS

DATASET_ID = "google/civil_comments"
MANIFEST_DIR = Path("evidence/dataset")  # committable receipts (no raw text)

# binarize emits these two literals; keep it in step with the taxonomy SSOT.
assert {"safe", "unsafe"} <= set(LABELS)


def binarize(toxicity: float, tau: float) -> str:
    """Collapse the annotator-vote fraction to one label: unsafe iff at least a
    fraction `tau` of raters flagged the comment (a >= boundary), else safe."""
    return "unsafe" if toxicity >= tau else "safe"


def content_hash(rows: list[dict]) -> str:
    """A stable sha256 over the pulled rows (text+label+toxicity), order-independent
    so the same slice hashes identically regardless of row order. This is the
    fingerprint a re-pull compares against to detect upstream drift."""
    canon = sorted(
        json.dumps(
            {"text": r["text"], "label": r["label"], "toxicity": r["toxicity"]},
            sort_keys=True,
            ensure_ascii=False,
        )
        for r in rows
    )
    return hashlib.sha256("\n".join(canon).encode("utf-8")).hexdigest()


def resolve_revision(revision: str | None) -> str:
    """Return the commit SHA to pin. If given, use it verbatim (reproduce a prior
    pull); else resolve the dataset's current HEAD SHA and record it, so future
    pulls can pin to exactly this snapshot."""
    if revision:
        return revision
    from huggingface_hub import HfApi  # ships with `datasets`; deferred import

    return HfApi().dataset_info(DATASET_ID).sha


def fetch_balanced(
    n: int, tau: float, seed: int, split: str, buffer_size: int, revision: str
) -> list[dict]:
    """Stream the dataset (pinned to `revision`) and collect n//2 rows per class,
    deduped on exact text. Deterministic under `seed` (reservoir shuffle over a
    `buffer_size` window). Raw text is not length-filtered here - that stays honest;
    length stats are reported so step (e) can watch sequence length."""
    from datasets import load_dataset  # heavy import, deferred so `import ft_cm` stays light

    per_class = n // 2
    ds = load_dataset(
        DATASET_ID, split=split, streaming=True, revision=revision
    ).shuffle(seed=seed, buffer_size=buffer_size)
    buckets: dict[str, list[dict]] = {"safe": [], "unsafe": []}
    seen: set[str] = set()
    for row in ds:
        text = (row["text"] or "").strip()
        if not text or text.lower() in seen:
            continue
        label = binarize(float(row["toxicity"]), tau)
        if len(buckets[label]) >= per_class:
            continue
        seen.add(text.lower())
        buckets[label].append(
            {"text": text, "label": label, "toxicity": round(float(row["toxicity"]), 6)}
        )
        if len(buckets["safe"]) >= per_class and len(buckets["unsafe"]) >= per_class:
            break
    return buckets["safe"] + buckets["unsafe"]


def check_drift(manifest_path: Path, new: dict) -> None:
    """If a manifest already exists for the same pull params, compare fingerprints
    and shout if the upstream revision or the content hash moved (that is drift);
    confirm reproducibility if both match."""
    if not manifest_path.exists():
        return
    old = json.loads(manifest_path.read_text())
    if not all(old.get(k) == new.get(k) for k in ("dataset_id", "split", "tau", "seed", "n")):
        return  # different slice params - not a drift comparison
    if old.get("revision") != new.get("revision"):
        print(
            f"[download] *** UPSTREAM REVISION CHANGED *** {old.get('revision')} -> "
            f"{new.get('revision')} (pin --revision to reproduce the earlier snapshot)"
        )
    if old.get("content_sha256") != new.get("content_sha256"):
        print("[download] *** CONTENT DRIFT *** identical pull params, different content hash")
    elif old.get("revision") == new.get("revision"):
        print("[download] reproducibility OK: same revision + content hash as the prior manifest")


def _print_eyeball(rows: list[dict]) -> None:
    """Print every row, grouped by label and sorted by toxicity, so the boundary
    (rows just under vs just over TAU) is readable at a glance."""
    for label in ("safe", "unsafe"):
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: r["toxicity"])
        print(f"\n--- {label.upper()} ({len(group)} rows, toxicity sorted) ---")
        for r in group:
            snippet = " ".join(r["text"].split())
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            print(f"  {r['toxicity']:.3f}  {snippet}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull a tiny balanced Civil Comments slice.")
    ap.add_argument("--n", type=int, default=24, help="total rows, split evenly per class")
    ap.add_argument("--tau", type=float, default=0.5, help="binarize threshold on toxicity")
    ap.add_argument("--seed", type=int, default=0, help="deterministic sample")
    ap.add_argument("--split", default="test", help="source split (test is smallest)")
    ap.add_argument("--buffer-size", type=int, default=10000, help="reservoir shuffle window")
    ap.add_argument(
        "--revision",
        default=None,
        help="pin a dataset commit SHA (default: resolve + record the current HEAD)",
    )
    ap.add_argument(
        "--out",
        default=str(REAL_DATA_DIR / "civil-comments-raw.jsonl"),
        help="output jsonl (must stay under data/real/ - git-ignored)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="manifest path (default: evidence/dataset/<out-stem>.manifest.json, committable)",
    )
    args = ap.parse_args()

    revision = resolve_revision(args.revision)
    rows = fetch_balanced(args.n, args.tau, args.seed, args.split, args.buffer_size, revision)

    out = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_DIR / f"{out.stem}.manifest.json"

    counts = {label: sum(r["label"] == label for r in rows) for label in ("safe", "unsafe")}
    lengths = sorted(len(r["text"]) for r in rows)
    retrieved = datetime.now(UTC).date().isoformat()
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "split": args.split,
        "tau": args.tau,
        "seed": args.seed,
        "n": args.n,
        "buffer_size": args.buffer_size,
        "retrieved": retrieved,
        "class_balance": counts,
        "row_count": len(rows),
        "content_sha256": content_hash(rows),
    }

    check_drift(manifest_path, manifest)  # compare BEFORE overwriting the old manifest

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[download] {DATASET_ID}@{revision[:12]} split={args.split} seed={args.seed} tau={args.tau}")
    print(f"[download] retrieved {retrieved} -> {out} (git-ignored, raw text NEVER committed)")
    print(f"[download] manifest -> {manifest_path} (committable receipt: SHA + content hash)")
    print(f"[download] content_sha256 {manifest['content_sha256']}")
    print(f"[download] class balance: {counts}  (balanced by design; unsafe is rare in the corpus)")
    if lengths:
        print(
            f"[download] text length chars: min={lengths[0]} "
            f"median={lengths[len(lengths) // 2]} max={lengths[-1]}"
        )
    _print_eyeball(rows)
    print(
        "\n[download] eyeball the boundary: rows just under TAU are 'safe', just over are"
        " 'unsafe'. If the line sits in the wrong place, re-run with a different --tau."
    )
    if counts["safe"] < args.n // 2 or counts["unsafe"] < args.n // 2:
        print("[download] WARNING: a class did not fill - stream exhausted or buffer too small.")


if __name__ == "__main__":
    main()
