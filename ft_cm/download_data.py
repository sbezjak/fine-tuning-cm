"""Pull a TINY balanced slice of the REAL moderation dataset into data/real/.

Public repo, toxic domain: raw comment text lands ONLY under data/real/
(git-ignored) and is NEVER committed - only derived labels/metrics are published
(see docs/dataset.md). This streams google/civil_comments (CC0, no login needed)
and labels each row with its harm GROUND via the taxonomy recipe (`assign_ground`),
then writes a class-balanced {text, label, toxicity, <sub-scores>} jsonl small
enough to eyeball every row before we scale.

GROUNDS, not a binary. `label` is now one of the taxonomy GROUNDS
(threat / identity_attack / sexual / insult / safe), assigned deterministically
from the harm sub-scores at threshold TAU by the priority ladder in taxonomy.py -
no hand-labeling, no judge. safe/unsafe DERIVES from the ground (taxonomy.verdict).
The sub-scores are carried into each row so the boundary stays eyeball-able and the
consistency test can be built from real rows.

`tau` is the annotator-vote fraction boundary (0.5 = a majority of raters agreed);
it is the one conceptual dial - print the slice, look at the boundary, then decide.

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
from ft_cm.taxonomy import LABELS, SUBSCORE_COLUMNS, assign_ground, verdict

DATASET_ID = "google/civil_comments"
MANIFEST_DIR = Path("evidence/dataset")  # committable receipts (no raw text)


def content_hash(rows: list[dict]) -> str:
    """A stable sha256 over the pulled rows (text + label + sub-scores), order-
    independent so the same slice hashes identically regardless of row order. This is
    the fingerprint a re-pull compares against to detect upstream drift."""
    keep = ("text", "label", "toxicity", *SUBSCORE_COLUMNS)
    canon = sorted(
        json.dumps({k: r[k] for k in keep}, sort_keys=True, ensure_ascii=False) for r in rows
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


def fetch_grounded(
    n: int, tau: float, seed: int, split: str, buffer_size: int, revision: str, max_scan: int
) -> list[dict]:
    """Stream the dataset (pinned to `revision`) and collect a BALANCED slice across
    the GROUNDS: n // len(LABELS) rows per ground, deduped on exact text. Each row is
    labeled by `assign_ground` on its harm sub-scores at TAU. Deterministic under
    `seed` (reservoir shuffle over a `buffer_size` window). `max_scan` caps how many
    rows we stream before giving up on the rare grounds (threat/sexual are <1% of the
    corpus, so filling them balanced needs a long scan)."""
    from datasets import load_dataset  # heavy import, deferred so `import ft_cm` stays light

    per_class = n // len(LABELS)
    ds = load_dataset(
        DATASET_ID, split=split, streaming=True, revision=revision
    ).shuffle(seed=seed, buffer_size=buffer_size)
    buckets: dict[str, list[dict]] = {label: [] for label in LABELS}
    seen: set[str] = set()
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max_scan:
            break
        text = (row["text"] or "").strip()
        if not text or text.lower() in seen:
            continue
        scores = {col: float(row[col]) for col in SUBSCORE_COLUMNS}
        label = assign_ground(scores, tau)
        if len(buckets[label]) >= per_class:
            continue
        seen.add(text.lower())
        record = {
            "text": text,
            "label": label,
            "toxicity": round(float(row["toxicity"]), 6),
            **{col: round(scores[col], 6) for col in SUBSCORE_COLUMNS},
        }
        buckets[label].append(record)
        if all(len(buckets[label]) >= per_class for label in LABELS):
            break
    return [r for label in LABELS for r in buckets[label]]


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
    """Print every row grouped by ground, each with its sub-scores, so the boundary
    (which sub-score tripped the ground, and how close the others were) is readable."""
    for label in LABELS:
        group = [r for r in rows if r["label"] == label]
        print(f"\n--- {label.upper()} ({len(group)} rows) -> verdict {verdict(label)} ---")
        for r in group:
            subs = " ".join(f"{c[:4]}={r[c]:.2f}" for c in SUBSCORE_COLUMNS)
            snippet = " ".join(r["text"].split())
            if len(snippet) > 110:
                snippet = snippet[:107] + "..."
            print(f"  [{subs}]  {snippet}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull a tiny balanced grounded Civil Comments slice.")
    ap.add_argument(
        "--n", type=int, default=len(LABELS) * 6, help="total rows, split evenly per ground"
    )
    ap.add_argument("--tau", type=float, default=0.7, help="sub-score threshold for the ground recipe")
    ap.add_argument("--seed", type=int, default=0, help="deterministic sample")
    ap.add_argument("--split", default="test", help="source split (test is smallest)")
    ap.add_argument("--buffer-size", type=int, default=10000, help="reservoir shuffle window")
    ap.add_argument(
        "--max-scan",
        type=int,
        default=200000,
        help="cap rows streamed before giving up on the rare grounds",
    )
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
    rows = fetch_grounded(
        args.n, args.tau, args.seed, args.split, args.buffer_size, revision, args.max_scan
    )

    out = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_DIR / f"{out.stem}.manifest.json"

    counts = {label: sum(r["label"] == label for r in rows) for label in LABELS}
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
        "max_scan": args.max_scan,
        "grounds": list(LABELS),
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
    print(f"[download] ground balance: {counts}  (balanced by design; harm grounds are rare)")
    if lengths:
        print(
            f"[download] text length chars: min={lengths[0]} "
            f"median={lengths[len(lengths) // 2]} max={lengths[-1]}"
        )
    _print_eyeball(rows)
    per_class = args.n // len(LABELS)
    short = {label: c for label, c in counts.items() if c < per_class}
    if short:
        print(
            f"\n[download] WARNING: these grounds did not fill to {per_class}: {short} "
            f"- raise --max-scan/--buffer-size or lower --n (rare grounds need a long scan)."
        )


if __name__ == "__main__":
    main()
