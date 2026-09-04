"""Pull a TINY balanced slice of the REAL moderation dataset into data/real/.

Public repo, toxic domain: raw comment text lands ONLY under data/real/
(git-ignored) and is NEVER committed - only derived labels/metrics are published
(see docs/dataset.md). This TARGETS google/civil_comments (CC0, no login) with
DuckDB over the HF-hosted parquet: each ground is a `WHERE subscore >= tau` query
(predicate pushdown), so we pull only the rows we want instead of streaming and
discarding ~99% - the difference that makes the RARE harm grounds (threat, sexual
~0.07% of the corpus) fillable at all. It writes a class-balanced
{text, label, toxicity, <sub-scores>} jsonl small enough to eyeball every row.

GROUNDS, not a binary. `label` is one of the taxonomy GROUNDS
(threat / identity_attack / sexual / insult / safe), assigned by the priority ladder
in taxonomy.py; each per-ground query is priority-EXCLUSIVE (a row that also trips a
more serious ground is left to that ground), so the SQL exactly mirrors
`assign_ground` - which we assert on every pulled row. safe/unsafe DERIVES from the
ground (taxonomy.verdict). The sub-scores are carried into each row so the boundary
stays eyeball-able and the consistency test can be built from real rows.

`tau` (default 0.7) is the annotator-vote fraction boundary - the one conceptual dial.

BALANCED BUT NON-REPRESENTATIVE, on purpose: harm is <0.1% of the real corpus, so a
balanced slice is right for TRAIN/VALID (enough per class to learn) but would OVERSTATE
real-world accuracy if used as the test set. The held-out read stays honest by reporting
per-class + reweighting to the true prevalences (recorded in the manifest).

REPRODUCIBILITY / DRIFT: `seed` fixes OUR sampling (a deterministic hash order), and the
pull PINS a dataset revision (a HuggingFace commit SHA) + writes a small committable
MANIFEST (SHA + a content hash of the rows). A re-pull with identical params but a
different SHA or content hash prints a loud drift warning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ft_cm.config import REAL_DATA_DIR
from ft_cm.taxonomy import GROUND_RECIPE, LABELS, SUBSCORE_COLUMNS, assign_ground, verdict

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
    pull); else resolve the dataset's current HEAD SHA and record it."""
    from huggingface_hub import HfApi  # deferred import

    return revision or HfApi().dataset_info(DATASET_ID).sha


def download_split_parquet(revision: str, split: str) -> list[str]:
    """Download the split's parquet file(s) to the local HF cache (pinned to
    `revision`) and return the LOCAL paths. Querying local files instead of the
    remote URL turns DuckDB's many per-row-group range requests into one cached,
    CDN-backed download - avoids the HTTP 429 rate-limiting that repeated remote
    scans hit, and makes a re-pull instant (the cache is reused). Listing the repo
    (rather than hardcoding filenames) survives a re-shard upstream."""
    from huggingface_hub import HfApi, hf_hub_download  # deferred import

    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=revision)
    matches = sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    if not matches:
        raise ValueError(f"no parquet files for split {split!r} in {DATASET_ID}@{revision[:12]}")
    return [
        hf_hub_download(DATASET_ID, filename=f, repo_type="dataset", revision=revision)
        for f in matches
    ]


def ground_where_clauses(tau: float) -> dict[str, str]:
    """Build one priority-EXCLUSIVE SQL predicate per ground, straight from the SSOT
    recipe: a ground trips when any of its sub-scores >= tau AND no more-serious
    ground's sub-score does; `safe` is none of them. This is `assign_ground` expressed
    in SQL (asserted equal on every pulled row).

    The columns are cast to DOUBLE so the SQL comparison happens in the SAME float64
    space Python uses. Without the cast DuckDB casts the literal `tau` DOWN to the
    column's float32 (0.7 -> 0.69999998), so a value stored as float32-0.7 passes in
    SQL but fails `assign_ground` on the widened float64 - a boundary mislabel the
    per-row assertion caught."""
    def ge(col: str) -> str:
        return f"CAST({col} AS DOUBLE) >= {tau}"

    clauses: dict[str, str] = {}
    higher: list[str] = []
    for ground, cols in GROUND_RECIPE:
        cond = "(" + " OR ".join(ge(c) for c in cols) + ")"
        if higher:
            cond += " AND NOT (" + " OR ".join(ge(c) for c in higher) + ")"
        clauses[ground] = cond
        higher.extend(cols)
    clauses["safe"] = "NOT (" + " OR ".join(ge(c) for c in higher) + ")"
    return clauses


def fetch_grounded(n: int, tau: float, seed: int, split: str, revision: str) -> list[dict]:
    """Query the pinned parquet for n // len(LABELS) rows PER GROUND, deterministic
    under `seed` (a stable hash order), deduped on exact text. Each ground is its own
    predicate-pushdown query, so the rare grounds are pulled directly rather than
    streamed past. Asserts the SQL label matches the SSOT `assign_ground`."""
    import duckdb  # heavy import, deferred so `import ft_cm` stays light

    per_class = n // len(LABELS)
    paths = download_split_parquet(revision, split)
    clauses = ground_where_clauses(tau)
    select_cols = "text, toxicity, " + ", ".join(SUBSCORE_COLUMNS)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")  # keep logs clean (no ascii progress spam)
    src = "read_parquet(" + json.dumps(paths) + ")"  # LOCAL cached parquet, no remote requests

    rows: list[dict] = []
    seen: set[str] = set()
    for label in LABELS:
        q = (
            f"SELECT {select_cols} FROM {src} "
            f"WHERE ({clauses[label]}) AND text IS NOT NULL AND length(trim(text)) > 0 "
            # over-fetch a little so exact-dup drops do not starve the class
            f"ORDER BY hash(text || '|{seed}') LIMIT {per_class + 25}"
        )
        got = 0
        for row in con.execute(q).fetchall():
            text = str(row[0]).strip()
            if not text or text.lower() in seen:
                continue
            scores = {col: float(v) for col, v in zip(SUBSCORE_COLUMNS, row[2:], strict=True)}
            assert assign_ground(scores, tau) == label, f"SQL/recipe mismatch: {label}"
            seen.add(text.lower())
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "toxicity": round(float(row[1]), 6),
                    **{col: round(scores[col], 6) for col in SUBSCORE_COLUMNS},
                }
            )
            got += 1
            if got >= per_class:
                break
    return rows


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
        "--n", type=int, default=len(LABELS) * 120, help="total rows, split evenly per ground"
    )
    ap.add_argument("--tau", type=float, default=0.7, help="sub-score threshold for the ground recipe")
    ap.add_argument("--seed", type=int, default=0, help="deterministic sample")
    ap.add_argument("--split", default="train", help="source parquet split (train has the most rows)")
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
    rows = fetch_grounded(args.n, args.tau, args.seed, args.split, revision)

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
            f"- the corpus may not hold that many at tau={args.tau}; lower --n or --tau."
        )


if __name__ == "__main__":
    main()
