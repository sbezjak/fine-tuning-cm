#!/usr/bin/env bash
# Pull a TINY balanced slice of the REAL moderation dataset into data/real/.
#
#   scripts/download-dataset.sh                 # defaults: 24 rows, TAU 0.5, seed 0
#   scripts/download-dataset.sh --n 30 --tau 0.3
#
# TARGETS google/civil_comments (CC0, no login) with DuckDB over the HF parquet
# (WHERE subscore >= TAU per ground - predicate pushdown, not streamed), labels each
# row with its harm GROUND via the taxonomy recipe, and writes a class-balanced
# {text,label,toxicity,<sub-scores>} jsonl under data/real/ (git-ignored).
#
# Hard rules (PUBLIC repo, content-moderation domain):
#   1. Output stays under data/real/ (git-ignored). Raw text is NEVER committed.
#   2. Source + license are cited in docs/dataset.md (CC0) before first use.
#   3. Only DERIVED labels/metrics are ever published, not the corpus.
#
# Start TINY so every row can be eyeballed; scale only after the slice is reviewed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/..")" || exit 1

# Safety net beyond .gitignore: refuse to run if data/real/ is somehow tracked.
if git ls-files --error-unmatch data/real >/dev/null 2>&1; then
  echo "ABORT: data/real/ is tracked by git - raw text must stay git-ignored." >&2
  exit 1
fi

echo "[download] pulling a tiny balanced grounded slice (data group installs on first run)"
uv run --group data python -m ft_cm.download_data "$@"
