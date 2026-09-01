#!/usr/bin/env bash
# Download the REAL moderation dataset into data/real/ (git-ignored).
#
# STUB until the source is chosen (a human-task: source + license + citation).
# This is deliberately empty of a hardcoded URL so no toxic dataset gets pulled or
# committed by accident. When the source is picked, fill in the fetch below and
# record the source + license in docs/dataset.md.
#
# Hard rules for this repo (PUBLIC, content-moderation domain):
#   1. Output goes ONLY under data/real/ (git-ignored). Never commit raw text.
#   2. Cite the source and license in docs/dataset.md before first use.
#   3. Prefer a dataset whose license permits research use and redistribution of
#      DERIVED labels/metrics (which is all this repo publishes), not the raw text.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0")/.." 2>/dev/null || true
mkdir -p data/real

echo "download-dataset.sh is a stub."
echo "Pick a source + license first (see docs/dataset.md), then wire the fetch here."
echo "Candidate sources to evaluate (check each license before use):"
echo "  - Jigsaw Toxic Comment Classification (Kaggle)"
echo "  - OpenAI / other public moderation eval sets"
echo "  - HatEval / OLID / Civil Comments (research licenses vary)"
exit 1
