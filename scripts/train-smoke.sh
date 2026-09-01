#!/usr/bin/env bash
# Smoke training run. Prepares the fabricated data, then trains a LoRA adapter,
# with the FULL training stream preserved to evidence/raw-logs/ (never truncated).
#
#   scripts/train-smoke.sh
#
# The log name is timestamped so successive runs never clobber each other's
# evidence. Watch the run in another shell with:
#   tail -f evidence/raw-logs/train-smoke-*.log
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0")/.." 2>/dev/null || true

echo "[train-smoke] preparing data/smoke/smoke.jsonl -> data/smoke/prepared/"
uv run python -m ft_cm.data_prep data/smoke/smoke.jsonl data/smoke/prepared --seed 0

ts=$(date -u +%Y%m%dT%H%M%SZ)
echo "[train-smoke] training LoRA adapter (full log preserved)"
scripts/run-evidence.sh "train-smoke-${ts}" -- \
  uv run --group train mlx_lm.lora --config configs/lora-smoke.yaml
