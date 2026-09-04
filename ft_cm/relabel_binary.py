"""Relabel the frozen GROUNDS split onto the binary safe/unsafe verdict axis.

The harm-binary ablation asks: was the lift from the RELABEL (off Civil Comments
sub-scores, insult -> safe) or from predicting the multi-class GROUND? The only fair
comparison is the SAME rows and SAME split trained to name the verdict directly - so
this does NOT re-run data_prep (which would re-stratify by 2 classes and move rows).
It maps the existing data/real/prepared/ rows ground -> verdict in place:

- chat rows (train/valid/test.jsonl): swap the system message to the binary
  SYSTEM_PROMPT and the assistant target to taxonomy.verdict(ground); user untouched
  (build_prompt is task-independent, so the input text is byte-identical across runs).
- raw holdout.jsonl ({text, label}): map label -> verdict(label), text untouched.

Output goes to data/real/prepared-binary/ (git-ignored). Run under FT_CM_TASK=binary
so SYSTEM_PROMPT is the binary framing; the script refuses otherwise.

    FT_CM_TASK=binary uv run python -m ft_cm.relabel_binary
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from ft_cm import taxonomy

SRC = Path("data/real/prepared")
DST = Path("data/real/prepared-binary")
CHAT_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl")
RAW_FILES = ("holdout.jsonl",)


def _relabel_chat(row: dict) -> tuple[dict, str]:
    """Rewrite one chat row's system + assistant; return (new_row, ground) for tally."""
    ground = None
    out = []
    for msg in row["messages"]:
        if msg["role"] == "system":
            out.append({"role": "system", "content": taxonomy.SYSTEM_PROMPT})
        elif msg["role"] == "assistant":
            ground = msg["content"]
            if ground not in taxonomy.GROUNDS:
                raise ValueError(f"assistant target {ground!r} not a known ground")
            out.append({"role": "assistant", "content": taxonomy.verdict(ground)})
        else:
            out.append(msg)
    if ground is None:
        raise ValueError("chat row had no assistant message")
    return {"messages": out}, ground


def _relabel_raw(row: dict) -> tuple[dict, str]:
    ground = row["label"]
    if ground not in taxonomy.GROUNDS:
        raise ValueError(f"holdout label {ground!r} not a known ground")
    return {"text": row["text"], "label": taxonomy.verdict(ground)}, ground


def main() -> None:
    if taxonomy.TASK != "binary":
        sys.exit("refusing: run under FT_CM_TASK=binary (else the grounds prompt leaks in)")
    DST.mkdir(parents=True, exist_ok=True)
    for name in CHAT_FILES + RAW_FILES:
        src = SRC / name
        rows = [json.loads(line) for line in src.read_text().splitlines() if line]
        relabel = _relabel_chat if name in CHAT_FILES else _relabel_raw
        out_rows, grounds = [], []
        for row in rows:
            new_row, ground = relabel(row)
            out_rows.append(new_row)
            grounds.append(ground)
        (DST / name).write_text("".join(json.dumps(r) + "\n" for r in out_rows))
        # tally the ground -> verdict remap so the mapping is eyeballable, not on faith.
        by_verdict = collections.Counter(taxonomy.verdict(g) for g in grounds)
        by_ground = collections.Counter(grounds)
        print(f"{name}: {len(out_rows)} rows  verdicts={dict(by_verdict)}")
        print(f"    grounds={dict(by_ground)}")


if __name__ == "__main__":
    main()
