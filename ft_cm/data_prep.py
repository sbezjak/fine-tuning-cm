"""Turn raw {text, label} records into the files MLX-LM LoRA trains on.

MLX-LM reads `train.jsonl` / `valid.jsonl` / `test.jsonl` from a data directory.
This writes them in chat format, one message list per line:
    system (the moderation framing) -> user (the message) -> assistant (the label)
so the model is trained to emit the label under the exact framing the eval uses.

Alongside those, it writes `holdout.jsonl` in the raw {text, label} shape: the
before/after eval reads THAT, so the harness never has to parse MLX message lines
back into a gold label. The split is deterministic (seeded) and stratified by
label, so "held-out" means the same rows every run and both classes are present
in every split.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ft_cm.taxonomy import LABELS, SYSTEM_PROMPT, build_prompt


def load_records(path: str | Path) -> list[dict]:
    """Read a jsonl of {"text": str, "label": str}, validating labels."""
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "text" not in rec or "label" not in rec:
            raise ValueError(f"{path}:{i}: record needs 'text' and 'label': {rec!r}")
        if rec["label"] not in LABELS:
            raise ValueError(f"{path}:{i}: label {rec['label']!r} not in {LABELS}")
        records.append({"text": rec["text"], "label": rec["label"]})
    return records


def stratified_split(
    records: list[dict], seed: int = 0, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into (train, valid, test), stratified by label so every split holds
    both classes. Deterministic under `seed`."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[rec["label"]].append(rec)

    rng = random.Random(seed)
    train, valid, test = [], [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * ratios[0]))
        n_valid = max(1, round(n * ratios[1])) if n - n_train >= 2 else 0
        train += group[:n_train]
        valid += group[n_train : n_train + n_valid]
        test += group[n_train + n_valid :]
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def to_chat(record: dict) -> dict:
    """One MLX chat-format training line for a record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(record["text"])},
            {"role": "assistant", "content": record["label"]},
        ]
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def prepare(
    src_path: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, int]:
    """Read src, split, and write MLX train/valid/test.jsonl + a raw holdout.jsonl.
    Returns the per-split row counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(src_path)
    train, valid, test = stratified_split(records, seed=seed, ratios=ratios)

    _write_jsonl(out / "train.jsonl", [to_chat(r) for r in train])
    _write_jsonl(out / "valid.jsonl", [to_chat(r) for r in valid])
    _write_jsonl(out / "test.jsonl", [to_chat(r) for r in test])
    _write_jsonl(out / "holdout.jsonl", test)
    return {"train": len(train), "valid": len(valid), "test": len(test)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Prepare MLX-LM LoRA data from raw {text,label} jsonl."
    )
    ap.add_argument("src", help="source jsonl of {text, label}")
    ap.add_argument("out_dir", help="output directory for train/valid/test/holdout.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    counts = prepare(args.src, args.out_dir, seed=args.seed)
    print(f"prepared {counts} -> {args.out_dir}")
