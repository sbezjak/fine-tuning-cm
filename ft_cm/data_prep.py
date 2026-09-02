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
import re
from collections import defaultdict
from pathlib import Path

from ft_cm.taxonomy import LABELS, SYSTEM_PROMPT, build_prompt


def _tokens(text: str) -> set[str]:
    """Normalized word-token set for near-dup comparison (lowercase, word chars)."""
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Token-set overlap in [0,1]. 1.0 = same word set, 0.0 = disjoint."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def dedup_near(records: list[dict], threshold: float) -> list[dict]:
    """Drop rows that are near-duplicates of an already-kept row (token Jaccard >=
    threshold). Exact dupes are removed at download; this catches reposts/boilerplate
    that would leak between train and held-out and inflate the score by memorization.
    Deterministic: keeps the first occurrence in input order. Simple O(n^2) - fine at
    this N; production scale would use MinHash/LSH or embedding similarity instead."""
    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    for rec in records:
        toks = _tokens(rec["text"])
        if any(_jaccard(toks, kt) >= threshold for kt in kept_tokens):
            continue
        kept.append(rec)
        kept_tokens.append(toks)
    return kept


def max_cross_jaccard(a: list[dict], b: list[dict]) -> float:
    """Highest token-Jaccard of any (a, b) pair - the leakage guard: after the split,
    the max between held-out and train must stay under the dedup threshold."""
    a_tokens = [_tokens(r["text"]) for r in a]
    b_tokens = [_tokens(r["text"]) for r in b]
    return max((_jaccard(x, y) for x in a_tokens for y in b_tokens), default=0.0)


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
    dedup_threshold: float | None = None,
) -> dict[str, float]:
    """Read src, optionally near-dup dedup, stratified-split, and write MLX
    train/valid/test.jsonl + a raw holdout.jsonl. Returns per-split counts plus,
    when dedup is on, how many near-dupes were dropped and the max held-out-vs-train
    Jaccard (the leakage guard - must stay under the threshold). `dedup_threshold`
    defaults to None so smoke prep is byte-for-byte unchanged (its hand rows are
    unique); the real prep passes a threshold to catch reposts/boilerplate."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(src_path)
    n_before = len(records)
    if dedup_threshold is not None:
        records = dedup_near(records, dedup_threshold)
    train, valid, test = stratified_split(records, seed=seed, ratios=ratios)

    _write_jsonl(out / "train.jsonl", [to_chat(r) for r in train])
    _write_jsonl(out / "valid.jsonl", [to_chat(r) for r in valid])
    _write_jsonl(out / "test.jsonl", [to_chat(r) for r in test])
    _write_jsonl(out / "holdout.jsonl", test)

    result: dict[str, float] = {"train": len(train), "valid": len(valid), "test": len(test)}
    if dedup_threshold is not None:
        result["dropped_near_dup"] = n_before - len(records)
        # leakage guard: the held-out (test) vs everything it must be unseen against.
        result["max_holdout_train_jaccard"] = round(max_cross_jaccard(test, train + valid), 4)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Prepare MLX-LM LoRA data from raw {text,label} jsonl."
    )
    ap.add_argument("src", help="source jsonl of {text, label}")
    ap.add_argument("out_dir", help="output directory for train/valid/test/holdout.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dedup-threshold",
        type=float,
        default=None,
        help="near-dup Jaccard cutoff (e.g. 0.5 for real data); omit for smoke",
    )
    ap.add_argument(
        "--receipt",
        default=None,
        help="write a committable JSON receipt (counts + leakage guard, no raw text)",
    )
    args = ap.parse_args()
    counts = prepare(
        args.src, args.out_dir, seed=args.seed, dedup_threshold=args.dedup_threshold
    )
    print(f"prepared {counts} -> {args.out_dir}")
    if "max_holdout_train_jaccard" in counts:
        j = counts["max_holdout_train_jaccard"]
        status = "OK (no leakage)" if j < (args.dedup_threshold or 1.0) else "*** LEAK ***"
        print(f"leakage guard: max held-out vs train/valid Jaccard = {j}  -> {status}")
    if args.receipt:
        receipt = {
            "source": str(args.src),
            "seed": args.seed,
            "dedup_threshold": args.dedup_threshold,
            **counts,
        }
        rp = Path(args.receipt)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"receipt -> {rp}")
