"""Before/after evaluation on the held-out slice.

The whole point of the project lands here: run the SAME held-out rows through the
base model and the LoRA-tuned model and report what moved. Accuracy is the
headline; the calibration fields (per-label accuracy, the confusion counts, and
how many completions were non-answers or hedges) are what let you say WHERE the
fine-tune generalized versus where it just learned to emit the majority label.

`evaluate` is backend-agnostic: it takes any Provider, so the same function scores
the base MLX model, the tuned MLX model, or a live Ollama sanity check.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ft_cm.providers.base import Provider
from ft_cm.scorer import extract_label
from ft_cm.taxonomy import LABELS, SYSTEM_PROMPT, build_prompt


@dataclass
class RowResult:
    text: str
    gold: str
    pred: str | None
    raw: str
    ok: bool
    ambiguous: bool


@dataclass
class EvalResult:
    n: int
    correct: int
    accuracy: float
    per_label_accuracy: dict[str, float]
    confusion: dict[str, dict[str, int]]  # gold -> {pred_or_"none": count}
    n_none: int  # completions with no extractable label
    n_ambiguous: int  # completions that hedged (>1 label)
    rows: list[RowResult] = field(default_factory=list)


def load_holdout(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def evaluate(provider: Provider, holdout: list[dict]) -> EvalResult:
    per_label_total: dict[str, int] = {label: 0 for label in LABELS}
    per_label_correct: dict[str, int] = {label: 0 for label in LABELS}
    confusion: dict[str, dict[str, int]] = {g: {} for g in LABELS}
    rows: list[RowResult] = []
    n_none = n_ambiguous = correct = 0

    for item in holdout:
        gold = item["label"]
        raw = await provider.generate(build_prompt(item["text"]))
        ext = extract_label(raw)
        pred = ext.label
        ok = pred is not None and not ext.ambiguous and pred == gold

        per_label_total[gold] += 1
        if ok:
            correct += 1
            per_label_correct[gold] += 1
        if pred is None:
            n_none += 1
        if ext.ambiguous:
            n_ambiguous += 1

        key = pred if pred is not None else "none"
        confusion[gold][key] = confusion[gold].get(key, 0) + 1
        rows.append(RowResult(item["text"], gold, pred, raw, ok, ext.ambiguous))

    n = len(holdout)
    per_label_accuracy = {
        label: (
            per_label_correct[label] / per_label_total[label] if per_label_total[label] else 0.0
        )
        for label in LABELS
    }
    return EvalResult(
        n=n,
        correct=correct,
        accuracy=correct / n if n else 0.0,
        per_label_accuracy=per_label_accuracy,
        confusion=confusion,
        n_none=n_none,
        n_ambiguous=n_ambiguous,
        rows=rows,
    )


def _summary(tag: str, r: EvalResult) -> str:
    pl = ", ".join(f"{k}={v:.2f}" for k, v in r.per_label_accuracy.items())
    return (
        f"[{tag}] acc={r.accuracy:.3f} ({r.correct}/{r.n})  per-label: {pl}  "
        f"non-answers={r.n_none}  hedged={r.n_ambiguous}"
    )


def _print_rows(tag: str, r: EvalResult) -> None:
    """Print every held-out row with the model's RAW completion. The aggregate
    numbers say how much moved; only the raw text says WHERE and whether a 'safe'
    was a real judgment or a hedge/non-answer that happened to contain the word.
    Always on: a report without the raw replies hides the ground truth."""
    print(f"\n--- {tag} rows (raw completions) ---")
    for row in r.rows:
        mark = "ok" if row.ok else "XX"
        print(f"  {mark}  gold={row.gold:<6} pred={row.pred!s:<6} | {row.raw!r}")


def _strip_rows(result: dict) -> dict:
    """Return a copy with the per-row raw text removed - the derived-metrics-only
    view that is safe to commit for the REAL dataset (raw toxic comments never leave
    data/real/). Accuracy, per-label, the confusion COUNTS, and the non-answer/hedge
    tallies all survive; only the raw comment text + raw completions are dropped."""
    slim = copy.deepcopy(result)
    for key in ("baseline", "before", "after"):
        if isinstance(slim.get(key), dict):
            slim[key].pop("rows", None)
    return slim


async def baseline(
    model: str,
    holdout_path: str,
    receipts_path: str | None = None,
    metrics_path: str | None = None,
) -> dict:
    """Run ONLY the base model (no adapter) on the held-out slice - the honest
    starting number BEFORE any training (the 'before' half on its own, for Phase 2's
    baseline read when no adapter exists yet). Writes a full receipt (with raw rows,
    keep git-ignored) and/or a metrics-only receipt (no text, committable)."""
    from ft_cm.providers.mlx_provider import MLXProvider

    holdout = load_holdout(holdout_path)
    base = MLXProvider(model=model, adapter_path=None, system=SYSTEM_PROMPT)
    ev = await evaluate(base, holdout)

    print(_summary("baseline", ev))
    print(f"[grid] confusion gold->pred: {json.dumps(ev.confusion)}")
    _print_rows("baseline", ev)

    result = {
        "model": model,
        "adapter_path": None,
        "holdout": holdout_path,
        "n": ev.n,
        "baseline": asdict(ev),
    }
    if receipts_path:
        rp = Path(receipts_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(result, indent=2))
        print(f"[receipts:full] -> {rp} (contains raw comment text - keep git-ignored)")
    if metrics_path:
        mp = Path(metrics_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(_strip_rows(result), indent=2))
        print(f"[receipts:metrics] -> {mp} (derived metrics only, no raw text - committable)")
    return result


def _assert_adapter_present(adapter_path: str) -> None:
    """Fail loudly if the adapter is missing. A typo'd or absent adapter_path makes
    mlx_lm.load silently serve the BASE model, so `tuned` would equal `base` and the
    before/after delta would be a false 0 - indistinguishable from a saturated
    baseline. Checking the files exist before the run turns that silent no-op into an
    error. (It cannot catch an adapter that loads but is inert; the mlx-marked probe
    test does that, by asserting base and tuned differ on a known input.)"""
    d = Path(adapter_path)
    if not d.is_dir():
        raise FileNotFoundError(
            f"adapter_path {adapter_path!r} is not a directory - the tuned run would "
            f"silently be the base model. Train first, or fix the path."
        )
    if not (d / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{adapter_path}/adapter_config.json missing - {adapter_path!r} does not look "
            f"like a trained adapter, so the tuned run would silently be the base model."
        )


async def before_after(
    model: str,
    adapter_path: str,
    holdout_path: str,
    receipts_path: str | None = None,
    metrics_path: str | None = None,
) -> dict:
    """Run base (no adapter) then tuned (with adapter) on the held-out slice."""
    from ft_cm.providers.mlx_provider import MLXProvider

    _assert_adapter_present(adapter_path)
    holdout = load_holdout(holdout_path)
    base = MLXProvider(model=model, adapter_path=None, system=SYSTEM_PROMPT)
    tuned = MLXProvider(model=model, adapter_path=adapter_path, system=SYSTEM_PROMPT)

    before = await evaluate(base, holdout)
    after = await evaluate(tuned, holdout)

    print(_summary("before", before))
    print(_summary("after ", after))
    print(f"[delta] accuracy {after.accuracy - before.accuracy:+.3f}")
    _print_rows("before", before)
    _print_rows("after", after)

    result = {
        "model": model,
        "adapter_path": adapter_path,
        "holdout": holdout_path,
        "n": before.n,
        # Full per-row rows (with the raw completion) are kept, not stripped: the
        # receipt must be recomputable AND readable - the raw reply is the ground
        # truth behind every count, so it always ships in the report.
        "before": asdict(before),
        "after": asdict(after),
        "delta_accuracy": after.accuracy - before.accuracy,
    }
    if receipts_path:
        Path(receipts_path).write_text(json.dumps(result, indent=2))
        print(f"[receipts:full] -> {receipts_path} (contains raw comment text - keep git-ignored)")
    if metrics_path:
        mp = Path(metrics_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(_strip_rows(result), indent=2))
        print(f"[receipts:metrics] -> {mp} (derived metrics only, no raw text - committable)")
    return result


if __name__ == "__main__":
    import argparse

    from ft_cm.config import ADAPTER_PATH, BASE_MODEL

    ap = argparse.ArgumentParser(description="Before/after held-out eval of a LoRA adapter.")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default=str(ADAPTER_PATH))
    ap.add_argument("--holdout", default="data/smoke/prepared/holdout.jsonl")
    ap.add_argument("--receipts", default="evidence/smoke-before-after.json")
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="run only the base model (no adapter) - the pre-training baseline",
    )
    ap.add_argument(
        "--metrics",
        default=None,
        help="committable metrics-only receipt path (no raw text); works in both modes",
    )
    args = ap.parse_args()
    if args.baseline_only:
        asyncio.run(baseline(args.model, args.holdout, args.receipts, args.metrics))
    else:
        asyncio.run(
            before_after(args.model, args.adapter, args.holdout, args.receipts, args.metrics)
        )
