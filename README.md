# fine-tuning-cm

**A defensive content-classification study: fine-tuning a tiny model to flag unsafe messages for moderation.** This repository is a learning project for the fine-tuning toolchain (data preparation, LoRA/SFT, eval-in-the-loop). Its purpose is defensive: training a classifier that helps a platform detect abusive, threatening, or fraudulent content so it can be reviewed or removed.

It does not generate harmful content, and it does not publish any toxic training text. Real moderation datasets are downloaded locally and kept out of version control (see [Data handling](#data-handling)); only a small set of tame, fabricated examples is committed, purely to prove the training loop runs.

## What this is

The task is a single-label classifier: given a message, output exactly one label (`safe` or `unsafe`). Because the answer is one word from a closed set, grading is an exact, deterministic check, not an LLM judge.

The build order is deliberately smallest-thing-first:

1. **Smoke test on fabricated data.** Prove the training loop runs end to end and that a LoRA adapter actually changes held-out behavior, before touching any real dataset.
2. **Swap in a real dataset.** Source, license, and a download script (git-ignored output).
3. **Report the finding.** Before/after accuracy on a held-out slice, plus one calibration observation: where the fine-tune genuinely generalized versus where it just memorized the training distribution.

## Hardware and method

Built for an **Apple M1 MacBook Air, 8 GB unified memory**. Training uses **MLX-LM LoRA** (the Apple-Silicon path) on a **4-bit-quantized tiny base model** (`Qwen2.5-0.5B-Instruct`, with a planned swap to `1.5B` to feel the memory ceiling). The 8 GB budget is the binding constraint: small batch, short sequences, few adapter layers. The LoRA/SFT mechanics are identical at this size, which is all the learning goal needs. The production-versus-pragmatic trade (train on a free cloud T4 if a run outgrows 8 GB, keep the Mac for the eval harness) is recorded in the working notes.

## Layout

```
ft_cm/            classifier package: taxonomy, data prep, scorer, providers, eval
  taxonomy.py       label set + prompt framing (single source of truth)
  data_prep.py      raw {text,label} -> MLX chat-format train/valid/test + holdout
  scorer.py         tier-3 label extraction (exact/substring, word-boundary safe)
  eval.py           before/after held-out accuracy + calibration breakdown
  providers/        Provider seam: MLX (tuned model) and Ollama (local sanity)
configs/          MLX-LM LoRA configs (memory-tuned for 8 GB)
data/smoke/       fabricated, tame examples (committed); prepared/ is generated
data/real/        real dataset (git-ignored, never committed)
scripts/          train wrapper, dataset download stub, log/secret guards
tests/            mocked unit tests (scorer trap, split determinism, eval harness)
```

## Running it

```bash
uv sync                          # eval + test stack
uv run pytest -m mocked          # fast unit tests, no model or network
uv run --group train mlx_lm.lora --help   # confirm the MLX training path

scripts/train-smoke.sh           # prepare data + train the smoke adapter
uv run python -m ft_cm.eval      # before/after on the held-out smoke slice
```

## Data handling

This is a public repository in the content-moderation domain, so toxic content is handled like a security researcher would handle it:

- Raw and real datasets live only under `data/real/` and are **git-ignored**. They are never committed.
- Only `data/smoke/` (hand-written, tame, fabricated) is tracked, and only to demonstrate the pipeline.
- The real dataset's source and license are cited in `docs/dataset.md` before first use, and the repository publishes only derived labels and metrics, not raw text.
