# fine-tuning-cm

**Teaching a small model to flag unsafe messages, for content moderation.** A learning project for the fine-tuning toolchain: prepare data, train a small add-on to the model (a LoRA adapter), and measure whether it actually helped. Built end to end on an 8 GB laptop. The point is the method, how to run a fine-tune and check the result honestly, not to reach an impressive score.

It generates no harmful content and publishes no toxic training text. Real data is downloaded locally and kept out of version control; only a small set of tame, made-up examples is committed, just to show the loop runs. See [Data handling](#data-handling).

Full write-up: [`docs/walkthrough.md`](docs/walkthrough.md).

## The task

Given a message, the model outputs one word: `safe` or `unsafe`. Because the answer is one word from a fixed set, grading is a plain exact-match check, not a second model acting as a judge. The real examples come from [Civil Comments](docs/dataset.md) (public news-site comments, CC0); a comment counts as `unsafe` when at least 70% of reviewers flagged it toxic. Only counts and labels are published, never the comments.

## The result

Two readings, because they measure different things.

**Made-up smoke data shows the loop works.** On a small hand-built test set the adapter clearly changes the model's decisions, `0.75 -> 0.958` ([receipt](evidence/hard-1.5b-v2-before-after.json), [report](reports/hard-1.5b-v2-before-after.html)). This only shows the machinery runs; made-up examples cannot measure real moderation quality.

**Real data is the honest result.** On a 90-message held-out set from Civil Comments:

| model | safe caught | unsafe caught | real threats let through | overall |
|---|---|---|---|---|
| base (before) | 0.11 | 0.82 | 10 | 0.533 |
| tuned (after) | 0.89 | 0.72 | 15 | 0.789 |

Overall accuracy jumps from `0.533` to `0.789`, and the tuned model stops over-flagging safe speech (`0.11 -> 0.89` caught). But the safety-critical number moved the wrong way: of 54 genuinely unsafe messages it now lets 15 through, up from 10. It traded catching threats for raising fewer false alarms. Receipt: [real safe/unsafe run](evidence/dataset/real-binary-it200.metrics.json).

The habit the whole project is about: **read the full breakdown, not the single score.** The `0.789` looks solid until you split it, real label down the side, model's guess across the top:

| actual \ guessed | safe | unsafe |
|---|---|---|
| **safe** (36) | 32 | 4 |
| **unsafe** (54) | 15 | 39 |

The bottom row is the one a moderation model can least afford: 15 of 54 unsafe messages called safe and let through. A single score hides that; the grid shows it. (And don't trust a before/after until base and tuned run through the exact same setup on the exact same frozen test set.) What helped most was cleaner labels and better-chosen training examples, the full story is in the [walkthrough](docs/walkthrough.md).

## Method, and the 8 GB limit

Built for an **Apple M1 MacBook Air, 8 GB**. Training uses **MLX-LM LoRA** (the Apple-Silicon toolchain) on a 4-bit `Qwen2.5-1.5B-Instruct`. The limit shapes every choice, small batches, short inputs, few trained layers, but in the end it barely bit: a full run used about 1.7 GB, so renting a cloud GPU was not needed. The same training also runs on a **free Google Colab T4** and produced an adapter that behaved the same way, so the method does not depend on the expensive hardware.

## Running it

The smoke path runs the whole thing on made-up data in about 5 minutes, for free:

```bash
uv sync                          # test + eval tools
uv run pytest -m mocked          # fast unit tests, no model or network

scripts/train-smoke.sh           # prepare data + train the small smoke adapter
uv run python -m ft_cm.eval      # before/after on the held-out smoke set
uv run python -m ft_cm.report evidence/smoke-before-after.json   # readable HTML report
```

The real safe/unsafe pipeline on Civil Comments (downloads the base model and dataset on first use, takes longer):

```bash
scripts/download-dataset.sh                       # small balanced slice into data/real/ (git-ignored)
uv run python -m ft_cm.data_prep \
  data/real/civil-comments-raw.jsonl data/real/prepared --seed 0   # frozen train/valid/test/holdout split

FT_CM_TASK=binary uv run python -m ft_cm.relabel_binary            # map the labels to safe/unsafe
FT_CM_TASK=binary uv run --group train mlx_lm.lora --config configs/lora-real-binary.yaml

FT_CM_TASK=binary uv run python -m ft_cm.eval --adapter adapters/real-binary \
  --holdout data/real/prepared-binary/holdout.jsonl \
  --receipts data/real/eval/real-binary.json \
  --metrics evidence/dataset/real-binary.metrics.json            # before/after
```

By default the pipeline predicts the harm *type* (five grounds); `FT_CM_TASK=binary` selects the plain safe/unsafe run. The [walkthrough](docs/walkthrough.md) covers the free-Colab cloud path, checkpoint choice, the harm-type version, and why the cloud numbers differ from the Mac.

## Layout

```
ft_cm/         classifier package
  taxonomy.py    the label set + how the prompt is framed (one source of truth)
  data_prep.py   raw {text,label} -> chat-format train/valid/test + held-out set
  scorer.py      pulls the one-word label out of the reply (whole-word match)
  eval.py        before/after accuracy on the held-out set + a full breakdown
  report.py      turns a receipt into one self-contained HTML report
  providers/     the model backends: MLX (base + tuned) and Ollama (quick sanity)
configs/       MLX-LM LoRA settings (tuned for 8 GB)
data/smoke/    made-up, tame examples (committed); data/real/ is kept out of git
evidence/      the receipts (JSON) behind every number above
```

## Data handling

This is a public repository about content moderation, so toxic content is handled carefully:

- Raw and real data live only under `data/real/` and are **kept out of version control**, never committed.
- Only `data/smoke/` (hand-written, tame, made-up) is committed, and only to show the pipeline.
- The real data's source and license are in [`docs/dataset.md`](docs/dataset.md); the repository publishes only labels and numbers, never the raw comments.
