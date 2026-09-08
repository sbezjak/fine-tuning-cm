# Fine-tuning a content-moderation model on a laptop is easy. Trusting the result is not.

#ai #finetuning #python #machinelearning

I fine-tuned a small language model to sort messages into `safe` or `unsafe`, the kind of check a platform runs before letting a comment through. The whole thing runs on an 8 GB laptop, in a few minutes, for free.

The result was not a clean win, and that was the point: the overall score went up while the model got worse at what mattered most. What I came away with was the method - how to run a fine-tune, and how to tell whether it actually helped.

A learning project, written up for anyone who wants to try fine-tuning without a big machine or a big budget.

Repo: https://github.com/sbezjak/fine-tuning-cm

## First: should you even fine-tune?

Most of the time, the answer is no. If a better prompt gets you there, do that. If the model just needs facts it does not have, give it those at question time (that is what retrieval, or RAG, is for).

Fine-tuning earns its place in a narrower spot: one repeated, well-defined task you want done the same way every time, without a long prompt. Sorting messages into two labels is exactly that shape, so it is a good thing to learn on.

## The task, and why grading is easy

The model gets a message and answers with one word: `safe` or `unsafe`. Because that answer is one word from a fixed set, I can grade it with a plain string check, no second model acting as a judge and no scoring rubric. That keeps the scoring objective and the whole project free to run.

One catch: `safe` is contained inside `unsafe`, so a check that just looks for "safe" in the reply would mark every "unsafe" answer correct. Matching whole words only is a one-line fix, and the kind of thing that silently corrupts your numbers if you miss it.

## The data

The real training examples come from [Civil Comments](https://huggingface.co/datasets/google/civil_comments), a public set of reader comments from news sites. Each comment has a toxicity number, which is simply the share of human reviewers who marked it toxic. I turn that into a single label, calling a comment `unsafe` when at least 70% of reviewers flagged it.

The comments are real and some are genuinely toxic, so they never go into the public repo. A small download script pulls a tiny, class-balanced slice into a git-ignored folder (it is CC0, public domain; the source is cited in `docs/dataset.md`). Only the derived labels and the final numbers are ever committed, never the text itself: in a sensitive domain, you publish what you measured, not the corpus you measured it on.

## What a fine-tune actually changes

I did not retrain the model. That would need far more memory than a laptop has.

Instead I trained a small add-on called a LoRA adapter. The original model - a small open one, Qwen2.5-1.5B-Instruct - has more than a billion numbers inside it, and they all stay frozen. The adapter is a tiny extra set of numbers, trained from scratch, that sits on top and nudges the model's answers. In this project the adapter was about 0.17% of the model's size, a file of roughly 10 MB.

That is the fact that made fine-tuning click for me: I am not rewriting the model, I am training a small, cheap correction that clips onto it. Take the clip off and you have the original model back. Put it on and you have the tuned one. That is why adapters are so easy to store, swap, and compare.

## Training, and knowing when to stop

I used Apple's MLX toolchain, which trains on the Mac's own chip, and a small model kept in a compact 4-bit form to save memory. Every setting is shaped by the 8 GB limit: small batches, short inputs, few trained layers.

Training took about nine minutes and peaked at roughly 1.8 GB of memory. The 8 GB limit barely bit at this size, which was itself worth learning: the ceiling I planned around was not the one that mattered.

The important lesson was about when to stop. As training runs, you watch two numbers. One is how well the model fits the examples it is training on (the training loss). The other is how well it does on a separate set it never trains on (the validation loss). The training loss kept dropping. The validation loss dropped, hit a low point, and then started climbing again. That climb is the model starting to memorize the training examples instead of learning the general pattern, which is called overfitting. So the best adapter was not the last one saved: I trained for 400 steps but kept the checkpoint from step 200, where the validation loss bottomed out, right in the middle of the run.

## How to trust a before-and-after

This is the part the whole project is really about. The test set is "held out": a batch of labelled messages set aside at the start and never shown to the model during training, so grading it is a fair check on messages it has not already seen. Training and measuring are not two phases, they are one loop: you never train and just trust the result, you train and immediately re-run that same held-out test to see whether it actually helped.

To measure whether the adapter helped, I run the held-out messages through the model twice: once without the adapter (the "before"), once with it (the "after"). Everything else is identical. Same messages, same settings, and the model is set to be fully repeatable, so the same message always gives the same answer. That way any difference in the score is caused by the adapter and nothing else.

There is a quiet trap here. If you point the code at the wrong adapter folder, it just loads the plain model and reports a difference of zero, and you would never know the fine-tune did nothing. So the code now checks that the adapter is really there and stops loudly if it is not. A before-and-after you cannot trust is worse than no measurement at all.

## The result, and the trap in it

First, how to read these numbers, because the direction flips from the last section. Training and validation loss are errors, so lower is better. Everything in the table below is either an accuracy (the share the model got right, from 0 to 1, so higher is better) or a plain count of mistakes (lower is better). So a rising score is good news and more threats let through is bad news, even though both numbers go up.

I ran the whole loop first on a smaller 0.5B model with fabricated data, just to prove it worked end to end. That is not a result, so the numbers here are the real run: the 1.5B model, trained and measured on held-out Civil Comments.

The starting model over-flags badly. It catches only 11% of the safe messages, calling almost everything unsafe, and on 13 of the 90 it does not answer at all: instead of a label it falls back to a stock phrase ("contains a threat of violence"), because the untuned model was never taught to reply with one word from a fixed set. Training fixes that, and the non-answers drop to zero. But the interesting part is what it trades to get there.

| model | safe caught | unsafe caught | real threats let through | overall |
|---|---|---|---|---|
| base (before) | 0.11 | 0.82 | 10 | 0.533 |
| tuned (after) | 0.89 | 0.72 | 15 | 0.789 |

Read the overall column and the fine-tune is a clear win: 0.533 up to 0.789, and it almost stops over-flagging safe speech (11% caught up to 89%). But read across the row, not down that one column. Of 54 genuinely unsafe messages, the tuned model now lets 15 through, up from 10. It bought its lower false-alarm rate by catching fewer real threats, and letting a threat through is the one error a moderation model can least afford.

Two things did move it in the right direction. The first was cleaner labels: at the bare 50% cutoff the unsafe pile filled up with genuinely civil comments that only a couple of raters happened to flag, so raising the bar to 70% agreement dropped that noise and pointed "unsafe" at real hostility. The second was better-chosen training examples, aimed at the exact cases the model kept getting wrong. I also tried teaching the model to name the *type* of harm rather than just safe/unsafe, which helped a little; that is a thread for a future project, not this one.

To check the method did not depend on my particular laptop, I ran the same training on a free Google Colab T4 GPU. It behaved the same way, down to the same over-training climb: the recipe is about the method, not the hardware.

## What real systems do that this one doesn't

Worth being clear about the gap between this and production.

When a big lab ships a new model, it trains all of its numbers across huge data and hardware. The LoRA approach I used is the common choice for the other job: adapting an existing model to a narrow task cheaply, and keeping many small adapters over one shared base.

Real moderation systems also do things this one skips: they read a confidence score and send the uncertain cases to a human instead of forcing a yes/no; they weigh a missed threat as more costly than an over-flag; they test against much larger, carefully labeled sets. Those are the next steps, named here rather than built.

## Try it yourself

Start with the "smoke" path. It trains on the tiny set of made-up examples that ship with the repo, so it does not measure real moderation, but it turns the whole loop: preparing the data, training the adapter, and measuring the before-and-after. The point is to feel the machinery run end to end, not to get a real number:

```bash
uv sync
scripts/train-smoke.sh                                    # prepare + train on the fabricated set
uv run python -m ft_cm.eval                               # before/after on held-out examples
uv run python -m ft_cm.report evidence/smoke-before-after.json
```

To get a real result, point it at the real data instead. `scripts/download-dataset.sh` pulls a small, class-balanced slice of Civil Comments into the git-ignored `data/real/` folder (no login, CC0 licensed; provenance is in `docs/dataset.md`), and you train and evaluate on that the same way. The real text stays on your machine and never gets committed.

Either way, the habit is the same one from earlier: change something, retrain, re-run the before-and-after, and read the whole breakdown rather than the headline score. That is the fastest way to feel what a fine-tune does, and what it does not.

Repo: https://github.com/sbezjak/fine-tuning-cm
