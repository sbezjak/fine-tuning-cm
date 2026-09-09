# Walkthrough

The full record of what this project did, why, and what it found, in plain language. It draws on a private working log; this is the public version, and it aims to describe everything in the repo: every module, every experiment, and every test.

Readable before/after report (made-up data): [reports/hard-1.5b-v2-before-after.html](../reports/hard-1.5b-v2-before-after.html). Every number below traces to a receipt in [`evidence/`](../evidence).

## Why this project exists

I wanted to learn the fine-tuning toolchain end to end: prepare data, train a small add-on to a model, and measure honestly whether it helped. The task is a bounded, defensive one, sorting messages into `safe` or `unsafe` for content moderation, chosen because the answer is a single word from a fixed set, so grading is a plain exact-match check rather than a second model acting as a judge.

The binding constraint is the hardware: an Apple M1 laptop with 8 GB of memory. Every choice below is shaped by that limit, and naming where the limit did and did not bite is part of the finding.

The project has two layers. A **simple core** (a first fine-tune anyone could run and trust) and a set of **deeper experiments** that push on the honest result: a confidence score, calibration, a cloud port, and a relabelling that finally lifts the ceiling. The core is Findings F1-F7; the deeper work is its own section below.

### Vocabulary, translated

| Term in this repo | What it means |
|---|---|
| base model | The starting model, untouched. Here, `Qwen2.5-1.5B-Instruct`. |
| 4-bit / quantized | The model's numbers stored in a compact, low-precision form to save memory. |
| LoRA adapter | A small set of new numbers, trained from scratch, that clips onto the frozen base and nudges its answers. About 0.17% of the model's size (~10 MB here). |
| SFT (supervised fine-tuning) | Training on labeled examples (message -> correct label). What this project does. |
| MLX / MLX-LM | Apple's toolchain for training on the Mac's own chip. |
| PEFT / trl | The equivalent toolchain on the CUDA (NVIDIA) path, used for the cloud run. |
| held-out set | Test messages the model never trains on. The only honest place to measure quality. |
| before / after | The same held-out set run through the base model, then the tuned model. |
| step | One training pass: the model guesses a label, the guess is compared to the correct one, and every adapter number is nudged a tiny amount toward the right answer. Hundreds run in sequence; the learning is their accumulation. |
| checkpoint | A saved copy of the adapter partway through training. |
| val (validation) loss | How well the adapter does on messages it is not training on, during the run. |
| greedy / temperature 0 | The model always picks its most likely next word, so the same input gives the same output. |
| confusion grid | A small table of actual label vs the model's guess. Shows *which* errors happen. |
| decision-token score | The model's probability for `unsafe` vs `safe`, read directly, giving a 0-1 confidence rather than one word. |
| calibration / ECE | Whether a stated probability is trustworthy (a "0.7" is right about 70% of the time). ECE measures the gap. |
| ground | The *kind* of harm (threat, identity attack, sexual, insult, safe), a finer label than safe/unsafe. |
| tau | The agreement bar for calling a comment unsafe: the share of human reviewers who flagged it (0.7 here). |

## TL;DR (5 minutes)

**What this is.** A complete, small fine-tune: data prep, a LoRA adapter trained with MLX-LM on a 4-bit 1.5B model, and a before/after evaluation on a frozen held-out set, all on an 8 GB laptop. First on made-up "smoke" data to prove the loop, then on real public data ([Civil Comments](dataset.md)). Then deeper: a confidence dial, calibration, a cloud port, and a harm-type relabelling.

**What this isn't.** A production moderation classifier. The datasets are tiny, one small model is used, and the labels are noisy. Findings show the mechanism and the method, not moderation quality at scale.

**Three headline findings, in order.**

1. **Read the full breakdown, not the single score.** On real safe/unsafe data the tuned model scores 0.789 accuracy, which looks solid, but of 54 genuinely unsafe messages it lets 15 through, the one error a moderation model can least afford. The single score hides that; the confusion grid shows it.
2. **Data is the fix, size is the ceiling.** A bigger base started smarter but re-learned the same blunt bias from the same data. What moved the boundary was aiming the training examples at the cases the model kept getting wrong, and later, relabelling on the harm *type* rather than a noisy overall score.
3. **A before/after is only trustworthy if it is controlled.** Base and tuned must run through the same setup, set to be fully repeatable, on the same frozen held-out set, with a check that the adapter actually loaded.

**The habit to lift.** When you measure a change to a non-deterministic system, control everything except the one variable, freeze the test set, and read the breakdown rather than the headline.

## The build arc, at a glance

| Step | Data | Result | What it taught |
|---|---|---|---|
| Smoke run | made-up, easy | 1.000 -> 1.000 (no change) | The loop runs, but a saturated base can't show movement. |
| Hard slice, 0.5B | made-up, hard | 0.458 -> 0.542 | The adapter shifts the boundary; it doesn't get smarter. |
| Hard slice, 1.5B | same | base 0.708, tuned 0.750 | Bigger base is smarter, but re-learns the same bias. |
| Targeted data | made-up, aimed | 0.708 -> 0.958 | *Which* data is the fix: aim it at the errors. |
| Real data | Civil Comments | base 0.583, tuned straddles | The honest result: neither checkpoint balances. |
| Harm-type relabel | Civil Comments sub-scores | verdict 0.589 -> 0.878 | Better labels finally lift the ceiling. |

## Findings

Each finding: the question, what I did, the numbers, and the takeaway.

### F1. The loop runs, but an easy test set shows nothing

**Question.** Does the training loop actually work end to end?

**What I did.** Trained a LoRA adapter on 0.5B with a small made-up dataset, then ran a before/after on a made-up held-out set. Receipt: `evidence/smoke-before-after.json`.

**The numbers.** Before 1.000, after 1.000, difference 0.000. Peak memory 0.5 GB.

**Takeaway.** The loop is proven: data -> train -> adapter -> eval all run. But the base already scored a perfect 1.000, so there was no room to show an effect. A fine-tune can only show its value where the base fails. An easy test set cannot measure a fine-tune, which motivated a deliberately harder held-out set.

### F2. The adapter shifts the decision boundary, it doesn't get smarter

**Question.** On a hard set the base gets wrong, what does the adapter actually change?

**What I did.** Built a 24-message hard held-out set (benign messages that sound alarming, plus subtle threats) and ran before/after on 0.5B. Receipt: `evidence/hard-before-after.json`.

**The numbers.** Overall 0.458 -> 0.542. Per label: safe caught 0.17 -> 0.67 (better), unsafe caught 0.75 -> 0.42 (worse).

**Takeaway.** The small +0.08 overall hides the story. The adapter learned to lean toward `safe`: that helped it stop over-flagging harmless messages, and it hurt its ability to catch subtle threats, the direction that matters most for safety. The training data (mostly benign) taught a blunt "lean safe" habit.

### F3. A bigger base raises the ceiling but does not fix the bias

**Question.** Does a bigger model fix the subtle-threat problem on its own?

**What I did.** Changed one thing only, the base from 0.5B to 1.5B, keeping data and settings identical. Receipt: `evidence/hard-1.5b-before-after.json`.

**The numbers.** The 1.5B base was smarter out of the box (0.458 -> 0.708). After tuning on the same data it re-learned the same bias: safe caught 0.50 -> 0.75, unsafe caught 0.92 -> 0.75.

**Takeaway.** Size raised the ceiling but did not fix the regression: the same data re-imposed the same bias. **Size is the ceiling, data is the fix.** Peak memory went from 0.5 GB to 1.2 GB, still far from 8.

### F4. Which data is the fix: aim it at the errors

**Question.** If data is the fix, what data?

**What I did.** Two experiments on 1.5B, changing only the training examples: a broader set teaching the scary-but-safe vs genuinely-unsafe distinction, then a targeted swap adding examples in exactly the registers the model kept missing. Receipts: `evidence/hard-1.5b-{datafix,swap}-before-after.json`.

**The numbers** (same base, same held-out set):

| training data | overall | safe caught | unsafe caught | real threats let through |
|---|---|---|---|---|
| benign-heavy | 0.750 | 0.75 | 0.75 | 3 |
| broad distinction | 0.875 | 0.92 | 0.83 | 2 |
| targeted at the errors | 0.958 | 0.92 | 1.00 | 0 |

**Takeaway.** The model learns whatever the data makes cheapest. Aim the examples at the confusion region and the boundary moves there. A fresh, differently-worded held-out set confirmed real transfer, not memorization (0.75 -> 0.958). This is a clean win on made-up data, so it shows the mechanism, not a real classifier's quality.

### F5. Real data is messier, and the honest result

**Question.** What happens on real, human-labeled data?

**What I did.** Moved to [Civil Comments](dataset.md): real reader comments, each carrying the share of reviewers who marked it toxic; `unsafe` when at least 70% flagged it. Trained on 112 examples, evaluated on a frozen, balanced 24-message held-out set. Receipts: `evidence/dataset/real-*.metrics.json`.

**The numbers.** The base over-flags:

| actual \ guessed | safe | unsafe |
|---|---|---|
| **safe** (12) | 3 | 9 |
| **unsafe** (12) | 1 | 11 |

That is 0.583 accuracy, 0.92 unsafe caught, but only 0.25 safe caught. After tuning, neither saved checkpoint got the balance right: the early save labeled everything unsafe (safe 0.00), the later save recovered safe messages (0.25 -> 0.83) but let 7 real threats through.

**Takeaway.** The adapter moved behavior, but the balanced middle sits *between* the saved checkpoints, and picking a saved checkpoint cannot reach it. An honest negative result, more useful than the tidy 0.958 on made-up data. It also exposed a limit: with only a yes/no output there are just two coarse settings and no dial between them (see F8).

### F6. When to stop training: the down-then-up curve

**Question.** Which checkpoint do you keep?

**What I did.** Each step nudges the adapter's numbers a little further, so more steps is not automatically better - past a point the numbers bend to fit the exact training examples. To find that point, I watched validation loss across the real run.

**The numbers.** Validation loss fell then rose: 4.890 -> 0.962 (halfway) -> 1.734 (end), while training-set loss kept dropping.

**Takeaway.** The rise is the model memorizing the training examples (and their labeling noise) at the cost of generalizing, so the best adapter is from the middle of the run, not the last save. Caution: validation loss ranked the halfway checkpoint best, but on the actual moderation goal that checkpoint collapsed to all-unsafe. Loss is a proxy; the held-out breakdown is the real judge.

### F7. The silent no-op: a wrong path reports a false zero

**Question.** How do you know the adapter is even loaded?

**What I did.** Added a check that the adapter folder exists and is valid before the tuned run (`_assert_adapter_present`, covered by `tests/test_adapter_effect.py`).

**Takeaway.** If you point the evaluation at the wrong adapter folder, the code quietly loads the plain base model and reports a before/after difference of zero, indistinguishable from a real "the fine-tune did nothing." A saturated base (F1) hides this doubly. The guard makes a missing or misnamed adapter fail loudly instead of faking a null result.

## Going further: the deeper experiments

Everything above is the simple core. These experiments push on the honest real-data result (F5). They are the reason the repo has more code than a first fine-tune needs, and they are deliberately out of scope for the plain-language article.

### F8. A confidence dial: it works, but the threshold does not transfer at n=24

Instead of reading the model's one-word answer, the code (`ft_cm/logprob.py`) reads its probability for `unsafe` vs `safe` at the decision point, a continuous 0-1 score. That gives a dial: choose a threshold, and messages above it are called unsafe.

The dial genuinely improved the ranking of messages (area-under-curve rose from 0.736 to 0.812 after tuning), and at threshold 0.5 it reproduced the one-word result exactly, so the score is trustworthy. But choosing the threshold on one 24-message split and applying it to another failed: the two splits disagreed on the best cutoff by 0.16 on a 0-1 scale, so a threshold picked on the validation split left 4 real threats through on the held-out set. At this size, threshold selection is too noisy to transfer. The honest response is to report the whole curve, not one number, and to route the uncertain middle to a human rather than force a yes/no. Receipt: `evidence/dataset/real-logprob-160-tau70.metrics.json`.

### F9. Calibration: temperature scaling can't fix a ranking limit

`ft_cm/calibration.py` measures whether the confidence scores are trustworthy (a reliability diagram and ECE, the expected calibration error) and tries the standard fix, temperature scaling. The scores were off by about 0.147 ECE, under-confident in one narrow band where safe and unsafe genuinely overlap. Temperature scaling, even fitted directly on the test set (the best possible case), did nothing useful: it is a single global confidence rescale, but the problem is a localized overlap, a limit of how *separable* the two classes are, not how the confidence is scaled. That overlap traces back to the labels (see F11), not the method. A clean example of a fix that cannot work because it targets the wrong problem.

### F10. The cloud port: a faithful config is not a faithful result

To learn the cloud toolchain (and test the planned 8 GB fallback), I ported the same training onto a free Google Colab T4 GPU, which runs a different toolchain (HuggingFace PEFT + trl) because MLX is Apple-only. Reading the MLX source rather than assuming caught two settings that would have quietly weakened the port: MLX's `scale` of 20 equals PEFT's `lora_alpha` of 160 (a direct multiplier, not divided by rank), and MLX trains all seven projections in the top eight layers, not the common two-projection default. With those matched, the config was faithful.

The result was not. MLX lifted the ranking by +0.076; the best cloud checkpoint only +0.021. The smoking gun: even the base model shifted (0.736 -> 0.701) with no adapter at all, purely from the different 4-bit number format used in training vs the full-precision format used in local evaluation. So "same recipe, different number format" alone moves the result. The overfitting signature and checkpoint ordering still reproduced across toolchains, even though the absolute quality did not. The port is real, behind the same `Provider` seam (`ft_cm/providers/peft_provider.py`). Receipts: `evidence/dataset/real-peft-logprob-iter{50,100}-160-tau70.metrics.json`; executed notebook `notebooks/cloud_lora_peft-executed.ipynb`.

A second lesson: the 8 GB ceiling that never bit training (~1.7 GB) *did* bite this local evaluation, because a PEFT adapter must be evaluated in full precision (~3 GB per model). The fix was to load one model at a time.

### F11. Relabel on harm sub-scores: the ceiling finally lifts

The real-data wall (F5, F8) was the labels: Civil Comments' overall toxicity measures general incivility, not the threat/harm a moderation system cares about. So the fix had to be better labels. Instead of an overall score, the harm-type sub-scores (`threat`, `identity_attack`, `sexual_explicit`) were used to assign each message a **ground**, its kind of harm: threat, identity attack, sexual, insult, or safe. The safe/unsafe verdict then derives from the ground (insult and profanity count as safe-to-publish; threat, identity, and sexual as unsafe).

Two things this surfaced, both from reading raw rows rather than summaries: in Civil Comments `obscene` means profanity, not sexual content (an early version wrongly folded it into the sexual ground; fixed to use `sexual_explicit` only), and the data had to be pulled by targeting the rare harmful rows directly (a database query over the dataset) rather than streaming and discarding 99%.

Training the 1.5B model to predict the five-way ground, same settings as before, lifted the ceiling the binary task could not: verdict accuracy 0.589 -> 0.878 (+0.289), ground accuracy 0.256 -> 0.656 (+0.400), on a frozen 90-message held-out set. The base defaulted to calling almost everything a threat; the tune taught it to tell the harm types apart. The lift came from correctly passing civil-but-rude messages: the base flagged 26 of 36 safe messages as unsafe, the tune passes 34 of 36. Receipts: `evidence/dataset/real-grounds-{baseline,it200}.metrics.json`. **Better labels lift the ceiling; reproducing the old config never could.**

### F12. Predicting the harm ground beats a plain binary by about +0.09

Did the *finer labelling* help, or just the cleaner relabel? To separate the two, I trained a plain safe/unsafe binary model on the exact same rows, split, and settings, with a prompt encoding the identical harm boundary, changing only the label granularity (`ft_cm/relabel_binary.py` maps the ground rows onto the binary axis with no re-split). On the same 90-message held-out set: grounds 0.878 (79/90) vs binary 0.789 (71/90), a +0.089 gap. Grounds won on both sides, catching more subtle harm (8 identity/sexual cases the binary missed) and raising fewer false alarms (3 rude-but-safe rows the binary over-flagged), losing only on its own documented harm-type confusions (3 rows). The finer target forces the model to build harm-type features, and those features are the boundary. Receipt: `evidence/dataset/real-binary-it200.metrics.json`. The effect is real and consistent but modest (n=90), and it is project-5 theory-testing territory, so the article mentions it in one line.

### F13. Consistency: treat like cases alike

`tests/test_consistency.py` encodes the theory-testing shape as a judge-free contract on the grounds model: structurally identical cases must get the same ground. It passes 6 of 7. Swapping an inert detail holds the ground (party swap: "the {Republican/Democratic/Green} party are complete morons" all read as insult; name swap: "come find {Alex/Priya/Chen/Omar} and make them pay" all read as threat), and varying the harm type moves the ground as it should. The one strict expected-failure: an identity attack read as a plain insult, the same hard boundary the held-out set showed. It is written as an `xfail(strict=True)` so that if the model ever learns the distinction, the test breaks loudly and forces the finding to be updated rather than lost.

## How the code fits together

Everything runs through a small set of modules, grouped by role.

**The label contract.**
- `ft_cm/taxonomy.py` is the single source of truth for the label set and the prompt framing; data prep, scorer, and eval all import it so the trained labels can never drift from what the scorer accepts. A `FT_CM_TASK` switch selects the grounds label space or the binary one.
- `ft_cm/config.py` holds run configuration, including `FT_CM_BASE_MODEL` (the base model is a config var, so the 0.5B/1.5B swap is one setting).

**Data.**
- `ft_cm/download_data.py` pulls a tiny balanced slice of the real dataset by targeting the rare harmful rows directly (a database query over the dataset's files), at a set agreement bar (tau). Each pull pins the dataset revision and writes a manifest with a content hash, so a re-pull that returns different rows raises a loud drift warning. Raw text stays git-ignored; only the manifest commits.
- `ft_cm/data_prep.py` turns raw {text, label} rows into the chat-format train/valid/test files MLX-LM trains on, with a stratified split (so the held-out set keeps both classes) and a near-duplicate check (so nothing in the test set leaks from training).
- `ft_cm/relabel_binary.py` remaps the frozen grounds split onto the binary verdict for the F12 ablation, without re-splitting.

**Scoring.**
- `ft_cm/scorer.py` extracts the single label from a completion by exact, whole-word match (so `safe` is never matched inside `unsafe`).
- `ft_cm/logprob.py` reads the model's confidence at the decision token instead of its word (F8), summing the probability of every surface form of each label.
- `ft_cm/calibration.py` is pure, tested math over those confidence scores: reliability diagram, ECE, and temperature scaling (F9).

**Running and reporting.**
- `ft_cm/eval.py` is the before/after harness. It runs base then tuned through the same held-out set, and supports a baseline-only mode, the logprob mode, and a `--backend {mlx,peft}` switch. It writes two receipts: a full one (with raw text, git-ignored) and a stripped metrics-only one (no text, committable).
- `ft_cm/report.py` renders a receipt into one self-contained HTML report (summary, confusion grid, and every row's raw completion), generated on demand so it is never committed with raw text.
- `ft_cm/providers/` is the backend seam: `base.py` (the interface), `mlx_provider.py` (base and tuned served identically by toggling the adapter path), `peft_provider.py` (the CUDA-path twin for the cloud adapter), and `ollama.py` (a quick local sanity backend).

**Reproducibility scaffolding.**
- `configs/` holds the four LoRA settings files (smoke, real binary, real grounds, and the shared real config), each tuned for 8 GB.
- `scripts/` holds the training wrapper (`train-smoke.sh`), a full-log recorder (`run-evidence.sh`), and guard hooks that warn on truncated logs or committed secrets.
- `notebooks/` holds the Colab notebook for the cloud run, and the executed copy kept as the durable record.

## Tests

The tests check the code, not the model's quality (that is what the eval measures). They run in seconds with no model or network.

- `test_scorer.py` locks the whole-word label extraction, including the `safe`-inside-`unsafe` case.
- `test_data_prep.py` locks the split determinism, the stratification, and the leakage check.
- `test_download_data.py` locks the ground-recipe semantics (which sub-score maps to which ground).
- `test_smoke_mocked.py` runs the whole loop against a fake provider.
- `test_adapter_effect.py` guards the silent no-op (F7): a missing or misnamed adapter must fail loudly.
- `test_logprob.py` covers the decision-token scorer and the threshold math.
- `test_report.py` renders HTML from a receipt with no model.
- `test_calibration.py` encodes measured behaviors of the tuned model as live contracts.
- `test_consistency.py` is the grounds consistency contract (F13), including the one strict expected-failure.

## Limitations

**The datasets are tiny on purpose.** A 24- or 90-message held-out set means each message is a large share of the score, so the confidence range around any number is wide. Enough to show a mechanism, not to make a quality claim.

**The labels are noisy and slightly off-target.** Civil Comments measures general incivility by crowd vote, not the specific harm a moderation system cares about. The harm-type relabel (F11) narrows this but does not erase it: even the sub-scores carry boundary noise (some profanity trips the sexual score).

**Made-up data measures the machinery, not moderation.** Every made-up result (including the 0.958) shows the loop works. None is a statement about real moderation quality.

**A single yes/no output has no dial.** Because the model emits one word, there is no built-in confidence to threshold-tune or route to a human. F8's confidence read is the partial workaround, and it does not transfer cleanly at this size.

**What a production version would add.** All the model's numbers trained rather than a small add-on (what a lab does to ship a base model); a confidence score with a tuned threshold and a human-review queue for the uncertain middle; a missed threat weighted as more costly than an over-flag; and a much larger, carefully labeled evaluation set so the numbers carry real statistical weight. The method here scales up; the numbers do not.
