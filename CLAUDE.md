# CLAUDE.md

## Project

`fine-tuning-cm` is the first real fine-tune in the portfolio: learn the
fine-tuning toolchain (data prep -> LoRA/SFT -> eval-in-the-loop) on a bounded,
DEFENSIVE content-moderation classifier. Single-label task (`safe`/`unsafe`),
so the reward is a tier-3 exact/substring check, not an LLM judge. The output is
a short educational article; basic-but-finished-and-understood beats ambitious.

**Public repo, content-moderation domain.** Frame everything defensively. Never
commit raw toxic text: real datasets stay under `data/real/` (git-ignored) via a
download script; only the tame fabricated `data/smoke/` set is tracked, and only
to prove the loop. Cite source + license in `docs/dataset.md` before first use.

**Hardware is the binding constraint: Apple M1, 8 GB, macOS 26.1.** Train with
MLX-LM LoRA (Apple-Silicon path) on a 4-bit tiny base (`Qwen2.5-0.5B-Instruct`,
planned swap to `1.5B`). Base model is a config var (`ft_cm/config.py`,
`FT_CM_BASE_MODEL`). Small batch, short sequences, few adapter layers. If a run
outgrows 8 GB, train on a free cloud T4 and keep the Mac for the eval harness -
write that production-vs-pragmatic trade into `notes.md`.

## Commands

```bash
uv sync                                   # eval + test stack
uv run pytest -m mocked                   # fast unit tests, no model/network
uv run pytest -m ollama                   # live local Ollama sanity (free)
uv run ruff check .
scripts/train-smoke.sh                    # prepare data + train smoke adapter (full log preserved)
uv run python -m ft_cm.eval               # before/after held-out eval
```

## Test conventions (configured in pyproject.toml)

- `mocked`: fast, respx-mocked, no model or network. The default gate.
- `ollama`: requires live Ollama at localhost:11434 (free local).
- `mlx`: requires the MLX training stack + a downloaded base model (slow, local).
- Async by default (`asyncio_mode = "auto"`). ruff line length 100.

## Architecture intent

- `ft_cm/taxonomy.py` is the single source of truth for the label set and the
  prompt framing; data prep, scorer, and eval all import it so the trained label
  space can never drift from what the scorer accepts.
- `Provider` is the backend seam (`generate(prompt) -> str`). The MLX provider
  serves base (no adapter) and tuned (with adapter) identically, so the SAME
  held-out slice runs through both - that is the before/after measurement.
- The finding that lifts it above a tutorial: before/after held-out accuracy
  PLUS one calibration read (generalized vs memorized), using the confusion +
  non-answer + hedge counts in `EvalResult`.

## Working style with this user

- **`human-tasks.md` is the channel for anything only the user can do.**
  Whenever the next step needs the user (post an update, paste a real URL,
  choose a dataset + license, run an interactive login), append a checkbox item
  to `human-tasks.md` instead of only mentioning it in chat - chat scrolls, the
  file persists. Read it when picking up work.
- **Prepare drafts/templates for any task the user has to do by hand.** Reduce
  the user's task to filling in the squishy parts; don't hand them a blank page.
- **Capture explanations to `notes.md` when teaching.** When the user asks
  "explain this to me" and the answer is non-trivial (why a LoRA run OOMs, what
  the adapter actually changed, why held-out accuracy moved), mirror it (lightly
  cleaned up) into `notes.md` as reference. The chat scrolls; the notes stay.
- **Two writing registers, no duplicate copies.** `notes.md` is the dense,
  finding-first record. Front-door artifacts (README, `docs/` explainers, the
  article) use the plain, layered voice. Same explanation in two registers
  drifts, so the registers serve different artifacts, never two copies of one.
- **Ground each front-door artifact in the previous projects' versions before
  drafting.** Read the LOCAL sibling repos (`../llm-benchmark/`,
  `../llm-eval-harness/`, `../llm-rag/`, `../llm-red/`) first - measurably better
  output (A/B confirmed).
- **Default to production / best-practice solutions; take the pragmatic shortcut
  only when justified for this project's scope, and call the trade-off out.**
  Name the production-grade pattern, name why we're not doing it here, write the
  trade-off into `notes.md` so the writeup shows a deliberate choice. (The 8 GB
  ceiling forces several of these - they are the good material.)
- **Validate a new integration cheaply before any long or expensive run.** Prove
  it works on the smallest input first (fabricated is fine) and confirm the
  output parses. ALWAYS arm a monitor on the log for any run that is not
  near-instant - grep for success AND failure signatures
  (`Traceback|Error|Exception|Killed|OOM|Metal|NaN|nan|Timeout|Failed`) so a
  mid-run failure surfaces at the failing step. Smoke test first, monitor second,
  both every time. On failure, fix the root cause and re-validate cheaply before
  re-running full.
- **Preserve the FULL run log for anything a claim rests on - never truncate it
  through `tail`/`head`/`tee | tail`.** Redirect the whole stream to a durable
  file: `scripts/run-evidence.sh <name> -- <cmd>`. A PreToolUse Bash hook
  (`scripts/guard-truncated-logs.sh`) warns when a run is piped through
  tail/head. Emit a machine-checkable receipts JSON next to any summarized
  finding (the eval writes `evidence/smoke-before-after.json`) so the summary is
  recomputable, not taken on faith.
- **Read the logs/receipts thoroughly - the raw output is the only ground truth.**
  Never conclude from a summary boolean or a count when the actual text is there.
- **Prefer the smallest solution that solves the problem; after writing, re-read
  and cut machinery the task didn't ask for.**
- In all prose (docs, comments, commit messages), join clauses with a single
  hyphen `-`, a comma, a period, or parentheses. The only dash in written text
  is a single `-`. No `Co-Authored-By` trailers. No CI workflows.
