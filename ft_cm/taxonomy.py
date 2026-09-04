"""The moderation label set and the prompt that frames the classification task.

Single source of truth: the data builder, the scorer, the logprob decision head,
and the eval harness all import from here, so the label space can never silently
drift between what the model is trained to say and what the scorer accepts.

The task is deliberately SINGLE-LABEL. The model emits exactly one GROUND word
from LABELS as the whole answer, which keeps the reward a tier-3 exact/substring
check, not a squishy LLM judge.

GROUNDS, not a bare binary. The label is the harm GROUND the verdict rests on
(threat / identity_attack / sexual / insult / safe), and safe/unsafe DERIVES from
it (see `verdict`). Grounding it this way makes the classifier consistency-testable
(treat same-ground cases alike) and lets one trained model be read under more than
one safe/unsafe boundary for free - the boundary is a post-hoc remap, not a retrain.
Why these five, and why NOT a judge: see notes.md "Grounds testing HERE".
"""

from __future__ import annotations

# The label space the model predicts AND the scorer accepts - one word, exact match.
# Ordered harm-first so the positive (unsafe) grounds lead and `safe` is the sink.
GROUNDS: tuple[str, ...] = ("threat", "identity_attack", "sexual", "insult", "safe")

# The scorer/logprob/eval/data_prep all import LABELS; grounds ARE the labels now.
LABELS: tuple[str, ...] = GROUNDS

# The grounds whose verdict is `unsafe`. `insult` and `safe` derive to `safe`:
# insult is the incivility axis (proven the wall in step g / calibration / cloud),
# NOT our harm target, so it is named but publishable. Flip this set to read the
# same model under the insult->unsafe boundary - no retrain (see notes.md pivot).
HARM_GROUNDS: frozenset[str] = frozenset({"threat", "identity_attack", "sexual"})

# The recipe that assigns each row its ONE gold ground from Civil Comments
# sub-scores, in priority order (most serious harm wins when several trip). Each
# ground lists the sub-score column(s) that map to it; `sexual` folds obscene and
# sexual_explicit (both the profanity/sexual axis, both rare). `severe_toxicity` is
# intentionally absent: its scale is compressed (never crossed 0.5 in a 20k scan),
# so it is intensity, not a harm type. `toxicity` is the OLD overall axis, not used.
GROUND_RECIPE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("threat", ("threat",)),
    ("identity_attack", ("identity_attack",)),
    ("sexual", ("obscene", "sexual_explicit")),
    ("insult", ("insult",)),
)

# The sub-score columns the recipe reads (for the data puller to request/validate).
SUBSCORE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(col for _, cols in GROUND_RECIPE for col in cols)
)

# Every recipe ground must be a real label, and `safe` (the sink) must exist.
assert {g for g, _ in GROUND_RECIPE} <= set(LABELS)
assert "safe" in LABELS
assert HARM_GROUNDS <= set(LABELS)

SYSTEM_PROMPT = (
    "You are a defensive content-moderation classifier. Read the user message and "
    "label the single most serious kind of harm it contains, using exactly one of "
    "these words:\n"
    "threat - threatens violence or serious harm to someone;\n"
    "identity_attack - attacks or demeans people for a protected identity "
    "(race, religion, nationality, gender, sexuality, disability);\n"
    "sexual - sexually explicit or obscene content;\n"
    "insult - rude, demeaning, or name-calling with none of the above;\n"
    "safe - none of these.\n"
    "Answer with exactly one word. Do not explain."
)


def build_prompt(text: str) -> str:
    """The user-turn content the model classifies. The system framing above is
    applied by the chat template at train and inference time."""
    return f"Message:\n{text}\n\nLabel:"


def assign_ground(scores: dict[str, float], tau: float = 0.5) -> str:
    """Collapse a row's harm sub-scores to ONE gold ground via the priority ladder:
    the first ground in GROUND_RECIPE with any mapped sub-score >= tau wins, else
    `safe`. This is the recipe that labels training rows deterministically - no
    hand-labeling, no judge. `tau` is the annotator-vote fraction boundary (0.5 =
    a majority of raters agreed), the one conceptual dial, same as the old binarize."""
    for ground, cols in GROUND_RECIPE:
        if any(float(scores.get(col, 0.0)) >= tau for col in cols):
            return ground
    return "safe"


def verdict(ground: str) -> str:
    """Derive the safe/unsafe verdict from a ground. The whole point of grounding:
    this is a lookup, so the same predictions can be re-read under a different
    HARM_GROUNDS boundary without retraining."""
    return "unsafe" if ground in HARM_GROUNDS else "safe"
