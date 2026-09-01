"""The moderation label set and the prompt that frames the classification task.

Single source of truth: the training data builder, the scorer, and the eval
harness all import from here, so the label space can never silently drift between
what the model is trained to say and what the scorer accepts.

The task is deliberately SINGLE-LABEL. The model must emit exactly one label from
LABELS as the whole answer, which makes the reward a tier-3 exact/substring check,
not a squishy LLM judge.
"""

from __future__ import annotations

# Start binary (safe / unsafe). A small taxonomy (harassment, hate, sexual,
# self-harm, violence, ...) is a later step: swap this tuple and the fixtures,
# the pipeline is unchanged.
LABELS: tuple[str, ...] = ("safe", "unsafe")

SYSTEM_PROMPT = (
    "You are a defensive content-moderation classifier. Read the user message and "
    "decide whether it is safe or unsafe to publish on a general-audience platform. "
    "Answer with exactly one word: safe or unsafe. Do not explain."
)


def build_prompt(text: str) -> str:
    """The user-turn content the model classifies. The system framing above is
    applied by the chat template at train and inference time."""
    return f"Message:\n{text}\n\nLabel:"
