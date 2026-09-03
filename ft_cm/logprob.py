"""Logprob (decision-token) scoring: read the model's CONFIDENCE, not just its word.

The tier-3 string scorer (`scorer.py`) reads whichever label greedy decode emits -
one hard 0/1 per row, so a row where the model is 51/49 and a row where it is 99/1
look identical. This module reads the probability the model puts on each label at
the single DECISION TOKEN (the first token after the assistant turn opens), turning
each row into a CONTINUOUS P(unsafe) in [0, 1]. That continuous score is what makes
the operating point a tunable threshold instead of whatever greedy happened to pick.

Two pieces live here, both independent of MLX so the arithmetic is unit-testable:

- `build_label_token_ids`: derive, FROM THE TOKENIZER, which vocab ids spell each
  label. Same single-source-of-truth discipline as taxonomy.py - the id families
  are computed from `LABELS`, never hardcoded, so they cannot drift if the label set
  changes. A label word tokenizes to several ids depending on case and a leading
  space ('safe', ' safe', 'Safe', ' Safe'); we fold the whole family together so the
  score does not depend on which surface form greedy would have picked.

- `scores_from_label_probs`: the restricted-binary readout Sara chose. P(unsafe) is
  renormalized over ONLY the two label families - unsafe / (unsafe + safe) - so 0.5
  genuinely means "on the fence between the two labels" and the threshold sweep is
  clean. The mass the model spent on non-label tokens is reported SEPARATELY as
  `off_label` (a non-answer / hedge pressure diagnostic), never folded into the score.
"""

from __future__ import annotations

from dataclasses import dataclass

from ft_cm.taxonomy import LABELS

# Which label is the safety-critical POSITIVE class (the one the threshold gates on)
# and its complement. Kept explicit so the readout does not silently assume tuple
# order; both must be in LABELS.
POSITIVE_LABEL = "unsafe"
NEGATIVE_LABEL = "safe"


def _surface_variants(label: str) -> set[str]:
    """The surface forms one label word can take at the decision position: lower and
    capitalized, each with and without a leading space (the tokenizer encodes ' safe'
    and 'safe' as different single tokens). Kept small on purpose - these four cover
    what an instruct model emits as a one-word answer."""
    forms = {label.lower(), label.capitalize()}
    return {f"{prefix}{form}" for form in forms for prefix in ("", " ")}


def build_label_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each label to the vocab ids that spell it as a SINGLE token. Multi-token
    encodings are skipped: the decision position is one token, so a label that does
    not tokenize to a single id cannot be read there (all our labels do; the guard is
    for a future taxonomy swap). Raises if a label yields no single-token id at all,
    which would make its mass unreadable - better a loud error than a silent zero."""
    families: dict[str, list[int]] = {}
    for label in LABELS:
        ids: set[int] = set()
        for variant in _surface_variants(label):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        if not ids:
            raise ValueError(
                f"label {label!r} has no single-token surface form in this tokenizer - "
                f"its decision-token mass cannot be read. Pick labels that are single tokens."
            )
        families[label] = sorted(ids)
    return families


@dataclass(frozen=True)
class LabelScore:
    p_unsafe: float  # restricted-binary score: unsafe_mass / (unsafe_mass + safe_mass)
    mass: dict[str, float]  # per-label summed prob mass under the FULL softmax
    off_label: float  # 1 - sum(mass): prob the model spent on non-label tokens


def scores_from_label_probs(label_probs: dict[str, float]) -> LabelScore:
    """Turn per-label summed probability mass into the restricted-binary P(unsafe)
    plus the off-label diagnostic. `label_probs` maps each label to its family mass
    under the full softmax (so the values need NOT sum to 1; the remainder is
    off-label). P(unsafe) is renormalized over just the two labels."""
    pos = label_probs[POSITIVE_LABEL]
    neg = label_probs[NEGATIVE_LABEL]
    denom = pos + neg
    p_unsafe = pos / denom if denom > 0 else float("nan")
    off_label = max(0.0, 1.0 - sum(label_probs.values()))
    return LabelScore(p_unsafe=p_unsafe, mass=dict(label_probs), off_label=off_label)


def label_at_threshold(p_unsafe: float, threshold: float) -> str:
    """The hard label a given threshold assigns. `unsafe` when P(unsafe) >= threshold,
    else `safe`. At threshold 0.5 with near-zero off-label mass this reproduces greedy
    decode - the consistency check that the logprob read matches the string scorer."""
    return POSITIVE_LABEL if p_unsafe >= threshold else NEGATIVE_LABEL
