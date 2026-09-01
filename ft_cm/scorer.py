"""Tier-3 scoring: extract a single label from a model completion, exactly.

No LLM judge. The task is single-label, so grading is a deterministic string
check. The one subtlety worth getting right (and testing): "safe" is a literal
substring of "unsafe", so naive `in` matching mislabels every "unsafe" output as
"safe". Word-boundary matching avoids that. When more than one label appears
(a hedging completion like "safe or unsafe"), the earliest by position wins, and
`ambiguous` is set so calibration analysis can separate clean answers from hedges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ft_cm.taxonomy import LABELS


@dataclass(frozen=True)
class Extraction:
    label: str | None  # the extracted label, or None if no label token appeared
    ambiguous: bool  # True if >1 distinct label matched (a hedge, not a clean answer)


def extract_label(completion: str) -> Extraction:
    """Pull the single moderation label out of a raw completion."""
    text = completion.lower()
    hits: list[tuple[int, str]] = []
    for label in LABELS:
        m = re.search(rf"\b{re.escape(label)}\b", text)
        if m:
            hits.append((m.start(), label))
    if not hits:
        return Extraction(label=None, ambiguous=False)
    hits.sort()
    distinct = {label for _, label in hits}
    return Extraction(label=hits[0][1], ambiguous=len(distinct) > 1)


def is_correct(completion: str, expected: str) -> bool:
    """True when the extracted label equals the gold label. A missing or ambiguous
    extraction is scored wrong: on a moderation classifier, a non-answer is a miss."""
    got = extract_label(completion)
    if got.label is None or got.ambiguous:
        return False
    return got.label == expected.lower()
