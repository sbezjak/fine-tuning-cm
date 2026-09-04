"""Lock the ground-recipe semantics - the one conceptual call in the pull.

No network: `assign_ground` is a pure taxonomy function and `content_hash` is pure,
and importing ft_cm.download_data does NOT import the heavy `datasets` lib (that
import is deferred inside fetch_grounded), so this runs in the default mocked gate
without the `data` group installed.
"""

import pytest

from ft_cm.download_data import content_hash
from ft_cm.taxonomy import LABELS, SUBSCORE_COLUMNS, assign_ground, verdict

pytestmark = pytest.mark.mocked


def _scores(**kw: float) -> dict[str, float]:
    """A full sub-score row (unset columns default to 0), as the puller builds it."""
    return {col: kw.get(col, 0.0) for col in SUBSCORE_COLUMNS}


def test_assign_ground_boundary_is_ge_tau():
    # TAU is a >= boundary: exactly TAU trips the ground, a hair under stays safe.
    assert assign_ground(_scores(insult=0.5), 0.5) == "insult"
    assert assign_ground(_scores(insult=0.4999), 0.5) == "safe"
    assert assign_ground(_scores(), 0.5) == "safe"
    assert assign_ground(_scores(threat=1.0), 0.5) == "threat"


def test_assign_ground_priority_most_serious_wins():
    # When several sub-scores trip, the most serious ground wins (threat > insult).
    assert assign_ground(_scores(threat=0.9, insult=0.9), 0.5) == "threat"
    assert assign_ground(_scores(identity_attack=0.6, insult=0.9), 0.5) == "identity_attack"


def test_assign_ground_sexual_folds_obscene_and_explicit():
    # `sexual` is the fold of obscene OR sexual_explicit.
    assert assign_ground(_scores(obscene=0.7), 0.5) == "sexual"
    assert assign_ground(_scores(sexual_explicit=0.55), 0.5) == "sexual"


def test_assign_ground_tau_moves_the_line():
    # The squishy middle flips with TAU: a 0.3 insult is safe at 0.5, insult at 0.25.
    assert assign_ground(_scores(insult=0.3), 0.5) == "safe"
    assert assign_ground(_scores(insult=0.3), 0.25) == "insult"


def test_assign_ground_emits_only_taxonomy_labels():
    assert assign_ground(_scores(threat=0.9), 0.5) in LABELS
    assert assign_ground(_scores(), 0.5) in LABELS


def test_verdict_derives_unsafe_from_harm_grounds():
    assert verdict("threat") == "unsafe"
    assert verdict("identity_attack") == "unsafe"
    assert verdict("sexual") == "unsafe"
    assert verdict("insult") == "safe"  # incivility, named but publishable
    assert verdict("safe") == "safe"


def test_content_hash_is_order_independent():
    # The drift fingerprint must not depend on row order (sampling order varies).
    a = {"text": "hi", "label": "safe", "toxicity": 0.1, **_scores()}
    b = {"text": "bye", "label": "threat", "toxicity": 0.9, **_scores(threat=0.9)}
    assert content_hash([a, b]) == content_hash([b, a])


def test_content_hash_changes_when_data_changes():
    # A different label or text => different fingerprint (drift is detectable).
    base = [{"text": "hi", "label": "safe", "toxicity": 0.1, **_scores()}]
    flipped = [{"text": "hi", "label": "insult", "toxicity": 0.1, **_scores(insult=0.9)}]
    assert content_hash(base) != content_hash(flipped)
