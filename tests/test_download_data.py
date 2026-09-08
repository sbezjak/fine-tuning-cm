"""Lock the ground-recipe semantics - the one conceptual call in the pull.

No network: `assign_ground` is a pure taxonomy function and `content_hash` is pure,
and importing ft_cm.download_data does NOT import the heavy `datasets` lib (that
import is deferred inside fetch_grounded), so this runs in the default mocked gate
without the `data` group installed.
"""

import pytest

from ft_cm.download_data import content_hash
from ft_cm.taxonomy import GROUNDS, SUBSCORE_COLUMNS, assign_ground, verdict

pytestmark = pytest.mark.mocked


def _scores(**kw: float) -> dict[str, float]:
    """A full sub-score row (unset columns default to 0), as the puller builds it."""
    return {col: kw.get(col, 0.0) for col in SUBSCORE_COLUMNS}


def test_assign_ground_boundary_is_ge_tau():
    # TAU is a >= boundary: exactly TAU trips the ground, a hair under stays safe.
    assert assign_ground(_scores(insult=0.7), 0.7) == "insult"
    assert assign_ground(_scores(insult=0.6999), 0.7) == "safe"
    assert assign_ground(_scores(), 0.7) == "safe"
    assert assign_ground(_scores(threat=1.0), 0.7) == "threat"


def test_assign_ground_default_tau_is_0_7():
    # The default boundary is 0.7 (strong majority): 0.6 stays safe, 0.7 trips.
    assert assign_ground(_scores(insult=0.6)) == "safe"
    assert assign_ground(_scores(insult=0.7)) == "insult"


def test_assign_ground_priority_most_serious_wins():
    # When several sub-scores trip, the most serious ground wins (threat > insult).
    assert assign_ground(_scores(threat=0.9, insult=0.9), 0.7) == "threat"
    assert assign_ground(_scores(identity_attack=0.8, insult=0.9), 0.7) == "identity_attack"


def test_assign_ground_sexual_is_explicit_only_obscene_is_incivility():
    # `sexual` comes ONLY from sexual_explicit; `obscene` (profanity) folds into insult.
    assert assign_ground(_scores(sexual_explicit=0.8), 0.7) == "sexual"
    assert assign_ground(_scores(obscene=0.9), 0.7) == "insult"


def test_assign_ground_tau_moves_the_line():
    # The squishy middle flips with TAU: a 0.5 insult is safe at 0.7, insult at 0.25.
    assert assign_ground(_scores(insult=0.5), 0.7) == "safe"
    assert assign_ground(_scores(insult=0.5), 0.25) == "insult"


def test_assign_ground_emits_only_taxonomy_labels():
    # assign_ground always emits a harm GROUND (grounds machinery stays live in both
    # tasks); the binary relabel derives safe/unsafe from it downstream via verdict().
    assert assign_ground(_scores(threat=0.9), 0.5) in GROUNDS
    assert assign_ground(_scores(), 0.5) in GROUNDS


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
