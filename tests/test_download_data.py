"""Lock the binarize threshold semantics - the one conceptual call in the pull.

No network: `binarize` is a pure function, and importing ft_cm.download_data does
NOT import the heavy `datasets` lib (that import is deferred inside fetch_balanced),
so this runs in the default mocked gate without the `data` group installed.
"""

import pytest

from ft_cm.download_data import binarize, content_hash
from ft_cm.taxonomy import LABELS

pytestmark = pytest.mark.mocked


def test_binarize_boundary_is_ge_tau():
    # TAU is a >= boundary: exactly TAU counts as unsafe, a hair under is safe.
    assert binarize(0.5, 0.5) == "unsafe"
    assert binarize(0.4999, 0.5) == "safe"
    assert binarize(0.0, 0.5) == "safe"
    assert binarize(1.0, 0.5) == "unsafe"


def test_binarize_tau_moves_the_line():
    # The squishy middle flips with TAU: a 0.3-toxicity row is safe at 0.5, unsafe at 0.25.
    assert binarize(0.3, 0.5) == "safe"
    assert binarize(0.3, 0.25) == "unsafe"


def test_binarize_emits_only_taxonomy_labels():
    assert binarize(0.9, 0.5) in LABELS
    assert binarize(0.1, 0.5) in LABELS


def test_content_hash_is_order_independent():
    # The drift fingerprint must not depend on row order (sampling order varies).
    a = {"text": "hi", "label": "safe", "toxicity": 0.1}
    b = {"text": "bye", "label": "unsafe", "toxicity": 0.9}
    assert content_hash([a, b]) == content_hash([b, a])


def test_content_hash_changes_when_data_changes():
    # A different label or text => different fingerprint (drift is detectable).
    base = [{"text": "hi", "label": "safe", "toxicity": 0.1}]
    flipped = [{"text": "hi", "label": "unsafe", "toxicity": 0.1}]
    assert content_hash(base) != content_hash(flipped)
