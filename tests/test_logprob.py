"""Mocked unit tests for the decision-token scorer (`logprob`) and the threshold
metrics (`calibration`). Pure arithmetic and a fake tokenizer - no MLX, no model, no
network, so they run in the default gate and pin the math the step-g finding rests on.
"""

import math
from typing import ClassVar

import pytest

from ft_cm.calibration import (
    auto_decision,
    average_precision,
    confusion_at,
    pick_operating_point,
    roc_auc,
    smallest_safe_band,
)
from ft_cm.logprob import (
    build_label_token_ids,
    label_at_threshold,
    scores_from_label_probs,
)


class FakeTokenizer:
    """Encodes each label surface form to a single distinct id; anything else to a
    multi-token sequence (so the single-token filter is exercised)."""

    VOCAB: ClassVar[dict[str, int]] = {
        "safe": 1, " safe": 2, "Safe": 3, " Safe": 4,
        "unsafe": 5, " unsafe": 6, "Unsafe": 7, " Unsafe": 8,
    }

    def encode(self, s, add_special_tokens=False):
        if s in self.VOCAB:
            return [self.VOCAB[s]]
        return [98, 99]  # pretend multi-token


@pytest.mark.mocked
def test_label_token_ids_fold_case_and_space():
    fams = build_label_token_ids(FakeTokenizer())
    assert fams["safe"] == [1, 2, 3, 4]
    assert fams["unsafe"] == [5, 6, 7, 8]


@pytest.mark.mocked
def test_label_token_ids_raise_when_no_single_token():
    class Empty:
        def encode(self, s, add_special_tokens=False):
            return [1, 2, 3]  # never single-token

    with pytest.raises(ValueError):
        build_label_token_ids(Empty())


@pytest.mark.mocked
def test_restricted_binary_and_off_label():
    sc = scores_from_label_probs({"safe": 0.02, "unsafe": 0.95})
    assert sc.p_unsafe == pytest.approx(0.95 / 0.97)
    assert sc.off_label == pytest.approx(0.03)
    # renormalized over just the two labels, so 0.5 is a true fence
    even = scores_from_label_probs({"safe": 0.3, "unsafe": 0.3})
    assert even.p_unsafe == pytest.approx(0.5)
    assert even.off_label == pytest.approx(0.4)


@pytest.mark.mocked
def test_label_at_threshold_reproduces_greedy_at_half():
    assert label_at_threshold(0.64, 0.5) == "unsafe"
    assert label_at_threshold(0.49, 0.5) == "safe"
    # raising the threshold recovers a near-fence over-flag
    assert label_at_threshold(0.64, 0.8) == "safe"


@pytest.mark.mocked
def test_roc_auc_separates_and_handles_ties():
    golds = ["unsafe", "unsafe", "safe", "safe"]
    assert roc_auc([0.9, 0.8, 0.2, 0.1], golds) == pytest.approx(1.0)  # perfect
    assert roc_auc([0.1, 0.2, 0.8, 0.9], golds) == pytest.approx(0.0)  # inverted
    # one unsafe and one safe tied -> chance
    assert roc_auc([0.5, 0.5], ["unsafe", "safe"]) == pytest.approx(0.5)
    assert math.isnan(roc_auc([0.5, 0.6], ["safe", "safe"]))  # one class absent


@pytest.mark.mocked
def test_average_precision_perfect_ranking():
    golds = ["unsafe", "unsafe", "safe", "safe"]
    assert average_precision([0.9, 0.8, 0.2, 0.1], golds) == pytest.approx(1.0)


@pytest.mark.mocked
def test_confusion_counts_and_derived():
    scores = [0.9, 0.4, 0.6, 0.2]
    golds = ["unsafe", "unsafe", "safe", "safe"]
    c = confusion_at(scores, golds, 0.5)
    assert (c.tp, c.fn, c.fp, c.tn) == (1, 1, 1, 1)  # 0.4 unsafe->safe miss; 0.6 over-flag
    assert c.misses == 1
    assert c.safe_recall == pytest.approx(0.5)
    assert c.unsafe_recall == pytest.approx(0.5)
    assert c.accuracy == pytest.approx(0.5)


@pytest.mark.mocked
def test_operating_point_respects_hard_constraint():
    # unsafe at 0.55/0.60, safe at 0.40/0.45: a cutoff in (0.45,0.55] gives 0 misses
    # AND perfect safe-recall. A cutoff above 0.55 would start missing unsafe rows.
    scores = [0.60, 0.55, 0.45, 0.40]
    golds = ["unsafe", "unsafe", "safe", "safe"]
    op = pick_operating_point(scores, golds, max_misses=0)
    assert op.misses == 0
    assert op.safe_recall == pytest.approx(1.0)
    assert op.unsafe_recall == pytest.approx(1.0)


@pytest.mark.mocked
def test_operating_point_trades_a_miss_for_safe_recall():
    # overlap: safe 0.52 sits ABOVE unsafe 0.48. At max_misses=0 the cutoff must stay
    # <=0.48 to catch both unsafe rows, so the 0.52 safe row is a forced over-flag
    # (safe-recall 0.5). Allowing 1 miss lets the cutoff rise past 0.52, recovering full
    # safe-recall at the cost of the 0.48 unsafe row - the asymmetric trade, in miniature.
    scores = [0.90, 0.48, 0.52, 0.10]
    golds = ["unsafe", "unsafe", "safe", "safe"]
    strict = pick_operating_point(scores, golds, max_misses=0)
    assert strict.misses == 0
    loose = pick_operating_point(scores, golds, max_misses=1)
    assert loose.misses <= 1
    assert loose.safe_recall >= strict.safe_recall


@pytest.mark.mocked
def test_auto_decision_and_smallest_safe_band():
    # unsafe 0.55 sits just above the 0.5 cutoff; a band of 0.06 pulls it (and the
    # 0.47 safe row) into review, leaving the auto path clean.
    scores = [0.95, 0.55, 0.47, 0.05]
    golds = ["unsafe", "unsafe", "safe", "safe"]
    ad = auto_decision(scores, golds, threshold=0.5, band=0.06)
    assert ad.review == 2
    assert ad.auto.n == 2
    widths = [x / 100 for x in range(1, 13)]
    w = smallest_safe_band(scores, golds, threshold=0.5, widths=widths)
    assert auto_decision(scores, golds, 0.5, w).auto.misses == 0
