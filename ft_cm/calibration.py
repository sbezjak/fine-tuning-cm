"""Threshold-sweep metrics over the continuous P(unsafe) scores from `logprob`.

Everything here is pure arithmetic over (score, gold) pairs - no MLX, no model - so
it unit-tests in the mocked gate and reruns for free from a receipt. The string
scorer gave one hard label per row; these functions read the CONTINUOUS score the
`logprob` module produces and answer the questions step g was built for:

  - `roc_auc` / `average_precision`: threshold-FREE ranking quality. Whether the
    fine-tune improved the ORDER of rows, independent of where any cutoff lands - the
    cleanest "did the adapter help" read.
  - `threshold_sweep` / `pick_operating_point`: turn the dial. Find the cutoff that
    maximizes safe-recall under the asymmetric HARD CONSTRAINT (unsafe->safe misses
    <= k), the calibrated operating point a single decode word cannot express.
  - `auto_decision`: selective classification. Route the rows within a band of the
    cutoff to human review and report the metrics over only what auto-decides - the
    robust answer when the exact threshold cannot be trusted at small N.

Positive class = unsafe (the safety-critical label), taken from `logprob` so the
label space cannot drift from the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ft_cm.logprob import NEGATIVE_LABEL, POSITIVE_LABEL


def roc_auc(scores: list[float], golds: list[str]) -> float:
    """Area under the ROC via the average-rank Mann-Whitney U identity (exact, with
    tie-averaged ranks). Positive=unsafe. 0.5 = no better than chance ordering, 1.0 =
    every unsafe row scored above every safe row. NaN if either class is absent."""
    pos = [s for s, g in zip(scores, golds) if g == POSITIVE_LABEL]
    neg = [s for s, g in zip(scores, golds) if g == NEGATIVE_LABEL]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(scores)) if golds[i] == POSITIVE_LABEL)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(scores: list[float], golds: list[str]) -> float:
    """Area under precision-recall (average precision), positive=unsafe. More honest
    than ROC-AUC when the positive class is what you care about protecting."""
    pairs = sorted(zip(scores, golds), key=lambda p: -p[0])
    n_pos = sum(1 for _, g in pairs if g == POSITIVE_LABEL)
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for _, g in pairs:
        if g == POSITIVE_LABEL:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


@dataclass(frozen=True)
class Confusion:
    threshold: float
    tp: int  # unsafe called unsafe
    fp: int  # safe called unsafe  (over-flag)
    tn: int  # safe called safe
    fn: int  # unsafe called safe  (the safety-critical MISS)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def misses(self) -> int:
        return self.fn

    @property
    def overflags(self) -> int:
        return self.fp

    @property
    def safe_recall(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else float("nan")

    @property
    def unsafe_recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "safe_recall": self.safe_recall,
            "unsafe_recall": self.unsafe_recall,
            "misses": self.misses,
            "overflags": self.overflags,
            "accuracy": self.accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def confusion_at(scores: list[float], golds: list[str], threshold: float) -> Confusion:
    """The 2x2 grid at one cutoff: pred=unsafe iff score >= threshold."""
    tp = fp = tn = fn = 0
    for s, g in zip(scores, golds):
        pred_unsafe = s >= threshold
        if g == POSITIVE_LABEL and pred_unsafe:
            tp += 1
        elif g == POSITIVE_LABEL:
            fn += 1
        elif g == NEGATIVE_LABEL and not pred_unsafe:
            tn += 1
        else:
            fp += 1
    return Confusion(threshold, tp, fp, tn, fn)


def candidate_thresholds(scores: list[float]) -> list[float]:
    """Every cutoff that produces a distinct partition: each observed score and a hair
    above it, plus 0. `>= t` at a score value keeps that row unsafe; the +eps variant
    flips it, so the pair covers both sides of every tie."""
    uniq = sorted(set(scores))
    eps = 1e-9
    return sorted(set([0.0] + uniq + [s + eps for s in uniq]))


def threshold_sweep(
    scores: list[float], golds: list[str], thresholds: list[float] | None = None
) -> list[Confusion]:
    ts = thresholds if thresholds is not None else candidate_thresholds(scores)
    return [confusion_at(scores, golds, t) for t in ts]


def pick_operating_point(
    scores: list[float], golds: list[str], max_misses: int
) -> Confusion:
    """The calibrated point: among cutoffs whose unsafe->safe misses <= max_misses,
    the one that MAXIMIZES safe-recall (ties broken by fewer misses, then accuracy).
    This is the asymmetric objective - protect the safety cell, then recover safe
    recall - expressed as a threshold instead of a checkpoint choice."""
    feasible = [c for c in threshold_sweep(scores, golds) if c.misses <= max_misses]
    if not feasible:
        raise ValueError(f"no threshold satisfies misses <= {max_misses}")
    return max(feasible, key=lambda c: (c.safe_recall, -c.misses, c.accuracy))


@dataclass(frozen=True)
class AutoDecision:
    threshold: float
    band: float
    review: int  # rows routed to a human (within +/- band of the threshold)
    auto: Confusion  # metrics over ONLY the auto-decided rows

    @property
    def review_fraction(self) -> float:
        total = self.review + self.auto.n
        return self.review / total if total else float("nan")


def auto_decision(
    scores: list[float], golds: list[str], threshold: float, band: float
) -> AutoDecision:
    """Selective classification: rows with |score - threshold| <= band go to human
    review; the rest auto-decide at `threshold`. Returns the confusion over the
    auto-decided rows only - the honest "what ships automatically" grid."""
    auto_scores: list[float] = []
    auto_golds: list[str] = []
    review = 0
    for s, g in zip(scores, golds):
        if abs(s - threshold) <= band:
            review += 1
        else:
            auto_scores.append(s)
            auto_golds.append(g)
    auto = confusion_at(auto_scores, auto_golds, threshold)
    return AutoDecision(threshold=threshold, band=band, review=review, auto=auto)


def smallest_safe_band(
    scores: list[float], golds: list[str], threshold: float, widths: list[float]
) -> float:
    """The smallest band width (from `widths`, ascending) whose auto path has zero
    unsafe->safe misses; the widest tried if none clears. The band is the mitigation
    for a threshold you cannot pin down - pick its width where the safety cell goes
    clean, not by eyeballing the test set."""
    for w in sorted(widths):
        if auto_decision(scores, golds, threshold, w).auto.misses == 0:
            return w
    return max(widths)
