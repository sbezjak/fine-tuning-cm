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

import math
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
class ReliabilityBin:
    """One bin of a reliability diagram: rows whose predicted P(unsafe) fell in
    [lo, hi). `mean_pred` is the model's average confidence in the bin, `obs_freq`
    the fraction actually unsafe. On the diagonal (gap ~ 0) the number is calibrated;
    obs_freq > mean_pred is UNDER-confident, obs_freq < mean_pred over-confident.
    Empty bins carry NaN stats and drop out of the ECE weighting."""

    lo: float
    hi: float
    count: int
    mean_pred: float  # mean predicted P(unsafe) over the rows in the bin
    obs_freq: float  # observed unsafe fraction over the rows in the bin

    @property
    def gap(self) -> float:
        return abs(self.mean_pred - self.obs_freq)

    def as_dict(self) -> dict:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "count": self.count,
            "mean_pred": self.mean_pred,
            "obs_freq": self.obs_freq,
            "gap": self.gap,
        }


def _reliability_bin(lo: float, hi: float, group: list[tuple[float, str]]) -> ReliabilityBin:
    if not group:
        return ReliabilityBin(lo, hi, 0, float("nan"), float("nan"))
    mean_pred = sum(s for s, _ in group) / len(group)
    obs_freq = sum(1 for _, g in group if g == POSITIVE_LABEL) / len(group)
    return ReliabilityBin(lo, hi, len(group), mean_pred, obs_freq)


def reliability_diagram(
    scores: list[float], golds: list[str], n_bins: int = 10, *, scheme: str = "width"
) -> list[ReliabilityBin]:
    """Bin rows by predicted P(unsafe) and report mean-predicted vs observed-unsafe per
    bin - the reliability diagram, the read that asks whether the SCORE is a calibrated
    probability (not just a good RANK, which `roc_auc` already answers).

    scheme="width": fixed equal-width edges (i/n_bins). The standard Guo-et-al. ECE
    binning; empty bins are KEPT because at small N their emptiness is itself the finding
    (e.g. a "flag less" adapter that never emits a high score). scheme="frequency":
    ~equal-count bins (edges are the empirical score range of each ~N/n_bins block), the
    robustness read when equal-width bins go sparse. Assumes scores in [0, 1]."""
    if scheme == "width":
        edges = [i / n_bins for i in range(n_bins + 1)]
        groups: list[list[tuple[float, str]]] = [[] for _ in range(n_bins)]
        for s, g in zip(scores, golds):
            b = min(int(s * n_bins), n_bins - 1)
            groups[b].append((s, g))
        return [_reliability_bin(edges[b], edges[b + 1], groups[b]) for b in range(n_bins)]
    if scheme == "frequency":
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        bins: list[ReliabilityBin] = []
        for b in range(n_bins):
            block = order[(b * len(order)) // n_bins : ((b + 1) * len(order)) // n_bins]
            group = [(scores[i], golds[i]) for i in block]
            lo = min(s for s, _ in group) if group else float("nan")
            hi = max(s for s, _ in group) if group else float("nan")
            bins.append(_reliability_bin(lo, hi, group))
        return bins
    raise ValueError(f"unknown binning scheme: {scheme!r}")


def expected_calibration_error(bins: list[ReliabilityBin]) -> float:
    """ECE: the average bin gap |mean_pred - obs_freq|, weighted by bin occupancy. One
    number for how far the probabilities sit off the diagonal. NaN if no rows. WARNING at
    small N this is dominated by 2-5-row bins and barely defined - read it as a method
    check, not a trustworthy value."""
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return sum((b.count / total) * b.gap for b in bins if b.count)


def _logit(p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def temperature_scale(scores: list[float], temperature: float) -> list[float]:
    """Rescale confidence by a single scalar T: p_T = sigmoid(logit(p) / T). Because the
    score is RESTRICTED-BINARY (renormalized over just safe/unsafe), the probability IS
    the full logit difference, so this is exact temperature scaling from p alone - no raw
    logits needed. T > 1 SOFTENS toward 0.5 (fix for over-confidence), T < 1 SHARPENS
    toward the extremes (fix for under-confidence). It is monotonic: ROC-AUC and which
    side of 0.5 a row lands are UNCHANGED - only the probability values move, so it fixes
    calibration WITHOUT touching ranking or the 0.5 decision."""
    return [_sigmoid(_logit(p) / temperature) for p in scores]


def negative_log_likelihood(scores: list[float], golds: list[str]) -> float:
    """Mean binary cross-entropy of the probabilities against the golds (positive=unsafe).
    The proper scoring rule temperature scaling minimizes - unlike ECE it is smooth in T
    and penalizes confident wrongness. Lower is better."""
    eps = 1e-12
    total = 0.0
    for p, g in zip(scores, golds):
        p = min(max(p, eps), 1 - eps)
        y = 1.0 if g == POSITIVE_LABEL else 0.0
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(scores) if scores else float("nan")


def fit_temperature(
    scores: list[float], golds: list[str], bounds: tuple[float, float] = (0.05, 20.0)
) -> float:
    """The standard post-hoc fix (Guo et al.): the single T minimizing NLL on this split.
    Golden-section search in log-T (T is multiplicative). Fit on VALID and apply to
    holdout for the honest number; fit on holdout for the optimistic best-case reference."""
    invphi = (5**0.5 - 1) / 2
    a, b = math.log(bounds[0]), math.log(bounds[1])

    def f(log_t: float) -> float:
        return negative_log_likelihood(temperature_scale(scores, math.exp(log_t)), golds)

    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return math.exp((a + b) / 2)


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
