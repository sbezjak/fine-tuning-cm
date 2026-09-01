"""The reporting step: render HTML off a receipts dict. Pure/mocked ($0, no model).

The report is an artifact of the receipts, so the test proves it renders the three
sections and - the load-bearing part - the verbatim raw completion for every row plus
the two marks the finding rests on: the safety-critical unsafe->safe confusion cell,
and the flip where the adapter changed the answer. See ft_cm/report.py.
"""

from __future__ import annotations

import pytest

from ft_cm.report import render_report


def _label_block(n: int, correct: int, safe_acc: float, unsafe_acc: float, conf: dict, rows: list[dict]) -> dict:
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "per_label_accuracy": {"safe": safe_acc, "unsafe": unsafe_acc},
        "confusion": conf,
        "n_none": 0,
        "n_ambiguous": 0,
        "rows": rows,
    }


def _receipts() -> dict:
    benign = "That final boss killed me twenty times."  # benign, base over-flags
    threat = "Be a pity if something happened to your shop."  # veiled threat
    before = _label_block(
        2, 1, 0.0, 1.0,
        {"safe": {"safe": 0, "unsafe": 1}, "unsafe": {"safe": 0, "unsafe": 1}},
        [
            {"text": benign, "gold": "safe", "pred": "unsafe", "raw": "unsafe", "ok": False},
            {"text": threat, "gold": "unsafe", "pred": "unsafe", "raw": "unsafe", "ok": True},
        ],
    )
    after = _label_block(
        2, 1, 1.0, 0.0,
        {"safe": {"safe": 1, "unsafe": 0}, "unsafe": {"safe": 1, "unsafe": 0}},
        [
            {"text": benign, "gold": "safe", "pred": "safe", "raw": "safe", "ok": True},
            {"text": threat, "gold": "unsafe", "pred": "safe", "raw": "safe", "ok": False},
        ],
    )
    return {
        "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "adapter_path": "adapters/smoke",
        "holdout": "data/smoke/prepared/hard-holdout.jsonl",
        "n": 2,
        "before": before,
        "after": after,
        "delta_accuracy": 0.0,
    }


@pytest.mark.mocked
def test_report_renders_sections_and_ground_truth(tmp_path):
    rep = _receipts()
    out = render_report(rep, tmp_path / "r.html")
    doc = out.read_text()

    # the three sections
    assert "Summary - before" in doc
    assert "Confusion grid" in doc
    assert "Full trace" in doc

    # the two marks the finding rests on
    assert "class='crit'" in doc  # safety-critical unsafe->safe confusion cell
    assert "MISS" in doc  # the unsafe->safe row after tuning
    assert doc.count("<span class='flip'>flip</span>") == 2  # both rows flipped base->tuned

    # ground truth: every row's verbatim text and raw completion is present
    for r in rep["after"]["rows"]:
        assert r["text"] in doc
    assert "base" in doc and "tuned" in doc


@pytest.mark.mocked
def test_report_handles_sparse_confusion(tmp_path):
    """Regression: a PERFECT column makes the confusion dict SPARSE - a cell that never
    occurred is simply absent (a flawless unsafe column has no unsafe->safe key), so the
    renderer must default missing cells to 0 instead of raising KeyError. This only ever
    fires on a SUCCESS, so a normal (dense) fixture never catches it. See ft_cm/report.py."""
    benign = "That final boss killed me twenty times."
    threat = "Be a pity if something happened to your shop."
    # after: safe 1/1, unsafe 1/1 -> the confusion holds ONLY the diagonal keys.
    after = _label_block(
        2, 2, 1.0, 1.0,
        {"safe": {"safe": 1}, "unsafe": {"unsafe": 1}},
        [
            {"text": benign, "gold": "safe", "pred": "safe", "raw": "safe", "ok": True},
            {"text": threat, "gold": "unsafe", "pred": "unsafe", "raw": "unsafe", "ok": True},
        ],
    )
    rep = _receipts()
    rep["after"] = after
    rep["delta_accuracy"] = 1.0

    out = render_report(rep, tmp_path / "sparse.html")  # must not raise KeyError
    doc = out.read_text()
    assert "Confusion grid" in doc
    # the absent unsafe->safe cell renders as 0, still flagged as the safety-critical cell
    assert "class='crit'>0<" in doc
