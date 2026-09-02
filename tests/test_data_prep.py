import json

import pytest

from ft_cm.data_prep import dedup_near, max_cross_jaccard, prepare, stratified_split, to_chat
from ft_cm.taxonomy import LABELS


def _balanced(n_each=10):
    return [{"text": f"safe {i}", "label": "safe"} for i in range(n_each)] + [
        {"text": f"unsafe {i}", "label": "unsafe"} for i in range(n_each)
    ]


@pytest.mark.mocked
def test_split_is_deterministic():
    recs = _balanced()
    a = stratified_split(recs, seed=7)
    b = stratified_split(recs, seed=7)
    assert a == b


@pytest.mark.mocked
def test_split_is_stratified_every_class_present():
    train, valid, test = stratified_split(_balanced(), seed=0)
    for split in (train, valid, test):
        labels = {r["label"] for r in split}
        assert labels == set(LABELS), f"a split is missing a class: {labels}"


@pytest.mark.mocked
def test_no_leakage_across_splits():
    train, valid, test = stratified_split(_balanced(), seed=3)
    texts = [r["text"] for r in train + valid + test]
    assert len(texts) == len(set(texts)), "a record appears in more than one split"


@pytest.mark.mocked
def test_to_chat_shape():
    row = to_chat({"text": "hello", "label": "safe"})
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][-1]["content"] == "safe"


@pytest.mark.mocked
def test_prepare_writes_all_files(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _balanced()))
    out = tmp_path / "prepared"
    counts = prepare(src, out, seed=0)

    for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "holdout.jsonl"):
        assert (out / name).is_file()
    assert counts["train"] + counts["valid"] + counts["test"] == 20

    holdout = [json.loads(x) for x in (out / "holdout.jsonl").read_text().splitlines()]
    assert all("text" in r and "label" in r for r in holdout)


@pytest.mark.mocked
def test_prepare_rejects_bad_label(tmp_path):
    src = tmp_path / "bad.jsonl"
    src.write_text(json.dumps({"text": "x", "label": "toxic"}) + "\n")
    with pytest.raises(ValueError):
        prepare(src, tmp_path / "out")


@pytest.mark.mocked
def test_dedup_near_drops_reposts_keeps_distinct():
    recs = [
        {"text": "the health care bill takes insurance from a baby", "label": "unsafe"},
        {"text": "The health care bill takes insurance from a baby!", "label": "unsafe"},  # near-dup
        {"text": "a completely different comment about gardening", "label": "safe"},
    ]
    kept = dedup_near(recs, threshold=0.5)
    assert len(kept) == 2  # the near-dup repost is dropped, the distinct row stays
    assert kept[0]["text"] == recs[0]["text"]  # first occurrence is the one kept


@pytest.mark.mocked
def test_dedup_off_by_default_smoke_unchanged(tmp_path):
    # dedup_threshold=None (default) must not drop anything: smoke prep stays identical.
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _balanced()))
    counts = prepare(src, tmp_path / "out", seed=0)
    assert counts["train"] + counts["valid"] + counts["test"] == 20
    assert "dropped_near_dup" not in counts  # no dedup path taken


@pytest.mark.mocked
def test_prepare_reports_leakage_guard(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _balanced()))
    counts = prepare(src, tmp_path / "out", seed=0, dedup_threshold=0.5)
    # distinct synthetic rows -> held-out shares no near-dup with train/valid.
    assert counts["max_holdout_train_jaccard"] < 0.5


@pytest.mark.mocked
def test_max_cross_jaccard_catches_overlap():
    a = [{"text": "one two three four", "label": "safe"}]
    b = [{"text": "one two three four", "label": "safe"}]  # identical token set
    assert max_cross_jaccard(a, b) == 1.0
