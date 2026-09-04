import pytest

from ft_cm.scorer import extract_label, is_correct


@pytest.mark.mocked
def test_clean_labels():
    assert extract_label("threat").label == "threat"
    assert extract_label("insult").label == "insult"
    assert extract_label("safe").label == "safe"


@pytest.mark.mocked
def test_unsafe_word_is_not_read_as_safe():
    """Word-boundary trap: 'safe' is a substring of 'unsafe'. A model that emits the
    old binary word 'unsafe' (there is no such ground now) must NOT be read as the
    'safe' ground - \\bsafe\\b does not match inside 'unsafe', so it extracts to None."""
    ext = extract_label("unsafe")
    assert ext.label is None
    assert not ext.ambiguous


@pytest.mark.mocked
def test_label_inside_a_sentence():
    assert extract_label("This message is safe to publish.").label == "safe"
    assert extract_label("I would mark this as a threat.").label == "threat"


@pytest.mark.mocked
def test_hedge_is_ambiguous_and_wrong():
    ext = extract_label("It could be an insult or a threat depending on context.")
    assert ext.ambiguous
    assert not is_correct("insult or threat", "insult")


@pytest.mark.mocked
def test_no_label_is_none_and_wrong():
    ext = extract_label("I cannot help with that.")
    assert ext.label is None
    assert not is_correct("I cannot help with that.", "safe")


@pytest.mark.mocked
def test_case_insensitive():
    assert extract_label("THREAT").label == "threat"
    assert is_correct("Safe", "safe")
