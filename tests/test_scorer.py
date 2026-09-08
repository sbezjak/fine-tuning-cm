import pytest

from ft_cm import taxonomy
from ft_cm.scorer import extract_label, is_correct

# The scorer matches against the ACTIVE label set (taxonomy.LABELS), which the task
# switch drives. The whole-word / hedge / none MECHANISM is task-independent, but the
# labels differ, so the label-specific cases are split by task: the gate runs the set
# for whichever task is active - binary by default, grounds under FT_CM_TASK=grounds.
binary_only = pytest.mark.skipif(
    taxonomy.TASK != "binary", reason="binary-task scorer case; default task"
)
grounds_only = pytest.mark.skipif(
    taxonomy.TASK != "grounds", reason="grounds-task scorer case; run under FT_CM_TASK=grounds"
)


# --- task-independent mechanism ---
@pytest.mark.mocked
def test_no_label_is_none_and_wrong():
    ext = extract_label("I cannot help with that.")
    assert ext.label is None
    assert not is_correct("I cannot help with that.", "safe")


# --- binary task: the article's safe/unsafe classifier ---
@pytest.mark.mocked
@binary_only
def test_binary_clean_labels():
    assert extract_label("safe").label == "safe"
    assert extract_label("unsafe").label == "unsafe"


@pytest.mark.mocked
@binary_only
def test_safe_is_not_read_inside_unsafe():
    r"""The article's trap: 'safe' is a substring of 'unsafe'. Whole-word matching
    (\bsafe\b) must not fire inside 'unsafe', so 'unsafe' extracts to 'unsafe', not 'safe'."""
    assert extract_label("unsafe").label == "unsafe"
    assert extract_label("This message is safe to publish.").label == "safe"


@pytest.mark.mocked
@binary_only
def test_binary_hedge_is_ambiguous():
    ext = extract_label("could be safe, but parts read as unsafe")
    assert ext.ambiguous


@pytest.mark.mocked
@binary_only
def test_binary_case_insensitive():
    assert extract_label("UNSAFE").label == "unsafe"
    assert is_correct("Safe", "safe")


# --- grounds task: the 5-way harm-ground ablation ---
@pytest.mark.mocked
@grounds_only
def test_grounds_clean_labels():
    assert extract_label("threat").label == "threat"
    assert extract_label("insult").label == "insult"
    assert extract_label("safe").label == "safe"


@pytest.mark.mocked
@grounds_only
def test_grounds_binary_word_extracts_to_none():
    r"""There is no 'unsafe' ground; the old binary word must extract to None, not be
    misread as the 'safe' ground (\bsafe\b does not match inside 'unsafe')."""
    ext = extract_label("unsafe")
    assert ext.label is None
    assert not ext.ambiguous


@pytest.mark.mocked
@grounds_only
def test_grounds_label_inside_a_sentence():
    assert extract_label("This message is safe to publish.").label == "safe"
    assert extract_label("I would mark this as a threat.").label == "threat"


@pytest.mark.mocked
@grounds_only
def test_grounds_hedge_is_ambiguous_and_wrong():
    ext = extract_label("It could be an insult or a threat depending on context.")
    assert ext.ambiguous
    assert not is_correct("insult or threat", "insult")


@pytest.mark.mocked
@grounds_only
def test_grounds_case_insensitive():
    assert extract_label("THREAT").label == "threat"
    assert is_correct("Safe", "safe")
