import pytest
import respx
from httpx import Response

from ft_cm.eval import _strip_rows, evaluate
from ft_cm.providers.base import Provider
from ft_cm.providers.ollama import OllamaProvider


@pytest.mark.mocked
@respx.mock
async def test_ollama_provider_returns_response():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(200, json={"response": "unsafe"})
    )
    out = await OllamaProvider().generate("classify this")
    assert out == "unsafe"


class _ScriptedProvider(Provider):
    """A fake backend that replays a fixed answer per input text, so the eval
    harness can be exercised end to end with no model."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    async def generate(self, prompt: str) -> str:
        for key, val in self.answers.items():
            if key in prompt:
                return val
        return ""


@pytest.mark.mocked
async def test_evaluate_scores_a_perfect_run():
    holdout = [{"text": "a kind hello", "label": "safe"}, {"text": "a threat", "label": "unsafe"}]
    provider = _ScriptedProvider({"kind hello": "safe", "threat": "unsafe"})
    result = await evaluate(provider, holdout)
    assert result.accuracy == 1.0
    assert result.n_none == 0


@pytest.mark.mocked
async def test_evaluate_counts_non_answers_and_confusion():
    holdout = [{"text": "a kind hello", "label": "safe"}, {"text": "a threat", "label": "unsafe"}]
    provider = _ScriptedProvider({"kind hello": "safe", "threat": "I cannot help"})
    result = await evaluate(provider, holdout)
    assert result.accuracy == 0.5
    assert result.n_none == 1
    assert result.confusion["unsafe"]["none"] == 1


@pytest.mark.mocked
def test_strip_rows_drops_raw_text_keeps_metrics():
    # The committable receipt for REAL data must carry NO raw comment text.
    full = {
        "n": 2,
        "baseline": {
            "accuracy": 0.5,
            "confusion": {"unsafe": {"safe": 1}},
            "rows": [{"text": "a real toxic comment", "raw": "safe"}],
        },
    }
    slim = _strip_rows(full)
    assert "rows" not in slim["baseline"]  # raw text gone
    assert slim["baseline"]["accuracy"] == 0.5  # metrics kept
    assert slim["baseline"]["confusion"] == {"unsafe": {"safe": 1}}
    assert full["baseline"]["rows"]  # original untouched (deep copy)
