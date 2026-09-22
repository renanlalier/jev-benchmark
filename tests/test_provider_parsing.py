import pytest

from jev_benchmark.providers.anthropic_provider import _strip_code_fence
from jev_benchmark.providers.jev import _confidence, extract_jev_score
from jev_benchmark.providers.openai_provider import _extract_output_text


def test_openai_extracts_top_level_and_nested_output_text() -> None:
    assert _extract_output_text({"output_text": "top-level"}) == "top-level"
    assert _extract_output_text({"output": [{"content": [{"text": "nested"}]}]}) == "nested"


def test_openai_rejects_response_without_text() -> None:
    with pytest.raises(ValueError, match="did not contain output text"):
        _extract_output_text({"output": []})


def test_anthropic_code_fence_and_jev_confidence_parsing() -> None:
    assert _strip_code_fence('```json\n{"intent": "OTHER"}\n```') == '{"intent": "OTHER"}'
    assert _strip_code_fence('{"intent": "OTHER"}') == '{"intent": "OTHER"}'
    assert _confidence({"confidence": 0.8}) == 0.8
    assert _confidence({"probability": 0.7}) == 0.7
    assert extract_jev_score({"score": 0.9}) == 0.9
    assert extract_jev_score(0.6) == 0.6
    assert _confidence("OTHER") is None
