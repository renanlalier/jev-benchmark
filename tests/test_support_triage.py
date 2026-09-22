import pytest

from jev_benchmark.support_triage import ANTHROPIC_TOOL_NAME, _anthropic_tool_input, _validated


def test_anthropic_tool_input_extracts_structured_labels() -> None:
    values = {
        "route": "BILLING",
        "urgency": "HIGH",
        "risk": "PAYMENT_DISPUTE",
        "action": "HUMAN_ESCALATION",
    }

    assert _anthropic_tool_input(
        {
            "content": [
                {"type": "text", "text": "unused"},
                {"type": "tool_use", "name": ANTHROPIC_TOOL_NAME, "input": values},
            ]
        }
    ) == values


def test_anthropic_tool_input_rejects_missing_tool_call() -> None:
    with pytest.raises(ValueError, match="triage tool input"):
        _anthropic_tool_input({"content": [{"type": "text", "text": "{}"}]})


def test_triage_validation_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="route"):
        _validated(
            {
                "route": "UNKNOWN",
                "urgency": "NORMAL",
                "risk": "NONE",
                "action": "AUTO_REPLY",
            }
        )
