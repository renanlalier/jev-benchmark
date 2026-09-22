import csv

import pytest

from jev_benchmark.support_triage import (
    ANTHROPIC_TOOL_NAME,
    SupportCase,
    SupportResult,
    _anthropic_tool_input,
    _validated,
    summarize_support,
    write_support_results,
)


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


def test_support_predictions_include_jev_scores(tmp_path) -> None:
    case = SupportCase("1", "billing issue", "BILLING", "HIGH", "PAYMENT_DISPUTE", "HUMAN_ESCALATION")
    result = SupportResult(
        "jev",
        "jev-latest",
        {
            "route": "BILLING",
            "urgency": "HIGH",
            "risk": "PAYMENT_DISPUTE",
            "action": "HUMAN_ESCALATION",
        },
        12.0,
        jev_scores={"route": 0.9, "urgency": 0.8, "risk": 0.7, "action": 0.6},
        input_tokens=1_000,
        output_tokens=10,
        estimated_cost_usd=0.123,
    )

    metrics = summarize_support([(case, result)])
    assert metrics["priced_samples"] == 1
    assert metrics["estimated_total_cost_usd"] == pytest.approx(0.123)
    assert metrics["estimated_cost_per_priced_request_usd"] == pytest.approx(0.123)

    summary = write_support_results({"jev": [(case, result)]}, tmp_path)
    with (summary.parent / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["jev_route_score"] == "0.9"
    assert row["jev_urgency_score"] == "0.8"
    assert row["jev_risk_score"] == "0.7"
    assert row["jev_action_score"] == "0.6"
    assert row["input_tokens"] == "1000"
    assert row["output_tokens"] == "10"
    assert row["estimated_cost_usd"] == "0.123"
