from jev_benchmark.models import BenchmarkCase, Prediction, ProviderResult
from jev_benchmark.runner import summarize


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="1",
        text="I spent 50 dollars at the supermarket",
        expected_intent="FINANCIAL_TRANSACTION",
        expected_sentiment="NEUTRAL",
        expected_escalation=False,
    )


def test_summary_reports_accuracy_latency_and_consistency() -> None:
    case = _case()
    records = [
        (
            case,
            ProviderResult(
                provider="fake",
                model="fake-1",
                prediction=Prediction(
                    intent="FINANCIAL_TRANSACTION",
                    sentiment="NEUTRAL",
                    escalation=False,
                ),
                latency_ms=100,
            ),
        ),
        (
            case,
            ProviderResult(
                provider="fake",
                model="fake-1",
                prediction=Prediction(
                    intent="FINANCIAL_TRANSACTION",
                    sentiment="NEUTRAL",
                    escalation=False,
                ),
                latency_ms=200,
            ),
        ),
    ]

    result = summarize(records)

    assert result["intent_accuracy"] == 1
    assert result["sentiment_accuracy"] == 1
    assert result["escalation_accuracy"] == 1
    assert result["overall_exact_match"] == 1
    assert result["latency_ms_mean"] == 150
    assert result["mean_run_consistency"] == 1
