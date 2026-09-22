import csv
import json

import pytest

from jev_benchmark.models import BenchmarkCase, Prediction, ProviderResult
from jev_benchmark.runner import load_cases, percentile, summarize, write_results


def make_case() -> BenchmarkCase:
    return BenchmarkCase(
        id="1",
        text="hello",
        expected_intent="GENERAL_CHAT",
        expected_sentiment="NEUTRAL",
        expected_escalation=False,
    )


def test_summary_mixed_predictions_includes_percentiles_consistency_and_cost() -> None:
    case = make_case()
    good = ProviderResult(
        provider="test",
        model="m",
        prediction=Prediction(intent="GENERAL_CHAT", sentiment="NEUTRAL", escalation=False),
        latency_ms=100,
        estimated_cost_usd=0.01,
    )
    bad = ProviderResult(
        provider="test",
        model="m",
        prediction=Prediction(intent="OTHER", sentiment="ANGRY", escalation=True),
        latency_ms=200,
        estimated_cost_usd=0.02,
    )

    summary = summarize([(case, good), (case, bad)])

    assert summary["intent_accuracy"] == 0.5
    assert summary["overall_exact_match"] == 0.5
    assert summary["latency_ms_p50"] == 150
    assert summary["latency_ms_p95"] == 195
    assert summary["mean_run_consistency"] == 0.5
    assert summary["estimated_total_cost_usd"] == pytest.approx(0.03)


def test_empty_summary_and_percentile_edge_case() -> None:
    assert summarize([]) == {}
    assert percentile([42], 0.95) == 42


def test_load_cases_and_write_results(tmp_path) -> None:
    dataset = tmp_path / "cases.csv"
    dataset.write_text(
        "id,text,expected_intent,expected_sentiment,expected_escalation\n1,hello,GENERAL_CHAT,NEUTRAL, TRUE \n",
        encoding="utf-8",
    )

    cases = load_cases(dataset)
    result = ProviderResult(
        provider="test",
        model="m",
        prediction=Prediction(intent="GENERAL_CHAT", sentiment="NEUTRAL", escalation=True),
        latency_ms=12,
    )
    output = write_results({"test": [(cases[0], result)]}, tmp_path / "results")
    predictions = output.parent / "predictions.csv"

    assert cases[0].expected_escalation is True
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["test"]["samples"] == 1
    with predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case_id"] == "1"
    assert rows[0]["predicted_escalation"] == "True"


@pytest.mark.asyncio
async def test_run_provider_repeats_cases() -> None:
    class Provider:
        name = "fake"
        model = "m"

        async def classify(self, case: BenchmarkCase) -> ProviderResult:
            prediction = Prediction(
                intent=case.expected_intent,
                sentiment=case.expected_sentiment,
                escalation=case.expected_escalation,
            )
            return ProviderResult(
                provider=self.name, model=self.model, prediction=prediction, latency_ms=1
            )

    from jev_benchmark.runner import run_provider

    assert len(await run_provider(Provider(), [make_case()], 2)) == 2
