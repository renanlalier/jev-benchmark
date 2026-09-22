from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from jev_benchmark.models import BenchmarkCase, Intent, ProviderResult, Sentiment
from jev_benchmark.providers.base import BenchmarkProvider

LOGGER = logging.getLogger(__name__)


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    rows: list[BenchmarkCase] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            escalation = row["expected_escalation"].strip().lower()
            if escalation not in {"true", "false"}:
                raise ValueError(f"Case {row['id']} has invalid expected_escalation: {escalation}")
            rows.append(
                BenchmarkCase(
                    id=row["id"],
                    text=row["text"],
                    expected_intent=Intent(row["expected_intent"]),
                    expected_sentiment=Sentiment(row["expected_sentiment"]),
                    expected_escalation=escalation == "true",
                )
            )
    LOGGER.info("Loaded %d case(s) from %s", len(rows), path)
    return rows


async def run_provider(
    provider: BenchmarkProvider,
    cases: list[BenchmarkCase],
    repetitions: int,
) -> list[tuple[BenchmarkCase, ProviderResult]]:
    output: list[tuple[BenchmarkCase, ProviderResult]] = []
    total = len(cases) * repetitions
    LOGGER.info("[%s] Starting %d classification(s)", provider.name, total)
    for _ in range(repetitions):
        for case in cases:
            started = asyncio.get_running_loop().time()
            try:
                result = await provider.classify(case)
            except (httpx.HTTPError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result = ProviderResult(
                    provider=provider.name,
                    model=provider.model,
                    latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
                LOGGER.warning("[%s] Case %s failed: %s", provider.name, case.id, result.error)
            output.append((case, result))
    LOGGER.info("[%s] Completed %d classification(s)", provider.name, len(output))
    return output


async def run_all(
    providers: list[BenchmarkProvider],
    cases: list[BenchmarkCase],
    repetitions: int,
) -> dict[str, list[tuple[BenchmarkCase, ProviderResult]]]:
    LOGGER.info("Running %d provider(s) concurrently", len(providers))
    results = await asyncio.gather(
        *(run_provider(provider, cases, repetitions) for provider in providers)
    )
    return {provider.name: result for provider, result in zip(providers, results, strict=True)}


def summarize(records: list[tuple[BenchmarkCase, ProviderResult]]) -> dict[str, object]:
    if not records:
        return {}

    successful = [(case, result) for case, result in records if result.prediction is not None]
    if not successful:
        return {
            "provider": records[0][1].provider,
            "model": records[0][1].model,
            "samples": len(records),
            "failure_count": len(records),
            "failure_rate": 1.0,
        }
    intent_hits = 0
    sentiment_hits = 0
    escalation_hits = 0
    latencies: list[float] = []
    predictions_by_case: dict[str, list[str]] = defaultdict(list)

    for case, result in successful:
        p = result.prediction
        assert p is not None
        intent_hits += int(p.intent == case.expected_intent)
        sentiment_hits += int(p.sentiment == case.expected_sentiment)
        escalation_hits += int(p.escalation == case.expected_escalation)
        latencies.append(result.latency_ms)
        predictions_by_case[case.id].append(f"{p.intent}|{p.sentiment}|{str(p.escalation).lower()}")

    n = len(successful)
    consistency_scores = []
    for values in predictions_by_case.values():
        counts = Counter(values)
        consistency_scores.append(max(counts.values()) / len(values))

    costs = [r.estimated_cost_usd for _, r in records if r.estimated_cost_usd is not None]

    return {
        "provider": records[0][1].provider,
        "model": records[0][1].model,
        "samples": len(records),
        "successful_samples": n,
        "failure_count": len(records) - n,
        "failure_rate": (len(records) - n) / len(records),
        "intent_accuracy": intent_hits / n,
        "sentiment_accuracy": sentiment_hits / n,
        "escalation_accuracy": escalation_hits / n,
        "overall_exact_match": sum(
            1
            for case, r in successful
            if r.prediction is not None
            and r.prediction.intent == case.expected_intent
            and r.prediction.sentiment == case.expected_sentiment
            and r.prediction.escalation == case.expected_escalation
        )
        / n,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "mean_run_consistency": statistics.mean(consistency_scores),
        "estimated_total_cost_usd": sum(costs) if costs else None,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def write_results(
    results: dict[str, list[tuple[BenchmarkCase, ProviderResult]]],
    output_dir: str | Path,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary = {name: summarize(records) for name, records in results.items()}
    LOGGER.info("Writing summary for %d provider(s)", len(summary))
    prediction_path = path / "predictions.csv"
    fieldnames = [
        "provider",
        "model",
        "case_id",
        "text",
        "expected_intent",
        "expected_sentiment",
        "expected_escalation",
        "predicted_intent",
        "predicted_sentiment",
        "predicted_escalation",
        "intent_confidence",
        "sentiment_confidence",
        "escalation_confidence",
        "jev_intent_score",
        "jev_sentiment_score",
        "jev_escalation_score",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "error",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for records in results.values():
            for case, result in records:
                prediction = result.prediction
                writer.writerow(
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "case_id": case.id,
                        "text": case.text,
                        "expected_intent": case.expected_intent.value,
                        "expected_sentiment": case.expected_sentiment.value,
                        "expected_escalation": case.expected_escalation,
                        "predicted_intent": prediction.intent.value if prediction else None,
                        "predicted_sentiment": prediction.sentiment.value if prediction else None,
                        "predicted_escalation": prediction.escalation if prediction else None,
                        "intent_confidence": prediction.intent_confidence if prediction else None,
                        "sentiment_confidence": prediction.sentiment_confidence
                        if prediction
                        else None,
                        "escalation_confidence": prediction.escalation_confidence
                        if prediction
                        else None,
                        "jev_intent_score": result.jev_scores.get("intent")
                        if result.jev_scores
                        else None,
                        "jev_sentiment_score": result.jev_scores.get("sentiment")
                        if result.jev_scores
                        else None,
                        "jev_escalation_score": result.jev_scores.get("escalation")
                        if result.jev_scores
                        else None,
                        "latency_ms": result.latency_ms,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "estimated_cost_usd": result.estimated_cost_usd,
                        "error": result.error,
                    }
                )
    LOGGER.info("Predictions written: %s", prediction_path)
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary_path
