from __future__ import annotations

import asyncio
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from jev_benchmark.models import BenchmarkCase, Intent, ProviderResult, Sentiment
from jev_benchmark.providers.base import BenchmarkProvider


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    rows: list[BenchmarkCase] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                BenchmarkCase(
                    id=row["id"],
                    text=row["text"],
                    expected_intent=Intent(row["expected_intent"]),
                    expected_sentiment=Sentiment(row["expected_sentiment"]),
                    expected_escalation=row["expected_escalation"].strip().lower() == "true",
                )
            )
    return rows


async def run_provider(
    provider: BenchmarkProvider,
    cases: list[BenchmarkCase],
    repetitions: int,
) -> list[tuple[BenchmarkCase, ProviderResult]]:
    output: list[tuple[BenchmarkCase, ProviderResult]] = []
    for _ in range(repetitions):
        for case in cases:
            result = await provider.classify(case)
            output.append((case, result))
    return output


async def run_all(
    providers: list[BenchmarkProvider],
    cases: list[BenchmarkCase],
    repetitions: int,
) -> dict[str, list[tuple[BenchmarkCase, ProviderResult]]]:
    results = await asyncio.gather(
        *(run_provider(provider, cases, repetitions) for provider in providers)
    )
    return {provider.name: result for provider, result in zip(providers, results, strict=True)}


def summarize(records: list[tuple[BenchmarkCase, ProviderResult]]) -> dict[str, object]:
    if not records:
        return {}

    intent_hits = 0
    sentiment_hits = 0
    escalation_hits = 0
    latencies: list[float] = []
    predictions_by_case: dict[str, list[str]] = defaultdict(list)

    for case, result in records:
        p = result.prediction
        intent_hits += int(p.intent == case.expected_intent)
        sentiment_hits += int(p.sentiment == case.expected_sentiment)
        escalation_hits += int(p.escalation == case.expected_escalation)
        latencies.append(result.latency_ms)
        predictions_by_case[case.id].append(f"{p.intent}|{p.sentiment}|{str(p.escalation).lower()}")

    n = len(records)
    consistency_scores = []
    for values in predictions_by_case.values():
        counts = Counter(values)
        consistency_scores.append(max(counts.values()) / len(values))

    costs = [r.estimated_cost_usd for _, r in records if r.estimated_cost_usd is not None]

    return {
        "provider": records[0][1].provider,
        "model": records[0][1].model,
        "samples": n,
        "intent_accuracy": intent_hits / n,
        "sentiment_accuracy": sentiment_hits / n,
        "escalation_accuracy": escalation_hits / n,
        "overall_exact_match": sum(
            1
            for case, r in records
            if r.prediction.intent == case.expected_intent
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
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary_path
