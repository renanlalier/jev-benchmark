from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from jev_benchmark.providers.openai_provider import _extract_output_text
from jev_benchmark.task_policies import SUPPORT_TRIAGE_CRITERIA, support_triage_prompt


@dataclass(frozen=True)
class SupportCase:
    id: str
    text: str
    route: str
    urgency: str
    risk: str
    action: str


@dataclass(frozen=True)
class SupportResult:
    provider: str
    model: str
    values: dict[str, str] | None
    latency_ms: float
    error: str | None = None


LOGGER = logging.getLogger(__name__)

FIELDS = {
    "route": [
        "BILLING",
        "TECHNICAL_SUPPORT",
        "ACCOUNT_ACCESS",
        "FRAUD_SECURITY",
        "CANCELLATION_RETENTION",
    ],
    "urgency": ["LOW", "NORMAL", "HIGH", "CRITICAL"],
    "risk": ["NONE", "PAYMENT_DISPUTE", "ACCOUNT_TAKEOVER", "SAFETY_PRIVACY"],
    "action": [
        "AUTO_REPLY",
        "REQUEST_INFORMATION",
        "PRIORITY_QUEUE",
        "HUMAN_ESCALATION",
        "SECURITY_FREEZE",
    ],
}


def load_support_cases(path: str | Path) -> list[SupportCase]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        cases = [
            SupportCase(
                id=row["id"],
                text=row["text"],
                route=row["expected_route"],
                urgency=row["expected_urgency"],
                risk=row["expected_risk"],
                action=row["expected_action"],
            )
            for row in csv.DictReader(handle)
        ]
    for case in cases:
        values = {
            "route": case.route,
            "urgency": case.urgency,
            "risk": case.risk,
            "action": case.action,
        }
        invalid = [name for name, labels in FIELDS.items() if values[name] not in labels]
        if invalid:
            raise ValueError(f"Case {case.id} has invalid labels: {', '.join(invalid)}")
    return cases


def _prompt() -> str:
    return support_triage_prompt()


async def _openai(case: SupportCase) -> SupportResult:
    payload = {
        "model": "gpt-5.6-luna",
        "input": [{"role": "system", "content": _prompt()}, {"role": "user", "content": case.text}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "customer_support_triage",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string", "enum": labels} for name, labels in FIELDS.items()
                    },
                    "required": list(FIELDS),
                    "additionalProperties": False,
                },
            }
        },
    }
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json=payload,
        )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    text = _extract_output_text(data)
    return SupportResult(
        "openai",
        str(payload["model"]),
        _validated(json.loads(text)),
        (time.perf_counter() - started) * 1000,
    )


ANTHROPIC_TOOL_NAME = "classify_customer_support"


def _anthropic_tool_input(data: dict[str, Any]) -> dict[str, Any]:
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == ANTHROPIC_TOOL_NAME:
            values = block.get("input")
            if isinstance(values, dict):
                return values
    raise ValueError("Anthropic response did not contain the triage tool input")


async def _anthropic(case: SupportCase) -> SupportResult:
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 160,
        "system": _prompt(),
        "messages": [{"role": "user", "content": case.text}],
        "tools": [
            {
                "name": ANTHROPIC_TOOL_NAME,
                "description": "Return the customer-support triage labels.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string", "enum": labels} for name, labels in FIELDS.items()
                    },
                    "required": list(FIELDS),
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": ANTHROPIC_TOOL_NAME},
    }
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return SupportResult(
        "anthropic",
        str(payload["model"]),
        _validated(_anthropic_tool_input(data)),
        (time.perf_counter() - started) * 1000,
    )


async def _jev(case: SupportCase) -> SupportResult:
    questions = {
        name: {
            "type": "choice",
            "instructions": f"Classify the support {name}.",
            "criteria": SUPPORT_TRIAGE_CRITERIA[name],
        }
        for name, labels in FIELDS.items()
    }
    payload = {"model": "jev-latest", "state": case.text, "questions": questions}
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"},
            json=payload,
        )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    answers = data["answers"]
    values = {name: str(answers[name]["choice"]) for name in FIELDS}
    return SupportResult(
        "jev", str(payload["model"]), _validated(values), (time.perf_counter() - started) * 1000
    )


def _validated(values: dict[str, Any]) -> dict[str, str]:
    parsed = {name: str(values[name]) for name in FIELDS}
    invalid = [name for name, labels in FIELDS.items() if parsed[name] not in labels]
    if invalid:
        raise ValueError(f"Invalid triage labels: {', '.join(invalid)}")
    return parsed


async def run_support_triage(
    providers: list[str], cases: list[SupportCase], repetitions: int
) -> dict[str, list[tuple[SupportCase, SupportResult]]]:
    functions = {"jev": _jev, "anthropic": _anthropic, "openai": _openai}

    async def run_one(provider: str) -> tuple[str, list[tuple[SupportCase, SupportResult]]]:
        LOGGER.info("[%s] Starting %d triage decision(s)", provider, len(cases) * repetitions)
        records: list[tuple[SupportCase, SupportResult]] = []
        for _ in range(repetitions):
            for case in cases:
                started = asyncio.get_running_loop().time()
                try:
                    result = await functions[provider](case)
                except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                    result = SupportResult(
                        provider,
                        {
                            "jev": "jev-latest",
                            "anthropic": "claude-haiku-4-5-20251001",
                            "openai": "gpt-5.6-luna",
                        }[provider],
                        None,
                        (asyncio.get_running_loop().time() - started) * 1000,
                        f"{type(exc).__name__}: {exc}",
                    )
                    LOGGER.warning("[%s] Case %s failed: %s", provider, case.id, result.error)
                records.append((case, result))
        LOGGER.info("[%s] Completed %d triage decision(s)", provider, len(records))
        return provider, records

    return dict(await asyncio.gather(*(run_one(provider) for provider in providers)))


def summarize_support(records: list[tuple[SupportCase, SupportResult]]) -> dict[str, object]:
    if not records:
        return {}
    successful = [(case, result) for case, result in records if result.values is not None]
    failures = len(records) - len(successful)
    base: dict[str, object] = {
        "provider": records[0][1].provider,
        "model": records[0][1].model,
        "samples": len(records),
        "successful_samples": len(successful),
        "failure_count": failures,
        "failure_rate": failures / len(records),
    }
    if not successful:
        return base
    expected = {"route": "route", "urgency": "urgency", "risk": "risk", "action": "action"}
    total = len(successful)
    accuracy = {
        name: sum(
            result.values is not None and result.values[name] == getattr(case, attribute)
            for case, result in successful
        )
        / total
        for name, attribute in expected.items()
    }
    latencies = [result.latency_ms for _, result in successful]
    base.update(
        {
            **{f"{name}_accuracy": score for name, score in accuracy.items()},
            "overall_exact_match": sum(
                all(
                    result.values is not None and result.values[name] == getattr(case, attribute)
                    for name, attribute in expected.items()
                )
                for case, result in successful
            )
            / total,
            "latency_ms_mean": statistics.mean(latencies),
            "latency_ms_p50": _percentile(latencies, 0.5),
            "latency_ms_p95": _percentile(latencies, 0.95),
        }
    )
    return base


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def write_support_results(
    results: dict[str, list[tuple[SupportCase, SupportResult]]], output_dir: str | Path
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    prediction_path = path / "predictions.csv"
    fieldnames = [
        "provider",
        "model",
        "case_id",
        "text",
        "expected_route",
        "expected_urgency",
        "expected_risk",
        "expected_action",
        "predicted_route",
        "predicted_urgency",
        "predicted_risk",
        "predicted_action",
        "latency_ms",
        "error",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for records in results.values():
            for case, result in records:
                writer.writerow(
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "case_id": case.id,
                        "text": case.text,
                        "expected_route": case.route,
                        "expected_urgency": case.urgency,
                        "expected_risk": case.risk,
                        "expected_action": case.action,
                        "predicted_route": result.values.get("route") if result.values else None,
                        "predicted_urgency": result.values.get("urgency")
                        if result.values
                        else None,
                        "predicted_risk": result.values.get("risk") if result.values else None,
                        "predicted_action": result.values.get("action") if result.values else None,
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                    }
                )
    LOGGER.info("Predictions written: %s", prediction_path)
    summary_path = path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {name: summarize_support(records) for name, records in results.items()}, indent=2
        ),
        encoding="utf-8",
    )
    return summary_path
