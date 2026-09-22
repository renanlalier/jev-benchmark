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
    values: dict[str, str]
    latency_ms: float


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
        return [
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


def _prompt() -> str:
    labels = "\n".join(f"{key}: {', '.join(values)}" for key, values in FIELDS.items())
    return f"""Classify this customer-support message. Return JSON only with route, urgency, risk, and action.
{labels}
Do not explain."""


async def _openai(case: SupportCase) -> SupportResult:
    payload = {
        "model": "gpt-5.6-luna",
        "input": [{"role": "system", "content": _prompt()}, {"role": "user", "content": case.text}],
        "text": {"format": {"type": "json_object"}},
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


async def _anthropic(case: SupportCase) -> SupportResult:
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 160,
        "system": _prompt(),
        "messages": [{"role": "user", "content": case.text}],
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
    text = "".join(
        part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"
    )
    return SupportResult(
        "anthropic",
        str(payload["model"]),
        _validated(json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())),
        (time.perf_counter() - started) * 1000,
    )


async def _jev(case: SupportCase) -> SupportResult:
    questions = {
        name: {
            "type": "choice",
            "instructions": f"Classify the support {name}.",
            "criteria": {label: label.replace("_", " ").lower() for label in labels},
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
        records = [
            (case, await functions[provider](case)) for _ in range(repetitions) for case in cases
        ]
        LOGGER.info("[%s] Completed %d triage decision(s)", provider, len(records))
        return provider, records

    return dict(await asyncio.gather(*(run_one(provider) for provider in providers)))


def summarize_support(records: list[tuple[SupportCase, SupportResult]]) -> dict[str, object]:
    if not records:
        return {}
    expected = {"route": "route", "urgency": "urgency", "risk": "risk", "action": "action"}
    total = len(records)
    accuracy = {
        name: sum(result.values[name] == getattr(case, attribute) for case, result in records)
        / total
        for name, attribute in expected.items()
    }
    return {
        "provider": records[0][1].provider,
        "model": records[0][1].model,
        "samples": total,
        **{f"{name}_accuracy": score for name, score in accuracy.items()},
        "overall_exact_match": sum(
            all(
                result.values[name] == getattr(case, attribute)
                for name, attribute in expected.items()
            )
            for case, result in records
        )
        / total,
        "latency_ms_mean": statistics.mean(result.latency_ms for _, result in records),
    }


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
                        "predicted_route": result.values["route"],
                        "predicted_urgency": result.values["urgency"],
                        "predicted_risk": result.values["risk"],
                        "predicted_action": result.values["action"],
                        "latency_ms": result.latency_ms,
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
