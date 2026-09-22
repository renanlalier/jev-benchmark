from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from jev_benchmark.models import BenchmarkCase, Prediction, ProviderResult
from jev_benchmark.pricing import estimate_cost_usd
from jev_benchmark.providers.base import BenchmarkProvider
from jev_benchmark.task_policies import assistant_system_prompt

SYSTEM_PROMPT = assistant_system_prompt()


class AnthropicProvider(BenchmarkProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model

    async def classify(self, case: BenchmarkCase) -> ProviderResult:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        payload = {
            "model": self.model,
            "max_tokens": 120,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": case.text}],
        }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        text = "".join(
            part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"
        ).strip()
        parsed = _parse_json_object(_strip_code_fence(text))
        usage = data.get("usage", {})

        return ProviderResult(
            provider=self.name,
            model=self.model,
            prediction=Prediction(**parsed),
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            estimated_cost_usd=estimate_cost_usd(
                self.name, self.model, usage.get("input_tokens"), usage.get("output_tokens")
            ),
            raw_output=data,
        )


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    if not isinstance(value, dict):
        raise TypeError("Anthropic response did not contain a JSON object")
    return value
