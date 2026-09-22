from __future__ import annotations

import json
import os
import time

import httpx

from jev_benchmark.models import BenchmarkCase, Prediction, ProviderResult
from jev_benchmark.pricing import estimate_cost_usd
from jev_benchmark.providers.base import BenchmarkProvider
from jev_benchmark.task_policies import assistant_system_prompt

SYSTEM_PROMPT = assistant_system_prompt()


class OpenAIProvider(BenchmarkProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-5.6-luna") -> None:
        self.model = model

    async def classify(self, case: BenchmarkCase) -> ProviderResult:
        api_key = os.environ["OPENAI_API_KEY"]
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": case.text},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "sentiment": {"type": "string"},
                            "escalation": {"type": "boolean"},
                        },
                        "required": ["intent", "sentiment", "escalation"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        text = _extract_output_text(data)
        parsed = json.loads(text)
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


def _extract_output_text(data: dict[str, object]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = data.get("output", [])
    if not isinstance(output, list):
        raise TypeError("OpenAI response did not contain output text")
    for item in output:
        if not isinstance(item, dict):
            continue
        content_blocks = item.get("content", [])
        if not isinstance(content_blocks, list):
            continue
        for content in content_blocks:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                return text
    raise ValueError("OpenAI response did not contain output text")
