from __future__ import annotations

import json
import os
import time
import httpx

from jev_benchmark.models import BenchmarkCase, Prediction, ProviderResult
from jev_benchmark.providers.base import BenchmarkProvider
from jev_benchmark.pricing import estimate_cost_usd


SYSTEM_PROMPT = """You are a strict classifier. Return JSON only with keys: intent, sentiment, escalation.
Intent must be one of FINANCIAL_TRANSACTION, SHOPPING_LIST, REMINDER, CALENDAR, WEATHER, GENERAL_CHAT, OTHER.
Sentiment must be one of SATISFIED, NEUTRAL, CONFUSED, FRUSTRATED, ANGRY.
Escalation must be true or false. Do not explain."""


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
            estimated_cost_usd=estimate_cost_usd(\n                self.name, self.model, usage.get("input_tokens"), usage.get("output_tokens")\n            ),
            raw_output=data,
        )


def _extract_output_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                return text
    raise ValueError("OpenAI response did not contain output text")
