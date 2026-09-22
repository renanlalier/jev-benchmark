from __future__ import annotations

import os
import time

import httpx

from jev_benchmark.models import BenchmarkCase, Intent, Prediction, ProviderResult, Sentiment
from jev_benchmark.pricing import estimate_cost_usd
from jev_benchmark.providers.base import BenchmarkProvider
from jev_benchmark.task_policies import ASSISTANT_CLASSIFICATION_POLICY


class JevProvider(BenchmarkProvider):
    name = "jev"
    model = "jev-latest"
    endpoint = "https://api.typesafe.ai/v1/systemone"

    async def classify(self, case: BenchmarkCase) -> ProviderResult:
        api_key = os.environ["JEV_API_KEY"]
        payload = {
            "model": self.model,
            "state": case.text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Classify the user intent using the benchmark policy: "
                    + ASSISTANT_CLASSIFICATION_POLICY,
                    "criteria": {
                        "FINANCIAL_TRANSACTION": "A financial transaction or payment.",
                        "SHOPPING_LIST": "An item to add to a shopping list.",
                        "REMINDER": "A reminder request.",
                        "CALENDAR": "A calendar or scheduling request.",
                        "WEATHER": "A weather request.",
                        "GENERAL_CHAT": "General conversation.",
                        "OTHER": "Anything not covered by the other labels.",
                    },
                },
                "sentiment": {
                    "type": "choice",
                    "instructions": "Classify the user sentiment using the benchmark policy: "
                    + ASSISTANT_CLASSIFICATION_POLICY,
                    "criteria": {
                        "SATISFIED": "Positive or satisfied.",
                        "NEUTRAL": "Neutral or factual.",
                        "CONFUSED": "Confused or uncertain.",
                        "FRUSTRATED": "Frustrated by a problem.",
                        "ANGRY": "Angry or highly negative.",
                    },
                },
                "escalation": {
                    "type": "noul",
                    "instructions": "Decide escalation using this benchmark policy: "
                    + ASSISTANT_CLASSIFICATION_POLICY,
                },
            },
        }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens") or usage.get("input")
        output_tokens = usage.get("output_tokens") or usage.get("output")

        # Jev is early access and gateways may expose slightly different envelopes.
        # Keep extraction isolated here so the benchmark core remains stable.
        answers = data.get("answers", data.get("result", data))
        intent = answers["intent"]
        sentiment = answers["sentiment"]
        escalation = answers["escalation"]

        intent_label = (
            intent.get("value", intent.get("choice", intent))
            if isinstance(intent, dict)
            else intent
        )
        sentiment_label = (
            sentiment.get("value", sentiment.get("choice", sentiment))
            if isinstance(sentiment, dict)
            else sentiment
        )
        escalation_value = (
            escalation.get("noul", escalation.get("probability", escalation.get("p_true")))
            if isinstance(escalation, dict)
            else escalation
        )
        if not isinstance(escalation_value, (int, float)):
            raise TypeError("Jev response did not contain a numeric escalation probability")
        escalation_prob = float(escalation_value)

        return ProviderResult(
            provider=self.name,
            model=self.model,
            prediction=Prediction(
                intent=Intent(str(intent_label)),
                sentiment=Sentiment(str(sentiment_label)),
                escalation=bool(escalation_prob >= 0.5),
                intent_confidence=_confidence(intent),
                sentiment_confidence=_confidence(sentiment),
                escalation_confidence=float(escalation_prob),
            ),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost_usd(
                self.name, self.model, input_tokens, output_tokens
            ),
            jev_scores={
                "intent": extract_jev_score(intent),
                "sentiment": extract_jev_score(sentiment),
                "escalation": extract_jev_score(escalation),
            },
            raw_output=data,
        )


def extract_jev_score(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return None
    for key in ("score", "confidence", "probability", "prob", "noul", "p_true"):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def _confidence(value: object) -> float | None:
    return extract_jev_score(value)
