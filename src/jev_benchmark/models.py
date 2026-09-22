from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Intent(StrEnum):
    FINANCIAL_TRANSACTION = "FINANCIAL_TRANSACTION"
    SHOPPING_LIST = "SHOPPING_LIST"
    REMINDER = "REMINDER"
    CALENDAR = "CALENDAR"
    WEATHER = "WEATHER"
    GENERAL_CHAT = "GENERAL_CHAT"
    OTHER = "OTHER"


class Sentiment(StrEnum):
    SATISFIED = "SATISFIED"
    NEUTRAL = "NEUTRAL"
    CONFUSED = "CONFUSED"
    FRUSTRATED = "FRUSTRATED"
    ANGRY = "ANGRY"


class BenchmarkCase(BaseModel):
    id: str
    text: str
    expected_intent: Intent
    expected_sentiment: Sentiment
    expected_escalation: bool


class Prediction(BaseModel):
    intent: Intent
    sentiment: Sentiment
    escalation: bool
    intent_confidence: float | None = Field(default=None, ge=0, le=1)
    sentiment_confidence: float | None = Field(default=None, ge=0, le=1)
    escalation_confidence: float | None = Field(default=None, ge=0, le=1)


class ProviderResult(BaseModel):
    provider: str
    model: str
    prediction: Prediction | None = None
    latency_ms: float
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    jev_scores: dict[str, float | None] | None = None
    raw_output: dict[str, object] | str | None = None
