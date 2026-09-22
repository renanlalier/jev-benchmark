from __future__ import annotations

ASSISTANT_CLASSIFICATION_POLICY = """Apply this benchmark policy exactly.

Intent:
- FINANCIAL_TRANSACTION: a payment, charge, transfer, balance, invoice, refund, card, or other financial operation.
- SHOPPING_LIST: adding, removing, creating, or reviewing items to buy or a shopping list.
- REMINDER: asking to remember, notify, or remind at a time or recurrence.
- CALENDAR: scheduling, changing, cancelling, sharing, or reviewing an event or appointment.
- WEATHER: weather, temperature, forecast, air quality, or related conditions.
- GENERAL_CHAT: conversation, thanks, requests for clarification, or feedback about the assistant or service.
- OTHER: a request outside the listed intents.

Sentiment:
- SATISFIED: explicit positive satisfaction, gratitude, or praise.
- NEUTRAL: factual or polite content without an expressed emotion.
- CONFUSED: explicit uncertainty or lack of understanding.
- FRUSTRATED: annoyance, disappointment, or a problem blocking the user.
- ANGRY: explicit anger, hostility, or severe negative language.
Urgency alone is not sentiment.

Escalation:
Set true only when the message explicitly requests a human, reports suspected fraud or unauthorized activity, requests an operation requiring human intervention, reports a repeated unresolved failure, or expresses severe dissatisfaction requiring intervention. Set false for ordinary informational requests, routine operations, and isolated confusion or frustration that can be handled automatically."""


def assistant_system_prompt() -> str:
    return f"""You are a strict classifier. Return JSON only with keys: intent, sentiment, escalation.
Intent must be one of FINANCIAL_TRANSACTION, SHOPPING_LIST, REMINDER, CALENDAR, WEATHER, GENERAL_CHAT, OTHER.
Sentiment must be one of SATISFIED, NEUTRAL, CONFUSED, FRUSTRATED, ANGRY.
Escalation must be true or false.

{ASSISTANT_CLASSIFICATION_POLICY}
Do not explain."""
