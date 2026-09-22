from __future__ import annotations

ASSISTANT_CLASSIFICATION_POLICY = """Apply this benchmark policy exactly.

Intent:
- FINANCIAL_TRANSACTION: a payment, charge, transfer, balance, invoice, refund, card, or other financial operation.
- SHOPPING_LIST: adding, removing, creating, or reviewing items to buy or a shopping list.
- REMINDER: asking to remember, notify, or remind at a time or recurrence.
- CALENDAR: scheduling, changing, cancelling, sharing, or reviewing an event or appointment.
- WEATHER: weather, temperature, forecast, air quality, or related conditions.
- GENERAL_CHAT: conversation, thanks, requests for clarification, feedback, or questions about the assistant or service itself.
- OTHER: a content or knowledge request outside the listed intents, such as translation, creative writing, general knowledge, or summarization.

Sentiment:
- SATISFIED: explicit positive satisfaction, gratitude, or praise.
- NEUTRAL: factual or polite content without an expressed emotion.
- CONFUSED: explicit uncertainty or lack of understanding.
- FRUSTRATED: annoyance, disappointment, or a problem blocking the user.
- ANGRY: explicit anger, hostility, or severe negative language.
Urgency alone is not sentiment. Do not infer an emotion solely from a request or question: use CONFUSED only for stated uncertainty or inability to understand, and FRUSTRATED only for stated annoyance or a stated blocking problem.

Escalation:
Set true only when the message explicitly requests a human, reports suspected fraud or unauthorized activity, requests a manual financial operation (a refund, duplicate-charge or incorrect-invoice dispute, cancellation of a completed transfer, or card freeze), reports a repeated unresolved failure, or expresses severe dissatisfaction requiring intervention. Set false for ordinary informational requests, routine operations, and isolated confusion or frustration that can be handled automatically."""


def assistant_system_prompt() -> str:
    return f"""You are a strict classifier. Return JSON only with keys: intent, sentiment, escalation.
Intent must be one of FINANCIAL_TRANSACTION, SHOPPING_LIST, REMINDER, CALENDAR, WEATHER, GENERAL_CHAT, OTHER.
Sentiment must be one of SATISFIED, NEUTRAL, CONFUSED, FRUSTRATED, ANGRY.
Escalation must be true or false.

{ASSISTANT_CLASSIFICATION_POLICY}
Do not explain."""


SUPPORT_TRIAGE_POLICY = """Apply this customer-support triage policy exactly. Route by the primary issue. For an explicit unauthorized-access, compromise, phishing, or data-exposure incident, use FRAUD_SECURITY; use ACCOUNT_ACCESS for routine password, invitation, profile, permission, or access-removal requests. A cancellation, downgrade, pause, or renewal request is CANCELLATION_RETENTION even when it also asks for a refund; a charge or refund failure after a cancellation is BILLING. Urgency measures operational time pressure, not sentiment. LOW is limited to static informational or documented self-service requests that do not require account-state inspection or a change. NORMAL covers routine account-specific requests that do not block access or a core workflow. HIGH covers a blocked account or core feature, material individual financial or privacy impact, or a near-term deadline. CRITICAL requires active ongoing unauthorized access or payment activity, or a business-critical or broad production workflow that is blocked. Risk records an active risk described by the message; an informational privacy or security question has risk NONE. PAYMENT_DISPUTE requires an actual unauthorized, duplicate, disputed, or unresolved charge. Action is the final workflow owner after routine data collection, not the next chat reply. Apply action precedence: SECURITY_FREEZE for an active credible compromise; PRIORITY_QUEUE only for time-sensitive technical or production disruption; HUMAN_ESCALATION for a manual financial dispute, exception, complaint, privacy/security review, or access-removal request; REQUEST_INFORMATION only when essential information is missing and no higher-precedence workflow applies; otherwise AUTO_REPLY."""

SUPPORT_TRIAGE_CRITERIA = {
    "route": {
        "BILLING": "Charges, invoices, payments, refunds, discounts, or subscriptions.",
        "TECHNICAL_SUPPORT": "Product defects, integrations, outages, configuration, or documentation.",
        "ACCOUNT_ACCESS": "Passwords, invitations, profiles, permissions, or workspace access.",
        "FRAUD_SECURITY": "Compromise, phishing, unrecognized access, data rights, or security incidents.",
        "CANCELLATION_RETENTION": "Cancellation, downgrade, pause, renewal, or retention discussion.",
    },
    "urgency": {
        "LOW": "Static informational or documented self-service request requiring no account-state inspection or change.",
        "NORMAL": "Routine account-specific request that does not block access or a core workflow.",
        "HIGH": "Blocked access or core feature, material individual financial or privacy impact, or a stated near-term deadline.",
        "CRITICAL": "Active ongoing unauthorized access/payment activity, or a business-critical or broad production workflow that is blocked.",
    },
    "risk": {
        "NONE": "No active payment, account-takeover, or privacy risk is described.",
        "PAYMENT_DISPUTE": "Unauthorized, duplicate, disputed, or unresolved financial charge.",
        "ACCOUNT_TAKEOVER": "Unauthorized access, account compromise, or credential misuse.",
        "SAFETY_PRIVACY": "Private-data exposure or a privacy-rights request.",
    },
    "action": {
        "AUTO_REPLY": "A documented informational or self-service response fully resolves the request.",
        "REQUEST_INFORMATION": "Essential account or diagnostic details are missing and no higher-precedence workflow applies.",
        "PRIORITY_QUEUE": "A time-sensitive technical or production disruption needs expedited specialist handling.",
        "HUMAN_ESCALATION": "A human owns a manual financial dispute, exception, complaint, privacy/security review, or access-removal request.",
        "SECURITY_FREEZE": "Immediately contain an active credible account or payment compromise.",
    },
}


def support_triage_prompt() -> str:
    sections = "\n".join(
        f"{field}: "
        + "; ".join(f"{label} = {description}" for label, description in criteria.items())
        for field, criteria in SUPPORT_TRIAGE_CRITERIA.items()
    )
    return f"""Classify this customer-support message. Return JSON only with route, urgency, risk, and action.

{SUPPORT_TRIAGE_POLICY}
{sections}
Do not explain."""
