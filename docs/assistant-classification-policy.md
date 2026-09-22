# Assistant classification policy

This policy is the normative contract for the `assistant` benchmark. Annotators and every provider receive the same definitions. A case is retained only when its label is justified by its text and this policy.

## Intent

- `FINANCIAL_TRANSACTION`: payment, charge, transfer, balance, invoice, refund, card, or another financial operation.
- `SHOPPING_LIST`: add, remove, create, or review items to buy or a shopping list.
- `REMINDER`: ask to remember, notify, or remind at a time or recurrence.
- `CALENDAR`: schedule, change, cancel, share, or review an event or appointment.
- `WEATHER`: weather, temperature, forecast, air quality, or related conditions.
- `GENERAL_CHAT`: conversation, thanks, clarification, or feedback about the assistant or service.
- `OTHER`: a request outside this taxonomy.

## Sentiment

- `SATISFIED`: explicit satisfaction, gratitude, or praise.
- `NEUTRAL`: factual or polite content without expressed emotion.
- `CONFUSED`: explicit uncertainty or lack of understanding.
- `FRUSTRATED`: annoyance, disappointment, or a blocking problem.
- `ANGRY`: explicit anger, hostility, or severe negative language.

Urgency does not imply anger or frustration.

## Escalation

Set `true` only for an explicit request for a human, suspected fraud or unauthorized activity, an operation requiring human intervention, a repeated unresolved failure, or severe dissatisfaction requiring intervention. Set `false` for ordinary informational requests, routine operations, and isolated confusion or frustration that can be handled automatically.

## Ambiguity policy

When two labels remain defensible after applying these definitions, rewrite the text to make the intended label clear or remove the case. Do not use ambiguous cases for headline results. The 2026-09 review changed case 19 to `NEUTRAL`, case 33 to `REMINDER`, and case 70 to `NEUTRAL` under this policy.
