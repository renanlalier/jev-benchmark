# Customer support triage

This starter use case routes inbound support messages before an agent handles them. It measures four independent decisions, rather than a single generic intent.

## Dataset

`datasets/customer_support_triage.csv` contains 50 synthetic seed cases. They are useful for exercising the harness and reviewing label definitions, but they are not a production-quality gold dataset. Before publishing comparative results, replace or supplement them with de-identified real conversations reviewed independently by at least two annotators.

## Decision contract

| Field | Labels |
|---|---|
| `expected_route` | `BILLING`, `TECHNICAL_SUPPORT`, `ACCOUNT_ACCESS`, `FRAUD_SECURITY`, `CANCELLATION_RETENTION` |
| `expected_urgency` | `LOW`, `NORMAL`, `HIGH`, `CRITICAL` |
| `expected_risk` | `NONE`, `PAYMENT_DISPUTE`, `ACCOUNT_TAKEOVER`, `SAFETY_PRIVACY` |
| `expected_action` | `AUTO_REPLY`, `REQUEST_INFORMATION`, `PRIORITY_QUEUE`, `HUMAN_ESCALATION`, `SECURITY_FREEZE` |

## Label policy

- **Route** is the primary issue, not every topic mentioned in the message. Unauthorized access, compromise, phishing, and data exposure are `FRAUD_SECURITY`; routine credentials, invitations, profiles, permissions, and access removal are `ACCOUNT_ACCESS`. A cancellation, downgrade, pause, or renewal request is `CANCELLATION_RETENTION` even when it asks for a refund; a charge or refund failure after cancellation is `BILLING`.
- **Urgency** is operational time pressure, not sentiment. `LOW` is limited to a static informational or documented self-service request that needs no account-state inspection or change. `NORMAL` is a routine account-specific request that does not block access or a core workflow. `HIGH` is blocked access or a core feature, material individual financial or privacy impact, or a near-term deadline. `CRITICAL` is active ongoing unauthorized access/payment activity or a business-critical or broad production workflow that is blocked.
- **Risk** is active risk described by the message. Informational privacy or security questions are `NONE`; private-data exposure or a privacy-rights request is `SAFETY_PRIVACY`; `PAYMENT_DISPUTE` requires an actual unauthorized, duplicate, disputed, or unresolved charge.
- **Action** identifies the final workflow owner after routine data collection, rather than the next chat reply. Apply this precedence: `SECURITY_FREEZE` for an active credible compromise; `PRIORITY_QUEUE` only for time-sensitive technical or production disruption; `HUMAN_ESCALATION` for manual financial disputes, exceptions, complaints, privacy/security reviews, and access removal; `REQUEST_INFORMATION` only when essential information is missing and no higher-precedence workflow applies; otherwise `AUTO_REPLY`.

When two labels remain defensible, rewrite or remove the case. The 2026-09 review made the `urgency` and `action` boundaries explicit, rewrote cases 1, 3, 7, 13, 19, 20, 27, 29, 31, 32, 37, 40, 42, 48, and 49 to state impact, scope, or the required workflow, and aligned the remaining labels with those rules.

## Evaluation

Report accuracy for each field and an exact-match score across all four. The most important error slices are false negatives for `ACCOUNT_TAKEOVER`, `PAYMENT_DISPUTE`, `SAFETY_PRIVACY`, and `CRITICAL` urgency. Those should be reported separately from aggregate accuracy.

## Operational policy

`SECURITY_FREEZE` and `HUMAN_ESCALATION` must be treated as recommendations to a surrounding workflow, not autonomous irreversible actions. The workflow should apply account, authorization, and audit controls before acting.
