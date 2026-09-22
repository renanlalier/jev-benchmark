from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    input_per_million_usd: float
    output_per_million_usd: float


# Standard direct-API prices captured for reproducible benchmark estimates.
# Update deliberately when provider pricing changes; published runs should record
# the benchmark commit so historical estimates remain reproducible.
PRICING: dict[tuple[str, str], TokenPricing] = {
    ("openai", "gpt-5.6-luna"): TokenPricing(0.20, 1.20),
    ("anthropic", "claude-haiku-4-5-20251001"): TokenPricing(1.00, 5.00),
    ("jev", "system-one"): TokenPricing(0.042, 0.0),
    ("jev", "jev-latest"): TokenPricing(0.042, 0.0),
}


def estimate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    pricing = PRICING.get((provider, model))
    if pricing is None or input_tokens is None:
        return None

    # Some decision APIs do not bill output tokens. Missing output usage is
    # therefore valid only when the configured output price is zero.
    if output_tokens is None and pricing.output_per_million_usd != 0:
        return None

    return (
        input_tokens * pricing.input_per_million_usd
        + (output_tokens or 0) * pricing.output_per_million_usd
    ) / 1_000_000
