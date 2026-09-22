from jev_benchmark.pricing import estimate_cost_usd


def test_openai_luna_cost() -> None:
    assert estimate_cost_usd("openai", "gpt-5.6-luna", 1_000_000, 1_000_000) == 1.40


def test_anthropic_haiku_cost() -> None:
    assert estimate_cost_usd("anthropic", "claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 6.0


def test_jev_has_free_output_tokens() -> None:
    assert estimate_cost_usd("jev", "system-one", 1_000_000, None) == 0.042


def test_unknown_model_does_not_invent_cost() -> None:
    assert estimate_cost_usd("openai", "unknown", 100, 20) is None


def test_missing_billable_usage_does_not_invent_cost() -> None:
    assert estimate_cost_usd("openai", "gpt-5.6-luna", 100, None) is None
