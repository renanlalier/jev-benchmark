# Contributing

Thanks for helping improve the benchmark.

## Principles

Contributions should improve reproducibility, fairness, transparency, or provider coverage. Headline claims must be supported by versioned benchmark artifacts and documented methodology.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

## Adding a provider

1. Implement `BenchmarkProvider` under `src/jev_benchmark/providers/`.
2. Normalize the provider response into `ProviderResult`.
3. Keep provider-specific parsing inside the adapter.
4. Do not change task semantics for one provider only.
5. Add unit tests with mocked HTTP responses.
6. Document exact model IDs and pricing assumptions.

## Adding or changing a dataset

Dataset changes should include annotation rationale and avoid silently changing labels used by published results. Large benchmark datasets should have explicit versioning and annotation guidance.

## Pull requests

Keep pull requests focused. Explain what changed, how it affects benchmark fairness, and how it was validated.
