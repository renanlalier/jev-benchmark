# JEV Benchmark

Open, reproducible benchmark for comparing **decision-oriented models** and **general-purpose LLMs** on structured classification tasks.

The first benchmark focuses on **user intent classification**, **sentiment classification**, and **escalation decisions** using the same labeled dataset and the same evaluation protocol across providers.

## Why this project

General-purpose LLMs can return structured outputs, but decision models such as Jev are optimized for classification, scoring, and probabilistic decisions. This repository measures the trade-offs rather than assuming one approach is better.

We compare providers on:

- Accuracy and macro F1
- Per-class precision / recall / F1
- Confusion matrix
- Latency (p50 / p95 / mean)
- Estimated API cost
- Run-to-run consistency
- Confidence calibration when the provider exposes probabilities
- Error / parse-failure rate

## Initial providers

| Provider | Default model | Role in the benchmark |
|---|---|---|
| Jev | System One | Decision model baseline |
| Anthropic | Claude Haiku 4.5 | Low-cost general-purpose LLM |
| OpenAI | GPT-5.6 Luna | Low-cost general-purpose LLM |

Providers are adapters. Adding a model should not require changing the benchmark core.

## Architecture

```text
Golden Dataset
      │
      ▼
Benchmark Runner
      │
      ├─────────────┬────────────────┐
      ▼             ▼                ▼
    Jev         Claude Haiku      OpenAI Luna
      │             │                │
      └─────────────┴────────────────┘
                    │
                    ▼
            Normalized Prediction
                    │
                    ▼
              Metrics Engine
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Quality    Latency     Cost
          │
          ├─ Accuracy / F1
          ├─ Confusion Matrix
          ├─ Consistency
          └─ Calibration
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Add the API keys you want to use. You do **not** need credentials for every provider to run a subset.

```bash
jev-bench run --providers jev anthropic openai
```

Run a single provider:

```bash
jev-bench run --providers jev
```

Run the benchmark multiple times to measure consistency:

```bash
jev-bench run --providers jev anthropic openai --repetitions 10
```

Results are written to `results/` as JSON and CSV.

## Dataset format

The initial dataset is intentionally human-readable and versioned in Git:

```csv
id,text,expected_intent,expected_sentiment,expected_escalation
1,"I spent 50 dollars at the supermarket",FINANCIAL_TRANSACTION,NEUTRAL,false
2,"Remind me tomorrow at 9 to call the doctor",REMINDER,NEUTRAL,false
3,"This is the third time this failed",GENERAL_CHAT,FRUSTRATED,true
```

The repository ships with a small smoke-test dataset. A larger benchmark dataset should be built with explicit annotation guidelines, review, and dataset versioning before publishing headline comparisons.

## Fairness principles

1. Same task taxonomy and dataset for every provider.
2. Deterministic parsing into one normalized prediction contract.
3. Provider-specific prompting is allowed only when required to express the same task correctly.
4. Raw provider outputs and benchmark configuration are persisted for reproducibility.
5. Published results must record model IDs, date, dataset version, repetitions, and pricing assumptions.
6. Confidence is compared only when semantically meaningful; synthetic confidence produced by an LLM is not treated as equivalent to provider-native probabilities.

See [`docs/methodology.md`](docs/methodology.md).

## Project status

This repository is an early benchmark harness. The initial goal is to establish a transparent protocol and provider abstraction before publishing large comparative claims.

## Contributing

Contributions are welcome: new providers, datasets, metrics, reproducibility improvements, and independent benchmark runs.

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
