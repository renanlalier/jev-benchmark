# Benchmark Methodology

This project is designed to compare decision-oriented models and general-purpose LLMs on the same structured classification tasks without conflating model capability with benchmark plumbing.

## Benchmark questions

The first suite measures three outputs for each user message:

1. **Intent** — one class from the benchmark taxonomy.
2. **Sentiment** — one class from the benchmark taxonomy.
3. **Escalation** — whether the message should be escalated to a human.

## Reproducibility contract

Every published result should record:

- Dataset file and git commit SHA
- Provider and exact model identifier
- Benchmark timestamp
- Number of repetitions
- Prompt / schema version
- API endpoint / gateway when relevant
- Measured latency
- Token usage when exposed by the provider
- Pricing table version used for cost estimates
- Parse or request failures

Raw outputs should be retained when licensing and privacy constraints allow it.

## Core quality metrics

The benchmark will evolve beyond simple accuracy. The intended metric set is:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision, recall and F1
- Confusion matrix
- Exact match across all requested decisions
- Failure / invalid-output rate

## Operational metrics

- Mean latency
- p50 latency
- p95 latency
- Total and per-request estimated cost
- Run-to-run consistency
- Confidence calibration when probabilities are natively available

## Confidence caveat

Provider-native calibrated probabilities and confidence values generated as ordinary LLM text are not assumed to be equivalent. The benchmark records their origin and should report calibration separately.

## Repetitions

Repeated evaluation of the same examples is important because one goal is to measure decision stability. A benchmark run with `--repetitions 10`, for example, sends every case ten times to every selected provider.

Consistency is reported separately from correctness. A model can be consistently wrong.

## Prompt fairness

General-purpose LLMs receive an explicit classification contract and structured output instructions. Jev receives the same semantic task expressed through its native decision primitives. The goal is semantic parity rather than byte-for-byte identical prompts.

Provider-specific optimizations must be documented. Hidden few-shot examples or provider-specific data that materially change the task are not allowed in headline comparisons.

## Dataset governance

The repository starts with a tiny `datasets/smoke.csv` file only to validate the harness. It is **not** suitable for leaderboard claims.

A publishable dataset should include:

- clear taxonomy definitions;
- annotation guidelines;
- ambiguous-case policy;
- at least two independent annotations for a validation subset;
- class-balance analysis;
- dataset versioning;
- checks for duplicates and leakage;
- a separate holdout set when tuning prompts.

## Cost methodology

Cost is calculated using the documented provider prices that apply on the run date and the token usage returned by the provider. Pricing assumptions must be versioned with benchmark results because API pricing changes over time.

Jev has a different charging model from general-purpose LLMs, so the benchmark reports absolute request cost and does not normalize away output-token differences.

## What this benchmark does not prove

Performance on this suite does not establish that one provider is globally superior. Results apply to the specified task taxonomy, dataset, prompts, model versions, and run date.
