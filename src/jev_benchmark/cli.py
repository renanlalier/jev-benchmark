from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from jev_benchmark.artifacts import create_run_directory, write_metadata
from jev_benchmark.providers.anthropic_provider import AnthropicProvider
from jev_benchmark.providers.jev import JevProvider
from jev_benchmark.providers.openai_provider import OpenAIProvider
from jev_benchmark.runner import load_cases, run_all, write_results
from jev_benchmark.support_triage import (
    load_support_cases,
    run_support_triage,
    write_support_results,
)

app = typer.Typer(no_args_is_help=True)
LOGGER = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """Run benchmark commands."""


@app.command()
def run(
    providers: list[str] = typer.Option(
        [],
        "--provider",
        "-p",
        help="Provider to benchmark; repeat this option to select multiple providers.",
    ),
    dataset: Path = typer.Option(Path("datasets/assistant_classification.csv"), "--dataset"),
    task: str = typer.Option(
        "assistant", "--task", help="Benchmark task: assistant or customer-support."
    ),
    repetitions: int = typer.Option(1, "--repetitions", min=1),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for results; defaults to results/<task>/latest.",
    ),
) -> None:
    """Run the benchmark against one or more providers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    provider_map = {
        "jev": JevProvider(),
        "anthropic": AnthropicProvider(),
        "openai": OpenAIProvider(),
    }

    if task not in {"assistant", "customer-support"}:
        raise typer.BadParameter("Task must be assistant or customer-support")

    resolved_output_dir, run_id = create_run_directory(task, output_dir)

    selected_names = providers or list(provider_map)
    unknown = [name for name in selected_names if name not in provider_map]
    if unknown:
        raise typer.BadParameter(f"Unknown provider(s): {', '.join(unknown)}")

    if task == "customer-support":
        support_dataset = (
            dataset
            if dataset != Path("datasets/assistant_classification.csv")
            else Path("datasets/customer_support_triage.csv")
        )
        LOGGER.info("Loading customer-support dataset: %s", support_dataset)
        support_cases = load_support_cases(support_dataset)
        LOGGER.info(
            "Starting customer-support triage: %d case(s), %d repetition(s), providers=%s",
            len(support_cases),
            repetitions,
            ", ".join(selected_names),
        )
        support_results = asyncio.run(
            run_support_triage(selected_names, support_cases, repetitions)
        )
        summary_path = write_support_results(support_results, resolved_output_dir)
        metadata_path = write_metadata(
            resolved_output_dir,
            task=task,
            dataset=support_dataset,
            providers=selected_names,
            repetitions=repetitions,
            run_id=run_id,
        )
        LOGGER.info("Results written: %s; metadata: %s", summary_path, metadata_path)
        typer.echo(f"Customer-support benchmark complete. Summary: {summary_path}")
        return

    LOGGER.info("Loading dataset: %s", dataset)
    cases = load_cases(dataset)
    selected = [provider_map[name] for name in selected_names]
    LOGGER.info(
        "Starting benchmark: %d case(s), %d repetition(s), providers=%s",
        len(cases),
        repetitions,
        ", ".join(selected_names),
    )
    results = asyncio.run(run_all(selected, cases, repetitions))
    LOGGER.info("Calculating metrics and writing results to: %s", resolved_output_dir)
    summary_path = write_results(results, resolved_output_dir)
    metadata_path = write_metadata(
        resolved_output_dir,
        task=task,
        dataset=dataset,
        providers=selected_names,
        repetitions=repetitions,
        run_id=run_id,
    )
    LOGGER.info("Results written: %s; metadata: %s", summary_path, metadata_path)
    typer.echo(f"Benchmark complete. Summary: {summary_path}")


if __name__ == "__main__":
    app()
