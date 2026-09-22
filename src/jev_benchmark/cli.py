from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from dotenv import load_dotenv

from jev_benchmark.providers.anthropic_provider import AnthropicProvider
from jev_benchmark.providers.jev import JevProvider
from jev_benchmark.providers.openai_provider import OpenAIProvider
from jev_benchmark.runner import load_cases, run_all, write_results

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    providers: list[str] = typer.Option(
        ["jev", "anthropic", "openai"], "--providers", help="Providers to benchmark"
    ),
    dataset: Path = typer.Option(Path("datasets/smoke.csv"), "--dataset"),
    repetitions: int = typer.Option(1, "--repetitions", min=1),
    output_dir: Path = typer.Option(Path("results/latest"), "--output-dir"),
) -> None:
    """Run the benchmark against one or more providers."""
    load_dotenv()
    provider_map = {
        "jev": JevProvider(),
        "anthropic": AnthropicProvider(),
        "openai": OpenAIProvider(),
    }

    unknown = [name for name in providers if name not in provider_map]
    if unknown:
        raise typer.BadParameter(f"Unknown provider(s): {', '.join(unknown)}")

    cases = load_cases(dataset)
    selected = [provider_map[name] for name in providers]
    results = asyncio.run(run_all(selected, cases, repetitions))
    summary_path = write_results(results, output_dir)
    typer.echo(f"Benchmark complete. Summary: {summary_path}")


if __name__ == "__main__":
    app()
