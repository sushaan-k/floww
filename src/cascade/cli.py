"""Command-line interface for cascade.

Provides a ``cascade`` CLI command for running simulations, comparing
strategies, and generating reports from JSON pipeline definitions.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from cascade.comparator import Comparator
from cascade.failures import FailureConfig
from cascade.pipeline import Pipeline, Step
from cascade.report import (
    build_report,
    export_comparison_json,
    format_report,
)
from cascade.strategies import (
    ResilienceStrategy,
    adaptive,
    checkpoint,
    fallback,
    naive,
    parallel,
    retry,
)

logger = logging.getLogger(__name__)

STRATEGY_REGISTRY: dict[str, ResilienceStrategy] = {
    "naive": naive(),
    "retry": retry(max_attempts=3),
    "fallback": fallback(),
    "parallel": parallel(n=3),
    "checkpoint": checkpoint(interval=2),
    "adaptive": adaptive(),
}


def _configure_logging(verbose: bool) -> None:
    """Set up logging based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_pipeline_from_json(path: Path) -> Pipeline:
    """Load a Pipeline from a JSON file.

    Expected format::

        {
            "name": "my-pipeline",
            "steps": [
                {"name": "step1", "model": "sonnet", "tools": ["web_search"]},
                {"name": "step2", "model": "opus", "depends_on": ["step1"]}
            ]
        }

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Pipeline object.
    """
    with open(path) as f:
        data = json.load(f)

    steps = [Step(**s) for s in data.get("steps", [])]
    return Pipeline(
        steps=steps,
        name=data.get("name", path.stem),
        description=data.get("description", ""),
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="cascade")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Cascade -- agent reliability simulator.

    Chaos engineering for AI agent pipelines. Run Monte Carlo simulations
    to measure end-to-end reliability under different failure modes and
    resilience strategies.
    """
    ctx.ensure_object(dict)
    _configure_logging(verbose)


@main.command()
@click.argument("pipeline_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n",
    "--simulations",
    default=1000,
    type=int,
    show_default=True,
    help="Number of simulation runs.",
)
@click.option(
    "-s",
    "--strategy",
    "strategy_name",
    default="naive",
    type=click.Choice(list(STRATEGY_REGISTRY.keys()), case_sensitive=False),
    show_default=True,
    help="Resilience strategy to apply.",
)
@click.option(
    "--hallucination-rate",
    default=0.05,
    type=float,
    show_default=True,
    help="Hallucination probability per step.",
)
@click.option(
    "--tool-failure-rate",
    default=0.03,
    type=float,
    show_default=True,
    help="Tool failure probability per step.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Export report as JSON to this path.",
)
def simulate(
    pipeline_file: Path,
    simulations: int,
    strategy_name: str,
    hallucination_rate: float,
    tool_failure_rate: float,
    seed: int | None,
    output: Path | None,
) -> None:
    """Run a simulation on a pipeline definition file."""
    pipeline = _load_pipeline_from_json(pipeline_file)
    failure_config = FailureConfig(
        hallucination_rate=hallucination_rate,
        tool_failure_rate=tool_failure_rate,
    )
    strategy = STRATEGY_REGISTRY[strategy_name.lower()]

    from cascade.simulator import Simulator

    sim = Simulator(
        pipeline=pipeline,
        failure_config=failure_config,
        n_simulations=simulations,
        strategy=strategy,
        seed=seed,
    )
    result = sim.run()
    report = build_report(result)
    click.echo(format_report(report))

    if output:
        from cascade.report import export_json

        export_json(report, output)
        click.echo(f"\nReport exported to {output}")


@main.command()
@click.argument("pipeline_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-n",
    "--simulations",
    default=1000,
    type=int,
    show_default=True,
    help="Number of simulation runs per strategy.",
)
@click.option(
    "--strategies",
    "strategy_names",
    default="naive,retry,parallel,checkpoint,adaptive",
    type=str,
    show_default=True,
    help="Comma-separated list of strategies to compare.",
)
@click.option(
    "--hallucination-rate",
    default=0.05,
    type=float,
    show_default=True,
)
@click.option(
    "--tool-failure-rate",
    default=0.03,
    type=float,
    show_default=True,
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Export comparison as JSON to this path.",
)
@click.option(
    "--pareto",
    default=None,
    type=click.Path(path_type=Path),
    help="Save Pareto frontier plot to this path.",
)
@click.option(
    "--heatmap",
    default=None,
    type=click.Path(path_type=Path),
    help="Save failure heatmap to this path.",
)
def compare(
    pipeline_file: Path,
    simulations: int,
    strategy_names: str,
    hallucination_rate: float,
    tool_failure_rate: float,
    seed: int | None,
    output: Path | None,
    pareto: Path | None,
    heatmap: Path | None,
) -> None:
    """Compare multiple resilience strategies on a pipeline."""
    pipeline = _load_pipeline_from_json(pipeline_file)
    failure_config = FailureConfig(
        hallucination_rate=hallucination_rate,
        tool_failure_rate=tool_failure_rate,
    )

    names = [n.strip().lower() for n in strategy_names.split(",")]
    strategies = []
    for name in names:
        if name not in STRATEGY_REGISTRY:
            click.echo(
                f"Unknown strategy: {name!r}. "
                f"Available: {', '.join(STRATEGY_REGISTRY.keys())}",
                err=True,
            )
            raise SystemExit(1)
        strategies.append(STRATEGY_REGISTRY[name])

    comp = Comparator(
        pipeline=pipeline,
        failure_config=failure_config,
        n_simulations=simulations,
        seed=seed,
    )
    comparison = comp.compare(strategies)

    comparison.print_table()
    click.echo()
    comparison.recommend()

    if output:
        export_comparison_json(comparison, output)
        click.echo(f"\nComparison exported to {output}")

    if pareto:
        comparison.plot_pareto(save_path=pareto)
        click.echo(f"Pareto plot saved to {pareto}")

    if heatmap:
        comparison.plot_failure_heatmap(save_path=heatmap)
        click.echo(f"Heatmap saved to {heatmap}")


if __name__ == "__main__":
    main()
