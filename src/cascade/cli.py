"""Command-line interface for cascade.

Provides a ``cascade`` CLI command for running simulations, comparing
strategies, and generating reports from JSON pipeline definitions.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal, cast

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
    StrategyType,
    adaptive,
    checkpoint,
    fallback,
    human_in_loop,
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
    "human": human_in_loop(),
    "adaptive": adaptive(),
}

STRATEGY_SPEC_HELP = (
    "Strategy spec. Built-ins: naive, retry[:attempts], fallback[:model+...], "
    "parallel[:n[:majority|unanimous|any]], checkpoint[:interval], "
    "human[:step+...[:accuracy]], adaptive[:threshold[:strategy]]."
)


def _parse_positive_int(value: str, label: str) -> int:
    """Parse a positive integer for a strategy option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise click.BadParameter(f"{label} must be an integer") from exc

    if parsed < 1:
        raise click.BadParameter(f"{label} must be at least 1")
    return parsed


def _parse_non_negative_int(value: str, label: str) -> int:
    """Parse a non-negative integer for zero-based step indexes."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise click.BadParameter(f"{label} must be an integer") from exc

    if parsed < 0:
        raise click.BadParameter(f"{label} must be at least 0")
    return parsed


def _parse_probability(value: str, label: str) -> float:
    """Parse a probability in the inclusive range [0, 1]."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise click.BadParameter(f"{label} must be a number") from exc

    if not 0.0 <= parsed <= 1.0:
        raise click.BadParameter(f"{label} must be between 0 and 1")
    return parsed


def _split_list(value: str) -> list[str]:
    """Split a user-facing list parameter.

    ``+`` is the documented separator because compare specs are comma-separated,
    but accepting commas keeps single-strategy specs forgiving.
    """
    separator = "+" if "+" in value else ","
    return [item.strip() for item in value.split(separator) if item.strip()]


def _parse_step_list(value: str) -> list[int]:
    """Parse one or more human-review step indices."""
    steps = [_parse_non_negative_int(item, "human step") for item in _split_list(value)]
    if not steps:
        raise click.BadParameter("human strategy requires at least one step")
    return steps


def parse_strategy_spec(spec: str) -> ResilienceStrategy:
    """Parse a CLI strategy spec into a ResilienceStrategy.

    Supported forms include ``retry:5``, ``parallel:5:any``,
    ``fallback:opus+sonnet``, ``human:2+5:0.99``, and
    ``adaptive:2:fallback``. Bare strategy names use the built-in defaults.
    """
    raw = spec.strip()
    if not raw:
        raise click.BadParameter("strategy spec cannot be empty")

    name, *parts = [part.strip().lower() for part in raw.split(":")]
    if name == "human_in_loop":
        name = "human"

    if name == "naive":
        if parts:
            raise click.BadParameter("naive does not accept parameters")
        return naive()

    if name == "retry":
        if len(parts) > 1:
            raise click.BadParameter("retry accepts at most one parameter: attempts")
        if not parts:
            return retry()
        return retry(max_attempts=_parse_positive_int(parts[0], "retry attempts"))

    if name == "fallback":
        if len(parts) > 1:
            raise click.BadParameter("fallback accepts one parameter: model+model")
        return fallback(models=_split_list(parts[0])) if parts else fallback()

    if name == "parallel":
        if len(parts) > 2:
            raise click.BadParameter("parallel accepts parameters: n[:vote]")
        vote_methods = {"majority", "unanimous", "any"}
        n = _parse_positive_int(parts[0], "parallel n") if parts else 3
        vote: Literal["majority", "unanimous", "any"] = "majority"
        if len(parts) == 2:
            if parts[1] not in vote_methods:
                raise click.BadParameter(
                    "parallel vote must be one of: majority, unanimous, any"
                )
            vote = cast(Literal["majority", "unanimous", "any"], parts[1])
        return parallel(n=n, vote=vote)

    if name == "checkpoint":
        if len(parts) > 1:
            raise click.BadParameter("checkpoint accepts one parameter: interval")
        if not parts:
            return checkpoint()
        return checkpoint(interval=_parse_positive_int(parts[0], "checkpoint interval"))

    if name == "human":
        if len(parts) > 2:
            raise click.BadParameter("human accepts parameters: step+step[:accuracy]")
        steps = _parse_step_list(parts[0]) if parts else None
        accuracy = (
            _parse_probability(parts[1], "human accuracy") if len(parts) == 2 else 0.95
        )
        return human_in_loop(at_steps=steps, accuracy=accuracy)

    if name == "adaptive":
        if len(parts) > 2:
            raise click.BadParameter(
                "adaptive accepts parameters: threshold[:strategy]"
            )
        threshold = _parse_positive_int(parts[0], "adaptive threshold") if parts else 2
        escalation = parts[1] if len(parts) == 2 else "parallel"
        try:
            return adaptive(
                escalation_threshold=threshold,
                escalation_strategy=escalation,
            )
        except ValueError as exc:
            available = ", ".join(strategy_type.value for strategy_type in StrategyType)
            raise click.BadParameter(
                f"adaptive escalation strategy is unknown: {escalation!r}. "
                f"Available: {available}"
            ) from exc

    available = ", ".join(STRATEGY_REGISTRY.keys())
    raise click.BadParameter(f"unknown strategy {name!r}. Available: {available}")


def parse_strategy_specs(specs: str) -> list[ResilienceStrategy]:
    """Parse a comma-separated list of CLI strategy specs."""
    raw_specs = [item.strip() for item in specs.split(",") if item.strip()]
    if not raw_specs:
        raise click.BadParameter("at least one strategy is required")
    return [parse_strategy_spec(item) for item in raw_specs]


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
    show_default=True,
    help=STRATEGY_SPEC_HELP,
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
    strategy = parse_strategy_spec(strategy_name)

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
    help=f"Comma-separated strategy specs. {STRATEGY_SPEC_HELP}",
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

    strategies = parse_strategy_specs(strategy_names)

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
