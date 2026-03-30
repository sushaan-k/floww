"""Report generation and visualization for simulation results.

Generates structured reports from SimulationResult and StrategyComparison
objects, including text summaries, JSON export, and optional charts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cascade.comparator import StrategyComparison
from cascade.simulator import SimulationResult
from cascade.stats import (
    ConfidenceInterval,
    DistributionSummary,
    mean_ci,
    proportion_ci,
    summarize,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationReport:
    """Structured report for a single simulation result.

    Attributes:
        strategy_name: Name of the strategy used.
        n_simulations: Number of simulation runs.
        success_rate: Overall success rate.
        success_ci: Confidence interval for the success rate.
        cost_summary: Distribution summary of per-run costs.
        cost_ci: Confidence interval for mean cost.
        latency_summary: Distribution summary of per-run latencies.
        latency_ci: Confidence interval for mean latency.
        failure_counts: Count of each failure type.
        mean_steps_to_failure: Average step index of first failure.
        recovery_rate: Fraction of failing runs that recovered.
    """

    strategy_name: str
    n_simulations: int
    success_rate: float
    success_ci: ConfidenceInterval
    cost_summary: DistributionSummary
    cost_ci: ConfidenceInterval
    latency_summary: DistributionSummary
    latency_ci: ConfidenceInterval
    failure_counts: dict[str, int] = field(default_factory=dict)
    mean_steps_to_failure: float = 0.0
    recovery_rate: float = 0.0


def build_report(result: SimulationResult) -> SimulationReport:
    """Build a structured report from a SimulationResult.

    Args:
        result: Aggregated simulation result.

    Returns:
        SimulationReport with statistical summaries and CIs.
    """
    s_ci = proportion_ci(result.success_count, result.n_simulations)
    c_summary = summarize(result.costs)
    c_ci = (
        mean_ci(result.costs)
        if len(result.costs) >= 2
        else ConfidenceInterval(
            point=c_summary.mean,
            lower=c_summary.mean,
            upper=c_summary.mean,
            confidence=0.95,
        )
    )
    l_summary = summarize(result.latencies)
    l_ci = (
        mean_ci(result.latencies)
        if len(result.latencies) >= 2
        else ConfidenceInterval(
            point=l_summary.mean,
            lower=l_summary.mean,
            upper=l_summary.mean,
            confidence=0.95,
        )
    )

    return SimulationReport(
        strategy_name=result.strategy_name,
        n_simulations=result.n_simulations,
        success_rate=result.success_rate,
        success_ci=s_ci,
        cost_summary=c_summary,
        cost_ci=c_ci,
        latency_summary=l_summary,
        latency_ci=l_ci,
        failure_counts=result.failure_counts,
        mean_steps_to_failure=result.mean_steps_to_failure,
        recovery_rate=result.recovery_rate,
    )


def format_report(report: SimulationReport) -> str:
    """Format a SimulationReport as human-readable text.

    Args:
        report: The report to format.

    Returns:
        Multi-line string with the formatted report.
    """
    lines = [
        f"Simulation Report: {report.strategy_name}",
        "=" * 60,
        f"Simulations:          {report.n_simulations:,}",
        (
            f"Success Rate:         {report.success_rate:.1%} "
            f"[{report.success_ci.lower:.1%} - "
            f"{report.success_ci.upper:.1%}]"
        ),
        f"Mean Cost:            ${report.cost_ci.point:.4f} "
        f"[${report.cost_ci.lower:.4f} - ${report.cost_ci.upper:.4f}]",
        f"Mean Latency:         {report.latency_ci.point:.2f}s "
        f"[{report.latency_ci.lower:.2f}s - {report.latency_ci.upper:.2f}s]",
        f"Mean Steps to Fail:   {report.mean_steps_to_failure:.1f}",
        f"Recovery Rate:        {report.recovery_rate:.1%}",
        "",
        "Cost Distribution:",
        f"  Median: ${report.cost_summary.median:.4f}  "
        f"P5: ${report.cost_summary.p5:.4f}  "
        f"P95: ${report.cost_summary.p95:.4f}",
        "",
        "Latency Distribution:",
        f"  Median: {report.latency_summary.median:.2f}s  "
        f"P5: {report.latency_summary.p5:.2f}s  "
        f"P95: {report.latency_summary.p95:.2f}s",
    ]

    if report.failure_counts:
        lines.append("")
        lines.append("Failure Breakdown:")
        for ftype, count in sorted(report.failure_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {ftype:<30} {count:>6}")

    lines.append("=" * 60)
    return "\n".join(lines)


def export_json(
    report: SimulationReport,
    path: str | Path,
) -> None:
    """Export a SimulationReport to JSON.

    Args:
        report: The report to export.
        path: File path to write the JSON output.
    """
    data = asdict(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info("Report exported to %s", path)


def export_comparison_json(
    comparison: StrategyComparison,
    path: str | Path,
) -> None:
    """Export a StrategyComparison to JSON.

    Args:
        comparison: The comparison to export.
        path: File path to write the JSON output.
    """
    reports = [build_report(r) for r in comparison.results]
    data = {
        "pipeline_name": comparison.pipeline_name,
        "n_simulations": comparison.n_simulations,
        "strategies": [asdict(r) for r in reports],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info("Comparison report exported to %s", path)


def print_comparison_report(comparison: StrategyComparison) -> str:
    """Print detailed reports for all strategies in a comparison.

    Args:
        comparison: The StrategyComparison to print.

    Returns:
        Full formatted report string.
    """
    sections: list[str] = []
    for result in comparison.results:
        report = build_report(result)
        sections.append(format_report(report))
    full = "\n\n".join(sections)
    print(full)
    return full


def _json_default(obj: object) -> object:
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
