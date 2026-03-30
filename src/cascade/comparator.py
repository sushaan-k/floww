"""Strategy comparison engine.

Runs a Pipeline under multiple ResilienceStrategies and produces a
StrategyComparison that supports tabular display, Pareto frontier
visualization, failure heatmaps, and automated recommendations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from cascade.failures import FailureConfig
from cascade.pipeline import Pipeline
from cascade.simulator import SimulationResult, Simulator
from cascade.stats import pareto_frontier, proportion_ci
from cascade.strategies import ResilienceStrategy

logger = logging.getLogger(__name__)


class StrategyComparison(BaseModel):
    """Container for multi-strategy comparison results.

    Attributes:
        results: SimulationResult for each strategy, in order.
        pipeline_name: Name of the pipeline that was simulated.
        n_simulations: Number of simulations per strategy.
    """

    results: list[SimulationResult]
    pipeline_name: str = "pipeline"
    n_simulations: int = 0

    def print_table(self) -> str:
        """Format a comparison table as a string and print it.

        Returns:
            The formatted table string.
        """
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(
                title=(
                    f"Strategy Comparison ({self.n_simulations:,} simulations each)"
                ),
                show_lines=True,
            )
            table.add_column("Strategy", style="bold cyan")
            table.add_column("Success", justify="right", style="green")
            table.add_column("Avg Cost", justify="right")
            table.add_column("Avg Time", justify="right")
            table.add_column("Failures", justify="right", style="red")

            for r in self.results:
                ci = proportion_ci(r.success_count, r.n_simulations)
                total_failures = r.n_simulations - r.success_count
                table.add_row(
                    r.strategy_name,
                    f"{r.success_rate:.1%} [{ci.lower:.1%}-{ci.upper:.1%}]",
                    f"${r.mean_cost_usd:.4f}",
                    f"{r.mean_latency_s:.1f}s",
                    f"{total_failures:,}",
                )

            console = Console()
            console.print(table)

            # Also return a plain-text version
            return self._plain_table()
        except ImportError:
            text = self._plain_table()
            print(text)
            return text

    def _plain_table(self) -> str:
        """Generate a plain-text table without rich."""
        header = (
            f"{'Strategy':<25} {'Success':>10} {'Avg Cost':>10} "
            f"{'Avg Time':>10} {'Failures':>10}"
        )
        sep = "-" * len(header)
        lines = [
            f"Strategy Comparison ({self.n_simulations:,} simulations each)",
            sep,
            header,
            sep,
        ]
        for r in self.results:
            total_failures = r.n_simulations - r.success_count
            lines.append(
                f"{r.strategy_name:<25} {r.success_rate:>9.1%} "
                f"{'$' + f'{r.mean_cost_usd:.4f}':>10} "
                f"{r.mean_latency_s:>9.1f}s "
                f"{total_failures:>10,}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def plot_pareto(
        self,
        x_metric: str = "cost",
        y_metric: str = "success_rate",
        save_path: str | Path | None = None,
    ) -> None:
        """Plot the Pareto frontier of cost vs. reliability.

        Args:
            x_metric: Metric for x-axis ("cost" or "latency").
            y_metric: Metric for y-axis ("success_rate").
            save_path: If provided, save the figure to this path.
        """
        import matplotlib.pyplot as plt

        costs = [r.mean_cost_usd for r in self.results]
        latencies = [r.mean_latency_s for r in self.results]
        rates = [r.success_rate for r in self.results]
        names = [r.strategy_name for r in self.results]

        x_vals = costs if x_metric == "cost" else latencies
        x_label = "Average Cost (USD)" if x_metric == "cost" else "Average Latency (s)"

        frontier_indices = pareto_frontier(x_vals, rates)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_vals, rates, s=100, zorder=5)

        for i, name in enumerate(names):
            ax.annotate(
                name,
                (x_vals[i], rates[i]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
            )

        # Draw Pareto frontier line
        if len(frontier_indices) > 1:
            fx = [x_vals[i] for i in frontier_indices]
            fy = [rates[i] for i in frontier_indices]
            ax.plot(fx, fy, "r--", alpha=0.7, linewidth=2, label="Pareto frontier")
            ax.legend()

        ax.set_xlabel(x_label)
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Cost vs. Reliability — {self.pipeline_name}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Pareto plot saved to %s", save_path)
        else:
            plt.show()

        plt.close(fig)

    def plot_failure_heatmap(
        self,
        save_path: str | Path | None = None,
    ) -> None:
        """Plot a heatmap of failure mode frequency per strategy.

        Args:
            save_path: If provided, save the figure to this path.
        """
        import matplotlib.pyplot as plt

        # Collect all failure types across strategies
        all_types: set[str] = set()
        for r in self.results:
            all_types.update(r.failure_counts.keys())
        failure_types = sorted(all_types)

        if not failure_types:
            logger.warning("No failures recorded; skipping heatmap.")
            return

        strategy_names = [r.strategy_name for r in self.results]
        matrix = np.zeros((len(self.results), len(failure_types)))

        for i, r in enumerate(self.results):
            for j, ft in enumerate(failure_types):
                matrix[i, j] = r.failure_counts.get(ft, 0)

        fig, ax = plt.subplots(
            figsize=(max(8, len(failure_types) * 2), max(4, len(self.results)))
        )
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

        ax.set_xticks(range(len(failure_types)))
        ax.set_xticklabels(
            [ft.replace("_", " ").title() for ft in failure_types],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(strategy_names)))
        ax.set_yticklabels(strategy_names)

        # Add text annotations
        for i in range(len(self.results)):
            for j in range(len(failure_types)):
                val = int(matrix[i, j])
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, label="Failure Count")
        ax.set_title(f"Failure Mode Heatmap — {self.pipeline_name}")
        fig.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Heatmap saved to %s", save_path)
        else:
            plt.show()

        plt.close(fig)

    def recommend(self) -> str:
        """Recommend the best strategy based on cost-efficiency.

        Selects the cheapest strategy that achieves > 95% success rate.
        Falls back to the highest success rate if none exceed 95%.

        Returns:
            Human-readable recommendation string.
        """
        # Strategies exceeding 95% success
        high_reliability = [r for r in self.results if r.success_rate > 0.95]

        if high_reliability:
            best = min(high_reliability, key=lambda r: r.mean_cost_usd)
            baseline_cost = self.results[0].mean_cost_usd
            cost_ratio = (
                best.mean_cost_usd / baseline_cost if baseline_cost > 0 else 1.0
            )
            msg = (
                f"Recommendation: {best.strategy_name} "
                f"({best.success_rate:.1%} success at "
                f"{cost_ratio:.1f}x baseline cost)"
            )
        else:
            best = max(self.results, key=lambda r: r.success_rate)
            msg = (
                f"Recommendation: {best.strategy_name} "
                f"({best.success_rate:.1%} success) "
                f"— no strategy exceeded 95% reliability"
            )

        print(msg)
        return msg


class Comparator:
    """Orchestrates strategy comparison for a pipeline.

    Args:
        pipeline: The pipeline to test.
        failure_config: Failure injection configuration.
        n_simulations: Simulations per strategy.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        failure_config: FailureConfig,
        n_simulations: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.failure_config = failure_config
        self.n_simulations = n_simulations
        self.seed = seed

    def compare(
        self,
        strategies: list[ResilienceStrategy],
    ) -> StrategyComparison:
        """Run simulations for each strategy and produce a comparison.

        Args:
            strategies: List of strategies to evaluate.

        Returns:
            StrategyComparison containing all results.
        """
        sim = Simulator(
            pipeline=self.pipeline,
            failure_config=self.failure_config,
            n_simulations=self.n_simulations,
            seed=self.seed,
        )
        results = sim.compare_strategies(strategies)

        return StrategyComparison(
            results=results,
            pipeline_name=self.pipeline.name,
            n_simulations=self.n_simulations,
        )
