"""Tests for cascade.comparator module."""

from __future__ import annotations

from cascade.comparator import Comparator, StrategyComparison
from cascade.failures import FailureConfig
from cascade.simulator import SimulationResult
from cascade.strategies import adaptive, checkpoint, fallback, naive, parallel, retry


class TestComparator:
    """Tests for the Comparator class."""

    def test_compare_returns_comparison(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        strategies = [naive(), retry(max_attempts=3)]
        result = comp.compare(strategies)
        assert isinstance(result, StrategyComparison)
        assert len(result.results) == 2
        assert result.n_simulations == 50

    def test_comparison_pipeline_name(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        result = comp.compare([naive()])
        assert result.pipeline_name == "simple"


class TestStrategyComparison:
    """Tests for the StrategyComparison model."""

    def test_print_table(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        comparison = comp.compare([naive(), retry(max_attempts=3)])
        text = comparison.print_table()
        assert "Naive" in text
        assert "Retry(3)" in text

    def test_recommend(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        comparison = comp.compare(
            [
                naive(),
                retry(max_attempts=3),
                parallel(n=3),
            ]
        )
        rec = comparison.recommend()
        assert "Recommendation" in rec

    def test_recommend_no_high_reliability(self):
        """When no strategy exceeds 95%, recommend the best available."""
        from cascade.simulator import SimulationResult

        results = [
            SimulationResult(
                n_simulations=100,
                success_count=70,
                success_rate=0.70,
                mean_cost_usd=0.10,
                mean_latency_s=1.0,
                costs=[0.10] * 100,
                latencies=[1.0] * 100,
                strategy_name="Naive",
            ),
            SimulationResult(
                n_simulations=100,
                success_count=85,
                success_rate=0.85,
                mean_cost_usd=0.15,
                mean_latency_s=1.5,
                costs=[0.15] * 100,
                latencies=[1.5] * 100,
                strategy_name="Retry(3)",
            ),
        ]
        comparison = StrategyComparison(
            results=results,
            pipeline_name="test",
            n_simulations=100,
        )
        rec = comparison.recommend()
        assert "Retry(3)" in rec
        assert "no strategy exceeded 95%" in rec

    def test_plain_table_formatting(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        comparison = comp.compare([naive()])
        text = comparison._plain_table()
        assert "Strategy" in text
        assert "Success" in text

    def test_recommend_cheapest_high_reliability(self):
        """When multiple strategies exceed 95%, pick the cheapest."""
        results = [
            SimulationResult(
                n_simulations=100,
                success_count=96,
                success_rate=0.96,
                mean_cost_usd=0.50,
                mean_latency_s=2.0,
                costs=[0.50] * 100,
                latencies=[2.0] * 100,
                strategy_name="Expensive",
            ),
            SimulationResult(
                n_simulations=100,
                success_count=97,
                success_rate=0.97,
                mean_cost_usd=0.20,
                mean_latency_s=1.5,
                costs=[0.20] * 100,
                latencies=[1.5] * 100,
                strategy_name="Cheap",
            ),
        ]
        comparison = StrategyComparison(
            results=results, pipeline_name="test", n_simulations=100
        )
        rec = comparison.recommend()
        assert "Cheap" in rec
        assert "baseline cost" in rec

    def test_recommend_with_zero_baseline_cost(self):
        """Edge case: baseline cost is zero (cost_ratio defaults to 1.0)."""
        results = [
            SimulationResult(
                n_simulations=100,
                success_count=96,
                success_rate=0.96,
                mean_cost_usd=0.0,
                mean_latency_s=1.0,
                costs=[0.0] * 100,
                latencies=[1.0] * 100,
                strategy_name="FreeBaseline",
            ),
            SimulationResult(
                n_simulations=100,
                success_count=98,
                success_rate=0.98,
                mean_cost_usd=0.10,
                mean_latency_s=2.0,
                costs=[0.10] * 100,
                latencies=[2.0] * 100,
                strategy_name="NotFree",
            ),
        ]
        comparison = StrategyComparison(
            results=results, pipeline_name="test", n_simulations=100
        )
        rec = comparison.recommend()
        assert "Recommendation" in rec

    def test_recommend_single_strategy(self):
        """Recommendation with only one strategy."""
        results = [
            SimulationResult(
                n_simulations=100,
                success_count=80,
                success_rate=0.80,
                mean_cost_usd=0.10,
                mean_latency_s=1.0,
                costs=[0.10] * 100,
                latencies=[1.0] * 100,
                strategy_name="Only",
            ),
        ]
        comparison = StrategyComparison(
            results=results, pipeline_name="test", n_simulations=100
        )
        rec = comparison.recommend()
        assert "Only" in rec
        assert "no strategy exceeded 95%" in rec

    def test_recommend_all_perfect(self):
        """All strategies at 100% -- should pick cheapest."""
        results = [
            SimulationResult(
                n_simulations=100,
                success_count=100,
                success_rate=1.0,
                mean_cost_usd=0.30,
                mean_latency_s=3.0,
                costs=[0.30] * 100,
                latencies=[3.0] * 100,
                strategy_name="Pricey",
            ),
            SimulationResult(
                n_simulations=100,
                success_count=100,
                success_rate=1.0,
                mean_cost_usd=0.10,
                mean_latency_s=1.0,
                costs=[0.10] * 100,
                latencies=[1.0] * 100,
                strategy_name="Cheapest",
            ),
        ]
        comparison = StrategyComparison(
            results=results, pipeline_name="test", n_simulations=100
        )
        rec = comparison.recommend()
        assert "Cheapest" in rec

    def test_plain_table_multiple_strategies(self):
        """Plain table with multiple strategies shows all names."""
        results = [
            SimulationResult(
                n_simulations=50,
                success_count=40,
                success_rate=0.80,
                mean_cost_usd=0.10,
                mean_latency_s=1.0,
                costs=[0.10] * 50,
                latencies=[1.0] * 50,
                strategy_name="Alpha",
            ),
            SimulationResult(
                n_simulations=50,
                success_count=45,
                success_rate=0.90,
                mean_cost_usd=0.20,
                mean_latency_s=2.0,
                costs=[0.20] * 50,
                latencies=[2.0] * 50,
                strategy_name="Beta",
            ),
        ]
        comparison = StrategyComparison(
            results=results, pipeline_name="test", n_simulations=50
        )
        text = comparison._plain_table()
        assert "Alpha" in text
        assert "Beta" in text
        assert "Avg Cost" in text
        assert "Failures" in text


class TestStrategyComparisonPlots:
    """Tests for plot methods on StrategyComparison."""

    def test_plot_pareto_save(self, simple_pipeline, default_failures):
        """Test plot_pareto with save_path."""
        import tempfile

        comp = Comparator(simple_pipeline, default_failures, n_simulations=30, seed=42)
        comparison = comp.compare([naive(), retry(max_attempts=3)])
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            save_path = Path(tmpdir) / "pareto.png"
            comparison.plot_pareto(save_path=save_path)
            assert save_path.exists()

    def test_plot_pareto_latency_metric(self, simple_pipeline, default_failures):
        """Test plot_pareto with x_metric='latency'."""
        import tempfile

        comp = Comparator(simple_pipeline, default_failures, n_simulations=30, seed=42)
        comparison = comp.compare([naive(), retry(max_attempts=3)])
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            save_path = Path(tmpdir) / "pareto_latency.png"
            comparison.plot_pareto(x_metric="latency", save_path=save_path)
            assert save_path.exists()

    def test_plot_failure_heatmap_save(self, simple_pipeline, default_failures):
        """Test plot_failure_heatmap with save_path."""
        import tempfile

        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        comparison = comp.compare([naive(), retry(max_attempts=3)])
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            save_path = Path(tmpdir) / "heatmap.png"
            comparison.plot_failure_heatmap(save_path=save_path)
            # May or may not have failures to produce a heatmap
            # If no failures, the function returns early

    def test_plot_failure_heatmap_no_failures(self, simple_pipeline):
        """Heatmap with zero failures should return early."""
        zero = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        comp = Comparator(simple_pipeline, zero, n_simulations=20, seed=42)
        comparison = comp.compare([naive()])
        # Should not raise even though there are no failures
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "heatmap.png"
            comparison.plot_failure_heatmap(save_path=save_path)
            # File should NOT exist since no failures to plot
            assert not save_path.exists()

    def test_plot_pareto_single_strategy(self, simple_pipeline, default_failures):
        """Pareto plot with single strategy (no frontier line)."""
        import tempfile

        comp = Comparator(simple_pipeline, default_failures, n_simulations=30, seed=42)
        comparison = comp.compare([naive()])
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            save_path = Path(tmpdir) / "pareto_single.png"
            comparison.plot_pareto(save_path=save_path)
            assert save_path.exists()


class TestComparatorMultiStrategy:
    """Tests for Comparator with various strategy combinations."""

    def test_compare_five_strategies(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=30, seed=42)
        strategies = [
            naive(),
            retry(max_attempts=3),
            fallback(),
            checkpoint(interval=2),
            adaptive(),
        ]
        result = comp.compare(strategies)
        assert len(result.results) == 5

    def test_compare_no_seed(self, simple_pipeline, default_failures):
        """Compare without a seed (non-deterministic)."""
        comp = Comparator(
            simple_pipeline, default_failures, n_simulations=20, seed=None
        )
        result = comp.compare([naive(), retry(max_attempts=3)])
        assert len(result.results) == 2
