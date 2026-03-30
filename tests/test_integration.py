"""Integration tests for the full cascade simulation pipeline.

These tests exercise the complete workflow from pipeline definition
through simulation, comparison, and reporting.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cascade import (
    Comparator,
    FailureConfig,
    Pipeline,
    Simulator,
    Step,
    build_report,
    export_comparison_json,
    export_json,
    format_report,
)
from cascade.strategies import (
    adaptive,
    checkpoint,
    fallback,
    naive,
    parallel,
    retry,
)


class TestEndToEnd:
    """Full end-to-end integration tests."""

    def test_spec_example_pipeline(self):
        """Reproduce the exact example from the spec."""
        pipeline = Pipeline(
            steps=[
                Step(
                    name="research",
                    model="sonnet",
                    tools=["web_search", "read_file"],
                ),
                Step(
                    name="analyze",
                    model="sonnet",
                    tools=["python_exec"],
                    depends_on=["research"],
                ),
                Step(
                    name="draft",
                    model="sonnet",
                    tools=["write_file"],
                    depends_on=["analyze"],
                ),
                Step(
                    name="review",
                    model="opus",
                    tools=["read_file"],
                    depends_on=["draft"],
                ),
                Step(
                    name="revise",
                    model="sonnet",
                    tools=["write_file"],
                    depends_on=["review"],
                ),
                Step(
                    name="publish",
                    model="haiku",
                    tools=["api_call"],
                    depends_on=["revise"],
                ),
            ]
        )

        failures = FailureConfig(
            hallucination_rate=0.05,
            refusal_rate=0.02,
            tool_failure_rate=0.03,
            context_overflow_at=100000,
            cascade_propagation=0.8,
        )

        sim = Simulator(pipeline, failures, n_simulations=1000, seed=42)
        result = sim.run()

        assert result.n_simulations == 1000
        assert 0.0 < result.success_rate < 1.0
        assert result.mean_cost_usd > 0
        assert len(result.failure_counts) > 0

    def test_strategy_comparison_workflow(self, research_pipeline):
        """Full comparison workflow with multiple strategies."""
        failures = FailureConfig(
            hallucination_rate=0.05,
            refusal_rate=0.02,
            tool_failure_rate=0.03,
        )

        comp = Comparator(
            research_pipeline,
            failures,
            n_simulations=200,
            seed=42,
        )

        strategies = [
            naive(),
            retry(max_attempts=3),
            parallel(n=3, vote="majority"),
            checkpoint(interval=2),
        ]

        comparison = comp.compare(strategies)
        assert len(comparison.results) == 4

        # Verify ordering
        naive_result = comparison.results[0]
        retry_result = comparison.results[1]
        assert naive_result.strategy_name == "Naive"
        assert retry_result.strategy_name == "Retry(3)"

        # Table output
        table = comparison.print_table()
        assert "Naive" in table

        # Recommendation
        rec = comparison.recommend()
        assert "Recommendation" in rec

    def test_full_report_export(self, research_pipeline, default_failures):
        """Test full pipeline -> simulate -> report -> export."""
        sim = Simulator(
            research_pipeline,
            default_failures,
            n_simulations=100,
            seed=42,
        )
        result = sim.run()
        report = build_report(result)
        text = format_report(report)
        assert "Simulation Report" in text

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            export_json(report, path)
            data = json.loads(path.read_text())
            assert data["n_simulations"] == 100

    def test_comparison_json_export(self, simple_pipeline, default_failures):
        """Test comparison export to JSON."""
        comp = Comparator(
            simple_pipeline,
            default_failures,
            n_simulations=50,
            seed=42,
        )
        comparison = comp.compare([naive(), retry(max_attempts=3)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.json"
            export_comparison_json(comparison, path)
            data = json.loads(path.read_text())
            assert len(data["strategies"]) == 2

    def test_single_step_pipeline(self):
        """Even a 1-step pipeline should work."""
        pipeline = Pipeline(
            steps=[Step(name="only_step", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(hallucination_rate=0.1)
        sim = Simulator(pipeline, failures, n_simulations=100, seed=42)
        result = sim.run()
        assert result.n_simulations == 100
        assert result.success_rate > 0

    def test_large_pipeline(self):
        """Test with a 20-step pipeline."""
        steps = [Step(name=f"step_{i}", model="sonnet") for i in range(20)]
        for i in range(1, 20):
            steps[i].depends_on = [f"step_{i - 1}"]

        pipeline = Pipeline(steps=steps, name="large")
        failures = FailureConfig(hallucination_rate=0.05)
        sim = Simulator(pipeline, failures, n_simulations=100, seed=42)
        result = sim.run()
        assert result.n_simulations == 100
        # 20 steps at 5% failure => ~36% success naive
        assert result.success_rate < 0.8

    def test_zero_failure_all_strategies(self, simple_pipeline, zero_failures):
        """All strategies should achieve 100% with zero failures."""
        strategies = [
            naive(),
            retry(max_attempts=3),
            fallback(models=["sonnet", "haiku"]),
            checkpoint(interval=2),
            adaptive(),
        ]
        for strat in strategies:
            sim = Simulator(
                simple_pipeline,
                zero_failures,
                n_simulations=50,
                seed=42,
                strategy=strat,
            )
            result = sim.run()
            assert result.success_rate == 1.0, (
                f"{strat.display_name} should be 100% with zero failures"
            )
