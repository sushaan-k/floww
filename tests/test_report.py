"""Tests for cascade.report module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cascade.comparator import Comparator
from cascade.report import (
    SimulationReport,
    _json_default,
    build_report,
    export_comparison_json,
    export_json,
    format_report,
    print_comparison_report,
)
from cascade.simulator import SimulationResult, Simulator
from cascade.strategies import naive, retry


class TestBuildReport:
    """Tests for the build_report function."""

    def test_from_simulation(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        report = build_report(result)

        assert isinstance(report, SimulationReport)
        assert report.n_simulations == 100
        assert 0.0 <= report.success_rate <= 1.0
        assert report.success_ci.lower <= report.success_rate
        assert report.success_ci.upper >= report.success_rate

    def test_cost_ci(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        report = build_report(result)
        assert report.cost_ci.lower <= report.cost_ci.point
        assert report.cost_ci.upper >= report.cost_ci.point


class TestFormatReport:
    """Tests for the format_report function."""

    def test_output_format(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        result = sim.run()
        report = build_report(result)
        text = format_report(report)

        assert "Simulation Report" in text
        assert "Success Rate" in text
        assert "Mean Cost" in text
        assert "Mean Latency" in text

    def test_failure_breakdown_included(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=200, seed=42)
        result = sim.run()
        report = build_report(result)
        text = format_report(report)
        # Should have failures with default config
        if report.failure_counts:
            assert "Failure Breakdown" in text


class TestExportJson:
    """Tests for JSON export functions."""

    def test_export_report(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        result = sim.run()
        report = build_report(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            export_json(report, path)
            assert path.exists()

            data = json.loads(path.read_text())
            assert data["n_simulations"] == 50
            assert "success_rate" in data

    def test_export_comparison(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        comparison = comp.compare([naive(), retry(max_attempts=3)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.json"
            export_comparison_json(comparison, path)
            assert path.exists()

            data = json.loads(path.read_text())
            assert data["n_simulations"] == 50
            assert len(data["strategies"]) == 2

    def test_export_creates_parent_dirs(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=10, seed=42)
        result = sim.run()
        report = build_report(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "report.json"
            export_json(report, path)
            assert path.exists()


class TestPrintComparisonReport:
    """Tests for the print_comparison_report function."""

    def test_prints_all_strategies(self, simple_pipeline, default_failures):
        comp = Comparator(simple_pipeline, default_failures, n_simulations=50, seed=42)
        comparison = comp.compare([naive(), retry(max_attempts=3)])
        text = print_comparison_report(comparison)
        assert "Naive" in text
        assert "Retry(3)" in text


class TestJsonDefault:
    """Tests for the _json_default fallback serializer."""

    def test_numpy_integer(self):
        val = np.int64(42)
        assert _json_default(val) == 42
        assert isinstance(_json_default(val), int)

    def test_numpy_floating(self):
        val = np.float64(3.14)
        result = _json_default(val)
        assert abs(result - 3.14) < 1e-9
        assert isinstance(result, float)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _json_default(arr)
        assert result == [1, 2, 3]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_numpy_int32(self):
        val = np.int32(10)
        assert _json_default(val) == 10

    def test_numpy_float32(self):
        val = np.float32(2.5)
        result = _json_default(val)
        assert abs(result - 2.5) < 0.01

    def test_numpy_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _json_default(arr)
        assert result == [[1, 2], [3, 4]]


class TestBuildReportEdgeCases:
    """Edge cases for build_report."""

    def test_single_simulation(self, simple_pipeline, zero_failures):
        """Build report from a single simulation run."""
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim.run()
        report = build_report(result)
        assert report.n_simulations == 1
        # With 1 simulation, CI should collapse to point
        assert report.cost_ci.lower == report.cost_ci.point
        assert report.cost_ci.upper == report.cost_ci.point
        assert report.latency_ci.lower == report.latency_ci.point
        assert report.latency_ci.upper == report.latency_ci.point

    def test_report_with_no_failures(self, simple_pipeline, zero_failures):
        """Report when there are zero failures."""
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=50, seed=42)
        result = sim.run()
        report = build_report(result)
        assert report.success_rate == 1.0
        assert report.failure_counts == {}

    def test_format_report_no_failures(self, simple_pipeline, zero_failures):
        """Format a report with no failure breakdown."""
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=50, seed=42)
        result = sim.run()
        report = build_report(result)
        text = format_report(report)
        assert "Simulation Report" in text
        # Should NOT have "Failure Breakdown" with no failures
        assert "Failure Breakdown" not in text

    def test_report_with_high_failures(self, simple_pipeline, high_failure_config):
        """Report when there are many failures."""
        sim = Simulator(
            simple_pipeline, high_failure_config, n_simulations=100, seed=42
        )
        result = sim.run()
        report = build_report(result)
        assert report.success_rate < 1.0
        text = format_report(report)
        if report.failure_counts:
            assert "Failure Breakdown" in text


class TestReportMinimalData:
    """Tests with minimal/synthetic SimulationResult data."""

    def test_build_report_from_synthetic_result(self):
        """Build a report from a manually-created SimulationResult."""
        result = SimulationResult(
            n_simulations=10,
            success_count=7,
            success_rate=0.7,
            mean_cost_usd=0.15,
            mean_latency_s=2.0,
            failure_counts={"hallucination": 3},
            costs=[0.15] * 10,
            latencies=[2.0] * 10,
            strategy_name="TestStrategy",
        )
        report = build_report(result)
        assert report.strategy_name == "TestStrategy"
        assert report.n_simulations == 10
        assert abs(report.success_rate - 0.7) < 1e-9

    def test_format_report_from_synthetic(self):
        """Format a report from synthetic data."""
        result = SimulationResult(
            n_simulations=5,
            success_count=5,
            success_rate=1.0,
            mean_cost_usd=0.10,
            mean_latency_s=1.0,
            costs=[0.10] * 5,
            latencies=[1.0] * 5,
            strategy_name="Perfect",
        )
        report = build_report(result)
        text = format_report(report)
        assert "Perfect" in text
        assert "100.0%" in text
