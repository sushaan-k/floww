"""Tests for cascade.cli module.

Exercises the CLI commands (simulate and compare) using Click's CliRunner,
including various option combinations, edge cases, and error handling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from cascade.cli import (
    STRATEGY_REGISTRY,
    _configure_logging,
    _load_pipeline_from_json,
    main,
)


def _write_pipeline_json(directory: Path, name: str = "test-pipeline") -> Path:
    """Write a minimal pipeline JSON file and return its path."""
    data = {
        "name": name,
        "steps": [
            {"name": "step_a", "model": "sonnet", "tools": ["web_search"]},
            {"name": "step_b", "model": "sonnet", "depends_on": ["step_a"]},
            {"name": "step_c", "model": "haiku", "depends_on": ["step_b"]},
        ],
    }
    path = directory / "pipeline.json"
    path.write_text(json.dumps(data))
    return path


class TestLoadPipelineFromJson:
    """Tests for _load_pipeline_from_json helper."""

    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            pipeline = _load_pipeline_from_json(path)
            assert pipeline.name == "test-pipeline"
            assert len(pipeline.steps) == 3

    def test_name_defaults_to_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"steps": [{"name": "s1", "model": "sonnet"}]}
            path = Path(tmpdir) / "my_pipe.json"
            path.write_text(json.dumps(data))
            pipeline = _load_pipeline_from_json(path)
            assert pipeline.name == "my_pipe"

    def test_empty_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"name": "empty", "steps": []}
            path = Path(tmpdir) / "empty.json"
            path.write_text(json.dumps(data))
            pipeline = _load_pipeline_from_json(path)
            assert len(pipeline.steps) == 0

    def test_description_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "name": "desc",
                "description": "A test pipeline",
                "steps": [{"name": "s1"}],
            }
            path = Path(tmpdir) / "desc.json"
            path.write_text(json.dumps(data))
            pipeline = _load_pipeline_from_json(path)
            assert pipeline.description == "A test pipeline"


class TestConfigureLogging:
    """Tests for _configure_logging helper."""

    def test_verbose_sets_debug(self):
        """Should not raise and set logging level."""
        _configure_logging(verbose=True)

    def test_non_verbose_sets_warning(self):
        _configure_logging(verbose=False)


class TestStrategyRegistry:
    """Tests for the STRATEGY_REGISTRY constant."""

    def test_all_expected_strategies_present(self):
        expected = {"naive", "retry", "fallback", "parallel", "checkpoint", "adaptive"}
        assert set(STRATEGY_REGISTRY.keys()) == expected

    def test_registry_values_are_strategies(self):
        from cascade.strategies import ResilienceStrategy

        for name, strat in STRATEGY_REGISTRY.items():
            assert isinstance(strat, ResilienceStrategy), f"{name} is not a strategy"


class TestCLISimulateCommand:
    """Tests for the 'simulate' CLI command."""

    def test_simulate_basic(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            result = runner.invoke(
                main, ["simulate", str(path), "-n", "50", "--seed", "42"]
            )
            assert result.exit_code == 0
            assert "Simulation Report" in result.output

    def test_simulate_with_strategy(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            result = runner.invoke(
                main,
                [
                    "simulate",
                    str(path),
                    "-n",
                    "30",
                    "-s",
                    "retry",
                    "--seed",
                    "1",
                ],
            )
            assert result.exit_code == 0
            assert "Simulation Report" in result.output

    def test_simulate_with_output_json(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = _write_pipeline_json(Path(tmpdir))
            output_path = Path(tmpdir) / "output.json"
            result = runner.invoke(
                main,
                [
                    "simulate",
                    str(pipeline_path),
                    "-n",
                    "20",
                    "--seed",
                    "42",
                    "-o",
                    str(output_path),
                ],
            )
            assert result.exit_code == 0
            assert "Report exported" in result.output
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert data["n_simulations"] == 20

    def test_simulate_custom_failure_rates(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            result = runner.invoke(
                main,
                [
                    "simulate",
                    str(path),
                    "-n",
                    "20",
                    "--hallucination-rate",
                    "0.1",
                    "--tool-failure-rate",
                    "0.1",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0

    def test_simulate_all_strategies(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            for strategy_name in STRATEGY_REGISTRY:
                result = runner.invoke(
                    main,
                    [
                        "simulate",
                        str(path),
                        "-n",
                        "10",
                        "-s",
                        strategy_name,
                        "--seed",
                        "42",
                    ],
                )
                assert result.exit_code == 0, (
                    f"Strategy {strategy_name} failed: {result.output}"
                )

    def test_simulate_verbose(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            result = runner.invoke(
                main, ["-v", "simulate", str(path), "-n", "10", "--seed", "1"]
            )
            assert result.exit_code == 0

    def test_simulate_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["simulate", "/nonexistent/file.json"])
        assert result.exit_code != 0


class TestCLICompareCommand:
    """Tests for the 'compare' CLI command."""

    def test_compare_basic(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            result = runner.invoke(
                main,
                [
                    "compare",
                    str(path),
                    "-n",
                    "30",
                    "--strategies",
                    "naive,retry",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0
            assert "Recommendation" in result.output

    def test_compare_with_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = _write_pipeline_json(Path(tmpdir))
            output_path = Path(tmpdir) / "comparison.json"
            result = runner.invoke(
                main,
                [
                    "compare",
                    str(pipeline_path),
                    "-n",
                    "20",
                    "--strategies",
                    "naive,retry",
                    "--seed",
                    "42",
                    "-o",
                    str(output_path),
                ],
            )
            assert result.exit_code == 0
            assert "Comparison exported" in result.output
            assert output_path.exists()

    def test_compare_unknown_strategy(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            result = runner.invoke(
                main,
                [
                    "compare",
                    str(path),
                    "-n",
                    "10",
                    "--strategies",
                    "naive,bogus_strategy",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code != 0

    def test_compare_all_builtin_strategies(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_pipeline_json(Path(tmpdir))
            all_names = ",".join(STRATEGY_REGISTRY.keys())
            result = runner.invoke(
                main,
                [
                    "compare",
                    str(path),
                    "-n",
                    "10",
                    "--strategies",
                    all_names,
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0

    def test_compare_with_pareto_plot(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = _write_pipeline_json(Path(tmpdir))
            pareto_path = Path(tmpdir) / "pareto.png"
            result = runner.invoke(
                main,
                [
                    "compare",
                    str(pipeline_path),
                    "-n",
                    "20",
                    "--strategies",
                    "naive,retry",
                    "--seed",
                    "42",
                    "--pareto",
                    str(pareto_path),
                ],
            )
            assert result.exit_code == 0
            assert "Pareto plot saved" in result.output
            assert pareto_path.exists()

    def test_compare_with_heatmap(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = _write_pipeline_json(Path(tmpdir))
            heatmap_path = Path(tmpdir) / "heatmap.png"
            result = runner.invoke(
                main,
                [
                    "compare",
                    str(pipeline_path),
                    "-n",
                    "50",
                    "--strategies",
                    "naive,retry",
                    "--seed",
                    "42",
                    "--hallucination-rate",
                    "0.2",
                    "--heatmap",
                    str(heatmap_path),
                ],
            )
            assert result.exit_code == 0
            assert "Heatmap saved" in result.output
            assert heatmap_path.exists()

    def test_compare_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compare", "/nonexistent/file.json"])
        assert result.exit_code != 0


class TestCLIVersionAndHelp:
    """Tests for version and help output."""

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Cascade" in result.output

    def test_simulate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "simulations" in result.output.lower()

    def test_compare_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "strategies" in result.output.lower()
