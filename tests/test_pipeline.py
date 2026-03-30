"""Tests for cascade.pipeline module."""

from __future__ import annotations

import pytest

from cascade.pipeline import (
    MODEL_COST_PER_1K_INPUT,
    MODEL_COST_PER_1K_OUTPUT,
    Pipeline,
    Step,
)


class TestStep:
    """Tests for the Step model."""

    def test_defaults(self):
        s = Step(name="test")
        assert s.model == "sonnet"
        assert s.tools == []
        assert s.depends_on == []
        assert s.input_tokens == 500
        assert s.output_tokens == 200

    def test_cost_calculation(self):
        s = Step(name="test", model="sonnet")
        expected = (
            500 / 1000 * MODEL_COST_PER_1K_INPUT["sonnet"]
            + 200 / 1000 * MODEL_COST_PER_1K_OUTPUT["sonnet"]
        )
        assert abs(s.cost_usd() - expected) < 1e-9

    @pytest.mark.parametrize(
        "model,expected_latency",
        [
            ("opus", 2.0),
            ("sonnet", 1.0),
            ("haiku", 0.3),
        ],
    )
    def test_latency_by_model(self, model, expected_latency):
        s = Step(name="test", model=model)
        assert s.latency_s() == expected_latency

    def test_custom_latency_override(self):
        s = Step(name="test", model="sonnet", base_latency_s=5.0)
        assert s.latency_s() == 5.0

    def test_unknown_model_defaults(self):
        s = Step(name="test", model="unknown_model")
        # Should fall back to default rates
        assert s.cost_usd() > 0
        assert s.latency_s() == 1.0


class TestPipeline:
    """Tests for the Pipeline model."""

    def test_valid_linear_pipeline(self, simple_pipeline):
        assert len(simple_pipeline.steps) == 3
        order = simple_pipeline.topological_order()
        names = [s.name for s in order]
        assert names.index("step_a") < names.index("step_b")
        assert names.index("step_b") < names.index("step_c")

    def test_duplicate_step_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate step name"):
            Pipeline(
                steps=[
                    Step(name="a"),
                    Step(name="a"),
                ],
            )

    def test_missing_dependency_rejected(self):
        with pytest.raises(ValueError, match="unknown step"):
            Pipeline(
                steps=[
                    Step(name="a", depends_on=["nonexistent"]),
                ],
            )

    def test_cycle_rejected(self):
        with pytest.raises(ValueError, match="cycle"):
            Pipeline(
                steps=[
                    Step(name="a", depends_on=["b"]),
                    Step(name="b", depends_on=["a"]),
                ],
            )

    def test_diamond_dag(self):
        """A->B, A->C, B->D, C->D should work."""
        p = Pipeline(
            steps=[
                Step(name="a"),
                Step(name="b", depends_on=["a"]),
                Step(name="c", depends_on=["a"]),
                Step(name="d", depends_on=["b", "c"]),
            ],
        )
        order = p.topological_order()
        names = [s.name for s in order]
        assert names[0] == "a"
        assert names[-1] == "d"

    def test_step_by_name(self, simple_pipeline):
        step = simple_pipeline.step_by_name("step_b")
        assert step.name == "step_b"

    def test_step_by_name_missing(self, simple_pipeline):
        with pytest.raises(KeyError, match="no_such_step"):
            simple_pipeline.step_by_name("no_such_step")

    def test_downstream_of(self, research_pipeline):
        downstream = research_pipeline.downstream_of("analyze")
        assert "draft" in downstream
        assert "review" in downstream
        assert "revise" in downstream
        assert "publish" in downstream
        assert "research" not in downstream

    def test_downstream_of_leaf(self, research_pipeline):
        downstream = research_pipeline.downstream_of("publish")
        assert downstream == []

    def test_downstream_of_missing_step(self, simple_pipeline):
        with pytest.raises(KeyError):
            simple_pipeline.downstream_of("nonexistent")

    def test_total_baseline_cost(self, simple_pipeline):
        cost = simple_pipeline.total_baseline_cost()
        expected = sum(s.cost_usd() for s in simple_pipeline.steps)
        assert abs(cost - expected) < 1e-9

    def test_total_baseline_latency(self, simple_pipeline):
        latency = simple_pipeline.total_baseline_latency()
        expected = sum(s.latency_s() for s in simple_pipeline.steps)
        assert abs(latency - expected) < 1e-9

    def test_total_baseline_latency_diamond_critical_path(self):
        p = Pipeline(
            steps=[
                Step(name="a", base_latency_s=1.0),
                Step(name="b", base_latency_s=5.0, depends_on=["a"]),
                Step(name="c", base_latency_s=1.0, depends_on=["a"]),
                Step(name="d", base_latency_s=1.0, depends_on=["b", "c"]),
            ],
        )
        assert p.total_baseline_latency() == 7.0

    def test_single_step_pipeline(self):
        p = Pipeline(steps=[Step(name="only")])
        assert p.topological_order()[0].name == "only"
        assert p.downstream_of("only") == []

    def test_downstream_diamond_visits_once(self):
        """In a diamond DAG A->B, A->C, B->D, C->D, downstream_of(A)
        should list B, C, D once each (exercises the continue on visited)."""
        p = Pipeline(
            steps=[
                Step(name="a"),
                Step(name="b", depends_on=["a"]),
                Step(name="c", depends_on=["a"]),
                Step(name="d", depends_on=["b", "c"]),
            ],
        )
        downstream = p.downstream_of("a")
        assert downstream == ["b", "c", "d"]
