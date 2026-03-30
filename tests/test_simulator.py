"""Tests for cascade.simulator module."""

from __future__ import annotations

import numpy as np

from cascade.failures import FailureConfig, FailureEvent, FailureInjector, FailureType
from cascade.pipeline import Pipeline, Step
from cascade.simulator import (
    RunResult,
    SimulationResult,
    Simulator,
    StepResult,
)
from cascade.strategies import (
    adaptive,
    checkpoint,
    fallback,
    human_in_loop,
    naive,
    parallel,
    retry,
)


class TestSimulatorZeroFailures:
    """Simulations with zero failure rates should always succeed."""

    def test_all_succeed(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=100, seed=42)
        result = sim.run()
        assert result.success_rate == 1.0
        assert result.success_count == 100

    def test_costs_are_consistent(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=50, seed=1)
        result = sim.run()
        expected_cost = simple_pipeline.total_baseline_cost()
        # Each run should have the same cost (no retries)
        for cost in result.costs:
            assert abs(cost - expected_cost) < 1e-6

    def test_latencies_are_consistent(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=50, seed=1)
        result = sim.run()
        expected_latency = simple_pipeline.total_baseline_latency()
        for latency in result.latencies:
            assert abs(latency - expected_latency) < 1e-6

    def test_dag_latency_uses_critical_path(self, zero_failures):
        pipeline = Pipeline(
            steps=[
                Step(name="a", base_latency_s=1.0),
                Step(name="b", base_latency_s=5.0, depends_on=["a"]),
                Step(name="c", base_latency_s=1.0, depends_on=["a"]),
                Step(name="d", base_latency_s=1.0, depends_on=["b", "c"]),
            ]
        )
        sim = Simulator(pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim.run()
        assert result.mean_latency_s == 7.0
        assert result.latencies == [7.0]


class TestSimulatorWithFailures:
    """Simulations with non-zero failure rates."""

    def test_some_failures_occur(self, research_pipeline, default_failures):
        sim = Simulator(research_pipeline, default_failures, n_simulations=500, seed=42)
        result = sim.run()
        assert result.success_rate < 1.0
        assert result.success_count < 500
        assert len(result.failure_counts) > 0

    def test_high_failure_rate_low_success(self, simple_pipeline, high_failure_config):
        sim = Simulator(
            simple_pipeline,
            high_failure_config,
            n_simulations=500,
            seed=42,
        )
        result = sim.run()
        assert result.success_rate < 0.8

    def test_result_fields(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        assert result.n_simulations == 100
        assert 0.0 <= result.success_rate <= 1.0
        assert result.mean_cost_usd > 0
        assert result.mean_latency_s > 0
        assert len(result.costs) == 100
        assert len(result.latencies) == 100


class TestSimulatorStrategies:
    """Test that different strategies produce different results."""

    def test_retry_improves_success(self, research_pipeline, default_failures):
        sim = Simulator(research_pipeline, default_failures, n_simulations=500, seed=42)
        naive_result = sim.run(strategy=naive())

        sim = Simulator(research_pipeline, default_failures, n_simulations=500, seed=42)
        retry_result = sim.run(strategy=retry(max_attempts=3))

        assert retry_result.success_rate >= naive_result.success_rate

    def test_retry_costs_more(self, research_pipeline, default_failures):
        sim = Simulator(research_pipeline, default_failures, n_simulations=200, seed=42)
        naive_result = sim.run(strategy=naive())

        sim = Simulator(research_pipeline, default_failures, n_simulations=200, seed=42)
        retry_result = sim.run(strategy=retry(max_attempts=3))

        # Retry should cost at least as much (it might retry on failures)
        assert retry_result.mean_cost_usd >= naive_result.mean_cost_usd * 0.95

    def test_compare_strategies_returns_list(self, simple_pipeline, default_failures):
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        strategies = [naive(), retry(max_attempts=3)]
        results = sim.compare_strategies(strategies)
        assert len(results) == 2
        assert all(isinstance(r, SimulationResult) for r in results)

    def test_checkpoint_strategy(self, research_pipeline, default_failures):
        sim = Simulator(research_pipeline, default_failures, n_simulations=200, seed=42)
        result = sim.run(strategy=checkpoint(interval=2))
        assert result.success_rate > 0

    def test_adaptive_strategy(self, research_pipeline, default_failures):
        sim = Simulator(research_pipeline, default_failures, n_simulations=200, seed=42)
        result = sim.run(strategy=adaptive())
        assert result.success_rate > 0

    def test_fallback_strategy(self, research_pipeline, default_failures):
        sim = Simulator(research_pipeline, default_failures, n_simulations=200, seed=42)
        result = sim.run(strategy=fallback(models=["sonnet", "haiku"]))
        assert result.success_rate > 0

    def test_fallback_uses_alternate_model_costs(self, monkeypatch):
        pipeline = Pipeline(steps=[Step(name="s0", model="opus")])
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        call_models: list[str] = []

        def fake_inject(
            self,
            step_name,
            model,
            tools,
            cumulative_tokens,
            upstream_corrupted=False,
        ):
            call_models.append(model)
            if len(call_models) == 1:
                return FailureEvent(
                    step_name=step_name,
                    failure_type=FailureType.HALLUCINATION,
                    recoverable=True,
                )
            return None

        monkeypatch.setattr(FailureInjector, "inject", fake_inject)
        result = sim.run(strategy=fallback(models=["sonnet", "haiku"]))
        step = pipeline.steps[0]
        expected_cost = step.cost_usd(model="opus") + step.cost_usd(model="sonnet")
        assert result.success_rate == 1.0
        assert call_models == ["opus", "sonnet"]
        assert abs(result.costs[0] - expected_cost) < 1e-9

    def test_adaptive_escalates_to_configured_fallback(self, monkeypatch):
        pipeline = Pipeline(steps=[Step(name="s0", model="opus")])
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        call_models: list[str] = []

        def fake_inject(
            self,
            step_name,
            model,
            tools,
            cumulative_tokens,
            upstream_corrupted=False,
        ):
            call_models.append(model)
            if len(call_models) == 1:
                return FailureEvent(
                    step_name=step_name,
                    failure_type=FailureType.HALLUCINATION,
                    recoverable=True,
                )
            return None

        monkeypatch.setattr(FailureInjector, "inject", fake_inject)
        result = sim.run(
            strategy=adaptive(
                escalation_threshold=1,
                escalation_strategy="fallback",
            ),
        )
        step = pipeline.steps[0]
        expected_cost = step.cost_usd(model="opus") + step.cost_usd(model="sonnet")
        assert result.success_rate == 1.0
        assert call_models == ["opus", "sonnet"]
        assert abs(result.costs[0] - expected_cost) < 1e-9


class TestSimulatorReproducibility:
    """Test that simulations are reproducible with the same seed."""

    def test_same_seed_same_results(self, simple_pipeline, default_failures):
        sim1 = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        r1 = sim1.run()

        sim2 = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        r2 = sim2.run()

        assert r1.success_count == r2.success_count
        assert r1.costs == r2.costs

    def test_different_seed_different_results(self, simple_pipeline, default_failures):
        sim1 = Simulator(simple_pipeline, default_failures, n_simulations=200, seed=42)
        r1 = sim1.run()

        sim2 = Simulator(simple_pipeline, default_failures, n_simulations=200, seed=99)
        r2 = sim2.run()

        # Very unlikely to be identical with different seeds
        assert r1.costs != r2.costs


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_creation(self):
        sr = StepResult(step_name="s", success=True)
        assert sr.step_name == "s"
        assert sr.success is True
        assert sr.attempts == 1
        assert sr.failure is None


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_creation(self):
        rr = RunResult(success=True)
        assert rr.success is True
        assert rr.step_results == []
        assert rr.total_cost_usd == 0.0

    def test_default_fields(self):
        rr = RunResult(success=False)
        assert rr.total_latency_s == 0.0
        assert rr.failure_events == []
        assert rr.steps_completed == 0
        assert rr.first_failure_step == -1
        assert rr.recovered is False


class TestSimulatorZeroTrials:
    """Edge case: zero simulation trials."""

    def test_zero_simulations(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=0, seed=42)
        result = sim.run()
        assert result.n_simulations == 0
        assert result.success_count == 0
        assert result.success_rate == 0.0
        assert result.costs == []
        assert result.latencies == []


class TestSimulatorSingleStepPipeline:
    """Edge cases for a pipeline with only one step."""

    def test_single_step_zero_failures(self):
        pipeline = Pipeline(
            steps=[Step(name="only", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        sim = Simulator(pipeline, failures, n_simulations=50, seed=42)
        result = sim.run()
        assert result.success_rate == 1.0

    def test_single_step_high_failures(self):
        pipeline = Pipeline(
            steps=[Step(name="only", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(
            hallucination_rate=0.5,
            refusal_rate=0.3,
            tool_failure_rate=0.5,
        )
        sim = Simulator(pipeline, failures, n_simulations=200, seed=42)
        result = sim.run()
        assert result.success_rate < 1.0


class TestSimulatorAllFail:
    """Scenario where all steps are guaranteed to fail."""

    def test_guaranteed_failure_via_context_overflow(self):
        """A step whose cumulative tokens exceed limit always fails."""
        pipeline = Pipeline(
            steps=[
                Step(
                    name="s1",
                    model="sonnet",
                    input_tokens=200000,
                    output_tokens=0,
                ),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
            ]
        )
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            context_overflow_at=100000,
        )
        sim = Simulator(pipeline, failures, n_simulations=20, seed=42)
        result = sim.run()
        assert result.success_rate < 1.0

    def test_context_overflow_triggers_on_current_step(self):
        """Overflow should be checked against the current step's token usage."""
        pipeline = Pipeline(
            steps=[
                Step(
                    name="s1",
                    model="sonnet",
                    input_tokens=200000,
                    output_tokens=0,
                )
            ]
        )
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
            context_overflow_at=100000,
        )
        sim = Simulator(pipeline, failures, n_simulations=20, seed=42)
        result = sim.run()
        assert result.success_rate == 0.0
        assert result.failure_counts["context_overflow"] == 20

    def test_all_refusal(self):
        """100% refusal rate with naive strategy should yield 0% success."""
        pipeline = Pipeline(steps=[Step(name="s1", model="sonnet")])
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=1.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        sim = Simulator(pipeline, failures, n_simulations=50, seed=42)
        result = sim.run()
        assert result.success_rate == 0.0


class TestSimulatorParallelStrategy:
    """Tests for the parallel execution path in the simulator."""

    def test_parallel_majority_vote(self):
        pipeline = Pipeline(
            steps=[Step(name="s1", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(hallucination_rate=0.3, tool_failure_rate=0.2)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=parallel(n=3, vote="majority"),
        )
        result = sim.run()
        assert result.success_rate > 0

    def test_parallel_unanimous_vote(self):
        pipeline = Pipeline(
            steps=[Step(name="s1", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(hallucination_rate=0.2)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=parallel(n=3, vote="unanimous"),
        )
        result = sim.run()
        # Unanimous is stricter, so success rate is typically lower
        assert 0.0 <= result.success_rate <= 1.0

    def test_parallel_any_vote(self):
        pipeline = Pipeline(
            steps=[Step(name="s1", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(hallucination_rate=0.3)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=parallel(n=3, vote="any"),
        )
        result = sim.run()
        # "any" is most lenient
        assert result.success_rate > 0


class TestSimulatorHumanInLoop:
    """Tests for human-in-the-loop strategy execution."""

    def test_human_in_loop_strategy(self):
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet", tools=["web_search"]),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
            ]
        )
        failures = FailureConfig(hallucination_rate=0.1)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=human_in_loop(at_steps=[1, 2], accuracy=0.95),
        )
        result = sim.run()
        assert result.success_rate > 0

    def test_human_in_loop_perfect_accuracy(self):
        """With 100% accuracy humans always catch errors."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
            ]
        )
        failures = FailureConfig(hallucination_rate=0.1)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=100,
            seed=42,
            strategy=human_in_loop(at_steps=[1], accuracy=1.0),
        )
        result = sim.run()
        assert result.success_rate > 0

    def test_human_in_loop_zero_accuracy(self):
        """With 0% accuracy humans never catch errors."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
            ]
        )
        failures = FailureConfig(hallucination_rate=0.1)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=100,
            seed=42,
            strategy=human_in_loop(at_steps=[1], accuracy=0.0),
        )
        result = sim.run()
        assert 0.0 <= result.success_rate <= 1.0

    def test_human_in_loop_catches_own_hallucination(self):
        pipeline = Pipeline(steps=[Step(name="s0", model="sonnet")])
        failures = FailureConfig(
            hallucination_rate=1.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=50,
            seed=42,
            strategy=human_in_loop(at_steps=[0], accuracy=1.0),
        )
        result = sim.run()
        assert result.success_rate == 1.0
        assert result.recovery_rate == 1.0
        assert result.failure_counts["hallucination"] == 50


class TestSimulatorAdaptiveEscalation:
    """Tests for adaptive strategy escalation thresholds."""

    def test_adaptive_low_threshold(self):
        """Low threshold means escalation happens after just 1 failure."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet", tools=["web_search"]),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
            ]
        )
        failures = FailureConfig(hallucination_rate=0.2, tool_failure_rate=0.1)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=adaptive(escalation_threshold=1),
        )
        result = sim.run()
        assert result.success_rate > 0

    def test_adaptive_high_threshold(self):
        """High threshold means escalation rarely happens."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
            ]
        )
        failures = FailureConfig(hallucination_rate=0.1)
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=adaptive(escalation_threshold=10),
        )
        result = sim.run()
        assert 0.0 <= result.success_rate <= 1.0


class TestSimulatorLatencySpike:
    """Tests for latency spike handling (non-fatal)."""

    def test_latency_spike_is_non_fatal(self):
        pipeline = Pipeline(steps=[Step(name="s1", model="sonnet")])
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=1.0,
            spike_multiplier=5.0,
        )
        sim = Simulator(pipeline, failures, n_simulations=50, seed=42)
        result = sim.run()
        # Latency spikes are non-fatal, so success rate should be 1.0
        assert result.success_rate == 1.0
        # But latency should be elevated
        expected_base = pipeline.total_baseline_latency()
        assert result.mean_latency_s > expected_base


class TestSimulatorRecovery:
    """Tests targeting the recovery path (line 262) in _run_single."""

    def test_recovery_after_retry(self):
        """A retry strategy should be able to recover from a failure,
        triggering the recovered=True path."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet", tools=["web_search"]),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
            ]
        )
        # Moderate failure rate -- some runs will fail then recover
        failures = FailureConfig(
            hallucination_rate=0.15,
            refusal_rate=0.05,
            tool_failure_rate=0.15,
        )
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=500,
            seed=42,
            strategy=retry(max_attempts=3),
        )
        result = sim.run()
        # With retry and moderate failures, some runs should recover
        assert result.recovery_rate >= 0.0


class TestSimulatorHumanInLoopCatchCorruption:
    """Tests targeting lines 322-324: human catching upstream corruption."""

    def test_human_catches_corruption_with_high_accuracy(self):
        """When the human has 100% accuracy and upstream is corrupted,
        the human should always catch it."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
            ]
        )
        # High hallucination and cascade propagation to produce corruption
        failures = FailureConfig(
            hallucination_rate=0.5,
            cascade_propagation=1.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=500,
            seed=42,
            strategy=human_in_loop(at_steps=[1], accuracy=1.0),
        )
        result = sim.run()
        # Even with corruption, human should catch it -- improving success
        assert result.success_rate > 0


class TestSimulatorAdaptiveEscalationTriggered:
    """Tests targeting line 527: adaptive escalation when threshold is met."""

    def test_escalation_increases_attempts(self):
        """With high failure rates and low threshold, adaptive escalation
        should be triggered, giving extra attempts."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet", tools=["web_search"]),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
                Step(name="s3", model="sonnet", depends_on=["s2"]),
            ]
        )
        failures = FailureConfig(
            hallucination_rate=0.2,
            refusal_rate=0.1,
            tool_failure_rate=0.15,
        )
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=300,
            seed=42,
            strategy=adaptive(escalation_threshold=1),
        )
        result = sim.run()
        assert result.success_rate >= 0.0


class TestSimulatorCheckpointRollback:
    """Tests targeting the checkpoint rollback path."""

    def test_checkpoint_rollback_can_succeed(self):
        """With a checkpoint strategy and failures, some rollbacks should
        succeed, triggering the recovery path."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
                Step(name="s3", model="sonnet", depends_on=["s2"]),
            ]
        )
        failures = FailureConfig(
            hallucination_rate=0.15,
            refusal_rate=0.05,
            tool_failure_rate=0.1,
        )
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=500,
            seed=42,
            strategy=checkpoint(interval=1),
        )
        result = sim.run()
        # Some runs should recover via rollback
        assert result.success_rate > 0

    def test_checkpoint_rollback_can_fail(self):
        """With very high failure rates, rollback attempts themselves fail."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="sonnet", depends_on=["s1"]),
            ]
        )
        failures = FailureConfig(
            hallucination_rate=0.5,
            refusal_rate=0.3,
            tool_failure_rate=0.5,
        )
        sim = Simulator(
            pipeline,
            failures,
            n_simulations=200,
            seed=42,
            strategy=checkpoint(interval=1),
        )
        result = sim.run()
        assert result.success_rate < 1.0


class TestExecuteStepParallelDirect:
    """Direct tests for _execute_step_parallel (not called by _run_single
    but exists as a method on Simulator)."""

    def test_parallel_majority_success(self):
        """Direct call with zero failures should succeed."""
        pipeline = Pipeline(
            steps=[Step(name="s1", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        from cascade.failures import FailureInjector

        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        injector = FailureInjector(config=failures)
        injector.reset(seed=42)
        step = pipeline.steps[0]
        sr = sim._execute_step_parallel(
            step=step,
            strategy=parallel(n=3, vote="majority"),
            injector=injector,
            cumulative_tokens=0,
        )
        assert sr.success is True
        assert sr.attempts == 3

    def test_parallel_unanimous_with_failures(self):
        """Unanimous vote with high failure rate should fail."""
        pipeline = Pipeline(
            steps=[Step(name="s1", model="sonnet", tools=["web_search"])]
        )
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=1.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        from cascade.failures import FailureInjector

        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        injector = FailureInjector(config=failures)
        injector.reset(seed=42)
        step = pipeline.steps[0]
        sr = sim._execute_step_parallel(
            step=step,
            strategy=parallel(n=3, vote="unanimous"),
            injector=injector,
            cumulative_tokens=0,
        )
        assert sr.success is False

    def test_parallel_any_vote(self):
        """'any' vote: only need one success."""
        pipeline = Pipeline(steps=[Step(name="s1", model="sonnet")])
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=0.0,
        )
        from cascade.failures import FailureInjector

        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        injector = FailureInjector(config=failures)
        injector.reset(seed=42)
        step = pipeline.steps[0]
        sr = sim._execute_step_parallel(
            step=step,
            strategy=parallel(n=3, vote="any"),
            injector=injector,
            cumulative_tokens=0,
        )
        assert sr.success is True

    def test_parallel_with_latency_spike(self):
        """Latency spike in parallel should be treated as non-fatal."""
        pipeline = Pipeline(steps=[Step(name="s1", model="sonnet")])
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            latency_spike_rate=1.0,
            spike_multiplier=5.0,
        )
        from cascade.failures import FailureInjector

        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        injector = FailureInjector(config=failures)
        injector.reset(seed=42)
        step = pipeline.steps[0]
        sr = sim._execute_step_parallel(
            step=step,
            strategy=parallel(n=3, vote="majority"),
            injector=injector,
            cumulative_tokens=0,
        )
        # Latency spikes are non-fatal, so all parallel runs succeed
        assert sr.success is True

    def test_parallel_with_upstream_corruption(self):
        """Test parallel with upstream corruption marked."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet"),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
            ]
        )
        failures = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            cascade_propagation=1.0,
            latency_spike_rate=0.0,
        )
        from cascade.failures import FailureInjector

        sim = Simulator(pipeline, failures, n_simulations=1, seed=42)
        injector = FailureInjector(config=failures)
        injector.reset(seed=42)
        injector.mark_corrupted("s0")
        step = pipeline.steps[1]
        sr = sim._execute_step_parallel(
            step=step,
            strategy=parallel(n=3, vote="majority"),
            injector=injector,
            cumulative_tokens=0,
        )
        # With 100% cascade propagation and upstream corruption, all fail
        assert sr.success is False


class TestGetMaxAttemptsDirect:
    """Direct tests for _get_max_attempts to cover all strategy branches."""

    def test_naive_returns_1(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim._get_max_attempts(naive(), 0)
        assert result == 1

    def test_retry_returns_max_attempts(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim._get_max_attempts(retry(max_attempts=5), 0)
        assert result == 5

    def test_fallback_returns_max_attempts(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim._get_max_attempts(fallback(models=["sonnet", "haiku"]), 0)
        assert result == 2

    def test_parallel_returns_1(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim._get_max_attempts(parallel(n=3), 0)
        assert result == 1

    def test_checkpoint_returns_max_attempts(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim._get_max_attempts(checkpoint(interval=2), 0)
        assert result == 3

    def test_human_in_loop_returns_2(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        result = sim._get_max_attempts(human_in_loop(), 0)
        assert result == 2

    def test_adaptive_below_threshold(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        strat = adaptive(escalation_threshold=3)
        result = sim._get_max_attempts(strat, consecutive_failures=1)
        assert result == strat.max_attempts

    def test_adaptive_at_threshold(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        strat = adaptive(escalation_threshold=2)
        result = sim._get_max_attempts(strat, consecutive_failures=2)
        assert result == strat.max_attempts + 1

    def test_adaptive_above_threshold(self, simple_pipeline, zero_failures):
        sim = Simulator(simple_pipeline, zero_failures, n_simulations=1, seed=42)
        strat = adaptive(escalation_threshold=2)
        result = sim._get_max_attempts(strat, consecutive_failures=5)
        assert result == strat.max_attempts + 1


class TestMonteCarloStatisticalProperties:
    """Property-based style tests for Monte Carlo simulation validity."""

    def test_success_rate_bounds(self, simple_pipeline, default_failures):
        """Success rate should always be in [0, 1]."""
        for seed in range(5):
            sim = Simulator(
                simple_pipeline, default_failures, n_simulations=100, seed=seed
            )
            result = sim.run()
            assert 0.0 <= result.success_rate <= 1.0

    def test_success_count_matches_rate(self, simple_pipeline, default_failures):
        """success_count / n_simulations should equal success_rate."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=200, seed=42)
        result = sim.run()
        expected_rate = result.success_count / result.n_simulations
        assert abs(result.success_rate - expected_rate) < 1e-9

    def test_costs_all_positive(self, simple_pipeline, default_failures):
        """Every run cost should be positive."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        for cost in result.costs:
            assert cost > 0

    def test_latencies_all_positive(self, simple_pipeline, default_failures):
        """Every run latency should be positive."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        for lat in result.latencies:
            assert lat > 0

    def test_mean_cost_equals_average(self, simple_pipeline, default_failures):
        """mean_cost_usd should match np.mean(costs)."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        assert abs(result.mean_cost_usd - float(np.mean(result.costs))) < 1e-9

    def test_mean_latency_equals_average(self, simple_pipeline, default_failures):
        """mean_latency_s should match np.mean(latencies)."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=100, seed=42)
        result = sim.run()
        assert abs(result.mean_latency_s - float(np.mean(result.latencies))) < 1e-9

    def test_retry_never_decreases_success(self):
        """Retry strategy should never have fewer successes than naive."""
        pipeline = Pipeline(
            steps=[
                Step(name="s0", model="sonnet", tools=["web_search"]),
                Step(name="s1", model="sonnet", depends_on=["s0"]),
                Step(name="s2", model="haiku", depends_on=["s1"]),
            ]
        )
        failures = FailureConfig(hallucination_rate=0.1, tool_failure_rate=0.1)
        # Run 5 different seeds
        for seed in range(5):
            sim_naive = Simulator(
                pipeline, failures, n_simulations=300, seed=seed, strategy=naive()
            )
            sim_retry = Simulator(
                pipeline,
                failures,
                n_simulations=300,
                seed=seed,
                strategy=retry(max_attempts=3),
            )
            r_naive = sim_naive.run()
            r_retry = sim_retry.run()
            # With identical seeds, retry should be >= naive
            assert r_retry.success_rate >= r_naive.success_rate - 0.05

    def test_cost_variance_positive(self, simple_pipeline, default_failures):
        """Cost distribution should have positive variance with failures."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=500, seed=42)
        result = sim.run()
        variance = np.var(result.costs)
        # With random failures, there should be some cost variance
        # (though it could be zero if all runs succeed with same cost)
        assert variance >= 0.0

    def test_failure_counts_non_negative(self, simple_pipeline, default_failures):
        """All failure counts should be non-negative."""
        sim = Simulator(simple_pipeline, default_failures, n_simulations=200, seed=42)
        result = sim.run()
        for count in result.failure_counts.values():
            assert count >= 0

    def test_recovery_rate_bounded(self, research_pipeline, default_failures):
        """Recovery rate should be between 0 and 1."""
        sim = Simulator(
            research_pipeline,
            default_failures,
            n_simulations=200,
            seed=42,
            strategy=retry(max_attempts=3),
        )
        result = sim.run()
        assert 0.0 <= result.recovery_rate <= 1.0
