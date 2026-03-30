"""Tests for cascade.failures module."""

from __future__ import annotations

import pytest

from cascade.failures import (
    FailureConfig,
    FailureEvent,
    FailureInjector,
    FailureType,
    HallucinationSubtype,
)


class TestFailureConfig:
    """Tests for FailureConfig validation."""

    def test_defaults(self):
        fc = FailureConfig()
        assert fc.hallucination_rate == 0.05
        assert fc.refusal_rate == 0.02
        assert fc.tool_failure_rate == 0.03
        assert fc.cascade_propagation == 0.8

    def test_rates_bounded(self):
        with pytest.raises(ValueError):
            FailureConfig(hallucination_rate=1.5)
        with pytest.raises(ValueError):
            FailureConfig(refusal_rate=-0.1)

    def test_zero_config(self, zero_failures):
        assert zero_failures.hallucination_rate == 0.0
        assert zero_failures.tool_failure_rate == 0.0


class TestFailureEvent:
    """Tests for the FailureEvent dataclass."""

    def test_creation(self):
        fe = FailureEvent(
            step_name="step_a",
            failure_type=FailureType.HALLUCINATION,
            details="test",
            recoverable=True,
        )
        assert fe.step_name == "step_a"
        assert fe.failure_type == FailureType.HALLUCINATION
        assert fe.recoverable is True

    def test_hallucination_subtype(self):
        fe = FailureEvent(
            step_name="s",
            failure_type=FailureType.HALLUCINATION,
            hallucination_subtype=HallucinationSubtype.FABRICATED_DATA,
        )
        assert fe.hallucination_subtype == HallucinationSubtype.FABRICATED_DATA


class TestFailureInjector:
    """Tests for the FailureInjector."""

    def test_no_failures_with_zero_rates(self, zero_failures):
        injector = FailureInjector(config=zero_failures)
        injector.reset(seed=42)
        for _ in range(100):
            result = injector.inject(
                step_name="s",
                model="sonnet",
                tools=["web_search"],
                cumulative_tokens=0,
            )
            assert result is None

    def test_context_overflow_deterministic(self):
        fc = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            context_overflow_at=1000,
        )
        injector = FailureInjector(config=fc)
        injector.reset(seed=1)
        result = injector.inject(
            step_name="s",
            model="sonnet",
            tools=[],
            cumulative_tokens=1500,
        )
        assert result is not None
        assert result.failure_type == FailureType.CONTEXT_OVERFLOW
        assert not result.recoverable

    def test_cascading_corruption(self):
        fc = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            cascade_propagation=1.0,
        )
        injector = FailureInjector(config=fc)
        injector.reset(seed=1)
        result = injector.inject(
            step_name="s",
            model="sonnet",
            tools=[],
            cumulative_tokens=0,
            upstream_corrupted=True,
        )
        assert result is not None
        assert result.failure_type == FailureType.CASCADING_CORRUPTION

    def test_cascading_corruption_zero_propagation(self):
        fc = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=0.0,
            cascade_propagation=0.0,
        )
        injector = FailureInjector(config=fc)
        injector.reset(seed=1)
        result = injector.inject(
            step_name="s",
            model="sonnet",
            tools=[],
            cumulative_tokens=0,
            upstream_corrupted=True,
        )
        assert result is None

    def test_reset_clears_corruption(self):
        fc = FailureConfig()
        injector = FailureInjector(config=fc)
        injector.mark_corrupted("s1")
        assert injector.is_corrupted("s1")
        injector.reset(seed=99)
        assert not injector.is_corrupted("s1")

    def test_mark_and_clear_corruption(self):
        fc = FailureConfig()
        injector = FailureInjector(config=fc)
        injector.mark_corrupted("s1")
        assert injector.is_corrupted("s1")
        injector.clear_corruption("s1")
        assert not injector.is_corrupted("s1")

    def test_reproducibility_with_seed(self):
        fc = FailureConfig(hallucination_rate=0.5)
        results_a = []
        results_b = []
        for seed, results in [(42, results_a), (42, results_b)]:
            injector = FailureInjector(config=fc)
            injector.reset(seed=seed)
            for i in range(20):
                r = injector.inject(
                    step_name=f"s{i}",
                    model="sonnet",
                    tools=["t"],
                    cumulative_tokens=0,
                )
                results.append(r is not None)
        assert results_a == results_b

    def test_tool_failure_requires_tools(self):
        """Tool failures should only happen when the step has tools."""
        fc = FailureConfig(
            hallucination_rate=0.0,
            refusal_rate=0.0,
            tool_failure_rate=1.0,  # 100% tool failure
            latency_spike_rate=0.0,
        )
        injector = FailureInjector(config=fc)
        injector.reset(seed=1)
        # No tools => no tool failure
        result = injector.inject(
            step_name="s",
            model="sonnet",
            tools=[],
            cumulative_tokens=0,
        )
        # Should not get a tool failure (might get refusal or hallucination)
        if result is not None:
            assert result.failure_type != FailureType.TOOL_FAILURE

    def test_failure_types_enum(self):
        assert FailureType.HALLUCINATION.value == "hallucination"
        assert FailureType.LATENCY_SPIKE.value == "latency_spike"
        assert len(FailureType) == 6

    def test_hallucination_subtypes_enum(self):
        assert len(HallucinationSubtype) == 4
        assert HallucinationSubtype.WRONG_TOOL_ARGS.value == "wrong_tool_args"
