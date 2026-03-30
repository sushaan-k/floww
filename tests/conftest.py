"""Shared fixtures for cascade tests."""

from __future__ import annotations

import pytest

from cascade.failures import FailureConfig
from cascade.pipeline import Pipeline, Step
from cascade.strategies import (
    adaptive,
    checkpoint,
    fallback,
    naive,
    parallel,
    retry,
)


@pytest.fixture
def simple_pipeline() -> Pipeline:
    """A 3-step linear pipeline for basic tests."""
    return Pipeline(
        name="simple",
        steps=[
            Step(name="step_a", model="sonnet", tools=["web_search"]),
            Step(name="step_b", model="sonnet", depends_on=["step_a"]),
            Step(name="step_c", model="haiku", depends_on=["step_b"]),
        ],
    )


@pytest.fixture
def research_pipeline() -> Pipeline:
    """A 6-step pipeline mirroring the spec example."""
    return Pipeline(
        name="research",
        steps=[
            Step(name="research", model="sonnet", tools=["web_search", "read_file"]),
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
        ],
    )


@pytest.fixture
def default_failures() -> FailureConfig:
    """Default failure configuration matching the spec."""
    return FailureConfig(
        hallucination_rate=0.05,
        refusal_rate=0.02,
        tool_failure_rate=0.03,
        cascade_propagation=0.8,
    )


@pytest.fixture
def zero_failures() -> FailureConfig:
    """A failure config with all rates set to zero."""
    return FailureConfig(
        hallucination_rate=0.0,
        refusal_rate=0.0,
        tool_failure_rate=0.0,
        cascade_propagation=0.0,
        latency_spike_rate=0.0,
    )


@pytest.fixture
def high_failure_config() -> FailureConfig:
    """An aggressive failure config for stress testing."""
    return FailureConfig(
        hallucination_rate=0.30,
        refusal_rate=0.10,
        tool_failure_rate=0.15,
        cascade_propagation=0.9,
        latency_spike_rate=0.05,
    )


@pytest.fixture
def all_strategies():
    """Return a list of all built-in strategies."""
    return [
        naive(),
        retry(max_attempts=3),
        fallback(models=["sonnet", "haiku"]),
        parallel(n=3, vote="majority"),
        checkpoint(interval=2),
        adaptive(escalation_threshold=2, escalation_strategy="parallel"),
    ]
