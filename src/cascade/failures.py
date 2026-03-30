"""Failure mode models for agent reliability simulation.

Each failure type is modeled as a stochastic event with configurable
probability. The FailureInjector draws from these distributions during
simulation to determine which (if any) failure occurs at each step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FailureType(StrEnum):
    """Enumeration of supported failure modes."""

    HALLUCINATION = "hallucination"
    REFUSAL = "refusal"
    TOOL_FAILURE = "tool_failure"
    CONTEXT_OVERFLOW = "context_overflow"
    CASCADING_CORRUPTION = "cascading_corruption"
    LATENCY_SPIKE = "latency_spike"


class HallucinationSubtype(StrEnum):
    """Subtypes of hallucination failures."""

    WRONG_TOOL_ARGS = "wrong_tool_args"
    FABRICATED_DATA = "fabricated_data"
    INCORRECT_REASONING = "incorrect_reasoning"
    FORMAT_ERROR = "format_error"


class FailureConfig(BaseModel):
    """Configuration for failure injection probabilities and parameters.

    Attributes:
        hallucination_rate: Probability a step produces incorrect output.
        refusal_rate: Probability of a false-positive safety refusal.
        tool_failure_rate: Probability of an external tool/API failure.
        tool_timeout_ms: Timeout threshold for tool calls in milliseconds.
        context_overflow_at: Token count at which context overflow occurs.
        overflow_behavior: How the agent handles overflow ("truncate_early"
            or "summarize").
        cascade_propagation: Probability that a hallucinated output corrupts
            the next dependent step.
        latency_spike_rate: Probability of a latency spike on any step.
        spike_multiplier: Factor by which latency increases during a spike.
    """

    hallucination_rate: float = Field(
        default=0.05, ge=0.0, le=1.0, description="P(hallucination) per step"
    )
    refusal_rate: float = Field(
        default=0.02, ge=0.0, le=1.0, description="P(false refusal) per step"
    )
    tool_failure_rate: float = Field(
        default=0.03, ge=0.0, le=1.0, description="P(tool failure) per step"
    )
    tool_timeout_ms: int = Field(
        default=5000, gt=0, description="Tool timeout threshold in ms"
    )
    context_overflow_at: int = Field(
        default=128_000,
        gt=0,
        description="Token count triggering context overflow",
    )
    overflow_behavior: str = Field(
        default="truncate_early",
        description="How overflow is handled: truncate_early or summarize",
    )
    cascade_propagation: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="P(bad output corrupts next step)",
    )
    latency_spike_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="P(latency spike) per step",
    )
    spike_multiplier: float = Field(
        default=10.0, gt=1.0, description="Latency multiplier during spikes"
    )


@dataclass
class FailureEvent:
    """Record of a failure that occurred during simulation.

    Attributes:
        step_name: The step where the failure occurred.
        failure_type: The category of failure.
        details: Human-readable description of what happened.
        recoverable: Whether a resilience strategy could retry this.
        latency_added_s: Extra latency caused by this failure.
        hallucination_subtype: Subtype if failure_type is HALLUCINATION.
    """

    step_name: str
    failure_type: FailureType
    details: str = ""
    recoverable: bool = True
    latency_added_s: float = 0.0
    hallucination_subtype: HallucinationSubtype | None = None


@dataclass
class FailureInjector:
    """Stochastic failure injector for simulation runs.

    Maintains a random number generator for reproducibility and tracks
    which steps have been corrupted by upstream hallucinations.

    Attributes:
        config: The failure configuration controlling probabilities.
        rng: Numpy random generator for reproducible draws.
        corrupted_steps: Set of step names whose inputs are tainted.
    """

    config: FailureConfig
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    corrupted_steps: set[str] = field(default_factory=set)

    def reset(self, seed: int | None = None) -> None:
        """Reset state for a new simulation run.

        Args:
            seed: Optional seed for the random generator.
        """
        self.corrupted_steps.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def inject(
        self,
        step_name: str,
        model: str,
        tools: list[str],
        cumulative_tokens: int,
        upstream_corrupted: bool = False,
    ) -> FailureEvent | None:
        """Attempt to inject a failure at the given step.

        Failures are checked in priority order. Only the first triggered
        failure is returned (an agent step fails in one way at a time).

        Args:
            step_name: Name of the step being executed.
            model: Model identifier for this step.
            tools: Tools available at this step.
            cumulative_tokens: Total tokens consumed so far in the pipeline.
            upstream_corrupted: Whether an upstream step produced bad output.

        Returns:
            A FailureEvent if a failure was injected, otherwise None.
        """
        # 1. Context overflow
        if cumulative_tokens >= self.config.context_overflow_at:
            return FailureEvent(
                step_name=step_name,
                failure_type=FailureType.CONTEXT_OVERFLOW,
                details=(
                    f"Context overflow at {cumulative_tokens} tokens "
                    f"(limit: {self.config.context_overflow_at})"
                ),
                recoverable=False,
            )

        # 2. Cascading corruption from upstream hallucination
        if upstream_corrupted and self.rng.random() < self.config.cascade_propagation:
            self.corrupted_steps.add(step_name)
            return FailureEvent(
                step_name=step_name,
                failure_type=FailureType.CASCADING_CORRUPTION,
                details=(
                    f"Step {step_name!r} received corrupted input "
                    f"from upstream hallucination"
                ),
                recoverable=False,
            )

        # 3. Tool failure (only if the step uses tools)
        if tools and self.rng.random() < self.config.tool_failure_rate:
            tool_idx = int(self.rng.integers(0, len(tools)))
            failed_tool = tools[tool_idx]
            return FailureEvent(
                step_name=step_name,
                failure_type=FailureType.TOOL_FAILURE,
                details=f"Tool {failed_tool!r} failed (API error/timeout)",
                recoverable=True,
                latency_added_s=self.config.tool_timeout_ms / 1000,
            )

        # 4. Refusal
        if self.rng.random() < self.config.refusal_rate:
            return FailureEvent(
                step_name=step_name,
                failure_type=FailureType.REFUSAL,
                details=(
                    f"Step {step_name!r} refused by safety filter (false positive)"
                ),
                recoverable=True,
            )

        # 5. Hallucination
        if self.rng.random() < self.config.hallucination_rate:
            subtypes = list(HallucinationSubtype)
            idx = int(self.rng.integers(0, len(subtypes)))
            subtype = subtypes[idx]
            self.corrupted_steps.add(step_name)
            return FailureEvent(
                step_name=step_name,
                failure_type=FailureType.HALLUCINATION,
                details=(f"Step {step_name!r} hallucinated ({subtype.value})"),
                recoverable=True,
                hallucination_subtype=subtype,
            )

        # 6. Latency spike (non-fatal but adds latency)
        if self.rng.random() < self.config.latency_spike_rate:
            return FailureEvent(
                step_name=step_name,
                failure_type=FailureType.LATENCY_SPIKE,
                details=f"Latency spike on step {step_name!r}",
                recoverable=True,
                latency_added_s=0.0,  # Handled by simulator via multiplier
            )

        return None

    def is_corrupted(self, step_name: str) -> bool:
        """Check whether a step's output is marked as corrupted."""
        return step_name in self.corrupted_steps

    def mark_corrupted(self, step_name: str) -> None:
        """Explicitly mark a step's output as corrupted."""
        self.corrupted_steps.add(step_name)

    def clear_corruption(self, step_name: str) -> None:
        """Remove corruption mark from a step (e.g. after successful retry)."""
        self.corrupted_steps.discard(step_name)
