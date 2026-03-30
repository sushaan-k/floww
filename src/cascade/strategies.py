"""Resilience strategies for agent pipelines.

Strategies wrap the simulation engine's step-execution logic to add
retry, fallback, redundancy, checkpointing, and other resilience patterns.
Each strategy is a Pydantic model so it can be serialized and compared.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StrategyType(StrEnum):
    """Enumeration of built-in resilience strategy types."""

    NAIVE = "naive"
    RETRY = "retry"
    FALLBACK = "fallback"
    PARALLEL = "parallel"
    CHECKPOINT = "checkpoint"
    HUMAN_IN_LOOP = "human_in_loop"
    ADAPTIVE = "adaptive"


class ResilienceStrategy(BaseModel):
    """Base configuration for a resilience strategy.

    Every strategy carries a type tag, a display name, and parameters
    that the simulator interprets during execution.

    Attributes:
        strategy_type: Which strategy pattern to apply.
        display_name: Human-readable label for reports.
        max_attempts: Maximum retries per step (used by retry, adaptive).
        fallback_models: Ordered list of models to try on failure.
        parallel_n: Number of parallel agents for redundancy.
        vote_method: How parallel results are reconciled.
        checkpoint_interval: Steps between checkpoints.
        human_at_steps: Step indices where a human verifies output.
        human_accuracy: Probability that human catches an error.
        escalation_threshold: Failures before strategy escalation.
        escalation_strategy: Strategy to escalate to after threshold.
        cost_multiplier: Estimated cost multiplier versus naive execution.
    """

    strategy_type: StrategyType
    display_name: str = ""
    max_attempts: int = Field(default=1, ge=1)
    fallback_models: list[str] = Field(default_factory=list)
    parallel_n: int = Field(default=1, ge=1)
    vote_method: Literal["majority", "unanimous", "any"] = "majority"
    checkpoint_interval: int = Field(default=0, ge=0)
    human_at_steps: list[int] = Field(default_factory=list)
    human_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    escalation_threshold: int = Field(default=2, ge=1)
    escalation_strategy: StrategyType = StrategyType.PARALLEL
    cost_multiplier: float = Field(default=1.0, ge=0.0)

    def model_post_init(self, __context: object) -> None:
        """Set display_name from strategy_type if not provided."""
        if not self.display_name:
            self.display_name = _default_display_name(self)


def _default_display_name(strategy: ResilienceStrategy) -> str:
    """Generate a human-readable display name for the strategy."""
    st = strategy.strategy_type
    if st == StrategyType.NAIVE:
        return "Naive"
    if st == StrategyType.RETRY:
        return f"Retry({strategy.max_attempts})"
    if st == StrategyType.FALLBACK:
        models = ", ".join(strategy.fallback_models)
        return f"Fallback({models})"
    if st == StrategyType.PARALLEL:
        return f"Parallel({strategy.parallel_n})"
    if st == StrategyType.CHECKPOINT:
        return f"Checkpoint({strategy.checkpoint_interval})"
    if st == StrategyType.HUMAN_IN_LOOP:
        steps = ", ".join(str(s) for s in strategy.human_at_steps)
        return f"HumanInLoop([{steps}])"
    if st == StrategyType.ADAPTIVE:
        return (
            f"Adaptive(threshold={strategy.escalation_threshold}, "
            f"escalate={strategy.escalation_strategy.value})"
        )
    return st.value  # pragma: no cover


# ---------------------------------------------------------------
# Factory functions matching the API surface in the spec
# ---------------------------------------------------------------


def naive() -> ResilienceStrategy:
    """Create a naive (no-retry, fail-fast) strategy."""
    return ResilienceStrategy(
        strategy_type=StrategyType.NAIVE,
        cost_multiplier=1.0,
    )


def retry(max_attempts: int = 3) -> ResilienceStrategy:
    """Create a simple retry strategy.

    Args:
        max_attempts: Maximum number of attempts per step.
    """
    return ResilienceStrategy(
        strategy_type=StrategyType.RETRY,
        max_attempts=max_attempts,
        cost_multiplier=1.0 + (max_attempts - 1) * 0.3,
    )


def fallback(models: list[str] | None = None) -> ResilienceStrategy:
    """Create a model-fallback strategy.

    Args:
        models: Ordered list of models to try (e.g. ["sonnet", "haiku"]).
    """
    models = models or ["sonnet", "haiku"]
    return ResilienceStrategy(
        strategy_type=StrategyType.FALLBACK,
        fallback_models=models,
        max_attempts=len(models),
        cost_multiplier=1.2,
    )


def parallel(
    n: int = 3,
    vote: Literal["majority", "unanimous", "any"] = "majority",
) -> ResilienceStrategy:
    """Create a parallel-redundancy strategy with voting.

    Args:
        n: Number of parallel agents.
        vote: Voting method to reconcile results.
    """
    return ResilienceStrategy(
        strategy_type=StrategyType.PARALLEL,
        parallel_n=n,
        vote_method=vote,
        cost_multiplier=float(n),
    )


def checkpoint(interval: int = 5) -> ResilienceStrategy:
    """Create a checkpoint-and-rollback strategy.

    Args:
        interval: Number of steps between checkpoints.
    """
    return ResilienceStrategy(
        strategy_type=StrategyType.CHECKPOINT,
        checkpoint_interval=interval,
        max_attempts=3,
        cost_multiplier=1.5,
    )


def human_in_loop(
    at_steps: list[int] | None = None,
    accuracy: float = 0.95,
) -> ResilienceStrategy:
    """Create a human-in-the-loop verification strategy.

    Args:
        at_steps: Step indices where human verification occurs.
        accuracy: Probability human catches an error (0-1).
    """
    at_steps = at_steps or [5, 10, 15]
    return ResilienceStrategy(
        strategy_type=StrategyType.HUMAN_IN_LOOP,
        human_at_steps=at_steps,
        human_accuracy=accuracy,
        cost_multiplier=1.3,
    )


def adaptive(
    escalation_threshold: int = 2,
    escalation_strategy: str = "parallel",
) -> ResilienceStrategy:
    """Create an adaptive strategy that escalates after repeated failures.

    Args:
        escalation_threshold: Number of failures before escalation.
        escalation_strategy: Strategy type to escalate to.
    """
    esc_type = StrategyType(escalation_strategy)
    return ResilienceStrategy(
        strategy_type=StrategyType.ADAPTIVE,
        escalation_threshold=escalation_threshold,
        escalation_strategy=esc_type,
        max_attempts=3,
        cost_multiplier=1.4,
    )
