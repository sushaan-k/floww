"""Tests for cascade.strategies module."""

from __future__ import annotations

import pytest

from cascade.strategies import (
    ResilienceStrategy,
    StrategyType,
    adaptive,
    checkpoint,
    fallback,
    human_in_loop,
    naive,
    parallel,
    retry,
)


class TestStrategyFactories:
    """Tests for the strategy factory functions."""

    def test_naive(self):
        s = naive()
        assert s.strategy_type == StrategyType.NAIVE
        assert s.max_attempts == 1
        assert s.cost_multiplier == 1.0
        assert s.display_name == "Naive"

    def test_retry_default(self):
        s = retry()
        assert s.strategy_type == StrategyType.RETRY
        assert s.max_attempts == 3
        assert s.display_name == "Retry(3)"

    def test_retry_custom(self):
        s = retry(max_attempts=5)
        assert s.max_attempts == 5
        assert "5" in s.display_name

    def test_fallback_default(self):
        s = fallback()
        assert s.strategy_type == StrategyType.FALLBACK
        assert s.fallback_models == ["sonnet", "haiku"]
        assert s.max_attempts == 2

    def test_fallback_custom(self):
        s = fallback(models=["opus", "sonnet", "haiku"])
        assert s.fallback_models == ["opus", "sonnet", "haiku"]
        assert s.max_attempts == 3

    def test_parallel_default(self):
        s = parallel()
        assert s.strategy_type == StrategyType.PARALLEL
        assert s.parallel_n == 3
        assert s.vote_method == "majority"
        assert s.cost_multiplier == 3.0

    def test_parallel_custom_vote(self):
        s = parallel(n=5, vote="unanimous")
        assert s.parallel_n == 5
        assert s.vote_method == "unanimous"

    def test_checkpoint_default(self):
        s = checkpoint()
        assert s.strategy_type == StrategyType.CHECKPOINT
        assert s.checkpoint_interval == 5
        assert s.max_attempts == 3

    def test_checkpoint_custom(self):
        s = checkpoint(interval=2)
        assert s.checkpoint_interval == 2
        assert "2" in s.display_name

    def test_human_in_loop_default(self):
        s = human_in_loop()
        assert s.strategy_type == StrategyType.HUMAN_IN_LOOP
        assert s.human_at_steps == [5, 10, 15]
        assert s.human_accuracy == 0.95

    def test_human_in_loop_custom(self):
        s = human_in_loop(at_steps=[1, 3], accuracy=0.99)
        assert s.human_at_steps == [1, 3]
        assert s.human_accuracy == 0.99

    def test_adaptive_default(self):
        s = adaptive()
        assert s.strategy_type == StrategyType.ADAPTIVE
        assert s.escalation_threshold == 2
        assert s.escalation_strategy == StrategyType.PARALLEL

    def test_adaptive_custom(self):
        s = adaptive(escalation_threshold=3, escalation_strategy="retry")
        assert s.escalation_threshold == 3
        assert s.escalation_strategy == StrategyType.RETRY


class TestResilienceStrategy:
    """Tests for ResilienceStrategy validation."""

    def test_display_name_auto(self):
        s = ResilienceStrategy(strategy_type=StrategyType.NAIVE)
        assert s.display_name == "Naive"

    def test_display_name_override(self):
        s = ResilienceStrategy(
            strategy_type=StrategyType.NAIVE,
            display_name="Custom Name",
        )
        assert s.display_name == "Custom Name"

    def test_max_attempts_validation(self):
        with pytest.raises(ValueError):
            ResilienceStrategy(
                strategy_type=StrategyType.RETRY,
                max_attempts=0,
            )

    def test_human_accuracy_bounds(self):
        with pytest.raises(ValueError):
            ResilienceStrategy(
                strategy_type=StrategyType.HUMAN_IN_LOOP,
                human_accuracy=1.5,
            )

    def test_serialization_roundtrip(self):
        s = retry(max_attempts=3)
        data = s.model_dump()
        rebuilt = ResilienceStrategy(**data)
        assert rebuilt.strategy_type == s.strategy_type
        assert rebuilt.max_attempts == s.max_attempts
        assert rebuilt.display_name == s.display_name


class TestStrategyType:
    """Tests for the StrategyType enum."""

    def test_all_types_present(self):
        expected = {
            "naive",
            "retry",
            "fallback",
            "parallel",
            "checkpoint",
            "human_in_loop",
            "adaptive",
        }
        actual = {t.value for t in StrategyType}
        assert actual == expected


class TestHumanInLoopEdgeCases:
    """Edge cases for human_in_loop strategy."""

    def test_accuracy_zero(self):
        s = human_in_loop(accuracy=0.0)
        assert s.human_accuracy == 0.0

    def test_accuracy_one(self):
        s = human_in_loop(accuracy=1.0)
        assert s.human_accuracy == 1.0

    def test_empty_steps_defaults_to_preset(self):
        """Empty list is falsy, so factory defaults to [5, 10, 15]."""
        s = human_in_loop(at_steps=[])
        assert s.human_at_steps == [5, 10, 15]

    def test_single_step(self):
        s = human_in_loop(at_steps=[0])
        assert s.human_at_steps == [0]

    def test_display_name_format(self):
        s = human_in_loop(at_steps=[2, 5])
        assert "2" in s.display_name
        assert "5" in s.display_name
        assert "HumanInLoop" in s.display_name


class TestAdaptiveEdgeCases:
    """Edge cases for adaptive strategy."""

    def test_threshold_one(self):
        s = adaptive(escalation_threshold=1)
        assert s.escalation_threshold == 1

    def test_escalation_to_retry(self):
        s = adaptive(escalation_strategy="retry")
        assert s.escalation_strategy == StrategyType.RETRY

    def test_escalation_to_fallback(self):
        s = adaptive(escalation_strategy="fallback")
        assert s.escalation_strategy == StrategyType.FALLBACK

    def test_escalation_to_checkpoint(self):
        s = adaptive(escalation_strategy="checkpoint")
        assert s.escalation_strategy == StrategyType.CHECKPOINT

    def test_display_name_includes_threshold(self):
        s = adaptive(escalation_threshold=5, escalation_strategy="parallel")
        assert "5" in s.display_name
        assert "parallel" in s.display_name


class TestParallelEdgeCases:
    """Edge cases for parallel strategy."""

    def test_parallel_n_one(self):
        """Parallel with n=1 is effectively no redundancy."""
        s = parallel(n=1)
        assert s.parallel_n == 1
        assert s.cost_multiplier == 1.0

    def test_parallel_any_vote(self):
        s = parallel(n=5, vote="any")
        assert s.vote_method == "any"

    def test_parallel_unanimous_vote(self):
        s = parallel(n=2, vote="unanimous")
        assert s.vote_method == "unanimous"


class TestFallbackEdgeCases:
    """Edge cases for fallback strategy."""

    def test_single_model_fallback(self):
        s = fallback(models=["opus"])
        assert s.fallback_models == ["opus"]
        assert s.max_attempts == 1

    def test_many_model_fallback(self):
        models = ["opus", "sonnet", "haiku"]
        s = fallback(models=models)
        assert s.max_attempts == 3
        assert s.fallback_models == models


class TestRetryEdgeCases:
    """Edge cases for retry strategy."""

    def test_retry_max_attempts_one(self):
        """Retry with 1 attempt is effectively naive."""
        s = retry(max_attempts=1)
        assert s.max_attempts == 1
        assert s.cost_multiplier == 1.0

    def test_retry_large_attempts(self):
        s = retry(max_attempts=10)
        assert s.max_attempts == 10
        assert s.cost_multiplier > 1.0


class TestCheckpointEdgeCases:
    """Edge cases for checkpoint strategy."""

    def test_checkpoint_interval_one(self):
        """Checkpoint after every single step."""
        s = checkpoint(interval=1)
        assert s.checkpoint_interval == 1

    def test_checkpoint_large_interval(self):
        """Checkpoint interval larger than most pipelines."""
        s = checkpoint(interval=100)
        assert s.checkpoint_interval == 100
