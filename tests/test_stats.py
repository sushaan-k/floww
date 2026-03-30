"""Tests for cascade.stats module."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.stats import (
    mean_ci,
    pareto_frontier,
    proportion_ci,
    summarize,
)


class TestSummarize:
    """Tests for the summarize function."""

    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = summarize(values)
        assert s.n == 5
        assert abs(s.mean - 3.0) < 1e-9
        assert abs(s.median - 3.0) < 1e-9
        assert s.min == 1.0
        assert s.max == 5.0

    def test_single_value(self):
        s = summarize([42.0])
        assert s.mean == 42.0
        assert s.std == 0.0
        assert s.n == 1

    def test_numpy_array(self):
        arr = np.array([10.0, 20.0, 30.0])
        s = summarize(arr)
        assert abs(s.mean - 20.0) < 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            summarize([])

    def test_percentiles(self):
        values = list(range(1, 101))  # 1..100
        s = summarize(values)
        assert abs(s.p5 - 5.95) < 1.0
        assert abs(s.p95 - 95.05) < 1.0


class TestProportionCI:
    """Tests for Wilson score proportion CI."""

    def test_basic(self):
        ci = proportion_ci(750, 1000)
        assert abs(ci.point - 0.75) < 1e-9
        assert ci.lower < 0.75
        assert ci.upper > 0.75
        assert ci.confidence == 0.95

    def test_all_success(self):
        ci = proportion_ci(100, 100)
        assert ci.point == 1.0
        assert ci.lower < 1.0
        assert ci.upper == 1.0

    def test_all_failure(self):
        ci = proportion_ci(0, 100)
        assert ci.point == 0.0
        assert ci.lower < 1e-6  # Essentially zero (floating point)
        assert ci.upper > 0.0

    def test_zero_trials_raises(self):
        with pytest.raises(ValueError, match="zero"):
            proportion_ci(0, 0)

    def test_successes_exceed_total_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            proportion_ci(10, 5)

    def test_custom_confidence(self):
        ci = proportion_ci(500, 1000, confidence=0.99)
        assert ci.confidence == 0.99
        # 99% CI should be wider than default 95%
        ci95 = proportion_ci(500, 1000, confidence=0.95)
        assert (ci.upper - ci.lower) > (ci95.upper - ci95.lower)


class TestMeanCI:
    """Tests for t-distribution mean CI."""

    def test_basic(self):
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        ci = mean_ci(values)
        assert ci.lower < ci.point < ci.upper
        assert ci.confidence == 0.95

    def test_tight_with_many_samples(self):
        rng = np.random.default_rng(42)
        values = rng.normal(100, 1, size=10000).tolist()
        ci = mean_ci(values)
        assert ci.upper - ci.lower < 0.1  # Very tight interval

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            mean_ci([5.0])


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_basic(self):
        costs = [1.0, 2.0, 3.0, 4.0]
        success = [0.7, 0.85, 0.9, 0.95]
        frontier = pareto_frontier(costs, success)
        # All points are on the frontier (monotonically increasing)
        assert frontier == [0, 1, 2, 3]

    def test_dominated_points(self):
        costs = [1.0, 2.0, 1.5, 3.0]
        success = [0.8, 0.75, 0.9, 0.95]
        frontier = pareto_frontier(costs, success)
        # Point at index 1 (cost=2.0, success=0.75) is dominated by index 2
        assert 1 not in frontier

    def test_empty(self):
        assert pareto_frontier([], []) == []

    def test_single_point(self):
        assert pareto_frontier([1.0], [0.5]) == [0]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            pareto_frontier([1.0, 2.0], [0.5])

    def test_all_same_cost(self):
        """All strategies with the same cost -- highest success wins."""
        costs = [1.0, 1.0, 1.0]
        success = [0.7, 0.9, 0.8]
        frontier = pareto_frontier(costs, success)
        # Among equal costs, only highest success should be on frontier
        assert 1 in frontier

    def test_all_same_success(self):
        """All strategies with the same success rate -- cheapest wins."""
        costs = [3.0, 1.0, 2.0]
        success = [0.8, 0.8, 0.8]
        frontier = pareto_frontier(costs, success)
        assert frontier == [1]  # index 1 is cheapest


class TestSummarizeExtremeValues:
    """Edge cases for summarize with extreme data."""

    def test_all_identical(self):
        values = [42.0] * 100
        s = summarize(values)
        assert s.mean == 42.0
        assert s.std == 0.0
        assert s.p5 == 42.0
        assert s.p95 == 42.0
        assert s.min == 42.0
        assert s.max == 42.0

    def test_very_large_values(self):
        values = [1e15, 2e15, 3e15]
        s = summarize(values)
        assert s.mean == 2e15
        assert s.min == 1e15
        assert s.max == 3e15

    def test_very_small_values(self):
        values = [1e-15, 2e-15, 3e-15]
        s = summarize(values)
        assert abs(s.mean - 2e-15) < 1e-25
        assert s.min == 1e-15
        assert s.max == 3e-15

    def test_negative_values(self):
        values = [-3.0, -1.0, -2.0]
        s = summarize(values)
        assert abs(s.mean - (-2.0)) < 1e-9
        assert s.min == -3.0
        assert s.max == -1.0

    def test_mixed_positive_negative(self):
        values = [-10.0, 0.0, 10.0]
        s = summarize(values)
        assert abs(s.mean - 0.0) < 1e-9
        assert s.median == 0.0

    def test_two_values(self):
        values = [1.0, 3.0]
        s = summarize(values)
        assert s.n == 2
        assert abs(s.mean - 2.0) < 1e-9
        assert s.std > 0


class TestMeanCIEdgeCases:
    """Edge cases for mean_ci."""

    def test_two_values_minimum(self):
        ci = mean_ci([1.0, 3.0])
        assert ci.point == 2.0
        assert ci.lower < 2.0
        assert ci.upper > 2.0

    def test_identical_values(self):
        ci = mean_ci([5.0, 5.0, 5.0, 5.0])
        assert ci.point == 5.0
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_custom_confidence_99(self):
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        ci99 = mean_ci(values, confidence=0.99)
        ci95 = mean_ci(values, confidence=0.95)
        assert ci99.confidence == 0.99
        # 99% CI should be wider
        assert (ci99.upper - ci99.lower) > (ci95.upper - ci95.lower)

    def test_large_sample_tight_ci(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=5000).tolist()
        ci = mean_ci(values)
        assert ci.upper - ci.lower < 0.1


class TestProportionCIEdgeCases:
    """Edge cases for proportion_ci."""

    def test_single_trial_success(self):
        ci = proportion_ci(1, 1)
        assert ci.point == 1.0
        assert ci.lower > 0.0
        assert ci.upper == 1.0

    def test_single_trial_failure(self):
        ci = proportion_ci(0, 1)
        assert ci.point == 0.0
        assert ci.lower == 0.0
        assert ci.upper < 1.0

    def test_very_large_sample(self):
        ci = proportion_ci(50000, 100000)
        assert abs(ci.point - 0.5) < 1e-9
        # CI should be very tight
        assert ci.upper - ci.lower < 0.01

    def test_near_zero_proportion(self):
        ci = proportion_ci(1, 10000)
        assert ci.point == 0.0001
        assert ci.lower >= 0.0
        assert ci.upper > ci.point

    def test_near_one_proportion(self):
        ci = proportion_ci(9999, 10000)
        assert ci.upper <= 1.0
        assert ci.lower < ci.point
