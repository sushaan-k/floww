"""Statistical utilities for simulation analysis.

Provides confidence intervals, distribution summaries, and percentile
calculations used by the report generator and comparator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for a numeric distribution.

    Attributes:
        mean: Arithmetic mean.
        std: Standard deviation.
        median: 50th percentile.
        p5: 5th percentile.
        p25: 25th percentile.
        p75: 75th percentile.
        p95: 95th percentile.
        min: Minimum value.
        max: Maximum value.
        n: Sample count.
    """

    mean: float
    std: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float
    min: float
    max: float
    n: int


def summarize(values: NDArray[np.floating] | list[float]) -> DistributionSummary:
    """Compute summary statistics for a numeric array.

    Args:
        values: Array of numeric observations.

    Returns:
        A DistributionSummary with percentiles and moments.

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty array")

    return DistributionSummary(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
        p5=float(np.percentile(arr, 5)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n=int(arr.size),
    )


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval for a point estimate.

    Attributes:
        point: Point estimate (e.g. sample mean).
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence: Confidence level (e.g. 0.95).
    """

    point: float
    lower: float
    upper: float
    confidence: float


def proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation for small samples
    and proportions near 0 or 1.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence: Desired confidence level (0-1).

    Returns:
        ConfidenceInterval for the true proportion.

    Raises:
        ValueError: If total is zero or successes exceeds total.
    """
    if total == 0:
        raise ValueError("Cannot compute CI with zero trials")
    if successes > total:
        raise ValueError(f"successes ({successes}) cannot exceed total ({total})")

    p_hat = successes / total
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / total
    centre = p_hat + z**2 / (2 * total)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))

    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator

    return ConfidenceInterval(
        point=p_hat,
        lower=max(0.0, float(lower)),
        upper=min(1.0, float(upper)),
        confidence=confidence,
    )


def mean_ci(
    values: NDArray[np.floating] | list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Confidence interval for the mean using the t-distribution.

    Args:
        values: Array of observations.
        confidence: Desired confidence level (0-1).

    Returns:
        ConfidenceInterval for the population mean.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("Need at least 2 values for a mean CI")

    n = arr.size
    mean = float(np.mean(arr))
    se = float(sp_stats.sem(arr))
    t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)

    return ConfidenceInterval(
        point=mean,
        lower=mean - t_crit * se,
        upper=mean + t_crit * se,
        confidence=confidence,
    )


def pareto_frontier(
    costs: list[float],
    success_rates: list[float],
) -> list[int]:
    """Compute the Pareto frontier (lower cost, higher success is better).

    Returns indices of strategies on the Pareto frontier, sorted by
    increasing cost.

    Args:
        costs: Cost values for each strategy.
        success_rates: Success rates for each strategy.

    Returns:
        List of indices on the Pareto frontier.
    """
    if len(costs) != len(success_rates):
        raise ValueError("costs and success_rates must have the same length")

    n = len(costs)
    if n == 0:
        return []

    # Sort by cost ascending, then by success rate descending
    indices = sorted(range(n), key=lambda i: (costs[i], -success_rates[i]))
    frontier: list[int] = []
    best_success = -1.0

    for idx in indices:
        if success_rates[idx] > best_success:
            frontier.append(idx)
            best_success = success_rates[idx]

    return frontier
