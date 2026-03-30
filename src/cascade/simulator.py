"""Monte Carlo simulation engine for agent pipeline reliability.

The Simulator runs N independent trials of a Pipeline under a given
FailureConfig and ResilienceStrategy, collecting per-run metrics that
are then aggregated into SimulationResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from cascade.failures import (
    FailureConfig,
    FailureEvent,
    FailureInjector,
    FailureType,
)
from cascade.pipeline import Pipeline, Step
from cascade.strategies import (
    ResilienceStrategy,
    StrategyType,
)
from cascade.strategies import (
    checkpoint as checkpoint_strategy,
)
from cascade.strategies import (
    fallback as fallback_strategy,
)
from cascade.strategies import (
    human_in_loop as human_in_loop_strategy,
)
from cascade.strategies import (
    naive as naive_strategy,
)
from cascade.strategies import (
    parallel as parallel_strategy,
)
from cascade.strategies import (
    retry as retry_strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Outcome of executing a single pipeline step in one simulation run.

    Attributes:
        step_name: Name of the step.
        success: Whether the step ultimately succeeded.
        attempts: Number of attempts made (including retries).
        failure: The failure event if the step failed, else None.
        cost_usd: Total cost for this step across all attempts.
        latency_s: Total wall-clock time for this step.
        corrupted_output: Whether the output is tainted.
    """

    step_name: str
    success: bool
    attempts: int = 1
    failure: FailureEvent | None = None
    failure_events: list[FailureEvent] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    corrupted_output: bool = False


@dataclass
class RunResult:
    """Outcome of a single end-to-end simulation run.

    Attributes:
        success: Whether the entire pipeline completed successfully.
        step_results: Per-step outcomes in execution order.
        total_cost_usd: Aggregate cost for this run.
        total_latency_s: Aggregate latency for this run.
        failure_events: All failure events encountered.
        steps_completed: Number of steps that completed (successfully or not).
        first_failure_step: Index of the first fatal failure, or -1.
        recovered: Whether the pipeline recovered from at least one failure.
    """

    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    failure_events: list[FailureEvent] = field(default_factory=list)
    steps_completed: int = 0
    first_failure_step: int = -1
    recovered: bool = False


class SimulationResult(BaseModel):
    """Aggregated results from all simulation runs.

    Attributes:
        n_simulations: Number of simulation runs.
        success_count: Number of fully successful runs.
        success_rate: Fraction of runs that succeeded.
        mean_cost_usd: Average total cost across all runs.
        mean_latency_s: Average total latency across all runs.
        failure_counts: Count of each failure type across all runs.
        mean_steps_to_failure: Average step index of first failure.
        recovery_rate: Fraction of runs that recovered from a failure.
        costs: List of per-run costs.
        latencies: List of per-run latencies.
        strategy_name: Display name of the strategy used.
    """

    n_simulations: int
    success_count: int
    success_rate: float
    mean_cost_usd: float
    mean_latency_s: float
    failure_counts: dict[str, int] = Field(default_factory=dict)
    mean_steps_to_failure: float = 0.0
    recovery_rate: float = 0.0
    costs: list[float] = Field(default_factory=list)
    latencies: list[float] = Field(default_factory=list)
    strategy_name: str = "Naive"


class Simulator:
    """Monte Carlo agent pipeline simulator.

    Runs N independent stochastic simulations of a Pipeline, injecting
    failures according to a FailureConfig and applying a ResilienceStrategy.

    Args:
        pipeline: The agent pipeline to simulate.
        failure_config: Failure injection configuration.
        n_simulations: Number of simulation runs.
        strategy: Resilience strategy to apply.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        failure_config: FailureConfig,
        n_simulations: int = 1000,
        strategy: ResilienceStrategy | None = None,
        seed: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.failure_config = failure_config
        self.n_simulations = n_simulations
        self.strategy = strategy or ResilienceStrategy(strategy_type=StrategyType.NAIVE)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        strategy: ResilienceStrategy | None = None,
    ) -> SimulationResult:
        """Execute all simulation runs and return aggregated results.

        Args:
            strategy: Optional strategy override for this run.

        Returns:
            SimulationResult with aggregated metrics.
        """
        active_strategy = strategy or self.strategy
        runs: list[RunResult] = []

        for _i in range(self.n_simulations):
            run_seed = int(self._rng.integers(0, 2**31))
            result = self._run_single(active_strategy, run_seed)
            runs.append(result)

        return self._aggregate(runs, active_strategy)

    def compare_strategies(
        self,
        strategies: list[ResilienceStrategy],
    ) -> list[SimulationResult]:
        """Run simulations for multiple strategies and return results.

        Args:
            strategies: List of strategies to compare.

        Returns:
            List of SimulationResult, one per strategy, in the same order.
        """
        results: list[SimulationResult] = []
        for strat in strategies:
            # Reset RNG so each strategy gets the same random sequence
            self._rng = np.random.default_rng(self.seed)
            result = self.run(strategy=strat)
            results.append(result)
        return results

    def _run_single(
        self,
        strategy: ResilienceStrategy,
        seed: int,
    ) -> RunResult:
        """Execute a single simulation run.

        Args:
            strategy: The resilience strategy in effect.
            seed: Random seed for this specific run.

        Returns:
            RunResult for this run.
        """
        injector = FailureInjector(config=self.failure_config)
        injector.reset(seed)

        steps = self.pipeline.topological_order()
        step_results: list[StepResult] = []
        failure_events: list[FailureEvent] = []
        total_cost = 0.0
        wall_clock_latency = 0.0
        cumulative_tokens = 0
        run_success = True
        first_failure_step = -1
        recovered = False
        consecutive_failures = 0
        step_finish_times: dict[str, float] = {}

        # Checkpointing state
        last_checkpoint_idx = 0

        for step_idx, step in enumerate(steps):
            if strategy.strategy_type == StrategyType.PARALLEL:
                sr = self._execute_step_parallel(
                    step=step,
                    strategy=strategy,
                    injector=injector,
                    cumulative_tokens=cumulative_tokens,
                )
            else:
                sr = self._execute_step(
                    step=step,
                    step_idx=step_idx,
                    strategy=strategy,
                    injector=injector,
                    cumulative_tokens=cumulative_tokens,
                    consecutive_failures=consecutive_failures,
                )

            step_results.append(sr)
            total_cost += sr.cost_usd
            cumulative_tokens += step.input_tokens + step.output_tokens

            if sr.failure_events:
                failure_events.extend(sr.failure_events)

            step_start = max(
                (step_finish_times.get(dep, 0.0) for dep in step.depends_on),
                default=0.0,
            )
            step_finish = step_start + sr.latency_s
            wall_clock_latency = max(wall_clock_latency, step_finish)
            step_finish_times[step.name] = step_finish

            had_failure_event = any(
                fe.failure_type != FailureType.LATENCY_SPIKE for fe in sr.failure_events
            )

            if had_failure_event and first_failure_step == -1:
                first_failure_step = step_idx

            if not sr.success:
                if first_failure_step == -1:
                    first_failure_step = step_idx
                consecutive_failures += 1

                # Checkpoint rollback: retry from last checkpoint
                if (
                    strategy.strategy_type == StrategyType.CHECKPOINT
                    and strategy.checkpoint_interval > 0
                ):
                    rollback_to = last_checkpoint_idx
                    rollback_success = self._attempt_rollback(
                        steps=steps,
                        from_idx=rollback_to,
                        to_idx=step_idx,
                        strategy=strategy,
                        injector=injector,
                        cumulative_tokens=cumulative_tokens,
                    )
                    if rollback_success:
                        recovered = True
                        consecutive_failures = 0
                        sr.success = True
                        sr.corrupted_output = False
                        sr.failure = None
                        injector.clear_corruption(step.name)
                        total_cost += step.cost_usd() * 1.5
                        rollback_latency = step.latency_s() * 2
                        wall_clock_latency = max(
                            wall_clock_latency, step_finish + rollback_latency
                        )
                        step_finish_times[step.name] = wall_clock_latency
                        continue

                run_success = False
                break
            else:
                if consecutive_failures > 0 or had_failure_event:
                    recovered = True
                consecutive_failures = 0

                # Update checkpoint position
                if (
                    strategy.strategy_type == StrategyType.CHECKPOINT
                    and strategy.checkpoint_interval > 0
                    and (step_idx + 1) % strategy.checkpoint_interval == 0
                ):
                    last_checkpoint_idx = step_idx + 1

        return RunResult(
            success=run_success,
            step_results=step_results,
            total_cost_usd=total_cost,
            total_latency_s=wall_clock_latency,
            failure_events=failure_events,
            steps_completed=len([sr for sr in step_results if sr.success]),
            first_failure_step=first_failure_step,
            recovered=recovered,
        )

    def _execute_step(
        self,
        step: Step,
        step_idx: int,
        strategy: ResilienceStrategy,
        injector: FailureInjector,
        cumulative_tokens: int,
        consecutive_failures: int,
    ) -> StepResult:
        """Execute a single step under the active resilience strategy.

        Args:
            step: The Step to execute.
            step_idx: Index of the step in topological order.
            strategy: Active resilience strategy.
            injector: Failure injector instance.
            cumulative_tokens: Tokens consumed so far.
            consecutive_failures: Consecutive failure count (for adaptive).

        Returns:
            StepResult for this step.
        """
        projected_tokens = (
            cumulative_tokens + step.input_tokens + step.output_tokens
        )

        upstream_corrupted = any(injector.is_corrupted(dep) for dep in step.depends_on)
        corrupted_output = upstream_corrupted
        attempt_failures: list[FailureEvent] = []
        is_adaptive = strategy.strategy_type == StrategyType.ADAPTIVE
        max_attempts = (
            strategy.max_attempts + 1
            if is_adaptive and strategy.escalation_threshold <= strategy.max_attempts
            else self._get_max_attempts(strategy, consecutive_failures)
        )
        failure_count = 0

        total_cost = 0.0
        total_latency = 0.0

        for attempt in range(1, max_attempts + 1):
            exec_strategy = strategy
            if is_adaptive:
                if failure_count >= strategy.escalation_threshold:
                    exec_strategy = self._resolve_adaptive_escalation_strategy(
                        strategy=strategy,
                        step=step,
                        step_idx=step_idx,
                    )
                else:
                    exec_strategy = retry_strategy(max_attempts=strategy.max_attempts)

                if exec_strategy.strategy_type == StrategyType.PARALLEL:
                    parallel_result = self._execute_step_parallel(
                        step=step,
                        strategy=exec_strategy,
                        injector=injector,
                        cumulative_tokens=cumulative_tokens,
                    )
                    parallel_result.cost_usd += total_cost
                    parallel_result.latency_s += total_latency
                    parallel_result.failure_events = (
                        attempt_failures + parallel_result.failure_events
                    )
                    parallel_result.corrupted_output = (
                        parallel_result.corrupted_output or corrupted_output
                    )
                    if parallel_result.corrupted_output:
                        injector.mark_corrupted(step.name)
                    else:
                        injector.clear_corruption(step.name)
                    return parallel_result

            st = exec_strategy.strategy_type
            human_checkpoint = (
                st == StrategyType.HUMAN_IN_LOOP
                and step_idx in exec_strategy.human_at_steps
            )
            current_model = step.model
            if st == StrategyType.FALLBACK and attempt > 1 and exec_strategy.fallback_models:
                fb_idx = min(attempt - 2, len(exec_strategy.fallback_models) - 1)
                current_model = exec_strategy.fallback_models[fb_idx]

            failure = injector.inject(
                step_name=step.name,
                model=current_model,
                tools=step.tools,
                cumulative_tokens=projected_tokens,
                upstream_corrupted=upstream_corrupted,
            )

            attempt_cost = step.cost_usd(model=current_model)
            attempt_latency = step.latency_s(model=current_model)

            if failure is not None:
                attempt_failures.append(failure)

                if failure.failure_type == FailureType.LATENCY_SPIKE:
                    # Latency spike is non-fatal, but still contributes to metrics.
                    attempt_latency *= self.failure_config.spike_multiplier
                    total_cost += attempt_cost
                    total_latency += attempt_latency
                    if (
                        human_checkpoint
                        and corrupted_output
                        and injector.rng.random() < exec_strategy.human_accuracy
                    ):
                        corrupted_output = False
                    if corrupted_output:
                        injector.mark_corrupted(step.name)
                    else:
                        injector.clear_corruption(step.name)
                    return StepResult(
                        step_name=step.name,
                        success=True,
                        attempts=attempt,
                        failure=None,
                        failure_events=attempt_failures,
                        cost_usd=total_cost,
                        latency_s=total_latency,
                        corrupted_output=corrupted_output,
                    )

                total_cost += attempt_cost
                total_latency += attempt_latency + failure.latency_added_s
                failure_count += 1

                if (
                    human_checkpoint
                    and failure.failure_type != FailureType.CONTEXT_OVERFLOW
                    and injector.rng.random() < exec_strategy.human_accuracy
                ):
                    corrupted_output = False
                    injector.clear_corruption(step.name)
                    return StepResult(
                        step_name=step.name,
                        success=True,
                        attempts=attempt,
                        failure=None,
                        failure_events=attempt_failures,
                        cost_usd=total_cost,
                        latency_s=total_latency,
                        corrupted_output=False,
                    )

                if not failure.recoverable or attempt == max_attempts:
                    injector.mark_corrupted(step.name)
                    return StepResult(
                        step_name=step.name,
                        success=False,
                        attempts=attempt,
                        failure=failure,
                        failure_events=attempt_failures,
                        cost_usd=total_cost,
                        latency_s=total_latency,
                        corrupted_output=True,
                    )

                continue

            total_cost += attempt_cost
            total_latency += attempt_latency

            if (
                human_checkpoint
                and corrupted_output
                and injector.rng.random() < exec_strategy.human_accuracy
            ):
                corrupted_output = False

            if corrupted_output:
                injector.mark_corrupted(step.name)
            else:
                injector.clear_corruption(step.name)

            return StepResult(
                step_name=step.name,
                success=True,
                attempts=attempt,
                failure=None,
                failure_events=attempt_failures,
                cost_usd=total_cost,
                latency_s=total_latency,
                corrupted_output=corrupted_output,
            )

        # Should not reach here, but safety fallback
        injector.mark_corrupted(step.name)
        return StepResult(
            step_name=step.name,
            success=False,
            attempts=max_attempts,
            failure_events=attempt_failures,
            cost_usd=total_cost,
            latency_s=total_latency,
            corrupted_output=True,
        )

    def _execute_step_parallel(
        self,
        step: Step,
        strategy: ResilienceStrategy,
        injector: FailureInjector,
        cumulative_tokens: int,
    ) -> StepResult:
        """Execute a step with parallel redundancy and voting.

        Args:
            step: The step to execute.
            strategy: Active strategy (must be PARALLEL).
            injector: Failure injector.
            cumulative_tokens: Tokens consumed so far.

        Returns:
            StepResult reflecting the voted outcome.
        """
        n = strategy.parallel_n
        successes = 0
        total_cost = 0.0
        max_latency = 0.0
        failure_events: list[FailureEvent] = []
        corrupted_input = any(injector.is_corrupted(dep) for dep in step.depends_on)

        for _ in range(n):
            upstream_corrupted = any(
                injector.is_corrupted(dep) for dep in step.depends_on
            )
            failure = injector.inject(
                step_name=step.name,
                model=step.model,
                tools=step.tools,
                cumulative_tokens=cumulative_tokens + step.input_tokens + step.output_tokens,
                upstream_corrupted=upstream_corrupted,
            )
            cost = step.cost_usd()
            latency = step.latency_s()

            if failure is not None:
                failure_events.append(failure)
                if failure.failure_type == FailureType.LATENCY_SPIKE:
                    latency *= self.failure_config.spike_multiplier
                    failure = None  # Spike is non-fatal

            total_cost += cost
            max_latency = max(max_latency, latency)

            if failure is None:
                successes += 1

        # Voting
        if strategy.vote_method == "majority":
            passed = successes > n / 2
        elif strategy.vote_method == "unanimous":
            passed = successes == n
        else:  # "any"
            passed = successes >= 1

        if passed:
            if corrupted_input:
                injector.mark_corrupted(step.name)
            else:
                injector.clear_corruption(step.name)

        return StepResult(
            step_name=step.name,
            success=passed,
            attempts=n,
            failure=failure_events[-1] if (failure_events and not passed) else None,
            failure_events=failure_events,
            cost_usd=total_cost,
            latency_s=max_latency,
            corrupted_output=(not passed) or corrupted_input,
        )

    def _resolve_adaptive_escalation_strategy(
        self,
        strategy: ResilienceStrategy,
        step: Step,
        step_idx: int,
    ) -> ResilienceStrategy:
        """Resolve adaptive escalation into a concrete strategy."""
        if strategy.escalation_strategy == StrategyType.RETRY:
            return retry_strategy(max_attempts=strategy.max_attempts + 1)
        if strategy.escalation_strategy == StrategyType.FALLBACK:
            return fallback_strategy(models=self._fallback_models_for(step.model))
        if strategy.escalation_strategy == StrategyType.PARALLEL:
            return parallel_strategy()
        if strategy.escalation_strategy == StrategyType.CHECKPOINT:
            return checkpoint_strategy()
        if strategy.escalation_strategy == StrategyType.HUMAN_IN_LOOP:
            return human_in_loop_strategy(
                at_steps=[step_idx],
                accuracy=strategy.human_accuracy,
            )

        return naive_strategy()

    def _fallback_models_for(self, model: str) -> list[str]:
        """Return fallback models distinct from the current model."""
        preferred_order = ["opus", "sonnet", "haiku"]
        return [candidate for candidate in preferred_order if candidate != model]

    def _get_max_attempts(
        self,
        strategy: ResilienceStrategy,
        consecutive_failures: int,
    ) -> int:
        """Determine how many attempts a step gets under the strategy.

        Args:
            strategy: Active resilience strategy.
            consecutive_failures: Current consecutive failure count.

        Returns:
            Maximum number of attempts for the current step.
        """
        st = strategy.strategy_type

        if st == StrategyType.NAIVE:
            return 1

        if st == StrategyType.RETRY:
            return strategy.max_attempts

        if st == StrategyType.FALLBACK:
            return strategy.max_attempts

        if st == StrategyType.PARALLEL:
            # Parallel uses a separate execution path
            return 1

        if st == StrategyType.CHECKPOINT:
            return strategy.max_attempts

        if st == StrategyType.HUMAN_IN_LOOP:
            return 2  # One retry after human verification

        if st == StrategyType.ADAPTIVE:
            if consecutive_failures >= strategy.escalation_threshold:
                return strategy.max_attempts + 1
            return strategy.max_attempts

        return 1  # pragma: no cover

    def _attempt_rollback(
        self,
        steps: list[Step],
        from_idx: int,
        to_idx: int,
        strategy: ResilienceStrategy,
        injector: FailureInjector,
        cumulative_tokens: int,
    ) -> bool:
        """Attempt to rollback and re-execute from a checkpoint.

        Args:
            steps: Full list of steps in topological order.
            from_idx: Checkpoint index to rollback to.
            to_idx: Index of the failed step.
            strategy: Active resilience strategy.
            injector: Failure injector.
            cumulative_tokens: Token count at time of failure.

        Returns:
            True if the rollback re-execution succeeded.
        """
        for idx in range(from_idx, to_idx + 1):
            step = steps[idx]
            failure = injector.inject(
                step_name=step.name,
                model=step.model,
                tools=step.tools,
                cumulative_tokens=cumulative_tokens,
                upstream_corrupted=False,
            )
            if failure is not None and failure.failure_type not in (
                FailureType.LATENCY_SPIKE,
            ):
                return False
        return True

    def _aggregate(
        self,
        runs: list[RunResult],
        strategy: ResilienceStrategy,
    ) -> SimulationResult:
        """Aggregate per-run results into a SimulationResult.

        Args:
            runs: List of individual run results.
            strategy: The strategy that was used.

        Returns:
            Aggregated SimulationResult.
        """
        n = len(runs)
        success_count = sum(1 for r in runs if r.success)
        costs = [r.total_cost_usd for r in runs]
        latencies = [r.total_latency_s for r in runs]

        failure_counts: dict[str, int] = {}
        failure_step_indices: list[int] = []
        recovery_count = 0

        for r in runs:
            for fe in r.failure_events:
                key = fe.failure_type.value
                failure_counts[key] = failure_counts.get(key, 0) + 1
            if r.first_failure_step >= 0:
                failure_step_indices.append(r.first_failure_step)
            if r.recovered:
                recovery_count += 1

        mean_steps_to_failure = (
            float(np.mean(failure_step_indices))
            if failure_step_indices
            else float(len(self.pipeline.steps))
        )

        runs_with_failures = sum(1 for r in runs if r.first_failure_step >= 0)
        recovery_rate = (
            recovery_count / runs_with_failures if runs_with_failures > 0 else 0.0
        )

        return SimulationResult(
            n_simulations=n,
            success_count=success_count,
            success_rate=success_count / n if n > 0 else 0.0,
            mean_cost_usd=float(np.mean(costs)) if costs else 0.0,
            mean_latency_s=float(np.mean(latencies)) if latencies else 0.0,
            failure_counts=failure_counts,
            mean_steps_to_failure=mean_steps_to_failure,
            recovery_rate=recovery_rate,
            costs=costs,
            latencies=latencies,
            strategy_name=strategy.display_name,
        )
