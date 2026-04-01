#!/usr/bin/env python3
"""Example: Coding agent pipeline reliability simulation.

Models a multi-step coding agent that plans, implements, tests, and
deploys code changes. Demonstrates how pipeline length and failure
rates compound into dramatic reliability drops.
"""

from cascade import (
    Comparator,
    FailureConfig,
    Pipeline,
    Simulator,
    Step,
)
from cascade.strategies import adaptive, checkpoint, naive, retry


def main() -> None:
    pipeline = Pipeline(
        name="coding-agent",
        description="Autonomous coding agent: plan -> implement -> test -> deploy",
        steps=[
            Step(name="understand_ticket", model="opus", tools=["read_file"]),
            Step(
                name="plan",
                model="opus",
                tools=["read_file"],
                depends_on=["understand_ticket"],
            ),
            Step(
                name="implement",
                model="sonnet",
                tools=["write_file", "python_exec"],
                depends_on=["plan"],
            ),
            Step(
                name="write_tests",
                model="sonnet",
                tools=["write_file"],
                depends_on=["implement"],
            ),
            Step(
                name="run_tests",
                model="haiku",
                tools=["python_exec"],
                depends_on=["write_tests"],
            ),
            Step(
                name="fix_failures",
                model="sonnet",
                tools=["write_file", "python_exec"],
                depends_on=["run_tests"],
            ),
            Step(
                name="code_review",
                model="opus",
                tools=["read_file"],
                depends_on=["fix_failures"],
            ),
            Step(
                name="apply_feedback",
                model="sonnet",
                tools=["write_file"],
                depends_on=["code_review"],
            ),
            Step(
                name="final_tests",
                model="haiku",
                tools=["python_exec"],
                depends_on=["apply_feedback"],
            ),
            Step(
                name="deploy",
                model="haiku",
                tools=["api_call"],
                depends_on=["final_tests"],
            ),
        ],
    )

    # Coding pipelines have higher tool failure rates (CI flakiness, etc.)
    failures = FailureConfig(
        hallucination_rate=0.04,
        refusal_rate=0.01,
        tool_failure_rate=0.06,
        context_overflow_at=128_000,
        cascade_propagation=0.7,
        latency_spike_rate=0.02,
    )

    # Show the compounding problem
    print("The Reliability Compounding Problem")
    print("=" * 60)
    print(f"Pipeline: {pipeline.name} ({len(pipeline.steps)} steps)")
    print(f"Baseline cost: ${pipeline.total_baseline_cost():.4f}")
    print(f"Baseline latency: {pipeline.total_baseline_latency():.1f}s")
    print()

    # Naive simulation
    sim = Simulator(pipeline, failures, n_simulations=5000, seed=42)
    result = sim.run(strategy=naive())
    print(f"Naive success rate: {result.success_rate:.1%}")
    print(
        f"  (That means {result.n_simulations - result.success_count:,} "
        f"failures out of {result.n_simulations:,} runs)"
    )
    print()

    # Compare strategies
    print("Strategy Comparison")
    print("=" * 60)
    comp = Comparator(pipeline, failures, n_simulations=5000, seed=42)
    comparison = comp.compare(
        [
            naive(),
            retry(max_attempts=3),
            checkpoint(interval=3),
            adaptive(escalation_threshold=2, escalation_strategy="retry"),
        ]
    )
    comparison.print_table()
    print()
    comparison.recommend()


if __name__ == "__main__":
    main()
