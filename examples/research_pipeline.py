#!/usr/bin/env python3
"""Example: Research pipeline reliability simulation.

Models a 6-step research agent pipeline (research -> analyze -> draft ->
review -> revise -> publish) and compares resilience strategies.
"""

from cascade import (
    Comparator,
    FailureConfig,
    Pipeline,
    Simulator,
    Step,
    build_report,
    format_report,
)
from cascade.strategies import (
    adaptive,
    checkpoint,
    fallback,
    naive,
    parallel,
    retry,
)


def main() -> None:
    # Define the pipeline from the spec
    pipeline = Pipeline(
        name="research-agent",
        description="End-to-end research and publishing pipeline",
        steps=[
            Step(
                name="research",
                model="sonnet",
                tools=["web_search", "read_file"],
            ),
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

    # Configure realistic failure injection
    failures = FailureConfig(
        hallucination_rate=0.05,
        refusal_rate=0.02,
        tool_failure_rate=0.03,
        context_overflow_at=100_000,
        cascade_propagation=0.8,
    )

    # Run a single simulation with the naive strategy
    print("=" * 60)
    print("Single Strategy Simulation (Naive)")
    print("=" * 60)
    sim = Simulator(pipeline, failures, n_simulations=10_000, seed=42)
    result = sim.run()
    report = build_report(result)
    print(format_report(report))
    print()

    # Compare multiple strategies
    print("=" * 60)
    print("Strategy Comparison")
    print("=" * 60)
    comp = Comparator(pipeline, failures, n_simulations=10_000, seed=42)
    comparison = comp.compare(
        [
            naive(),
            retry(max_attempts=3),
            fallback(models=["sonnet", "haiku"]),
            parallel(n=3, vote="majority"),
            checkpoint(interval=2),
            adaptive(escalation_threshold=2, escalation_strategy="parallel"),
        ]
    )

    comparison.print_table()
    print()
    comparison.recommend()


if __name__ == "__main__":
    main()
