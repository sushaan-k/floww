#!/usr/bin/env python3
"""Example: Customer support agent reliability simulation.

Models a customer support agent that handles tickets through
classification, research, response drafting, and escalation. Uses
a diamond-shaped dependency graph to show non-linear pipelines.
"""

from cascade import (
    Comparator,
    FailureConfig,
    Pipeline,
    Step,
)
from cascade.strategies import (
    adaptive,
    checkpoint,
    human_in_loop,
    naive,
    retry,
)


def main() -> None:
    # Diamond-shaped pipeline: classify fans out to two parallel research
    # paths, which converge before the response is drafted.
    pipeline = Pipeline(
        name="customer-support-agent",
        description="Ticket handling with parallel research paths",
        steps=[
            Step(
                name="classify",
                model="haiku",
                tools=["api_call"],
            ),
            Step(
                name="search_docs",
                model="sonnet",
                tools=["web_search", "read_file"],
                depends_on=["classify"],
            ),
            Step(
                name="search_history",
                model="sonnet",
                tools=["api_call"],
                depends_on=["classify"],
            ),
            Step(
                name="draft_response",
                model="sonnet",
                tools=["write_file"],
                depends_on=["search_docs", "search_history"],
            ),
            Step(
                name="quality_check",
                model="opus",
                tools=["read_file"],
                depends_on=["draft_response"],
            ),
            Step(
                name="send_response",
                model="haiku",
                tools=["api_call"],
                depends_on=["quality_check"],
            ),
        ],
    )

    # Customer support has lower hallucination tolerance but higher
    # tool failure rates (CRM, knowledge base outages)
    failures = FailureConfig(
        hallucination_rate=0.03,
        refusal_rate=0.02,
        tool_failure_rate=0.08,
        cascade_propagation=0.6,
        latency_spike_rate=0.03,
        spike_multiplier=8.0,
    )

    print(f"Pipeline: {pipeline.name}")
    print(f"Steps: {len(pipeline.steps)}")
    print("Topology: diamond (parallel research paths)")
    print(f"Baseline cost: ${pipeline.total_baseline_cost():.4f}")
    print()

    # Compare strategies including human-in-the-loop
    comp = Comparator(pipeline, failures, n_simulations=5000, seed=42)
    comparison = comp.compare([
        naive(),
        retry(max_attempts=3),
        checkpoint(interval=2),
        human_in_loop(at_steps=[3, 4], accuracy=0.95),
        adaptive(escalation_threshold=2, escalation_strategy="retry"),
    ])

    comparison.print_table()
    print()
    comparison.recommend()


if __name__ == "__main__":
    main()
