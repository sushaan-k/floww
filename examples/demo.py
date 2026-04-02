#!/usr/bin/env python3
"""Offline demo for cascade."""

from __future__ import annotations

from cascade import FailureConfig, Pipeline, Simulator, Step, build_report
from cascade.strategies import adaptive, naive


def build_pipeline() -> Pipeline:
    return Pipeline(
        name="release-review",
        steps=[
            Step(name="collect", model="sonnet", tools=["web_search"]),
            Step(name="analyze", model="sonnet", depends_on=["collect"]),
            Step(name="publish", model="haiku", depends_on=["analyze"]),
        ],
    )


def run_simulation(strategy_name: str) -> tuple[float, float]:
    strategies = {
        "naive": naive(),
        "adaptive": adaptive(escalation_strategy="parallel"),
    }
    result = Simulator(
        pipeline=build_pipeline(),
        failure_config=FailureConfig(hallucination_rate=0.08, tool_failure_rate=0.05),
        strategy=strategies[strategy_name],
        n_simulations=400,
        seed=42,
    ).run()
    report = build_report(result)
    return report.success_rate, report.cost_ci.point


def main() -> None:
    naive_success, naive_cost = run_simulation("naive")
    adaptive_success, adaptive_cost = run_simulation("adaptive")

    print("cascade demo")
    print(f"naive success rate: {naive_success:.1%} | avg cost: {naive_cost:.2f}")
    print(
        f"adaptive success rate: {adaptive_success:.1%} | avg cost: {adaptive_cost:.2f}"
    )


if __name__ == "__main__":
    main()
