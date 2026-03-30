"""Cascade -- agent reliability simulator.

Chaos engineering for AI agent pipelines. Model multi-step workflows,
inject realistic failure modes, and compare resilience strategies via
Monte Carlo simulation.

Example::

    from cascade import Pipeline, Step, Simulator, FailureConfig
    from cascade import strategies

    pipeline = Pipeline(steps=[
        Step(name="research", model="sonnet", tools=["web_search"]),
        Step(name="analyze", model="sonnet", depends_on=["research"]),
        Step(name="draft", model="sonnet", depends_on=["analyze"]),
    ])

    failures = FailureConfig(hallucination_rate=0.05)
    sim = Simulator(pipeline, failures, n_simulations=1000)
    results = sim.run()
"""

from cascade.comparator import Comparator, StrategyComparison
from cascade.failures import (
    FailureConfig,
    FailureEvent,
    FailureInjector,
    FailureType,
    HallucinationSubtype,
)
from cascade.pipeline import Pipeline, Step
from cascade.report import (
    SimulationReport,
    build_report,
    export_comparison_json,
    export_json,
    format_report,
)
from cascade.simulator import (
    RunResult,
    SimulationResult,
    Simulator,
    StepResult,
)
from cascade.stats import (
    ConfidenceInterval,
    DistributionSummary,
    mean_ci,
    pareto_frontier,
    proportion_ci,
    summarize,
)
from cascade.strategies import (
    ResilienceStrategy,
    StrategyType,
)

__version__ = "0.1.0"

__all__ = [
    # Comparator
    "Comparator",
    # Stats
    "ConfidenceInterval",
    "DistributionSummary",
    # Failures
    "FailureConfig",
    "FailureEvent",
    "FailureInjector",
    "FailureType",
    "HallucinationSubtype",
    # Pipeline
    "Pipeline",
    # Strategies
    "ResilienceStrategy",
    "RunResult",
    # Report
    "SimulationReport",
    "SimulationResult",
    # Simulator
    "Simulator",
    "Step",
    "StepResult",
    "StrategyComparison",
    "StrategyType",
    # Version
    "__version__",
    "build_report",
    "export_comparison_json",
    "export_json",
    "format_report",
    "mean_ci",
    "pareto_frontier",
    "proportion_ci",
    "summarize",
]
