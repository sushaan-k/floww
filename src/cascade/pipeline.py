"""Pipeline and Step definitions for agent workflow modeling.

A Pipeline is a directed acyclic graph of Steps, where each Step represents
a single agent action (LLM call, tool use, etc.). Steps can declare
dependencies on other steps, forming the execution order.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Default cost-per-token assumptions (USD per 1K tokens).
MODEL_COST_PER_1K_INPUT: dict[str, float] = {
    "opus": 0.015,
    "sonnet": 0.003,
    "haiku": 0.00025,
}
MODEL_COST_PER_1K_OUTPUT: dict[str, float] = {
    "opus": 0.075,
    "sonnet": 0.015,
    "haiku": 0.00125,
}

# Default latency assumptions (seconds) per step for each model.
MODEL_BASE_LATENCY: dict[str, float] = {
    "opus": 2.0,
    "sonnet": 1.0,
    "haiku": 0.3,
}

# Default tokens consumed per step (input + output).
DEFAULT_INPUT_TOKENS = 500
DEFAULT_OUTPUT_TOKENS = 200


class Step(BaseModel):
    """A single step in an agent pipeline.

    Attributes:
        name: Unique identifier for this step.
        model: The LLM model used (e.g. "sonnet", "opus", "haiku").
        tools: Tools available to the agent at this step.
        depends_on: Names of steps that must complete before this one.
        base_latency_s: Expected latency in seconds. Defaults to model preset.
        input_tokens: Estimated input tokens consumed per execution.
        output_tokens: Estimated output tokens produced per execution.
        context_tokens_used: Running context size entering this step.
        metadata: Arbitrary key-value pairs for user annotations.
    """

    name: str
    model: str = "sonnet"
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    base_latency_s: float | None = None
    input_tokens: int = DEFAULT_INPUT_TOKENS
    output_tokens: int = DEFAULT_OUTPUT_TOKENS
    context_tokens_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def cost_usd(self, model: str | None = None) -> float:
        """Calculate the estimated cost for a single execution of this step."""
        active_model = model or self.model
        input_rate = MODEL_COST_PER_1K_INPUT.get(active_model, 0.003)
        output_rate = MODEL_COST_PER_1K_OUTPUT.get(active_model, 0.015)
        return (
            self.input_tokens / 1000 * input_rate
            + self.output_tokens / 1000 * output_rate
        )

    def latency_s(self, model: str | None = None) -> float:
        """Return the base latency for this step."""
        active_model = model or self.model
        if self.base_latency_s is not None:
            return self.base_latency_s
        return MODEL_BASE_LATENCY.get(active_model, 1.0)


class Pipeline(BaseModel):
    """A directed acyclic graph of Steps defining an agent workflow.

    Attributes:
        steps: Ordered list of pipeline steps.
        name: Optional human-readable pipeline name.
        description: Optional description of what this pipeline does.
    """

    steps: list[Step]
    name: str = "pipeline"
    description: str = ""

    @model_validator(mode="after")
    def _validate_dag(self) -> Pipeline:
        """Ensure step names are unique and dependencies form a valid DAG."""
        names = {s.name for s in self.steps}
        if len(names) != len(self.steps):
            seen: set[str] = set()
            for s in self.steps:
                if s.name in seen:
                    raise ValueError(f"Duplicate step name: {s.name!r}")
                seen.add(s.name)

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in names:
                    raise ValueError(
                        f"Step {step.name!r} depends on unknown step {dep!r}"
                    )

        if self._has_cycle():
            raise ValueError("Pipeline contains a dependency cycle")

        return self

    def _has_cycle(self) -> bool:
        """Detect cycles using Kahn's algorithm."""
        adj: dict[str, list[str]] = {s.name: [] for s in self.steps}
        in_degree: dict[str, int] = {s.name: 0 for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                adj[dep].append(step.name)
                in_degree[step.name] += 1

        queue: deque[str] = deque(name for name, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited != len(self.steps)

    def topological_order(self) -> list[Step]:
        """Return steps in a valid topological execution order."""
        step_map = {s.name: s for s in self.steps}
        adj: dict[str, list[str]] = {s.name: [] for s in self.steps}
        in_degree: dict[str, int] = {s.name: 0 for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                adj[dep].append(step.name)
                in_degree[step.name] += 1

        queue: deque[str] = deque(name for name, deg in in_degree.items() if deg == 0)
        result: list[Step] = []
        while queue:
            node = queue.popleft()
            result.append(step_map[node])
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def total_baseline_cost(self) -> float:
        """Sum baseline cost across all steps (single execution, no retries)."""
        return sum(s.cost_usd() for s in self.steps)

    def total_baseline_latency(self) -> float:
        """Return the critical-path latency across the pipeline DAG."""
        finish_times: dict[str, float] = {}
        for step in self.topological_order():
            start_time = max(
                (finish_times[dep] for dep in step.depends_on),
                default=0.0,
            )
            finish_times[step.name] = start_time + step.latency_s()
        return max(finish_times.values(), default=0.0)

    def step_by_name(self, name: str) -> Step:
        """Look up a step by name.

        Raises:
            KeyError: If no step with the given name exists.
        """
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"No step named {name!r} in pipeline")

    def downstream_of(self, step_name: str) -> list[str]:
        """Return names of all steps that transitively depend on *step_name*."""
        step_map = {s.name: s for s in self.steps}
        if step_name not in step_map:
            raise KeyError(f"No step named {step_name!r} in pipeline")

        children: dict[str, list[str]] = {s.name: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                children[dep].append(step.name)

        visited: set[str] = set()
        queue: deque[str] = deque(children.get(step_name, []))
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(children.get(node, []))

        return sorted(visited)
