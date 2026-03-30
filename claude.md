# cascade

## Agent Reliability Simulator — Chaos Engineering for AI Agents

### The Problem

Here's the math that keeps every AI engineering team up at night:

| Steps | Per-Step Accuracy | End-to-End Success |
|-------|------------------|--------------------|
| 5     | 95%              | 77%                |
| 10    | 95%              | 60%                |
| 10    | 85%              | 20%                |
| 20    | 90%              | 12%                |
| 50    | 95%              | 8%                 |

Accuracy **compounds catastrophically**. A 95%-accurate agent on a 50-step task succeeds 8% of the time. This is the central unsolved problem in production AI agents — two-thirds of organizations are experimenting with agents but only one-quarter have made it to production.

Netflix built Chaos Monkey to test distributed systems resilience. Nobody has built the equivalent for AI agents. Teams deploy agents and pray. When they fail, debugging is a nightmare because the same input can produce wildly different execution paths.

### The Solution

`cascade` is a simulation framework for modeling multi-step AI agent pipelines, injecting realistic failure modes, and measuring end-to-end reliability under different architectural patterns (retry, fallback, parallel redundancy, consensus voting, human-in-the-loop).

### Architecture

```
┌───────────────────────────────────────────────────┐
│                    cascade                         │
│                                                    │
│  ┌──────────────┐  ┌───────────────────────────┐  │
│  │  Pipeline     │  │  Failure Injector          │  │
│  │  Definition   │  │                            │  │
│  │              │  │  - Hallucination (wrong     │  │
│  │  - Steps     │  │    tool args, bad output)   │  │
│  │  - Deps      │  │  - Refusal (safety filter)  │  │
│  │  - Tools     │  │  - Latency spike            │  │
│  │  - Models    │  │  - Context overflow          │  │
│  │              │  │  - Tool failure (API down)   │  │
│  └──────┬───────┘  │  - Cascading corruption     │  │
│         │          │    (bad output fed forward)  │  │
│         │          └──────────┬──────────────────┘  │
│         ▼                     ▼                     │
│  ┌──────────────────────────────────────────────┐  │
│  │            Simulation Engine                   │  │
│  │                                                │  │
│  │  Runs N simulations of the pipeline with       │  │
│  │  stochastic failure injection. Measures:       │  │
│  │  - End-to-end success rate                     │  │
│  │  - Mean steps to failure                       │  │
│  │  - Recovery rate after failure                 │  │
│  │  - Token cost distribution                     │  │
│  │  - Latency distribution                        │  │
│  │  - Failure mode frequency                      │  │
│  └──────────────────────────────────────────────┘  │
│                       │                             │
│                       ▼                             │
│  ┌──────────────────────────────────────────────┐  │
│  │        Resilience Strategy Comparator         │  │
│  │                                                │  │
│  │  Tests same pipeline under different           │  │
│  │  resilience strategies:                        │  │
│  │  - Naive (no retry)                            │  │
│  │  - Simple retry (up to K attempts)             │  │
│  │  - Exponential backoff                         │  │
│  │  - Fallback model (try cheaper, then better)   │  │
│  │  - Parallel redundancy (N agents, vote)        │  │
│  │  - Checkpoint + rollback                       │  │
│  │  - Human-in-the-loop at step K                 │  │
│  │  - Hybrid combinations                         │  │
│  └──────────────────────────────────────────────┘  │
│                       │                             │
│                       ▼                             │
│  ┌──────────────────────────────────────────────┐  │
│  │              Report Generator                  │  │
│  │  - Success rate comparison table               │  │
│  │  - Cost vs. reliability Pareto frontier        │  │
│  │  - Failure mode heatmap                        │  │
│  │  - Recommended strategy                        │  │
│  └──────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### Failure Models

Each failure mode is modeled with configurable probability and realistic behavior:

#### 1. Hallucination
The agent produces plausible but incorrect output. This is the hardest failure because downstream steps don't know the input is bad.
- **Subtypes**: Wrong tool arguments, fabricated data, incorrect reasoning, format errors
- **Config**: `hallucination_rate=0.05` (5% of steps produce wrong output)

#### 2. Refusal
Safety filters or guardrails block a legitimate action.
- **Config**: `refusal_rate=0.02` (2% false positive refusal rate)

#### 3. Tool Failure
External APIs return errors, timeouts, or rate limits.
- **Config**: `tool_failure_rate=0.03, tool_timeout_ms=5000`

#### 4. Context Overflow
The agent's context window fills up mid-task, losing important earlier information.
- **Config**: `context_limit=128000, overflow_behavior="truncate_early"`

#### 5. Cascading Corruption
A hallucination in step N produces bad data that is used by steps N+1, N+2, etc. The error amplifies.
- **Config**: `cascade_propagation=0.8` (80% chance bad output corrupts next step)

#### 6. Latency Spike
An individual step takes much longer than expected, potentially causing timeouts.
- **Config**: `latency_spike_rate=0.01, spike_multiplier=10`

### Resilience Strategies

```python
from cascade import strategies

# Built-in strategies
strategies.naive()                    # No retry, fail fast
strategies.retry(max_attempts=3)      # Simple retry
strategies.fallback(models=["sonnet", "haiku"])  # Try cheaper first
strategies.parallel(n=3, vote="majority")  # Run 3 agents, majority vote
strategies.checkpoint(interval=5)     # Checkpoint every 5 steps, rollback on failure
strategies.human_in_loop(at_steps=[5, 10, 15])  # Human verification at key steps
strategies.adaptive(                  # Dynamic strategy selection
    escalation_threshold=2,           # After 2 failures, escalate
    escalation_strategy="parallel"    # Escalate to parallel redundancy
)
```

### API Surface (Draft)

```python
from cascade import Pipeline, Step, Simulator, FailureConfig

# Define a pipeline
pipeline = Pipeline(
    steps=[
        Step("research", model="sonnet", tools=["web_search", "read_file"]),
        Step("analyze", model="sonnet", tools=["python_exec"], depends_on=["research"]),
        Step("draft", model="sonnet", tools=["write_file"], depends_on=["analyze"]),
        Step("review", model="opus", tools=["read_file"], depends_on=["draft"]),
        Step("revise", model="sonnet", tools=["write_file"], depends_on=["review"]),
        Step("publish", model="haiku", tools=["api_call"], depends_on=["revise"]),
    ]
)

# Configure failure injection
failures = FailureConfig(
    hallucination_rate=0.05,
    refusal_rate=0.02,
    tool_failure_rate=0.03,
    context_overflow_at=100000,
    cascade_propagation=0.8,
)

# Run simulation
sim = Simulator(pipeline, failures, n_simulations=10000)
results = sim.run()

# Compare strategies
comparison = sim.compare_strategies([
    strategies.naive(),
    strategies.retry(max_attempts=3),
    strategies.parallel(n=3, vote="majority"),
    strategies.checkpoint(interval=2),
])

# Output
comparison.print_table()
comparison.plot_pareto("cost", "success_rate")
comparison.plot_failure_heatmap()
comparison.recommend()  # "Use checkpoint(interval=2) — 94% success at 1.4x cost"
```

### Output Example

```
Strategy Comparison (10,000 simulations each):
┌─────────────────────┬──────────┬───────────┬──────────┬────────────┐
│ Strategy            │ Success  │ Avg Cost  │ Avg Time │ Failures   │
├─────────────────────┼──────────┼───────────┼──────────┼────────────┤
│ Naive               │ 73.2%    │ $0.12     │ 4.2s     │ 2,680      │
│ Retry(3)            │ 89.1%    │ $0.18     │ 5.8s     │ 1,090      │
│ Parallel(3)         │ 96.8%    │ $0.36     │ 4.5s     │ 320        │
│ Checkpoint(2)       │ 94.3%    │ $0.21     │ 6.1s     │ 570        │
│ Adaptive            │ 95.7%    │ $0.19     │ 5.3s     │ 430        │
└─────────────────────┴──────────┴───────────┴──────────┴────────────┘

Recommendation: Adaptive strategy (95.7% success, lowest cost among >95% strategies)
```

### Technical Stack

- **Language**: Python 3.11+
- **Simulation**: Custom discrete-event simulator (no heavy deps)
- **Visualization**: `matplotlib` + `rich` (terminal tables)
- **Statistics**: `numpy`, `scipy` for distributions and confidence intervals
- **Serialization**: `pydantic` for pipeline/config schemas

### What Makes This Novel

1. **First "chaos engineering" framework for AI agents** — entirely new category
2. **Quantifies the reliability compounding problem** — gives teams actual numbers
3. **Strategy comparison with cost modeling** — answers "what's the cheapest way to hit 95% reliability?"
4. **Cascading corruption modeling** — models the hardest failure mode (bad output propagating)
5. **Pareto frontier visualization** — cost vs. reliability tradeoff made visual

### Repo Structure

```
cascade/
├── README.md
├── pyproject.toml
├── src/
│   └── cascade/
│       ├── __init__.py
│       ├── pipeline.py         # Pipeline and Step definitions
│       ├── failures.py         # Failure mode models
│       ├── simulator.py        # Monte Carlo simulation engine
│       ├── strategies.py       # Resilience strategies
│       ├── comparator.py       # Strategy comparison
│       ├── report.py           # Report generation + visualization
│       └── stats.py            # Statistical utilities
├── tests/
├── examples/
│   ├── research_pipeline.py
│   ├── coding_pipeline.py
│   └── customer_support.py
└── docs/
    ├── failure_models.md
    ├── strategies.md
    └── interpreting_results.md
```

### Research References

- "5 Production Scaling Challenges for Agentic AI in 2026" (Machine Learning Mastery)
- Netflix Chaos Monkey (inspiration for the approach)
- "Reliability Engineering for AI Agents" (Google DeepMind internal, leaked summary)
- Agent Evaluation Framework 2026 (Galileo AI blog)
- Gartner 2026 predictions on agent production deployment rates
