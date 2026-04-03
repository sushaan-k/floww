# floww

[![CI](https://github.com/sushaan-k/floww/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/floww/actions)
[![PyPI](https://img.shields.io/pypi/v/floww.svg)](https://pypi.org/project/floww/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/floww.svg)](https://pypi.org/project/floww/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Chaos engineering toolkit for AI agent pipelines — know your reliability number before you ship.**

`floww` simulates failure cascades in multi-step agent workflows via Monte Carlo injection. It models hallucination, tool failure, latency spikes, context truncation, and output corruption — then computes end-to-end reliability curves and compares mitigation strategies.

---

## The Problem

You deploy an agent workflow. It works in testing. In production, it quietly fails 40% of the time because steps 3 and 7 have correlated failures nobody modeled. Classical software chaos engineering (Chaos Monkey, Gremlin) doesn't understand LLM-specific failure modes: hallucinated tool calls, soft refusals, corrupted chain-of-thought, probabilistic postcondition violations. There is no Netflix chaos monkey for AI.

## Solution

```python
from cascade import Pipeline, Step, FailureProfile, simulate

pipeline = Pipeline(steps=[
    Step("plan",    model="gpt-4o",       failure=FailureProfile(hallucination=0.05)),
    Step("search",  tool="web_search",    failure=FailureProfile(timeout=0.10, error=0.03)),
    Step("extract", model="gpt-4o-mini",  failure=FailureProfile(hallucination=0.12)),
    Step("write",   model="gpt-4o",       failure=FailureProfile(hallucination=0.04)),
    Step("verify",  tool="code_executor", failure=FailureProfile(error=0.08)),
])

report = simulate(pipeline, runs=10_000)

print(report.summary())
# End-to-end success rate:   58.3%  ← this is your real reliability
# P50 latency:               4.2s
# Most common failure:       Step 3 (extract) hallucination → cascade to Step 4
#
# Best mitigation:  checkpoint after Step 2 → 73.1% (+14.8pp)
# Cost: +$0.003/run

# Compare strategies
report.compare_strategies(["retry", "fallback", "checkpoint", "parallel"])
```

## At a Glance

- **Monte Carlo simulation** — 10k+ rollouts in seconds, not hours
- **LLM-specific failure modes** — hallucination, soft refusal, context loss, output corruption
- **Correlated failures** — model dependency and shared tool contention
- **Strategy comparison** — retry, fallback, checkpoint, parallel execution, human review
- **Cost modeling** — reliability gain vs. extra token spend per mitigation
- **HTML/JSON reports** — shareable reliability analysis for your team

## Install

```bash
pip install floww
```

## Failure Modes

| Mode | Description | Typical Rate |
|---|---|---|
| `hallucination` | Output is confidently wrong, downstream step consumes it | 3–15% |
| `soft_refusal` | Model refuses to complete the step despite valid input | 1–5% |
| `tool_error` | External tool returns an error or invalid response | 2–10% |
| `context_loss` | Critical context truncated in long pipelines | 5–20% |
| `latency_spike` | Step exceeds timeout, forcing fallback or failure | 2–8% |

## Architecture

```
Pipeline
 ├── Step              # unit of work with failure profile
 ├── FailureInjector   # injects failures per Monte Carlo draw
 ├── Simulator         # runs N pipeline rollouts in parallel
 ├── StrategyEvaluator # tests retry / fallback / checkpoint strategies
 └── ReportGenerator   # reliability curves, cost/benefit tables
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐
