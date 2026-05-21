# Resilience Strategies

Cascade supports seven built-in resilience strategies that can be applied to
any pipeline simulation.

## Naive

No retry, fail fast. This is the baseline for comparison.

```python
strategies.naive()
```

## Retry

Simple retry up to K attempts per step.

```python
strategies.retry(max_attempts=3)
```

## Fallback

Try models in order. If the primary model fails, fall back to alternatives.

```python
strategies.fallback(models=["sonnet", "haiku"])
```

## Parallel Redundancy

Run N agents in parallel and reconcile results via voting.

```python
strategies.parallel(n=3, vote="majority")  # or "unanimous", "any"
```

## Checkpoint + Rollback

Checkpoint pipeline state every N steps. On failure, rollback to the last
checkpoint and re-execute.

```python
strategies.checkpoint(interval=5)
```

## Human-in-the-Loop

Insert human verification at specific step indices. Humans catch errors
with configurable accuracy.

```python
strategies.human_in_loop(at_steps=[5, 10, 15], accuracy=0.95)
```

## Adaptive

Start with simple retry, then escalate to a stronger strategy after
repeated failures.

```python
strategies.adaptive(
    escalation_threshold=2,
    escalation_strategy="parallel",
)
```

## CLI strategy specs

The `cascade` CLI accepts configurable strategy specs anywhere a strategy name is
accepted. Bare names use the defaults above, while colon parameters tune the
strategy:

| Spec | Meaning |
| --- | --- |
| `naive` | Fail fast with no mitigation |
| `retry:5` | Retry each step up to five attempts |
| `fallback:opus+sonnet+haiku` | Try fallback models in the listed order |
| `parallel:5:any` | Run five parallel agents and pass if any succeeds |
| `checkpoint:3` | Checkpoint every three steps |
| `human:0+2:0.9` | Human review at zero-based steps 0 and 2 with 90% accuracy |
| `adaptive:1:fallback` | Escalate to fallback after one repeated failure |

For comparisons, separate strategy specs with commas:

```bash
cascade compare pipeline.json \
  --strategies naive,retry:4,parallel:5:any,checkpoint:3,human:1+2:0.8
```
