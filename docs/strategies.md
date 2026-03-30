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
