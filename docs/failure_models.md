# Failure Models

Cascade models six failure modes that occur in production AI agent pipelines.

## Hallucination

The agent produces plausible but incorrect output. This is the hardest failure
because downstream steps cannot detect that their input is bad.

**Subtypes:**
- Wrong tool arguments
- Fabricated data
- Incorrect reasoning
- Format errors

**Configuration:** `hallucination_rate=0.05`

## Refusal

Safety filters or guardrails block a legitimate action (false positive).

**Configuration:** `refusal_rate=0.02`

## Tool Failure

External APIs return errors, timeouts, or rate limits.

**Configuration:** `tool_failure_rate=0.03, tool_timeout_ms=5000`

## Context Overflow

The agent's context window fills up mid-task, losing important earlier information.

**Configuration:** `context_overflow_at=128000, overflow_behavior="truncate_early"`

## Cascading Corruption

A hallucination in step N produces bad data that is used by steps N+1, N+2, etc.
The error amplifies through the pipeline.

**Configuration:** `cascade_propagation=0.8`

## Latency Spike

An individual step takes much longer than expected, potentially causing timeouts.

**Configuration:** `latency_spike_rate=0.01, spike_multiplier=10`
