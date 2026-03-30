# Interpreting Results

## Key Metrics

### Success Rate

The fraction of simulation runs where the entire pipeline completed without
a fatal failure. A 95% confidence interval is provided via the Wilson score
method.

### Average Cost

Mean total cost (USD) across all simulation runs. Includes the cost of retries
and redundant executions. Compare against the baseline (Naive strategy) to
understand the cost of reliability.

### Average Latency

Mean total wall-clock time across all runs. Includes base step latency,
latency spikes, and retry delays. Parallel strategies may show lower latency
than retry strategies because parallel agents run concurrently.

### Failure Counts

Total number of each failure type across all runs. Use the failure heatmap
to identify which failure modes dominate under each strategy.

### Mean Steps to Failure

Average step index where the first failure occurs. Lower values indicate
that failures happen early in the pipeline.

### Recovery Rate

Fraction of runs that experienced a failure but ultimately succeeded
(via retry, rollback, etc.). Only meaningful for strategies that support
recovery.

## Strategy Comparison Table

The comparison table shows all metrics side by side. Use it to identify
the strategy that best fits your requirements:

- **Minimize cost:** Look for the cheapest strategy above your success
  rate threshold.
- **Maximize reliability:** Look for the highest success rate within
  your cost budget.
- **Pareto optimal:** Use `comparison.plot_pareto()` to visualize the
  cost-reliability tradeoff frontier.

## Recommendation Logic

`comparison.recommend()` selects the cheapest strategy that exceeds 95%
success rate. If no strategy exceeds 95%, it recommends the one with the
highest success rate.

## Confidence Intervals

All point estimates include confidence intervals:

- **Success rate:** Wilson score interval (accurate even near 0 or 1)
- **Mean cost/latency:** t-distribution interval

With 10,000 simulations, the 95% CI for a 90% success rate is roughly
89.4% to 90.6%.
