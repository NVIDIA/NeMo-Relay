---
name: run-latency-benchmark
description: Run, configure, troubleshoot, and interpret the NeMo Relay latency benchmark. Use when asked to execute a smoke test or full benchmark matrix, run from a TOML config, include custom middleware, locate generated reports, or explain benchmark metrics and results. Do not use for maintaining or expanding the fixture implementation.
---

# Run the Latency Benchmark

## Use the Run Guide

Read `scripts/latency_benchmark/README.md` completely before running the
benchmark. Treat it as the source of truth for prerequisites, commands,
configuration, middleware, output paths, metrics, and troubleshooting. Use
`just latency-benchmark --help` to confirm the current CLI without building
Relay.

Run commands from the repository root. Do not modify benchmark code, fixtures,
or configuration unless the user separately asks for an implementation change.

## Choose the Run

Choose the smallest run that answers the user's question:

- Use the smoke test for setup checks, troubleshooting, and unspecified test
  requests. Do not use smoke-test results for performance conclusions.
- Use the default matrix only when the user wants statistically meaningful
  performance data. Confirm that the operating system's temporary directory
  has ample free space because ATOF output can reach tens of gigabytes. A normal
  run removes this temporary data.
- Use `--config` for a repeatable custom matrix. Apply CLI overrides only when
  the user requests a one-off change.
- Keep the minimal, ATOF file-exporter, and OTLP variants. Add custom middleware
  with the README's `--middleware NAME=PATH` or `[[middleware]]` workflow.

Run the standard smoke test with:

```bash
just latency-benchmark \
  --tests gateway \
  --providers openai \
  --modes buffered \
  --payload-sizes 4096 \
  --concurrency 1 \
  --samples 5 \
  --warmup 1 \
  --response-bytes 1024
```

Run the complete default matrix with:

```bash
just latency-benchmark
```

## Report the Outcome

Report the exact command, whether it completed, the selected suites and matrix,
and any important warnings. Point the user to the persistent outputs unless
they overrode the result directory:

- `target/benchmark-results/nemo-relay-latency-report.html`
- `target/benchmark-results/nemo-relay-latency-report.json`

Use the HTML report for graphs and explanations. Use the JSON result for
machine-readable analysis. Interpret paired deltas as added milliseconds over
the named baseline, p50 as the median, and p95/p99 as tail observations. Treat
the median 95% confidence interval as uncertainty around the median paired
delta, not as a range containing 95% of samples.

For failures, follow the README troubleshooting section before changing the
command. Call out loopback permission errors, exporter-delivery failures,
invalid matrix values, and stale temporary directories explicitly.
