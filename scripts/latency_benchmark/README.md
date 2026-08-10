<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Latency Benchmark

Use this opt-in benchmark to measure the local latency that NeMo Relay adds
around OpenAI Responses, Anthropic Messages, Codex hooks, Claude Code hooks,
and Relay process startup. The fixture runs deterministic providers on
loopback, so network and model-service latency do not hide Relay overhead.

## Before You Run

Run all commands from the repository root. Start with the smoke test unless you
are collecting reportable performance results.

The default matrix can write about 25 GiB of temporary ATOF data. Keep at least
30 GiB free in the operating system's temporary directory. The benchmark
removes this data after a normal run.

## Prerequisites

Install the repository development prerequisites, including Rust, Python 3.11
or newer, `uv`, and `just`. The `just` recipe builds the release-mode Relay CLI
before running the benchmark.

## Fixture Layout

The benchmark keeps executable source under `src/`, repeatable TOML fixtures
under `config/`, static coding-agent fixtures under `data/`, and focused unit
tests under `tests/`. Add runtime behavior to `src/` and keep fixed test input
outside executable modules.

Run the fixture's fast unit tests with the following recipe:

```bash
just test-latency-benchmark
```

The recipe runs
`uv run --locked python -m pytest scripts/latency_benchmark/tests`. Keep the
`python -m` form so imports resolve from the repository root.

The `data/mock-codex.*` fixture is a small lifecycle stub used when Relay starts
a configured Codex command in transparent mode. It is not a simulated hook
client. The hooks suite invokes `nemo-relay hook-forward codex` and
`nemo-relay hook-forward claude` directly, so it does not launch either real
coding-agent executable and does not need a separate mock Claude command.

## Run a Smoke Test

Use a small matrix to verify the fixture and exporter paths:

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

Do not use a smoke-test result for performance conclusions. Its sample count
is only large enough to catch functional failures.

## Run the Default Matrix

After you check available disk space, run the default matrix with the following
command:

```bash
just latency-benchmark
```

The default configuration runs the following three suites:

| Suite | What It Measures |
| --- | --- |
| `gateway` | Request latency through direct, minimal Relay, ATOF file exporter, and OTLP exporter paths |
| `hooks` | Full Codex and Claude Code `nemo-relay hook-forward` subprocess wall time |
| `startup` | Cold Relay process launch through a healthy gateway |

During a run, the terminal prints an `[x/y]` indicator after each completed
benchmark test. Each gateway matrix scenario counts as one test. The hooks and
startup suites each count as one test.

### Gateway Suite

The gateway suite sends deterministic OpenAI Responses and Anthropic Messages
requests to a loopback provider. By default, each measurement cycle runs the
same request through four paths in a rotated order:

- `direct` calls the mock provider without Relay.
- `relay-minimal` measures Relay without an exporter and isolates the managed
  gateway pipeline.
- `relay-file` adds the ATOF file exporter.
- `relay-otlp` adds the OTLP exporter.

The suite varies provider protocol, buffered or streaming response mode,
request-content size, and concurrency. `total` measures from request start
until the buffered body is read or the streaming response reaches end of
stream. Streaming scenarios also report `first_content`, which measures from
request start until the first content-delta event arrives.

The primary paired comparisons subtract `direct` from each Relay path. The
`file_exporter_vs_minimal` and `otlp_exporter_vs_minimal` comparisons subtract
minimal Relay to isolate exporter overhead. Pairing measurements from the same
cycle reduces unrelated timing variation.

### Hooks Suite

The hooks suite measures the complete wall time of a new
`nemo-relay hook-forward` subprocess for Codex and Claude Code through minimal,
ATOF file exporter, and OTLP exporter configurations. Its paired comparisons
subtract a `nemo-relay --version` subprocess measured in the same cycle. This
process baseline estimates generic executable startup cost; it is not a no-op
hook.

### Startup Suite

The startup suite measures a cold Relay process from launch until its
`/healthz` endpoint reports ready for minimal, ATOF file exporter, and OTLP
exporter configurations. Its paired comparisons subtract the same
`nemo-relay --version` process baseline to make Relay-specific readiness work
easier to distinguish.

## Configure a Run

The benchmark resolves settings in this order:

1. Defaults from `config/default.toml`.
2. Values from the file passed with `--config`.
3. CLI arguments, which take final precedence.

A custom TOML file can contain only the settings that differ from the
defaults. For example:

```toml
tests = ["gateway"]
providers = ["openai"]
modes = ["streaming"]
samples = 50
warmup = 3
payload_sizes = [4096, 65536]
concurrency = [1, 4]
```

Run the custom configuration with the following command:

```bash
just latency-benchmark --config /path/to/benchmark.toml
```

Override any list from the command line with comma-separated values:

```bash
just latency-benchmark \
  --config /path/to/benchmark.toml \
  --tests gateway,startup \
  --providers openai,anthropic \
  --modes buffered \
  --concurrency 1,4,8
```

List every supported override without running the benchmark:

```bash
just latency-benchmark --help
```

## Benchmark Custom Middleware

Every run includes the minimal Relay (`relay-minimal`), ATOF file exporter
(`relay-file`), and OTLP exporter (`relay-otlp`) variants. Add middleware as an
opt-in variant by pointing the benchmark to a valid Relay plugin configuration.

The fixture includes `config/plugins-pii-redaction.toml` for the simplest
self-contained middleware test. It installs the built-in email detector and
does not require an external service. Run the following small gateway matrix:

```bash
just latency-benchmark \
  --tests gateway \
  --providers openai \
  --modes buffered \
  --payload-sizes 4096 \
  --concurrency 1 \
  --samples 5 \
  --warmup 1 \
  --response-bytes 1024 \
  --middleware pii-redaction=scripts/latency_benchmark/config/plugins-pii-redaction.toml
```

This smoke test verifies that Relay loads and executes the middleware and that
the reports include the custom variant. The deterministic benchmark payload
does not contain an email address, so use this command to test latency plumbing,
not redaction correctness. Increase the sample count before drawing performance
conclusions.

For repeatable custom runs, add one or more `[[middleware]]` tables to a
benchmark TOML file:

```toml
[[middleware]]
name = "pii-redaction"
plugin_config = "./plugins-pii-redaction.toml"
```

The `plugin_config` path is relative to the benchmark TOML file. Run the
configuration with the following command:

```bash
just latency-benchmark --config /path/to/benchmark.toml
```

For a one-off run, use `--middleware NAME=PATH`. Repeat the option to add more
than one variant:

```bash
just latency-benchmark \
  --middleware pii-redaction=/path/to/plugins-pii-redaction.toml \
  --middleware guardrails=/path/to/plugins-guardrails.toml
```

CLI middleware options replace the `[[middleware]]` entries from the benchmark
TOML file. Names must contain lowercase letters, digits, or hyphens and cannot
be `direct`, `minimal`, `file`, or `otlp`. Each custom variant runs across the
selected gateway, hook, and startup suites. Gateway results compare it with
both direct provider calls and `relay-minimal`.

Custom variants increase runtime in proportion to the number of variants. A
custom plugin configuration can also write additional data, so account for its
own storage behavior separately from the default ATOF estimate.

## Read the Results

The command prints a terminal summary and writes both of the following
persistent files:

- `target/benchmark-results/nemo-relay-latency-report.json` contains the
  complete, machine-readable result.
- `target/benchmark-results/nemo-relay-latency-report.html` is a self-contained
  report with interactive gateway graphs, hook and startup graphs, numeric
  tables, metric explanations, and the resolved run environment.

Open the HTML file directly in a browser. It embeds its styles, scripts, and
result data, so it does not require an Internet connection or a web server.
Use the following command to choose another output directory:

```bash
just output_dir=/tmp/relay-benchmarks latency-benchmark
```

Use `--report` to choose a different HTML path without changing the JSON path:

```bash
just latency-benchmark --report /tmp/nemo-relay-latency-report.html
```

The reports use the following statistics:

| Metric | Meaning |
| --- | --- |
| Absolute latency | Complete elapsed wall time for one measured path |
| Paired delta | Left path minus its baseline in the same cycle; positive is slower and small negative values can be measurement noise |
| p50 | Median observation |
| p95 and p99 | Tail percentiles that show slower observations |
| Min and max | Fastest and slowest observed values; these are sensitive to outliers |
| Median 95% CI | Deterministic bootstrap uncertainty interval around the median paired delta, not a range containing 95% of observations |

The exporter-delivery byte and request counts are correctness checks. They
confirm that the ATOF file exporter and OTLP exporter delivered data; they are
not latency metrics. If this validation fails, the command still writes the
JSON and HTML reports, records the messages in `validation_errors`, and then
exits with an error.

When comparing variants, prefer added milliseconds over percentages. Record
the commit, release build, hardware, operating system, matrix, and sample count
with any shared result. Small loopback baselines can make harmless absolute
differences look large as percentages.

## Troubleshoot

Use these checks to diagnose the most common benchmark failures:

- A loopback bind error means the environment must allow local HTTP listeners.
- An exporter-delivery error means the ATOF file exporter or OTLP exporter
  delivered no benchmark events. Rerun a small gateway suite to isolate the
  exporter path; Relay startup failures include their captured log output.
- A validation error names the invalid TOML or CLI value. Gateway samples must
  be at least as large as every requested concurrency value.
- A normal run removes the large temporary ATOF output and its
  `nemo-relay-latency-*` workspace, but a default run still needs at least 30
  GiB of free disk space while it is active. After an abrupt process
  termination, remove any stale benchmark directory from the operating
  system's temporary location.
