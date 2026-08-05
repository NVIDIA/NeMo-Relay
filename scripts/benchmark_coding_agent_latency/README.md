<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Coding-Agent Latency Benchmark

Use this opt-in benchmark to measure the local latency that NeMo Relay adds
around OpenAI Responses, Anthropic Messages, Codex hooks, Claude Code hooks,
and Relay process startup. The fixture runs deterministic providers on
loopback, so network and model-service latency do not hide Relay overhead.

Run all commands from the repository root. The default matrix is intentionally
large and its file-exporter scenarios can temporarily write tens of gigabytes
of ATOF data. Start with the smoke test unless you are collecting reportable
performance results.

## Prerequisites

Install the repository development prerequisites, including Rust, Python 3.11
or newer, `uv`, and `just`. The `just` recipe builds the release-mode Relay CLI
before running the benchmark.

## Run a Smoke Test

Use a small matrix to verify the fixture and exporter paths:

```bash
just benchmark-coding-agent-latency \
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
just benchmark-coding-agent-latency
```

The default configuration runs three suites:

| Suite | What It Measures |
| --- | --- |
| `gateway` | Direct loopback calls compared with minimal, ATOF file, and OTLP Relay gateways |
| `hooks` | Codex and Claude Code `hook-forward` subprocess wall time |
| `startup` | Cold Relay process startup through gateway readiness |

The gateway suite covers OpenAI and Anthropic, buffered and streaming
responses, multiple request sizes, and multiple concurrency levels.

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
just benchmark-coding-agent-latency --config /path/to/benchmark.toml
```

Override any list from the command line with comma-separated values:

```bash
just benchmark-coding-agent-latency \
  --config /path/to/benchmark.toml \
  --tests gateway,startup \
  --providers openai,anthropic \
  --modes buffered \
  --concurrency 1,4,8
```

List every supported override without running the benchmark:

```bash
uv run python -m scripts.benchmark_coding_agent_latency --help
```

## Read the Results

The command prints a terminal summary and writes
`target/benchmark-results/coding-agent-latency.json`. Use the following command
to choose another directory:

```bash
just output_dir=/tmp/relay-benchmarks benchmark-coding-agent-latency
```

The JSON report records the resolved matrix, environment, absolute latency,
paired latency differences, and exporter-delivery counts. Gateway results
include total latency; streaming results also include time to first content.
Summaries include p50, p95, p99, and a bootstrap 95% confidence interval for
the median difference.

When comparing variants, prefer added milliseconds over percentages. Record
the commit, release build, hardware, operating system, matrix, and sample count
with any shared result. Small loopback baselines can make harmless absolute
differences look large as percentages.

## Troubleshoot

- A loopback bind error means the environment must allow local HTTP listeners.
- An exporter-delivery error means the ATOF file or OTLP receiver observed no
  benchmark events. Rerun a small gateway suite to isolate the exporter path;
  Relay startup failures include their captured log output.
- A validation error names the invalid TOML or CLI value. Gateway samples must
  be at least as large as every requested concurrency value.
- An interrupted run removes its temporary workspace, but a default run still
  needs enough free disk space while it is active.
