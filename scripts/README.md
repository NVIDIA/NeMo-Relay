<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scripts

The canonical build and test surface now lives in the repository `justfile`.
Use `just --list` to discover supported developer workflows.

Keep `scripts/` focused on helpers that are still script-native:

## Top-Level Commands

- `build-docs.sh`: compatibility wrapper around the Fern documentation validation recipe; it regenerates ignored Fern API reference pages before checking the site
- `generate_attributions.sh`: regenerate attribution documents
- `test-install.sh`: Run live GitHub release and local interface checks for the curl-based CLI installer
- `test-install.ps1`: Run live GitHub release and local interface checks for the PowerShell CLI installer
- `test-install-mocks.sh`: Run installer scenarios that require simulated platforms or failures

## Opt-In Coding-Agent E2E Tests

These checks exercise installed coding-agent clients and are intentionally outside the default Rust and CI test suites. Run the recipe that matches an available local client:

- `just test-codex-plugin-e2e`
- `just test-claude-plugin-e2e`
- `just test-hermes-mcp-e2e`

## Opt-In Performance Benchmark

Run `just benchmark-coding-agent-latency` to build the release CLI and compare
direct provider requests with Relay's minimal, local-file, and local-OTLP
configurations. The benchmark also measures full hook subprocess and cold
gateway startup time. It writes structured results under
`target/benchmark-results/` by default and is intentionally outside regular CI.

The defaults live in
`scripts/benchmark_coding_agent_latency/config/default.toml`. Supply a partial
TOML file with `--config`, or override individual values on the command line.
For example, this runs only a small OpenAI gateway matrix:

```bash
just benchmark-coding-agent-latency \
  --tests gateway \
  --providers openai \
  --payload-sizes 4096 \
  --concurrency 1 \
  --samples 10
```

Run `uv run python -m scripts.benchmark_coding_agent_latency --help` to list
all overrides. The three selectable suites are `gateway`, `hooks`, and
`startup`. See
[`benchmark_coding_agent_latency/README.md`](benchmark_coding_agent_latency/README.md)
for the complete human-facing run guide.

## Internal Layout

- `docs/`: Fern reference-generation, migration cleanup, and `docs-website` branch sync helpers. Generated API reference output under `docs/reference/api/*-library-reference/` is ignored and recreated by `just docs`.
- `licensing/`: attribution generation helpers, including license inventory diff scripts
- `lint/`: pre-commit and local lint helpers
- `test-support/`: shared test utilities
