<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Flow)](https://github.com/NVIDIA/NeMo-Flow/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Flow/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Flow?color=green)](https://github.com/NVIDIA/NeMo-Flow/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Flow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Flow)
[![PyPI](https://img.shields.io/pypi/v/nemo-flow?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-flow/)
[![npm node](https://img.shields.io/npm/v/nemo-flow-node?label=nemo-flow-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-flow-wasm?label=nemo-flow-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow?label=nemo-flow&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-adaptive?label=nemo-flow-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-cli?label=nemo-flow-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Flow)

# nemo-flow-cli

`nemo-flow-cli` is the experimental coding-agent gateway CLI for NeMo Flow
observability. It installs the `nemo-flow` binary, which can configure
supported coding-agent hooks, run agents through an ephemeral gateway, and
diagnose local agent and exporter readiness.

The CLI is a Rust package in this repository, but most users should interact
with the installed `nemo-flow` command rather than link against the crate.

## Why Use It?

- 🧭 **Observe existing coding agents**: Run Claude Code, Codex, Cursor, or
  Hermes through a local NeMo Flow gateway without changing the agent itself.
- 🛠️ **Configure hooks interactively**: Use the setup wizard to write project or
  user config and install the hook files needed by supported agents.
- 📡 **Export local sessions**: Write ATIF trajectory files, ATOF event JSONL
  streams, or OpenInference spans from one shared config model.
- 🩺 **Diagnose the machine**: Check config layers, agent binaries, hook status,
  observability outputs, and shell completions with `nemo-flow doctor`.

## What You Get

- ✅ **`nemo-flow` binary**: The executable installed by the `nemo-flow-cli`
  Cargo package.
- ✅ **First-run setup**: Bare `nemo-flow` launches setup when no config exists,
  then runs doctor once config is present.
- ✅ **Agent shortcuts**: `nemo-flow claude`, `nemo-flow codex`,
  `nemo-flow cursor`, and `nemo-flow hermes` start observed agent runs.
- ✅ **Config-driven launch**: `nemo-flow run` resolves config, environment, and
  CLI overrides for deterministic non-interactive use.
- ✅ **Hook forwarding server**: A local gateway accepts agent hook events and
  provider-shaped OpenAI or Anthropic requests.

## Installation

Install the CLI:

```bash
cargo install nemo-flow-cli
```

That command installs the binary as:

```bash
nemo-flow --version
```

For local development, build and test the package directly:

```bash
cargo build -p nemo-flow-cli
cargo test -p nemo-flow-cli
```

## Getting Started

Run the first-time setup wizard:

```bash
nemo-flow
```

After setup, inspect local readiness:

```bash
nemo-flow doctor
```

Run a supported agent through the gateway:

```bash
nemo-flow codex
nemo-flow claude -- "summarize this repository"
```

Use `run --dry-run` to inspect resolved config without spawning the agent:

```bash
nemo-flow run --agent codex --dry-run
```

## Benchmark And Eval Runs

Use `nemo-flow run` for non-interactive benchmark tasks so each run gets an
ephemeral gateway and isolated observability artifacts. Prefer absolute
artifact directories when the benchmark harness runs from a different directory
than the task repository.

This example uses Codex CLI because it has a non-interactive `exec` mode. Use
the equivalent non-interactive command for another coding agent when one is
available.

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ART="/tmp/nemo-flow-eval/$RUN_ID"
mkdir -p "$ART/atif" "$ART/atof"

nemo-flow run \
  --atif-dir "$ART/atif" \
  --atof-dir "$ART/atof" \
  -- codex \
  --ask-for-approval never \
  exec \
  --cd /path/to/benchmark/repo \
  --sandbox workspace-write \
  --color never \
  "Run the benchmark task and report the result."

find "$ART" -maxdepth 3 -type f -print
```

The expected output includes one ATIF JSON file and one ATOF JSONL file for
the run. Remote or cloud-hosted tasks are outside the local gateway's LLM
capture boundary unless their model traffic reaches the local gateway.

## Configuration

System config lives at `/etc/nemo-flow/config.toml`, project config lives at
the nearest `./.nemo-flow/config.toml`, and user config lives at
`$XDG_CONFIG_HOME/nemo-flow/config.toml` or `~/.config/nemo-flow/config.toml`.
The CLI loads those layers in that order, then applies `NEMO_FLOW_*`
environment variables and command flags.

Exporter config uses nested per-backend tables:

```toml
[exporters.atif]
dir = "./atif"

[exporters.atof]
dir = "./atof"
mode = "append"
filename_template = "{session_id}.jsonl"

[exporters.openinference]
endpoint = "http://localhost:6006/v1/traces"
```

Use `plugin.toml` in the same system, project, or user config locations when
the gateway should activate process-level plugin config.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
