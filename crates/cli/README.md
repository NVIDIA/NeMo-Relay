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

`nemo-flow-cli` is the coding-agent gateway CLI for NeMo Flow observability.
It installs the `nemo-flow` binary, which can configure supported coding-agent
hooks, run agents through an ephemeral gateway, and diagnose local agent and
exporter readiness.

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

## Configuration

Project config lives at `./.nemo-flow/config.toml`; user config lives at
`~/.config/nemo-flow/config.toml` or `$XDG_CONFIG_HOME/nemo-flow/config.toml`.
The project layer overrides system config, and the user layer overrides the
project layer.

Observability exporters are configured through the plugin config. Edit the user
plugin config with:

```bash
nemo-flow plugins edit
```

The canonical plugin file is `plugins.toml`; user config lives at
`~/.config/nemo-flow/plugins.toml` or
`$XDG_CONFIG_HOME/nemo-flow/plugins.toml`. Project config lives at
`.nemo-flow/plugins.toml`.

Minimal ATIF example:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = "./atif"
```

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
