<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay

`nemo-relay-cli` installs the NeMo Relay CLI, the `nemo-relay` binary for local
coding-agent observability. It can configure supported coding-agent hooks, run
agents through an ephemeral gateway, and diagnose local agent and exporter
readiness.

The CLI is a Rust package in this repository, but most users should interact
with the installed `nemo-relay` command rather than link against the crate.

## Why Use It?

- 🧭 **Observe existing coding agents**: Run Claude Code, Codex, or Hermes
  Agent through a local NeMo Relay gateway without changing the agent
  itself.
- 🛠️ **Configure hooks interactively**: Use the setup wizard to write project or
  user config and install the hook files needed by supported agents.
- 📡 **Export local sessions**: Write ATIF trajectory files, ATOF event JSONL
  streams, or OpenInference spans from one shared config model.
- 🩺 **Diagnose the machine**: Check config layers, agent binaries, hook status,
  observability outputs, and shell completions with `nemo-relay doctor`.

## What You Get

- ✅ **`nemo-relay` binary**: The executable installed by the `nemo-relay-cli`
  Cargo package.
- ✅ **First-run setup**: Bare `nemo-relay` launches setup when no config exists,
  then runs doctor once config is present.
- ✅ **Agent shortcuts**: `nemo-relay claude`, `nemo-relay codex`, and
  `nemo-relay hermes` start observed agent runs.
- ✅ **Config-driven launch**: `nemo-relay run` resolves config, environment, and
  CLI overrides for deterministic non-interactive use.
- ✅ **Hook forwarding server**: A local gateway accepts agent hook events and
  provider-shaped OpenAI or Anthropic requests.

## Installation

Install the latest stable CLI release on Linux x86_64, Linux ARM64, or macOS
ARM64:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh
```

The installer downloads the matching GitHub Release binary, verifies its
published SHA-256 checksum, and installs it to `$HOME/.local/bin` without
using `sudo`. Verify the installation with:

```bash
nemo-relay --version
```

To install a specific release, pass its raw SemVer tag. An optional leading
`v` is accepted:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh -s -- <version>
```

`NEMO_RELAY_VERSION` provides the same version selection for automation. A
positional version takes precedence when both are supplied. Run the installer
with `--help` for the full interface, or use `--install-dir` to choose another
destination:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh -s -- --install-dir "$HOME/bin"
```

Rerunning the installer verifies the new download before atomically replacing
an existing `nemo-relay` binary. The installer never invokes `sudo` or edits
your shell configuration. If the destination is not on `PATH`, add it before
running `nemo-relay`.

For unsupported platforms or source-based installation, use Cargo. If you
already have `cargo-binstall`, it can use the same GitHub Release assets:

```bash
cargo install nemo-relay-cli
cargo binstall nemo-relay-cli
```

## Getting Started

Run the first-time setup wizard:

```bash
nemo-relay
```

After setup, inspect local readiness:

```bash
nemo-relay doctor
```

Run a supported agent through the gateway:

```bash
nemo-relay codex
nemo-relay claude -- "summarize this repository"
```

Use `run --dry-run` to inspect resolved config without spawning the agent:

```bash
nemo-relay run --agent codex --dry-run
```

## Configuration

Project config lives at `./.nemo-relay/config.toml`; user config lives at
`~/.config/nemo-relay/config.toml` or `$XDG_CONFIG_HOME/nemo-relay/config.toml`.
The project layer overrides system config, and the user layer overrides the
project layer.

General options are configured through the top-level config. Edit the config with:

```bash
nemo-relay config
```

Observability exporters are configured through the plugin config. Edit the user
plugin config with:

```bash
nemo-relay plugins edit
```

The canonical plugin file is `plugins.toml`; user config lives at
`~/.config/nemo-relay/plugins.toml` or
`$XDG_CONFIG_HOME/nemo-relay/plugins.toml`. Project config lives at
`.nemo-relay/plugins.toml`.

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

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
