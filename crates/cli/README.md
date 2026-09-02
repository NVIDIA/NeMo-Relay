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

The CLI is designed for these tasks:

- **Observe existing coding agents**: Run Claude Code, Codex, or pi through a local
  NeMo Relay gateway without changing the agent
  itself.
- **Configure transparent runs interactively**: Use the setup wizard to write
  project or user configuration for supported agents.
- **Export local sessions**: Write ATIF trajectory files, ATOF event JSONL
  streams, or typed OpenTelemetry spans from one shared config model.
- **Diagnose setup readiness**: Check config layers, `plugins.toml` discovery,
  agent binaries, persistent coding-agent integrations, hook status,
  observability outputs, and shell completions with `nemo-relay doctor`.

## What You Get

The CLI provides these capabilities:

- **`nemo-relay` binary**: The executable installed by the `nemo-relay-cli`
  Cargo package.
- **First-run setup**: Bare `nemo-relay` launches setup when no config exists,
  then runs doctor once config is present.
- **Agent shortcuts**: `nemo-relay claude` and `nemo-relay codex` start
  observed agent runs.
- **Config-driven launch**: `nemo-relay run` resolves config, environment, and
  CLI overrides for deterministic non-interactive use.
- **Hook forwarding server**: A local gateway accepts agent hook events and
  provider-shaped OpenAI or Anthropic requests.
- **Persistent agent integration**: `nemo-relay install` configures Codex or
  Claude Code with one generated MCP bootstrap and the host's
  canonical lifecycle hooks.
- **Shared gateway lifecycle**: Every persistent integration launches the same
  host-neutral `nemo-relay mcp` client. Concurrent clients share one native
  gateway on `127.0.0.1:47632`.

## Installation Options

Install the prebuilt CLI from PyPI:

```bash
pip install nemo-relay-cli-bin
```

Install the Python API and matching CLI with the optional extra:

```bash
pip install "nemo-relay[cli]"
```

Build and install the CLI from crates.io with Cargo:

```bash
cargo install nemo-relay-cli
```

Unix curl:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.ps1 | iex
```

For version pinning, custom installation directories, verification,
troubleshooting, and CLI usage, refer to the
[NeMo Relay installation guide](https://docs.nvidia.com/nemo/relay/getting-started/installation).

After installation, verify the binary with:

```bash
nemo-relay --version
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

To troubleshoot a specific configuration file, pass it explicitly. Doctor reports a missing or
invalid file in its configuration checks instead of stopping before diagnostics:

```bash
nemo-relay --config /path/to/config.toml doctor
```

Run a supported agent through the gateway:

```bash
nemo-relay codex
nemo-relay claude -- "summarize this repository"
```

Install persistent integrations for the supported agent CLIs on `PATH`:

```bash
nemo-relay install all
```

Use `run --dry-run` to inspect resolved config without spawning the agent:

```bash
nemo-relay run --agent codex --dry-run
```

## Configuration

User config lives at `~/.config/nemo-relay/config.toml` or
`$XDG_CONFIG_HOME/nemo-relay/config.toml`. Runtime files layer from lowest to
highest precedence as explicit-or-user, then system. An explicit `--config`
replaces the ambient user file without suppressing system configuration.
Repository-local `.nemo-relay/config.toml` files are ignored.

Set up agent entries in the top-level config with:

```bash
nemo-relay config
```

Edit gateway limits, provider upstreams, and operational logging with the
structured user-config editor:

```bash
nemo-relay config edit
```

Use `--global` for system configuration: `/etc/nemo-relay/config.toml` on Unix
or `%ProgramData%\nemo-relay\config.toml` on Windows. Global saves are
system-readable (`0644` on Unix) and reject authorization headers; use the
corresponding environment variables or a user config for credentials.

When the top-level CLI receives `--config path/to/config.toml`, the config
editor uses that exact file as its user target, so the default editor and
`config edit --user` both open it. Use `--global` to edit the system layer.

Observability exporters are configured through the plugin config. Edit the user
plugin config with:

```bash
nemo-relay plugins edit
```

When the top-level CLI receives `--plugin-config-path`, the editor uses that
exact file. Otherwise, `--config path/to/config.toml` makes the editor use the
sibling `path/to/plugins.toml`, matching runtime selection. The explicit file
replaces the user layer, so `--user` keeps that inherited target. `--global`
edits the system layer.

The top-level editor menu contains one entry per supported built-in, followed by
the dynamic plugin references in the selected physical `plugins.toml`. Dynamic
plugins with a manifest-declared JSON Schema provide structured field controls.
Other dynamic plugins use a raw JSON object editor.

At runtime, dynamic plugins that do not have an explicit host-policy override
are classified as required and must carry a valid signature from a configured
trusted public key. Installation and inspection remain available for unsigned
plugins, but Relay refuses to activate them. Worker processes inherit only the
small environment allowlist needed for process startup and TLS. Native plugins
run in-process and must be treated as trusted host code after verification.

The canonical plugin file is `plugins.toml`; user config lives at
`~/.config/nemo-relay/plugins.toml` or
`$XDG_CONFIG_HOME/nemo-relay/plugins.toml`. Use
`nemo-relay plugins edit --global` to edit `/etc/nemo-relay/plugins.toml` on
Unix or `%ProgramData%\nemo-relay\plugins.toml` on Windows. It is
system-readable (`0644` on Unix), so do not store credentials there. The editor
rejects schema-declared secret values in global plugin configuration.

Runtime plugin files layer from lowest to highest precedence as
explicit-or-user, then system. An explicit
`--plugin-config-path`, or a `plugins.toml` beside `--config`, replaces the
ambient XDG user file without suppressing system policy. Repository-local
`.nemo-relay/plugins.toml` files are ignored. Missing
files are skipped, and symlink aliases to one physical file are loaded once.

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
