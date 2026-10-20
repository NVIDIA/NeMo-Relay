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

For version pinning, custom installation directories, uninstalling a
script-installed CLI, verification, troubleshooting, and CLI usage, refer to the
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

User settings live at `~/.config/nemo-relay/config.toml` or
`$XDG_CONFIG_HOME/nemo-relay/config.toml`. Relay loads the user file first and the
system file second. System settings take priority. Pass `--config` to choose a
file in place of the user file. Relay still loads the system file. Relay ignores
repository-local `.nemo-relay/config.toml` files unless you select one explicitly.

### Set Up Agents

Run the following command to choose agents and save their settings in your user
`config.toml`:

```bash
nemo-relay config
```

### Edit Gateway Configuration

Use the config editor to change request limits, provider settings, agent
commands, and log output. Run the following command to open it:

```bash
nemo-relay config edit
```

By default, the editor opens your user `config.toml`. To open a different file,
pass `--config path/to/config.toml` before `config edit`. Both `config edit` and
`config edit --user` use that file. Use `--global` to edit the system file at
`/etc/nemo-relay/config.toml` on Unix or
`%ProgramData%\nemo-relay\config.toml` on Windows.

On Unix, the editor saves system files with mode `0644`, which lets other users
read them. The editor rejects authorization headers and static API keys in
global files. Store those values in a user config or environment variables.

The **Provider upstreams** menu includes URLs, response timeouts, and
credentials. For OpenAI and Anthropic, choose an API key, an authorization
header, or an API key helper. Setting one source removes the provider's other
sources from the selected file. The editor hides credentials and helper commands
and arguments during entry and in **Preview**.

Use **Operational logging** for `[logging]` and **Daemon logging** for
`[daemon.logging]`. Both support standard error (stderr) and file output. Log
files can use inline tables or `[[logging.sinks]]` and `[[daemon.logging.sinks]]`
entries. The editor keeps the format you use.

Select **Preview** to review your changes. Select **Save** to check and save the
file. These checks do not run API key helpers. If a check fails, the file stays
unchanged and the editor stays open. Select **Cancel** to leave without saving.

Refer to [Edit Gateway Configuration](../../docs/nemo-relay-cli/basic-usage.mdx#edit-gateway-configuration)
for menu details and helper setup.

### Per-Request Upstream Credentials

Configure `[upstream.openai_api_key_helper]` or
`[upstream.anthropic_api_key_helper]` in `config.toml`. Set `command` and `args`
to run your helper. Use `timeout_ms` to set its timeout in milliseconds.
The default timeout is 10 seconds.

Relay runs the helper for each outbound request attempt. It reads the token from
stdout and does not cache it. OpenAI uses bearer authentication. Anthropic uses
`x-api-key` by default. Set `auth_scheme = "bearer"` for an Anthropic endpoint
that requires bearer authentication.

Choose one credential source for each provider. You can use a helper, a static
API key, or a custom authorization header. If the helper fails, Relay stops the
request before sending it upstream.

With an OpenAI helper, Relay warns about and ignores `OPENAI_API_KEY` and
`NEMO_RELAY_OPENAI_AUTH_HEADER`. With an Anthropic helper, it warns about and
ignores `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and
`NEMO_RELAY_ANTHROPIC_AUTH_HEADER`. Without a helper, Relay can still use
environment API keys.

Refer to [Generate API Keys Per Request](../../docs/nemo-relay-cli/basic-usage.mdx#generate-api-keys-per-request)
for examples and credential routing rules.

### Edit Plugin Configuration

Use the plugin editor to change `plugins.toml`. These settings include
observability exporters and other Relay components. Run the following command
to open it:

```bash
nemo-relay plugins edit
```

The user plugin file lives at `~/.config/nemo-relay/plugins.toml` or
`$XDG_CONFIG_HOME/nemo-relay/plugins.toml`. Pass `--plugin-config-path` to choose
another file. If you pass `--config path/to/config.toml`, the editor uses
`path/to/plugins.toml` unless you also set `--plugin-config-path`. The default
editor and `--user` both open the selected file. Put path options before
`plugins edit`.

Use `--global` to edit `/etc/nemo-relay/plugins.toml` on Unix or
`%ProgramData%\nemo-relay\plugins.toml` on Windows. Other users can read this
file. On Unix, the editor saves it with mode `0644`. Keep credentials out of
global plugin files. The editor rejects values that the plugin schema marks
as secret.

The editor menu lists built-in plugins and the dynamic plugin references in
the selected file. Dynamic plugins with a JSON Schema provide structured field
controls. Other dynamic plugins use a JSON object editor.

The following example enables ATIF output:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = "./atif"
```

### Plugin Files and Policy

Relay loads the user or explicit plugin file first and the system file second.
An explicit file replaces the user file. System policy still applies. Relay
ignores repository-local `.nemo-relay/plugins.toml` files unless you select one
explicitly. Relay skips missing files. It loads each file only once, even if
multiple symlinks point to it.

With the default host policy, Relay treats dynamic plugins as required and
checks for a valid signature from a trusted public key. You can install and
inspect unsigned plugins. The default policy blocks their activation. An
explicit host policy can change these defaults.

Worker processes receive only the environment variables allowed for startup
and TLS. Native plugins run in the host process. Treat them as trusted host
code after verification.

## Documentation

Refer to [NeMo Relay Documentation](https://docs.nvidia.com/nemo/relay).
