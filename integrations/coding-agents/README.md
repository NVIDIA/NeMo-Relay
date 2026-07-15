<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Coding-Agent Observability Integrations

This directory contains hook integration bundles for coding agents that should
be observed by `nemo-relay`.

The gateway combines two observability paths:

- Agent lifecycle hooks for sessions, prompts, subagents, tool calls,
  compaction, responses, and stop events.
- A passthrough LLM gateway for OpenAI-compatible and Anthropic-compatible
  provider traffic.

Hook integrations preserve each coding agent's canonical hook payload. They do
not wrap the payload in a shared NeMo Relay envelope. Gateway-specific settings
travel through the transparent wrapper, hook command arguments, HTTP headers,
environment variables, or shared TOML config.

## Packages

- `claude-code/` is a Claude Code plugin package. The
  `nemo-relay install claude-code` command installs hook entries targeting
  `POST /hooks/claude-code` through `nemo-relay` on `PATH`.
- `codex/` is a Codex plugin package. `nemo-relay install codex` creates the
  marketplace, installs the plugin, enables `features.hooks = true`, and
  configures a local `nemo-relay-openai` provider alias. Codex plugin delivery
  uses hook-supervised lazy sidecar startup only, with no wrapper, user-level daemon,
  login item, launchd agent, systemd user service, scheduled task, or persistent
  supervisor.
- Hermes does not require a static bundle in this directory. The setup wizard
  (`nemo-relay config`) merges hook commands into `.hermes/config.yaml` when
  hermes is selected.

## Transparent Setup

Build or install the gateway binary so `nemo-relay` is on `PATH`.

Prefer the wrapper. It starts a gateway on a dynamic `127.0.0.1` port, injects
temporary hook and gateway configuration, runs the agent, and shuts the gateway
down when the agent exits.

```bash
nemo-relay run -- claude
nemo-relay run -- codex
nemo-relay run -- hermes
```

Use `--agent claude|codex|hermes` when a wrapper hides the agent
command name. Use `--dry-run --print` to inspect generated config without
launching.

Use `nemo-relay doctor` to inspect environment, config, agent commands, hook
readiness, observability outputs, and shell completions. Scope the report to one
agent when troubleshooting launch readiness:

```bash
nemo-relay doctor
nemo-relay doctor codex
nemo-relay doctor hermes --json
```

The command is read-only: it reports missing ATIF directories, hook files, and
agent commands instead of creating or patching them.

## Plugin Installation

The Claude Code and Codex plugins are installed by the `nemo-relay` CLI. The
CLI must already be installed and discoverable on `$PATH` or `%PATH%`; no
separate npm installer, release bundle download, or plugin-local Relay binary is
required.

Claude Code can start the sidecar from plugin hooks or helper commands and route
model traffic by setting `ANTHROPIC_BASE_URL` to the sidecar URL.

Codex does not use a daemon in plugin mode. The installed Codex hooks call the
`nemo-relay plugin-shim hook codex` command. The shim then checks `/healthz`, starts
the local `nemo-relay` sidecar, if needed, waits briefly for readiness, and then
forwards the hook payload. Codex model traffic is routed through the stable
provider alias at `http://127.0.0.1:47632`.

Install the local host marketplaces with:

```bash
nemo-relay install claude-code
nemo-relay install codex
nemo-relay install all
```

`nemo-relay install` writes local marketplace files, registers the selected host
plugin, and performs the required host provider and hook setup. Use
`nemo-relay uninstall <host>` to roll back and `nemo-relay doctor --plugin
<host>` to check an installed plugin.

If you are using Codex, add this repository as a marketplace for source/dev
discovery:

```bash
codex plugin marketplace add NVIDIA/NeMo-Relay
codex plugin add nemo-relay-plugin@nemo-relay
```

That path relies on `nemo-relay` being available on `PATH`; source plugin hooks
invoke `nemo-relay plugin-shim hook codex` directly.

Use the source marketplace path for discovery or manifest validation. Remove
the source-installed Codex plugin before running `nemo-relay install codex`;
keeping both active can forward the same Codex hook twice.

Claude Code users can add this repository as a marketplace the same way:

```bash
claude plugin marketplace add NVIDIA/NeMo-Relay \
  --sparse .claude-plugin integrations/coding-agents/claude-code
claude plugin install nemo-relay-plugin@nemo-relay --scope user
```

That path reads `.claude-plugin/marketplace.json` from the repository. Source
plugin hooks invoke `nemo-relay plugin-shim hook claude` directly. Use
`nemo-relay install claude-code` for the complete provider-routing setup.

Hermes transparent runs export the dynamic `NEMO_RELAY_GATEWAY_URL`, but Hermes
hooks must already be present in `.hermes/config.yaml` before they can call the
gateway. The setup wizard (`nemo-relay config`) writes that file for you when
you select hermes.

Shared TOML config is loaded from `/etc/nemo-relay/config.toml`, then nearest
project `.nemo-relay/config.toml`, then
`$XDG_CONFIG_HOME/nemo-relay/config.toml` or
`~/.config/nemo-relay/config.toml`.

```toml
[agents.codex]
command = "codex"

[agents.hermes]
command = "hermes"
```

Observability exporters are configured in `plugins.toml`. Run
`nemo-relay plugins edit --project` to create `.nemo-relay/plugins.toml`, or
write the plugin config directly:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = ".nemo-relay/atif"

[components.config.openinference]
enabled = true
endpoint = "http://127.0.0.1:4318/v1/traces"
```

During setup or launch, invalid shared TOML, malformed plugin config, unsupported exporter settings,
or unavailable exporter features will fail closed. The
wrapper does not start the coding agent against a configuration that cannot be
parsed, validated, or activated. Once the gateway and agent are running,
exporter delivery failures follow the observability plugin policy: application
work continues while the failing ATOF, ATIF, OpenTelemetry, or OpenInference
destination records, logs, or reports the failure.

## Hook Forwarding

The transparent wrapper hooks call `nemo-relay hook-forward <agent>` with the
canonical hook payload on stdin. The wrapper injects `NEMO_RELAY_GATEWAY_URL` so
the same hook command reaches the ephemeral per-run gateway; hermes hooks fall
back to an embedded `--gateway-url` when running outside the wrapper.

Claude Code and Codex plugin hooks call `nemo-relay plugin-shim hook <agent>`.
The plugin shim ensures the local sidecar is reachable, then forwards the hook
payload to the plugin sidecar endpoint.

Since hook forwarding fails open by default, gateway or sidecar outages do not
block the coding agent. The hook command exits successfully after logging the
forwarding problem, so the host agent can continue even though that hook
payload may be missing from telemetry. For wrapper-generated `hook-forward`
commands, add
`--fail-closed` when policy requires hook delivery to block the agent. For
plugin shim hooks, set `NEMO_RELAY_FAIL_CLOSED=1` in the hook execution
environment. In that mode, forwarding failures return a non-zero hook command
status to the host.

Useful wrapper options:

- `--session-metadata '<json>'` adds structured metadata to the agent begin
  event. For example, `--session-metadata '{"user_id":"alice"}'` exposes the
  string as `user.id` on OTLP trace roots.
- `--profile <name>` records a configuration profile in session metadata.
- `--gateway-mode hook-only|passthrough|required` records the expected gateway
  behavior in session metadata.

## LLM Gateway

Complete LLM lifecycle observability requires model traffic to pass through the
gateway. Hook-only mode observes agent, subagent, and tool lifecycle, but it
cannot observe provider request and response lifecycle when the coding agent
sends model traffic directly to an upstream provider or remote service.

The gateway exposes these passthrough routes:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`

Transparent runs configure provider routing automatically where the launched
agent supports local routing. Standalone gateway mode requires you to point the
agent's provider base URL at the gateway manually.

## Verify Export

Run a coding-agent session that starts, uses one tool, and ends. Then confirm
that ATIF was written:

```bash
ls .nemo-relay/atif
```

The gateway writes `<session-id>.atif.json` when it receives a session-end hook
for a session with ATIF configured.
