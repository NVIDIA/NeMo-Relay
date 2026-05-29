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

- `claude-code/` installs Claude Code hook entries targeting
  `POST /hooks/claude-code`.
- `codex/` installs Codex hook entries targeting `POST /hooks/codex` and enables
  `features.hooks = true`. Use `nemo-relay run` or a gateway provider alias
  for Codex LLM gateway routing.
- `cursor/` installs a Cursor `.cursor/hooks.json` bundle targeting
  `POST /hooks/cursor`.
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
nemo-relay run -- cursor-agent
nemo-relay run -- hermes
```

Use `--agent claude|codex|cursor|hermes` when a wrapper hides the agent
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

## Hook Forwarding

Hooks call `nemo-relay hook-forward <agent>` with the canonical hook payload on
stdin. The wrapper injects `NEMO_RELAY_GATEWAY_URL` so the same hook command
reaches the ephemeral per-run gateway; hermes hooks fall back to an embedded
`--gateway-url` when running outside the wrapper.

`hook-forward` prints the vendor-specific response and fails open by default
(observability outages do not block the coding agent). Add `--fail-closed` to
generated hook commands when policy requires hook delivery to block the agent.

Useful wrapper options:

- `--session-metadata '<json>'` adds structured metadata to the agent begin
  event.
- `--plugin-config '<json>'` records scope-local plugin configuration metadata.
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
