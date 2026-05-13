<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Coding-Agent Observability Integrations

This directory contains hook integration bundles for coding agents that should
be observed by `nemo-flow`.

The gateway combines two observability paths:

- Agent lifecycle hooks for sessions, prompts, subagents, tool calls,
  compaction, responses, and stop events.
- A passthrough LLM gateway for OpenAI-compatible and Anthropic-compatible
  provider traffic.

Hook integrations preserve each coding agent's canonical hook payload. They do
not wrap the payload in a shared NeMo Flow envelope. Gateway-specific settings
travel through the transparent wrapper, hook command arguments, HTTP headers,
environment variables, or shared TOML config.

## Packages

- `claude-code/` installs Claude Code hook entries targeting
  `POST /hooks/claude-code`.
- `codex/` installs Codex hook entries targeting `POST /hooks/codex` and enables
  `[features].hooks = true`. Use `nemo-flow run` or a gateway provider alias
  for Codex LLM gateway routing.
- `cursor/` installs a Cursor `.cursor/hooks.json` bundle targeting
  `POST /hooks/cursor`.
- Hermes does not require a static bundle in this directory. The setup wizard
  (`nemo-flow config`) merges hook commands into `.hermes/config.yaml` when
  hermes is selected.
- `hermes/` contains a native Hermes Python plugin prototype that writes ATIF
  from Hermes plugin middleware without running the gateway HTTP process.

## Transparent Setup

Build or install the gateway binary so `nemo-flow` is on `PATH`.

Prefer the wrapper. It starts a gateway on a dynamic `127.0.0.1` port, injects
temporary hook and gateway configuration, runs the agent, and shuts the gateway
down when the agent exits.

```bash
nemo-flow run -- claude
nemo-flow run -- codex
nemo-flow run -- cursor-agent
nemo-flow run -- hermes
```

Use `--agent claude|codex|cursor|hermes` when a wrapper hides the agent
command name. Use `--dry-run --print` to inspect generated config without
launching.

Use `nemo-flow doctor` to inspect environment, config, agent commands, hook
readiness, observability outputs, and shell completions. Scope the report to one
agent when troubleshooting launch readiness:

```bash
nemo-flow doctor
nemo-flow doctor codex
nemo-flow doctor hermes --json
```

The command is read-only: it reports missing ATIF directories, hook files, and
agent commands instead of creating or patching them.

Hermes transparent runs export the dynamic `NEMO_FLOW_GATEWAY_URL`, but Hermes
hooks must already be present in `.hermes/config.yaml` before they can call the
gateway. The setup wizard (`nemo-flow config`) writes that file for you when
you select hermes.

Shared TOML config is loaded from `/etc/nemo-flow/config.toml`, then nearest
project `.nemo-flow/config.toml`, then
`$XDG_CONFIG_HOME/nemo-flow/config.toml` or
`~/.config/nemo-flow/config.toml`.

```toml
[agents.codex]
command = "codex"

[agents.hermes]
command = "hermes"
```

Observability exporters are configured in `plugins.toml`. Run
`nemo-flow plugins edit --project` to create `.nemo-flow/plugins.toml`, or
write the plugin config directly:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = ".nemo-flow/atif"

[components.config.openinference]
enabled = true
endpoint = "http://127.0.0.1:4318/v1/traces"
```

## Hook Forwarding

Hooks call `nemo-flow hook-forward <agent>` with the canonical hook payload on
stdin. The wrapper injects `NEMO_FLOW_GATEWAY_URL` so the same hook command
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
ls .nemo-flow/atif
```

The gateway writes `<session-id>.atif.json` when it receives a session-end hook
for a session with ATIF configured.
