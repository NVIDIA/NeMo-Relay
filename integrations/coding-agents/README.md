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
  `codex_hooks = true`. Use `nemo-flow run` or a gateway provider alias
  for Codex LLM gateway routing.
- `cursor/` installs a Cursor `.cursor/hooks.json` bundle targeting
  `POST /hooks/cursor`.
- Hermes does not require a static bundle in this directory. Use
  `nemo-flow install hermes` to merge hook commands into
  `.hermes/config.yaml`.
- `hermes/` contains a native Hermes Python plugin prototype that writes ATIF
  from Hermes plugin middleware without running the gateway HTTP process.

## Transparent Setup

Build or install the gateway binary so `nemo-flow` is on `PATH`.

Prefer the wrapper. It starts a gateway on a dynamic `127.0.0.1` port, injects
temporary hook and gateway configuration, runs the agent, and shuts the gateway
down when the agent exits.

```bash
nemo-flow run --atif-dir .nemo-flow/atif -- claude
nemo-flow run --atif-dir .nemo-flow/atif -- codex
nemo-flow run --atif-dir .nemo-flow/atif -- cursor-agent
nemo-flow run --atif-dir .nemo-flow/atif -- hermes
```

Use `--agent claude|codex|cursor|hermes` when a wrapper hides the agent
command name. Use `--dry-run --print` to inspect generated config without
launching.

Hermes transparent runs export the dynamic `NEMO_FLOW_GATEWAY_URL`, but Hermes
hooks still need to be installed or approved in Hermes configuration before
they can call the gateway.

Shared TOML config is loaded from `/etc/nemo-flow/config.toml`, then nearest
project `.nemo-flow/config.toml`, then
`$XDG_CONFIG_HOME/nemo-flow/config.toml` or
`~/.config/nemo-flow/config.toml`.

```toml
[observability]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[export.openinference]
endpoint = "http://127.0.0.1:4318/v1/traces"

[agents.codex]
command = "codex"

[agents.hermes]
command = "hermes"
```

## Persistent Setup

Use `install` only when you want persistent hook configuration:

```bash
nemo-flow install claude-code --scope user --target cli --gateway-url http://127.0.0.1:4040
nemo-flow install codex --scope user --target both --gateway-url http://127.0.0.1:4040
nemo-flow install cursor --scope project --target gui --gateway-url http://127.0.0.1:4040
nemo-flow install hermes --scope user --target cli --gateway-url http://127.0.0.1:4040
```

Inspect generated changes before writing:

```bash
nemo-flow install codex \
  --scope user \
  --target both \
  --gateway-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif \
  --dry-run \
  --print
```

The installer backs up existing config files, merges only NeMo Flow hook
entries, and avoids adding duplicate NeMo Flow entries on repeated runs. In
persistent mode you start the gateway yourself and pass `--gateway-url` or set
`NEMO_FLOW_GATEWAY_URL` for hook forwarding.

## Common Options

Static bundles rely on `NEMO_FLOW_GATEWAY_URL` from `nemo-flow run` and
call:

```bash
nemo-flow hook-forward <agent>
```

Persistent installer output includes `--gateway-url` and any selected export or
session options in the generated command.

`hook-forward` reads the canonical hook JSON from standard input, forwards it to
the matching gateway endpoint, and prints the vendor-specific hook response.

Useful wrapper and install options:

- `--atif-dir <path>` writes ATIF trajectories on session end.
- `--openinference-endpoint <url>` exports OpenInference traces.
- `--session-metadata '<json>'` adds structured metadata to the agent begin
  event.
- `--plugin-config '<json>'` records scope-local plugin configuration metadata.
- `--profile <name>` records a configuration profile in session metadata.
- `--gateway-mode hook-only|passthrough|required` records the expected gateway
  behavior in session metadata.
- `--fail-closed` can be added to generated hook commands when the agent should
  block on hook delivery failures. The default is fail-open.

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
agent supports local routing. Persistent installs require you to point the
agent's provider base URL at the gateway manually.

## Verify Export

Run a coding-agent session that starts, uses one tool, and ends. Then confirm
that ATIF was written:

```bash
ls .nemo-flow/atif
```

The gateway writes `<session-id>.atif.json` when it receives a session-end hook
for a session with ATIF configured.
