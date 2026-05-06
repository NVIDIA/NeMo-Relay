<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Coding-Agent Observability Integrations

This directory contains hook integration bundles for coding agents that should
be observed by `nemo-flow-sidecar`.

The sidecar combines two observability paths:

- Agent lifecycle hooks for sessions, prompts, subagents, tool calls,
  compaction, responses, and stop events.
- A passthrough LLM gateway for OpenAI-compatible and Anthropic-compatible
  provider traffic.

Hook integrations preserve each coding agent's canonical hook payload. They do
not wrap the payload in a shared NeMo Flow envelope. Sidecar-specific settings
travel through the transparent wrapper, hook command arguments, HTTP headers,
environment variables, or shared TOML config.

## Packages

- `claude-code/` installs Claude Code hook entries targeting
  `POST /hooks/claude-code`.
- `codex/` installs Codex hook entries targeting `POST /hooks/codex` and enables
  `codex_hooks = true`.
- `cursor/` installs a Cursor `.cursor/hooks.json` bundle targeting
  `POST /hooks/cursor`.
- Hermes does not require a static bundle in this directory. Use
  `nemo-flow-sidecar install hermes` to merge hook commands into
  `.hermes/config.yaml`.

## Transparent Setup

Build or install the sidecar binary so `nemo-flow-sidecar` is on `PATH`.

Prefer the wrapper. It starts a sidecar on a dynamic `127.0.0.1` port, injects
temporary hook and gateway configuration, runs the agent, and shuts the sidecar
down when the agent exits.

```bash
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- claude
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- codex
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- cursor-agent
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- hermes
```

Use `--agent claude-code|codex|cursor|hermes` when a wrapper hides the agent
command name. Use `--dry-run --print` to inspect generated config without
launching.

Hermes transparent runs export the dynamic `NEMO_FLOW_SIDECAR_URL`, but Hermes
hooks still need to be installed or approved in Hermes configuration before
they can call the sidecar.

Shared TOML config is loaded from `/etc/nemo-flow/sidecar.toml`, then nearest
project `.nemo-flow/sidecar.toml`, then
`$XDG_CONFIG_HOME/nemo-flow/sidecar.toml` or
`~/.config/nemo-flow/sidecar.toml`.

```toml
[session]
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
nemo-flow-sidecar install claude-code --scope user --target cli --sidecar-url http://127.0.0.1:4040
nemo-flow-sidecar install codex --scope user --target both --sidecar-url http://127.0.0.1:4040
nemo-flow-sidecar install cursor --scope project --target gui --sidecar-url http://127.0.0.1:4040
nemo-flow-sidecar install hermes --scope user --target cli --sidecar-url http://127.0.0.1:4040
```

Inspect generated changes before writing:

```bash
nemo-flow-sidecar install codex \
  --scope user \
  --target both \
  --sidecar-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif \
  --dry-run \
  --print
```

The installer backs up existing config files, merges only NeMo Flow hook
entries, and avoids adding duplicate NeMo Flow entries on repeated runs. In
persistent mode you start the sidecar yourself and pass `--sidecar-url` or set
`NEMO_FLOW_SIDECAR_URL` for hook forwarding.

## Common Options

Static bundles rely on `NEMO_FLOW_SIDECAR_URL` from `nemo-flow-sidecar run` and
call:

```bash
nemo-flow-sidecar hook-forward <agent>
```

Persistent installer output includes `--sidecar-url` and any selected export or
session options in the generated command.

`hook-forward` reads the canonical hook JSON from standard input, forwards it to
the matching sidecar endpoint, and prints the vendor-specific hook response.

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
sidecar. Hook-only mode observes agent, subagent, and tool lifecycle, but it
cannot observe provider request and response lifecycle when the coding agent
sends model traffic directly to an upstream provider or remote service.

The sidecar exposes these passthrough routes:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`

Transparent runs configure provider routing automatically where the launched
agent supports local routing. Persistent installs require you to point the
agent's provider base URL at the sidecar manually.

## Verify Export

Run a coding-agent session that starts, uses one tool, and ends. Then confirm
that ATIF was written:

```bash
ls .nemo-flow/atif
```

The sidecar writes `<session-id>.atif.json` when it receives a session-end hook
for a session with ATIF configured.
