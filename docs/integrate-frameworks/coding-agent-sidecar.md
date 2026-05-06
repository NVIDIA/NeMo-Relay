<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Guide: Coding-Agent Gateway Sidecar

The `nemo-flow-sidecar` binary observes coding agents that do not expose every
LLM call site directly. It combines agent-specific hook endpoints with a
passthrough LLM gateway so NeMo Flow owns both the agent lifecycle and the model
request lifecycle.

Use the sidecar when you need one observability boundary for OpenAI Codex,
Claude Code, and Cursor without replacing each agent's canonical hook payload.

## Hook Endpoints

Each hook endpoint accepts the agent's native hook JSON directly. Do not wrap
the payload in a shared sidecar envelope.

- `POST /hooks/codex` accepts Codex hook JSON and returns the Codex-compatible
  hook response object.
- `POST /hooks/claude-code` accepts Claude Code hook JSON and returns
  Claude-compatible fields such as `continue` and permission decisions when the
  hook event supports them.
- `POST /hooks/cursor` accepts Cursor hook JSON and returns Cursor-compatible
  fields such as `continue`, `permission`, `user_message`, and `agent_message`
  when the hook event supports them.

The adapters preserve vendor fields such as session IDs, working directories,
transcript paths, model names, tool payloads, shell payloads, MCP payloads, file
payloads, user identity, and subagent metadata in NeMo Flow event metadata.

## Gateway Routes

Route all coding-agent LLM traffic through the sidecar when full LLM lifecycle
observability is required.

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`

The gateway forwards raw provider JSON without rewriting OpenAI or Anthropic
payload schemas. It removes only hop-by-hop transport headers, forwards
streaming responses as streams, and emits NeMo Flow LLM start and end events
under the active session scope.

## Transparent Run

Use `nemo-flow-sidecar run` for no-install local observability. The wrapper
starts a sidecar on a dynamic `127.0.0.1` port, injects the resolved hook and
gateway configuration into the launched coding agent, and stops the sidecar
when the agent exits.

```bash
nemo-flow-sidecar run -- codex
nemo-flow-sidecar run -- claude
nemo-flow-sidecar run -- cursor-agent
```

The wrapper infers the agent from the command basename. Use `--agent` when a
launcher or wrapper hides the real agent name:

```bash
nemo-flow-sidecar run --agent codex -- my-codex-wrapper
```

Use `--dry-run --print` to inspect the generated hook config, gateway
environment, sidecar URL, and final command without launching the agent.

## Shared Configuration

Shared TOML config is optional. The sidecar loads defaults, then global config,
then project config, then user config. User config takes priority over global
and project config. CLI flags and environment variables override file config.

Config file locations are:

- `/etc/nemo-flow/sidecar.toml`
- `.nemo-flow/sidecar.toml`
- `$XDG_CONFIG_HOME/nemo-flow/sidecar.toml`
- `~/.config/nemo-flow/sidecar.toml`

Example:

```toml
[server]
openai_base_url = "https://api.openai.com"
anthropic_base_url = "https://api.anthropic.com"

[session]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }
plugin_config = { components = [] }

[export.openinference]
endpoint = "http://127.0.0.1:4318/v1/traces"

[agents.claude-code]
command = "claude"

[agents.codex]
command = "codex"

[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true
```

Transparent runs always bind the managed sidecar to `127.0.0.1:0`. The selected
port is discovered by the wrapper and exposed to hooks through
`NEMO_FLOW_SIDECAR_URL`.

Common environment variables for direct sidecar server use are:

- `NEMO_FLOW_SIDECAR_BIND`
- `NEMO_FLOW_OPENAI_BASE_URL`
- `NEMO_FLOW_ANTHROPIC_BASE_URL`
- `NEMO_FLOW_OPENINFERENCE_ENDPOINT`
- `NEMO_FLOW_ATIF_DIR`

Per-session configuration controls the scope-local OpenInference subscriber,
the ATIF exporter, structured metadata on the top-level agent begin event, and
the plugin configuration metadata associated with the session.

## Runtime Mapping

The sidecar normalizes vendor hook payloads into private internal events before
calling NeMo Flow APIs.

- Agent start opens a top-level `ScopeType::Agent` scope on a dedicated
  `ScopeStackHandle`.
- Subagent start opens a child `ScopeType::Agent` scope. Subagent stop closes
  that scope when it is still active.
- Tool pre-use starts a NeMo Flow tool span. Tool post-use, denial, or failure
  closes it.
- Prompt, response, compaction, notification, and unknown hook events become
  mark events under the active session scope.
- Gateway requests emit NeMo Flow LLM start and end events under the active
  session scope.

Cursor hook-only mode observes agent, subagent, and tool lifecycle. To observe
Cursor LLM lifecycle completely, configure Cursor model traffic to use the
sidecar gateway.

## Persistent Install

The repository also includes installable integration packages under
`integrations/coding-agents/`. Use `install` when you want stable hook config
instead of the transparent wrapper.

```bash
nemo-flow-sidecar install claude-code --scope user --target cli --sidecar-url http://127.0.0.1:4040
nemo-flow-sidecar install codex --scope user --target both --sidecar-url http://127.0.0.1:4040
nemo-flow-sidecar install cursor --scope project --target gui --sidecar-url http://127.0.0.1:4040
```

Use `--dry-run` to see which files would be changed. Use `--print` to print the
merged file contents. Existing config files are backed up before the installer
writes replacement files, and generated hook entries are appended only when the
same NeMo Flow entry is not already present.

Common install options become hook-forwarding command arguments and sidecar
headers:

- `--atif-dir` sets `x-nemo-flow-atif-dir`.
- `--openinference-endpoint` sets `x-nemo-flow-openinference-endpoint`.
- `--session-metadata` sets `x-nemo-flow-session-metadata`.
- `--plugin-config` sets `x-nemo-flow-plugin-config`.

Static integration bundles rely on the wrapper-provided
`NEMO_FLOW_SIDECAR_URL` and run:

```bash
nemo-flow-sidecar hook-forward <agent>
```

Persistent installer output embeds `--sidecar-url` and any selected export or
session options directly in the generated hook command.

`hook-forward` reads the canonical hook payload from standard input, sends it to
the matching endpoint, and prints the endpoint response. In transparent runs it
discovers the sidecar through `NEMO_FLOW_SIDECAR_URL`; in persistent installs
you can still pass `--sidecar-url`. It fails open by default so observability
outages do not block the coding agent. Add `--fail-closed` only when policy
requires hook delivery to block the agent.

## Agent Guides

Use the per-agent guide for end-to-end setup, smoke tests, and GUI or
application-mode caveats.

- [Claude Code Sidecar Guide](coding-agent-claude-code.md)
- [Codex Sidecar Guide](coding-agent-codex.md)
- [Cursor Sidecar Guide](coding-agent-cursor.md)

Each guide covers transparent run setup, persistent installation, gateway
routing, hook smoke tests, ATIF export verification on session end, and
troubleshooting missing LLM lifecycle data.
