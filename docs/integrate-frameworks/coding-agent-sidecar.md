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
Claude Code, Cursor, and Hermes without replacing each agent's canonical hook
payload.

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
- `POST /hooks/hermes` accepts Hermes shell hook JSON and returns the empty JSON
  object expected by Hermes hook commands.

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
nemo-flow-sidecar run -- hermes
```

The wrapper infers the agent from the command basename. Use `--agent` when a
launcher or wrapper hides the real agent name:

```bash
nemo-flow-sidecar run --agent codex -- my-codex-wrapper
```

Hermes is different from the other transparent modes: `run --agent hermes`
starts the sidecar and exports the dynamic `NEMO_FLOW_SIDECAR_URL`, but Hermes
shell hooks still need to be installed or otherwise approved in Hermes config.

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

[agents.hermes]
command = "hermes"
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

`hook-forward` can also pass per-session configuration through headers:

- `x-nemo-flow-atif-dir`
- `x-nemo-flow-openinference-endpoint`
- `x-nemo-flow-config-profile`
- `x-nemo-flow-session-metadata`
- `x-nemo-flow-plugin-config`
- `x-nemo-flow-gateway-mode`

The accepted gateway mode values are `hook-only`, `passthrough`, and
`required`. The sidecar records this value as session metadata so downstream
exporters and review tooling can distinguish hook-only traces from sessions
where provider traffic was expected to pass through the gateway.

## Runtime Mapping

The sidecar normalizes vendor hook payloads into private internal events before
calling NeMo Flow APIs.

- Agent start opens a top-level `ScopeType::Agent` scope on a dedicated
  `ScopeStackHandle`.
- Subagent start opens a child `ScopeType::Agent` scope. Subagent stop closes
  that scope when it is still active.
- Tool pre-use starts a NeMo Flow tool span. Tool post-use, denial, or failure
  closes it.
- Prompt, response, agent-thought, and Hermes LLM hooks are retained as
  private correlation hints. They are not emitted as NeMo Flow events.
- Compaction, notification, and unknown hook events become mark events under
  the active session scope.
- Gateway requests emit NeMo Flow LLM start and end events under the active
  session scope. Before each LLM start, the sidecar uses explicit subagent
  headers, pending hints, shared conversation/generation/request identifiers,
  and the previous correlated owner to choose the parent scope.
- LLM responses that contain future tool-use suggestions are retained as
  private tool-call hints. The next matching tool hook can then inherit the
  subagent scope that owned the LLM response, even when the hook payload does
  not include a subagent id.

Gateway requests can provide explicit correlation identifiers with these
headers:

- `x-nemo-flow-session-id`
- `x-nemo-flow-subagent-id`
- `x-nemo-flow-conversation-id`
- `x-nemo-flow-generation-id`
- `x-nemo-flow-request-id`

When those headers are absent, the sidecar also looks for
`conversation_id`/`conversationId`/`conversation.id`,
`generation_id`/`generationId`/`generation.id`, and
`request_id`/`requestId`/`request.id` fields in the provider request body.
Correlation hints expire after five minutes. If the sidecar cannot select one
unambiguous hint, it falls back to the previous LLM owner, then to the only
active subagent, then to the top-level agent scope.

Every gateway LLM event includes `llm_correlation_status` metadata. Possible
values are `explicit`, `single_hint`, `matched_hint`, `sticky_last_owner`,
`active_subagent`, `agent_fallback`, and `ambiguous_fallback`. Matched hints can
also add `llm_correlation_source`, `llm_correlation_subagent_id`,
`llm_correlation_conversation_id`, `llm_correlation_generation_id`,
`llm_correlation_request_id`, and `llm_correlation_agent_type`.

Generated hook bundles subscribe to the events needed for that mapping:

| Agent | Correlation hint hooks | Scope, tool, and mark hooks |
| --- | --- | --- |
| Claude Code | `UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, `Stop` | `SessionStart`, `SessionEnd`, `SubagentStart`, `SubagentStop`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Notification`, `PreCompact` |
| Codex | `UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, `Stop` | `SessionStart`, `SessionEnd`, `SubagentStart`, `SubagentStop`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Notification`, `PreCompact` |
| Cursor | `beforeSubmitPrompt`, `afterAgentResponse`, `afterAgentThought` | `sessionStart`, `sessionEnd`, `subagentStart`, `subagentStop`, `preToolUse`, `postToolUse`, `beforeShellExecution`, `afterShellExecution`, `beforeMCPExecution`, `afterMCPExecution`, `preCompact`, `stop` |
| Hermes | `pre_llm_call`, `post_llm_call` | `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `subagent_start`, `subagent_stop`, `pre_tool_call`, `post_tool_call` |

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
nemo-flow-sidecar install hermes --scope user --target cli --sidecar-url http://127.0.0.1:4040
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
- `--profile` sets `x-nemo-flow-config-profile`.
- `--gateway-mode` sets `x-nemo-flow-gateway-mode`.

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
- [Hermes Sidecar Guide](coding-agent-hermes.md)

Each guide covers transparent run setup, persistent installation, gateway
routing, hook smoke tests, ATIF export verification on session end, and
troubleshooting missing LLM lifecycle data.
