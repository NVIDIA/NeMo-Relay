<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Basic Usage

The `nemo-flow` binary observes coding agents that do not expose every
LLM call site directly. It combines agent-specific hook endpoints with a
passthrough LLM gateway so NeMo Flow owns both the agent lifecycle and the model
request lifecycle.

Use the gateway when you need one observability boundary for OpenAI Codex,
Claude Code, Cursor, and Hermes without replacing each agent's canonical hook
payload.

## Hook Endpoints

Each hook endpoint accepts the agent's native hook JSON directly. Do not wrap
the payload in a shared gateway envelope.

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

Route all coding-agent LLM traffic through the gateway when full LLM lifecycle
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

Use the agent shortcuts for no-install local observability. The wrapper starts
a gateway on a dynamic `127.0.0.1` port, injects the resolved hook and gateway
configuration into the launched coding agent, and stops the gateway when the
agent exits.

```bash
nemo-flow codex
nemo-flow claude
nemo-flow cursor
nemo-flow hermes
```

Use `nemo-flow run -- <command>` when you want to launch an explicit command
instead of the built-in shortcut:

```bash
nemo-flow run -- codex
```

If a launcher or wrapper hides the real agent name, set that wrapper as the
configured command and pass `--agent`. The same pattern applies to Claude Code,
Codex, Cursor, and Hermes:

```toml
[agents.codex]
command = "my-codex-wrapper"
```

```bash
nemo-flow run --agent codex
```

Hermes is different from the other transparent modes: `run --agent hermes`
starts the gateway and exports the dynamic `NEMO_FLOW_GATEWAY_URL`, but Hermes
shell hooks still need to be installed or otherwise approved in Hermes config.

Use `--dry-run --print` to inspect the generated hook config, gateway
environment, gateway URL, and final command without launching the agent.

## Shared Configuration

Shared TOML config is optional. The gateway loads defaults, then system config,
then project config, then user config. User config takes priority over system
and project config. CLI flags and environment variables override file config.

Config file locations are:

- `/etc/nemo-flow/config.toml`
- `.nemo-flow/config.toml`
- `$XDG_CONFIG_HOME/nemo-flow/config.toml`
- `~/.config/nemo-flow/config.toml`

Example:

```toml
[upstream]
openai_base_url = "https://api.openai.com/v1"
anthropic_base_url = "https://api.anthropic.com"

[agents.claude]
command = "claude"

[agents.codex]
command = "codex"

[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true

[agents.hermes]
command = "hermes"
```

Observability exporters are configured in `plugins.toml`. Use
`nemo-flow plugins edit` for the user file, `nemo-flow plugins edit --project`
for `.nemo-flow/plugins.toml`, or write the plugin config directly:

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

Transparent runs always bind the managed gateway to `127.0.0.1:0`. The selected
port is discovered by the wrapper and exposed to hooks through
`NEMO_FLOW_GATEWAY_URL`.

Common environment variables for direct gateway server use are:

- `NEMO_FLOW_GATEWAY_BIND`
- `NEMO_FLOW_OPENAI_BASE_URL`
- `NEMO_FLOW_ANTHROPIC_BASE_URL`

Plugin configuration controls process-level Observability exporters. Per-session
configuration controls structured metadata on the top-level agent begin event
and the plugin configuration metadata associated with the session.

`hook-forward` can also pass per-session configuration through headers:

- `x-nemo-flow-config-profile`
- `x-nemo-flow-session-metadata`
- `x-nemo-flow-plugin-config`
- `x-nemo-flow-gateway-mode`

The accepted gateway mode values are `hook-only`, `passthrough`, and
`required`. The gateway records this value as session metadata so downstream
exporters and review tooling can distinguish hook-only traces from sessions
where provider traffic was expected to pass through the gateway.

## Runtime Mapping

The gateway normalizes vendor hook payloads into private internal events before
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
  session scope. Before each LLM start, the gateway uses explicit subagent
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

When those headers are absent, the gateway also looks for
`conversation_id`/`conversationId`/`conversation.id`,
`generation_id`/`generationId`/`generation.id`, and
`request_id`/`requestId`/`request.id` fields in the provider request body.
Correlation hints expire after five minutes. If the gateway cannot select one
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
gateway.

## Hook Forwarding

Hooks generated by the wrapper (Claude/Codex/Cursor ephemeral, Hermes via
setup) invoke `nemo-flow hook-forward <agent>` from stdin. Inside the wrapper
the gateway URL comes from `NEMO_FLOW_GATEWAY_URL` injected on every run;
outside the wrapper (Hermes standalone, IDE-launched Claude/Codex) the hook
command falls back to its embedded `--gateway-url`.

`hook-forward` reads the canonical hook payload from standard input, sends it
to the matching endpoint, and prints the endpoint response. It fails open by
default so observability outages do not block the coding agent. Add
`--fail-closed` only when policy requires hook delivery to block the agent.

Optional flags map to gateway headers:

- `--session-metadata` sets `x-nemo-flow-session-metadata`.
- `--plugin-config` sets `x-nemo-flow-plugin-config`.
- `--profile` sets `x-nemo-flow-config-profile`.
- `--gateway-mode` sets `x-nemo-flow-gateway-mode`.

## Agent Guides

Use the per-agent guide for end-to-end setup, smoke tests, and GUI or
application-mode caveats.

- [Claude Code](claude-code.md)
- [Codex](codex.md)
- [Cursor](cursor.md)
- [Hermes Agent](hermes.md)

Each guide covers transparent run setup, gateway routing, hook smoke tests,
ATIF export verification on session end, and troubleshooting missing LLM
lifecycle data.
