<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Code

Use this guide to observe Claude Code sessions with NeMo Relay. Claude Code is
the supported integration target. The Claude application, Claude web, and Claude
desktop sessions are unsupported unless they expose the same local hook and
gateway controls as Claude Code.

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-relay claude
```

Pass Claude Code arguments after `--`:

```bash
nemo-relay claude -- "summarize this repository"
```

This shortcut is equivalent to `nemo-relay run -- claude`. The wrapper starts a
gateway on a dynamic `127.0.0.1` port, creates a temporary Claude plugin
directory with NeMo Relay hooks, passes that plugin with `--plugin-dir`, and
sets `ANTHROPIC_BASE_URL` to the gateway URL for the launched process.

Inspect what would be launched without starting Claude Code:

```bash
nemo-relay run \
  --dry-run \
  --print \
  -- claude
```

## Shared Config

Create `.nemo-relay/config.toml` for project defaults or
`~/.config/nemo-relay/config.toml` for user defaults:

```toml
[agents.claude]
command = "claude"
```

Then configure observability with `nemo-relay plugins edit --project` or
`.nemo-relay/plugins.toml`:

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

Run `nemo-relay run --agent claude` to use the configured command and plugin
config. User config takes priority over project and system config.

## Standalone Gateway

Use the long-running gateway only when you want Claude Code running outside the
wrapper (e.g., already configured by an IDE):

```bash
nemo-relay --bind 127.0.0.1:4040
```

Launch Claude Code from another terminal with the gateway environment:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4040
claude
```

The gateway forwards Anthropic `/v1/messages`, `/v1/messages/count_tokens`, and
model routes without rewriting provider JSON. Hook events (tool calls, session
markers) are only captured when running through `nemo-relay claude` or
`nemo-relay run --agent claude`, which inject ephemeral hooks into the launched
process.

## Captured Events

Generated Claude Code hooks include `SessionStart`, `SessionEnd`,
`SubagentStart`, `SubagentStop`, `PreToolUse`, `PostToolUse`,
`PostToolUseFailure`, `Notification`, and `PreCompact` for scope, tool, and
mark events. `UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, and
`Stop` are retained as private LLM correlation hints and are not emitted as
standalone NeMo Relay events.

Tool hooks preserve canonical fields such as `tool_use_id`, `tool_name`,
`tool_input`, `error`, `duration_ms`, and `is_interrupt`. Subagent hooks use
`agent_id` as the subagent identifier and preserve `agent_type` in metadata.

## Smoke Test

Run a small Claude Code prompt that starts a session and uses one simple tool.
Then check that hook forwarding reaches the gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-claude","hook_event_name":"SessionStart"}' \
  | NEMO_RELAY_GATEWAY_URL=http://127.0.0.1:4040 nemo-relay hook-forward claude --fail-closed
```

The response should be valid Claude Code hook JSON. For most lifecycle events it
is an allow/continue response.

## Verify Export

End the Claude Code session and confirm that session-end closed the NeMo Relay
agent scope and wrote Agent Trajectory Interchange Format (ATIF):

```bash
ls .nemo-relay/atif
```

The gateway exports `<session-id>.atif.json` on session end. If no file appears,
confirm that `SessionEnd` hooks fire, `plugins.toml` enables the ATIF exporter,
and the gateway process can write to the configured directory.

## Troubleshoot LLM Lifecycle

Missing hooks usually means Claude Code did not load the local hook config or
the `nemo-relay` binary is not on `PATH`.

Missing LLM spans with present hook spans means Anthropic traffic is not routed
through the gateway. Verify `ANTHROPIC_BASE_URL` in the Claude Code process
environment and confirm that requests hit `/v1/messages`.

If LLM spans exist but attach to the session instead of a subagent, pass
`x-nemo-relay-subagent-id` on gateway requests or include shared
`conversation_id`, `generation_id`, or `request_id` values in both hook payloads
and provider requests.
