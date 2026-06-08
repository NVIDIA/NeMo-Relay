<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Claude Code Observability

This package contains Claude Code hook entries that forward canonical Claude
Code hook JSON to `nemo-relay` at `/hooks/claude-code`.

Claude Code is the supported Claude integration target. Claude application,
Claude web, and Claude desktop sessions are unsupported unless they expose the
same local hook and gateway controls as Claude Code.

## Files

- `.claude-plugin/plugin.json` describes the Claude Code hook package.
- `hooks/hooks.json` contains hook entries that run
  `nemo-relay hook-forward claude`.

## Captured Events

The bundle forwards `SessionStart`, `SessionEnd`, `SubagentStart`,
`SubagentStop`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`Notification`, and `PreCompact` as scope, tool, or mark events.
`UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, and `Stop`
provide private LLM correlation hints for gateway requests.

Claude Code observability is turn-oriented. A multi-turn session can produce one
root `claude-code-turn` span or ATIF trajectory per user turn. That is expected
when each turn has a real prompt input and assistant output. Known startup
probes, uncorrelatable late stop hooks, and other lifecycle-only noise are
excluded from exported user traces so they do not appear as synthetic `null`,
`user: test`, or `idle_timeout` turns. Startup probes are still logged by the
gateway as internal pre-turn probe bypasses for debugging.

## Transparent Setup

Build or install the gateway binary so `nemo-relay` is on `PATH`.

Run Claude Code through the wrapper:

```bash
nemo-relay run -- claude
```

The wrapper starts a per-invocation gateway on a dynamic localhost port,
creates a temporary Claude plugin directory, passes it with `--plugin-dir`, sets
`ANTHROPIC_BASE_URL` for the launched process, and removes the temporary plugin
when Claude exits.

Inspect the launch without starting Claude Code:

```bash
nemo-relay run \
  --dry-run \
  --print \
  -- claude
```

## Shared Config

Use `.nemo-relay/config.toml` for project defaults or
`~/.config/nemo-relay/config.toml` for user defaults:

```toml
[agents.claude]
command = "claude"
```

Configure observability with `nemo-relay plugins edit --project` or
`.nemo-relay/plugins.toml`:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = ".nemo-relay/atif"
```

Then run:

```bash
nemo-relay run --agent claude
```

## Standalone Gateway

Use the long-running gateway only when you do not want to launch Claude Code
through the wrapper. Start the gateway in one terminal:

```bash
nemo-relay --bind 127.0.0.1:4040
```

Launch Claude Code from another terminal with the gateway environment:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4040
claude
```

Hook events (tool calls, session markers) are only captured when running
through the wrapper, which injects ephemeral hooks per-run.

## Verify

Run a Claude Code session that starts, uses one simple tool, and ends. Confirm
that ATIF was written:

```bash
ls .nemo-relay/atif
```

For a direct endpoint smoke test against a manually started gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-claude","hook_event_name":"SessionStart"}' \
  | NEMO_RELAY_GATEWAY_URL=http://127.0.0.1:4040 nemo-relay hook-forward claude --fail-closed
```

If hooks arrive but LLM spans are missing, confirm the Claude Code process was
started by `nemo-relay run` or has `ANTHROPIC_BASE_URL` set to the
gateway URL.

If LLM spans are present but attached to the top-level agent instead of a
subagent, include `x-nemo-relay-subagent-id` on gateway requests or share
`conversation_id`, `generation_id`, or `request_id` values between hook payloads
and provider requests.

Relay records correlation diagnostics on exported spans instead of guessing
ownership. Inspect `llm_correlation_status`, `llm_correlation_source`, and
`llm_correlation_subagent_id` for LLM routing, and
`tool_correlation_status`, `tool_correlation_source`, and
`tool_correlation_subagent_id` for tool routing. Fallback statuses such as
`agent_fallback` and `ambiguous_fallback` mean Relay kept the span under the
active turn because the hook and gateway payloads did not prove a subagent
owner.

Late `SubagentStop` hooks with no matching `SubagentStart` are diagnostic-only.
When there is no active turn, Relay logs the missing subagent and suppresses the
hook from ATOF, OpenInference, and ATIF so it cannot create a null turn. When an
unknown subagent end arrives during an active turn, Relay may emit a
`subagent_end_without_start` mark under that turn.

Hook events are only available when Claude Code loads this plugin. A standalone
gateway observes Anthropic LLM traffic, but it cannot recover missing prompt,
tool, compaction, notification, or subagent hooks.
