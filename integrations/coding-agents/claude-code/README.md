<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Claude Code Observability

This package contains Claude Code hook entries that forward canonical Claude
Code hook JSON to `nemo-flow` at `/hooks/claude-code`.

Claude Code is the supported Claude integration target. Claude application,
Claude web, and Claude desktop sessions are unsupported unless they expose the
same local hook and gateway controls as Claude Code.

## Files

- `.claude-plugin/plugin.json` describes the Claude Code hook package.
- `hooks/hooks.json` contains hook entries that run
  `nemo-flow hook-forward claude`.

## Captured Events

The bundle forwards `SessionStart`, `SessionEnd`, `SubagentStart`,
`SubagentStop`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`Notification`, and `PreCompact` as scope, tool, or mark events.
`UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, and `Stop`
provide private LLM correlation hints for gateway requests.

## Transparent Setup

Build or install the gateway binary so `nemo-flow` is on `PATH`.

Run Claude Code through the wrapper:

```bash
nemo-flow run --atif-dir .nemo-flow/atif -- claude
```

The wrapper starts a per-invocation gateway on a dynamic localhost port,
creates a temporary Claude plugin directory, passes it with `--plugin-dir`, sets
`ANTHROPIC_BASE_URL` for the launched process, and removes the temporary plugin
when Claude exits.

Inspect the launch without starting Claude Code:

```bash
nemo-flow run \
  --atif-dir .nemo-flow/atif \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run \
  --print \
  -- claude
```

## Shared Config

Use `.nemo-flow/config.toml` for project defaults or
`~/.config/nemo-flow/config.toml` for user defaults:

```toml
[observability]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.claude]
command = "claude"
```

Then run:

```bash
nemo-flow run --agent claude
```

## Standalone Gateway

Use the long-running gateway only when you do not want to launch Claude Code
through the wrapper. Start the gateway in one terminal:

```bash
NEMO_FLOW_ATIF_DIR=.nemo-flow/atif nemo-flow --bind 127.0.0.1:4040
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
ls .nemo-flow/atif
```

For a direct endpoint smoke test against a manually started gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-claude","hook_event_name":"SessionStart"}' \
  | NEMO_FLOW_GATEWAY_URL=http://127.0.0.1:4040 nemo-flow hook-forward claude --fail-closed
```

If hooks arrive but LLM spans are missing, confirm the Claude Code process was
started by `nemo-flow run` or has `ANTHROPIC_BASE_URL` set to the
gateway URL.

If LLM spans are present but attached to the top-level agent instead of a
subagent, include `x-nemo-flow-subagent-id` on gateway requests or share
`conversation_id`, `generation_id`, or `request_id` values between hook payloads
and provider requests.
