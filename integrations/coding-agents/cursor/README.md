<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Cursor Observability

This package is a Cursor hook bundle, not an official Cursor plugin package. It
contains `.cursor/hooks.json` entries that forward canonical Cursor hook JSON to
`nemo-relay` at `/hooks/cursor`.

Cursor GUI or IDE sessions can provide agent, subagent, tool, shell, MCP, file,
and response lifecycle events through `.cursor/hooks.json`. Complete LLM
lifecycle observability additionally requires Cursor model traffic to route
through the gateway if the active Cursor build exposes provider base URL
configuration.

Cursor CLI builds require `.cursor/hooks.json` to set top-level `"version": 1`
and use direct command entries such as
`{"command": "nemo-relay hook-forward cursor", "timeout": 30}`. The nested
`{"matcher": "*", "hooks": [...]}` group shape used by Claude Code and Codex
does not fire in Cursor CLI.

> [!WARNING]
> Cursor CLI hook coverage is narrower than Cursor IDE hook coverage. Current
> headless CLI builds can emit fewer hook events than Cursor IDE sessions. Treat
> missing CLI hook events as a Cursor CLI limitation after `nemo-relay doctor
> cursor` confirms the hook file uses the direct versioned shape.

## Files

- `.cursor/hooks.json` contains hook entries that run
  `nemo-relay hook-forward cursor`.

## Captured Events

The bundle forwards `sessionStart`, `sessionEnd`, `subagentStart`,
`subagentStop`, `preToolUse`, `postToolUse`, `beforeShellExecution`,
`afterShellExecution`, `beforeMCPExecution`, `afterMCPExecution`, `preCompact`,
and `stop` as scope, tool, or mark events. `beforeSubmitPrompt`,
`afterAgentResponse`, and `afterAgentThought` provide private LLM correlation
hints for gateway requests.

Tool events preserve shell and MCP payloads in metadata and attach to
`subagent.id`, `subagent_id`, or `x-nemo-relay-subagent-id` when one is present.

## Transparent Setup

Build or install the gateway binary so `nemo-relay` is on `PATH`.

Run Cursor through the wrapper:

```bash
nemo-relay run -- cursor-agent
```

The wrapper starts a per-invocation gateway on a dynamic localhost port,
temporarily merges NeMo Relay hooks into project `.cursor/hooks.json`, launches
Cursor, and restores or removes the temporary hook file when Cursor exits. The
temporary Cursor hook file is written with top-level `"version": 1` and direct
command entries.

Inspect the launch without starting Cursor:

```bash
nemo-relay run \
  --dry-run \
  --print \
  -- cursor-agent
```

## Shared Config

Use `.nemo-relay/config.toml` for project defaults or
`~/.config/nemo-relay/config.toml` for user defaults:

```toml
[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true
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
nemo-relay run --agent cursor
```

## Standalone Gateway

Use the long-running gateway only when you do not want to launch Cursor
through the wrapper (e.g., the Cursor GUI). Start the gateway manually:

```bash
nemo-relay --bind 127.0.0.1:4040
```

Then point Cursor provider traffic at `http://127.0.0.1:4040` where Cursor
exposes provider base URL configuration. Hook events are only captured when
running through the wrapper.

## Verify

Run a Cursor session that starts, uses one simple tool, and ends. Confirm that
ATIF was written:

```bash
ls .nemo-relay/atif
```

For a direct endpoint smoke test against a manually started gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-cursor","hook_event_name":"sessionStart"}' \
  | NEMO_RELAY_GATEWAY_URL=http://127.0.0.1:4040 nemo-relay hook-forward cursor --fail-closed
```

If Cursor CLI hooks do not fire for the active `cursor-agent` version, treat
that CLI mode as hook-limited after confirming `.cursor/hooks.json` uses direct
versioned entries. User-managed Cursor hook files can be checked with
`nemo-relay doctor cursor`.

If LLM spans are present but attached to the top-level agent instead of a
subagent, include `x-nemo-relay-subagent-id` on gateway requests or share
`conversation_id`, `generation_id`, or `request_id` values between hook payloads
and provider requests.
