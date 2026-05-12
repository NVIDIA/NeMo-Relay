<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Cursor Observability

This package is a Cursor hook bundle, not an official Cursor plugin package. It
contains `.cursor/hooks.json` entries that forward canonical Cursor hook JSON to
`nemo-flow` at `/hooks/cursor`.

Cursor GUI or IDE sessions can provide agent, subagent, tool, shell, MCP, file,
and response lifecycle events through `.cursor/hooks.json`. Complete LLM
lifecycle observability additionally requires Cursor model traffic to route
through the gateway if the active Cursor build exposes provider base URL
configuration.

## Files

- `.cursor/hooks.json` contains hook entries that run
  `nemo-flow hook-forward cursor`.

## Captured Events

The bundle forwards `sessionStart`, `sessionEnd`, `subagentStart`,
`subagentStop`, `preToolUse`, `postToolUse`, `beforeShellExecution`,
`afterShellExecution`, `beforeMCPExecution`, `afterMCPExecution`, `preCompact`,
and `stop` as scope, tool, or mark events. `beforeSubmitPrompt`,
`afterAgentResponse`, and `afterAgentThought` provide private LLM correlation
hints for gateway requests.

Tool events preserve shell and MCP payloads in metadata and attach to
`subagent.id`, `subagent_id`, or `x-nemo-flow-subagent-id` when one is present.

## Transparent Setup

Build or install the gateway binary so `nemo-flow` is on `PATH`.

Run Cursor through the wrapper:

```bash
nemo-flow run --atif-dir .nemo-flow/atif -- cursor-agent
```

The wrapper starts a per-invocation gateway on a dynamic localhost port,
temporarily merges NeMo Flow hooks into project `.cursor/hooks.json`, launches
Cursor, and restores or removes the temporary hook file when Cursor exits.

Inspect the launch without starting Cursor:

```bash
nemo-flow run \
  --atif-dir .nemo-flow/atif \
  --dry-run \
  --print \
  -- cursor-agent
```

## Shared Config

Use `.nemo-flow/config.toml` for project defaults or
`~/.config/nemo-flow/config.toml` for user defaults:

```toml
[observability]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true
```

Then run:

```bash
nemo-flow run --agent cursor
```

## Standalone Gateway

Use the long-running gateway only when you do not want to launch Cursor
through the wrapper (e.g., the Cursor GUI). Start the gateway manually:

```bash
NEMO_FLOW_ATIF_DIR=.nemo-flow/atif nemo-flow --bind 127.0.0.1:4040
```

Then point Cursor provider traffic at `http://127.0.0.1:4040` where Cursor
exposes provider base URL configuration. Hook events are only captured when
running through the wrapper.

## Verify

Run a Cursor session that starts, uses one simple tool, and ends. Confirm that
ATIF was written:

```bash
ls .nemo-flow/atif
```

For a direct endpoint smoke test against a manually started gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-cursor","hook_event_name":"sessionStart"}' \
  | NEMO_FLOW_GATEWAY_URL=http://127.0.0.1:4040 nemo-flow hook-forward cursor --fail-closed
```

If Cursor CLI hooks do not fire for the active `cursor-agent` version, treat
that CLI mode as hook-limited and rely on gateway observability where provider
routing is available.

If LLM spans are present but attached to the top-level agent instead of a
subagent, include `x-nemo-flow-subagent-id` on gateway requests or share
`conversation_id`, `generation_id`, or `request_id` values between hook payloads
and provider requests.
