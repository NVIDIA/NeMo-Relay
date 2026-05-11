<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cursor Gateway Guide

Use this guide to observe Cursor hook lifecycle events with NeMo Flow. The
repository ships a Cursor hook bundle under `integrations/coding-agents/cursor/`
because this integration does not assume an official Cursor plugin package
format.

Cursor GUI or IDE sessions can provide agent, subagent, tool, shell, MCP, file,
and response lifecycle events through `.cursor/hooks.json`. Complete LLM
lifecycle observability additionally requires Cursor model traffic to route
through the gateway if your Cursor build exposes that configuration.

Cursor CLI support must be verified separately with `cursor-agent`. If CLI hooks
do not fire, treat Cursor CLI support as hook-limited and gateway-only where
model routing is configurable.

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-flow run --atif-dir .nemo-flow/atif -- cursor-agent
```

The wrapper infers Cursor from `cursor` or `cursor-agent`, starts a gateway on a
dynamic `127.0.0.1` port, temporarily merges NeMo Flow hook entries into the
project `.cursor/hooks.json`, launches Cursor, and restores the original hook
file after the agent exits.

Inspect what would be launched without starting Cursor:

```bash
nemo-flow run \
  --atif-dir .nemo-flow/atif \
  --dry-run \
  --print \
  -- cursor-agent
```

If a launcher hides the command name, pass the agent explicitly:

```bash
nemo-flow run --agent cursor -- my-cursor-wrapper
```

## Shared Config

Create `.nemo-flow/gateway.toml` for project defaults or
`~/.config/nemo-flow/gateway.toml` for user defaults:

```toml
[session]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true
```

Then run `nemo-flow run --agent cursor` to use the configured command.
User config takes priority over project and global config.

## Persistent Install

Use persistent hooks only when you want Cursor configured outside the wrapper:

```bash
nemo-flow install cursor \
  --scope project \
  --target gui \
  --gateway-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif
```

Then start the gateway manually and point Cursor provider traffic at
`http://127.0.0.1:4040` where Cursor exposes provider base URL configuration.
Hook-only Cursor mode observes agent and tool lifecycle but cannot provide
complete LLM lifecycle. Missing LLM spans are expected when Cursor sends model
traffic directly to the provider or through a remote service.

## Captured Events

Generated Cursor hooks include `sessionStart`, `sessionEnd`, `subagentStart`,
`subagentStop`, `preToolUse`, `postToolUse`, `beforeShellExecution`,
`afterShellExecution`, `beforeMCPExecution`, `afterMCPExecution`, `preCompact`,
and `stop` for scope, tool, and mark events. `beforeSubmitPrompt`,
`afterAgentResponse`, and `afterAgentThought` are retained as private LLM
correlation hints and are not emitted as standalone NeMo Flow events.

Tool events preserve Cursor shell and MCP payloads in metadata and use the
active `subagent.id`, `subagent_id`, or `x-nemo-flow-subagent-id` when present.
The transparent wrapper backs up the project hook file, merges NeMo Flow hook
entries for the run, and restores or removes the temporary file when the agent
exits.

## Smoke Test

Run a small Cursor GUI session that starts an agent and uses one simple tool.
Then check hook forwarding directly:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-cursor","hook_event_name":"sessionStart"}' \
  | NEMO_FLOW_GATEWAY_URL=http://127.0.0.1:4040 nemo-flow hook-forward cursor --fail-closed
```

For Cursor CLI, run an equivalent `cursor-agent` session and verify the gateway
receives hook requests. If no hook requests arrive, document that CLI version as
hook-limited and rely only on gateway observability where provider routing is
available.

## Verify Export

End the Cursor session and confirm ATIF exists:

```bash
ls .nemo-flow/atif
```

The gateway writes `<session-id>.atif.json` on session end. If the file is
missing, confirm Cursor loaded `.cursor/hooks.json`, the gateway binary is on
`PATH`, and `--atif-dir` or `NEMO_FLOW_ATIF_DIR` is configured.

## Troubleshoot LLM Lifecycle

If Cursor hook events appear but LLM spans are missing, provider traffic is not
routed through the gateway. Confirm the active Cursor GUI or CLI mode supports
provider base URL configuration for the model path being used.

If LLM spans exist but attach to the session instead of a subagent, pass
`x-nemo-flow-subagent-id` on gateway requests or include shared
`conversation_id`, `generation_id`, or `request_id` values in both hook payloads
and provider requests.
