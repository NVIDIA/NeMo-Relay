<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cursor Sidecar Guide

Use this guide to observe Cursor hook lifecycle events with NeMo Flow. The
repository ships a Cursor hook bundle under `integrations/coding-agents/cursor/`
because this integration does not assume an official Cursor plugin package
format.

Cursor GUI or IDE sessions can provide agent, subagent, tool, shell, MCP, file,
and response lifecycle events through `.cursor/hooks.json`. Complete LLM
lifecycle observability additionally requires Cursor model traffic to route
through the sidecar gateway if your Cursor build exposes that configuration.

Cursor CLI support must be verified separately with `cursor-agent`. If CLI hooks
do not fire, treat Cursor CLI support as hook-limited and gateway-only where
model routing is configurable.

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- cursor-agent
```

The wrapper infers Cursor from `cursor` or `cursor-agent`, starts a sidecar on a
dynamic `127.0.0.1` port, temporarily merges NeMo Flow hook entries into the
project `.cursor/hooks.json`, launches Cursor, and restores the original hook
file after the agent exits.

Inspect what would be launched without starting Cursor:

```bash
nemo-flow-sidecar run \
  --atif-dir .nemo-flow/atif \
  --dry-run \
  --print \
  -- cursor-agent
```

If a launcher hides the command name, pass the agent explicitly:

```bash
nemo-flow-sidecar run --agent cursor -- my-cursor-wrapper
```

## Shared Config

Create `.nemo-flow/sidecar.toml` for project defaults or
`~/.config/nemo-flow/sidecar.toml` for user defaults:

```toml
[session]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true
```

Then run `nemo-flow-sidecar run --agent cursor` to use the configured command.
User config takes priority over project and global config.

## Persistent Install

Use persistent hooks only when you want Cursor configured outside the wrapper:

```bash
nemo-flow-sidecar install cursor \
  --scope project \
  --target gui \
  --sidecar-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif
```

Then start the sidecar manually and point Cursor provider traffic at
`http://127.0.0.1:4040` where Cursor exposes provider base URL configuration.
Hook-only Cursor mode observes agent and tool lifecycle but cannot provide
complete LLM lifecycle. Missing LLM spans are expected when Cursor sends model
traffic directly to the provider or through a remote service.

## Smoke Test

Run a small Cursor GUI session that starts an agent and uses one simple tool.
Then check hook forwarding directly:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-cursor","hook_event_name":"sessionStart"}' \
  | NEMO_FLOW_SIDECAR_URL=http://127.0.0.1:4040 nemo-flow-sidecar hook-forward cursor --fail-closed
```

For Cursor CLI, run an equivalent `cursor-agent` session and verify the sidecar
receives hook requests. If no hook requests arrive, document that CLI version as
hook-limited and rely only on gateway observability where provider routing is
available.

## Verify Export

End the Cursor session and confirm ATIF exists:

```bash
ls .nemo-flow/atif
```

The sidecar writes `<session-id>.atif.json` on session end. If the file is
missing, confirm Cursor loaded `.cursor/hooks.json`, the sidecar binary is on
`PATH`, and `--atif-dir` or `NEMO_FLOW_ATIF_DIR` is configured.

## Troubleshoot LLM Lifecycle

If Cursor hook events appear but LLM spans are missing, provider traffic is not
routed through the sidecar. Confirm the active Cursor GUI or CLI mode supports
provider base URL configuration for the model path being used.
