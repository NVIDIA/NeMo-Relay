<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cursor

Use this guide to observe Cursor hook lifecycle events with NeMo Flow. The
repository ships a Cursor hook bundle under `integrations/coding-agents/cursor/`
because this integration does not assume an official Cursor plugin package
format.

Cursor GUI or IDE sessions can provide agent, subagent, tool, shell, MCP, file,
and response lifecycle events through `.cursor/hooks.json`. Complete LLM
lifecycle observability additionally requires Cursor model traffic to route
through the gateway if your Cursor build exposes that configuration.

Cursor CLI support must be verified separately with `cursor-agent`. Current
Cursor CLI builds require `.cursor/hooks.json` to set top-level `"version": 1`
and use direct command entries such as
`{"command": "nemo-flow hook-forward cursor", "timeout": 30}`. The nested
`{"matcher": "*", "hooks": [...]}` group shape used by Claude Code and Codex
does not fire in Cursor CLI. If CLI hooks still do not fire with direct
versioned entries, treat that Cursor CLI version as hook-limited and
gateway-only where model routing is configurable.

```{warning}
Cursor CLI hook coverage is not the same as Cursor IDE hook coverage. Current
headless CLI builds can emit fewer hook events than Cursor IDE sessions. Treat
missing CLI hook events as a Cursor CLI limitation after `nemo-flow doctor
cursor` confirms the hook file uses the direct versioned shape.
```

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-flow cursor
```

Pass Cursor arguments after `--`:

```bash
nemo-flow cursor -- agent --resume <session-id>
```

This shortcut is equivalent to `nemo-flow run -- cursor-agent`. The wrapper
starts a gateway on a dynamic `127.0.0.1` port, temporarily merges NeMo Flow
hook entries into the project `.cursor/hooks.json`, launches Cursor, and
restores the original hook file after the agent exits. The temporary Cursor hook
file is written with top-level `"version": 1` and direct command entries.

Inspect what would be launched without starting Cursor:

```bash
nemo-flow run \
  --dry-run \
  --print \
  -- cursor-agent
```

## Shared Config

Create `.nemo-flow/config.toml` for project defaults or
`~/.config/nemo-flow/config.toml` for user defaults:

```toml
[agents.cursor]
command = "cursor-agent"
patch_restore_hooks = true
```

Then configure observability with `nemo-flow plugins edit --project` or
`.nemo-flow/plugins.toml`:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = ".nemo-flow/atif"
```

Run `nemo-flow run --agent cursor` to use the configured command and plugin
config. User config takes priority over project and system config.

## Standalone Gateway

Use the long-running gateway only when you want Cursor running outside the
wrapper (e.g., the Cursor GUI). Start the gateway manually:

```bash
nemo-flow --bind 127.0.0.1:4040
```

Then point Cursor provider traffic at `http://127.0.0.1:4040` wherever Cursor
exposes provider base URL configuration. Without the wrapper, hook events are
not captured — Cursor GUI mode only emits LLM lifecycle as traffic passes
through the gateway. Missing LLM spans are expected when Cursor sends model
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
receives hook requests. If no hook requests arrive, confirm `.cursor/hooks.json`
uses top-level `"version": 1` with direct command entries, then document that
CLI version as hook-limited and rely only on gateway observability where
provider routing is available.

## Verify Export

End the Cursor session and confirm ATIF exists:

```bash
ls .nemo-flow/atif
```

The gateway writes `<session-id>.atif.json` on session end. If the file is
missing, confirm Cursor loaded `.cursor/hooks.json`, the gateway binary is on
`PATH`, `--atif-dir` or `NEMO_FLOW_ATIF_DIR` is configured, `plugins.toml`
enables the ATIF exporter with a writable `output_directory`, and user-managed
Cursor hooks pass `nemo-flow doctor cursor`.

## Troubleshoot LLM Lifecycle

If Cursor hook events appear but LLM spans are missing, provider traffic is not
routed through the gateway. Confirm the active Cursor GUI or CLI mode supports
provider base URL configuration for the model path being used.

If LLM spans exist but attach to the session instead of a subagent, pass
`x-nemo-flow-subagent-id` on gateway requests or include shared
`conversation_id`, `generation_id`, or `request_id` values in both hook payloads
and provider requests.
