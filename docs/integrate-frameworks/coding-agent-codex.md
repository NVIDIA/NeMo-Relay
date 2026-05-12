<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex Gateway Guide

Use this guide to observe local Codex CLI sessions and local Codex GUI or app
sessions that honor the same local config and gateway routing. Cloud or remote
Codex tasks are partial or unsupported for local gateway LLM capture because the
local gateway cannot observe provider traffic that never reaches the machine.

## Requirements

`codex-cli >= 0.129.0`. The gateway uses the `features.hooks` flag and the
`nemo-flow-openai` provider alias, both of which require this version. Earlier
versions either reject the provider override or do not recognize the hooks
feature flag.

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-flow run --atif-dir .nemo-flow/atif -- codex
```

The wrapper infers Codex from `codex`, starts a gateway on a dynamic
`127.0.0.1` port, enables Codex hooks with CLI config overrides, injects hook
commands that use `NEMO_FLOW_GATEWAY_URL`, and points Codex at a temporary
`nemo-flow-openai` provider alias that uses the gateway URL while preserving
Codex's OpenAI auth path.

Inspect what would be launched without starting Codex:

```bash
nemo-flow run \
  --atif-dir .nemo-flow/atif \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run \
  --print \
  -- codex
```

If a launcher hides the command name, pass the agent explicitly:

```bash
nemo-flow run --agent codex -- my-codex-wrapper
```

## Shared Config

Create `.nemo-flow/config.toml` for project defaults or
`~/.config/nemo-flow/config.toml` for user defaults:

```toml
[upstream]
openai_base_url = "https://api.openai.com"

[observability]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.codex]
command = "codex"
```

Then run `nemo-flow run --agent codex` to use the configured command.
User config takes priority over project and global config.

## Standalone Gateway

Use the long-running gateway only when you want Codex running outside the
wrapper:

```bash
NEMO_FLOW_ATIF_DIR=.nemo-flow/atif nemo-flow --bind 127.0.0.1:4040
```

Then configure local Codex to use a gateway provider alias instead of
overriding the reserved built-in `openai` provider:

```toml
model_provider = "nemo-flow-openai"

[model_providers.nemo-flow-openai]
name = "NeMo Flow OpenAI"
base_url = "http://127.0.0.1:4040"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
```

Local Codex GUI or app sessions have the same support level only when they read
the same local hook/plugin config and provider routing. Cloud tasks may still
emit some lifecycle hooks, but complete LLM lifecycle capture requires model
traffic to pass through the gateway.

## Captured Events

Generated Codex hooks include `SessionStart`, `SessionEnd`, `SubagentStart`,
`SubagentStop`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`Notification`, and `PreCompact` for scope, tool, and mark events.
`UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, and `Stop` are
retained as private LLM correlation hints and are not emitted as standalone
NeMo Flow events.

The transparent wrapper passes hook entries as Codex CLI config overrides and
sets `features.hooks=true` for that launched process. Persistent install writes
`.codex/config.toml` with `hooks = true` and merges generated hook entries into
`.codex/hooks.json`. (`features.codex_hooks` is the legacy alias of
`features.hooks`; new docs and configurations should prefer the canonical name.)

## Smoke Test

Run a small Codex prompt that starts a session and uses one simple tool. Then
check hook forwarding directly:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-codex","hook_event_name":"sessionStart"}' \
  | NEMO_FLOW_GATEWAY_URL=http://127.0.0.1:4040 nemo-flow hook-forward codex --fail-closed
```

The response should match Codex hook semantics. For most lifecycle events it is
an empty JSON object.

## Verify Export

End the Codex session and confirm ATIF exists:

```bash
ls .nemo-flow/atif
```

The gateway writes `<session-id>.atif.json` after every conversation turn for
Codex sessions (Codex's hook surface has no `SessionEnd`-equivalent event, so
the gateway uses each per-turn `Stop` hook to snapshot the trajectory; the file
grows cumulatively across turns and the final write reflects the full session).
For agents that do emit a session-end hook, the same file is written once on
session close. If the file is missing, confirm `hooks = true`, hook config
loading, and `--atif-dir` or `NEMO_FLOW_ATIF_DIR`.

## Troubleshoot LLM Lifecycle

If agent/tool events exist but LLM spans are missing, the active Codex provider
is not pointing at the gateway for the active Codex process. If only GUI
sessions are missing spans, confirm the GUI is using local provider
configuration rather than a remote execution path.

If LLM spans exist but attach to the session instead of a subagent, pass
`x-nemo-flow-subagent-id` on gateway requests or include shared
`conversation_id`, `generation_id`, or `request_id` values in both hook payloads
and provider requests.
