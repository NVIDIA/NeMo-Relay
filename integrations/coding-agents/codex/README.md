<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Codex Observability

This package contains Codex hook entries that forward canonical Codex hook JSON
to `nemo-relay` at `/hooks/codex`.

Codex CLI is fully supported for local sessions. Codex GUI or app sessions are
supported only when they run locally and honor the same hook/plugin config and
provider routing. Cloud or remote Codex tasks are partial or unsupported for
local gateway LLM capture.

Requires `codex-cli >= 0.129.0` (introduced the `features.hooks` flag and the
provider alias surface the gateway relies on).

## Files

- `.codex-plugin/plugin.json` describes the Codex plugin package.
- `hooks/hooks.json` contains hook entries that run
  `nemo-relay hook-forward codex`.

## Captured Events

The bundle forwards `SessionStart`, `SessionEnd`, `SubagentStart`,
`SubagentStop`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`Notification`, and `PreCompact` as scope, tool, or mark events.
`UserPromptSubmit`, `AfterAgentResponse`, `AfterAgentThought`, and `Stop`
provide private LLM correlation hints for gateway requests.

Transparent setup injects these hooks with CLI config overrides. Persistent
setup writes `features.hooks = true` in `.codex/config.toml` and merges the
hook entries into `.codex/hooks.json`.

## Transparent Setup

Build or install the gateway binary so `nemo-relay` is on `PATH`.

Run Codex through the wrapper:

```bash
nemo-relay run -- codex
```

The wrapper starts a per-invocation gateway on a dynamic localhost port,
enables Codex hooks with CLI config overrides, injects hook commands that use
`NEMO_RELAY_GATEWAY_URL`, and points Codex at a temporary `nemo-relay-openai`
provider alias that uses the gateway URL while preserving Codex's OpenAI auth
path.

Inspect the launch without starting Codex:

```bash
nemo-relay run \
  --dry-run \
  --print \
  -- codex
```

## Shared Config

Use `.nemo-relay/config.toml` for project defaults or
`~/.config/nemo-relay/config.toml` for user defaults:

```toml
[agents.codex]
command = "codex"
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
nemo-relay run --agent codex
```

## Standalone Gateway

Use the long-running gateway only when you do not want to launch Codex through
the wrapper. Start the gateway manually:

```bash
nemo-relay --bind 127.0.0.1:4040
```

Then configure local Codex to use a gateway provider alias instead of
overriding the reserved built-in `openai` provider:

```toml
model_provider = "nemo-relay-openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay OpenAI"
base_url = "http://127.0.0.1:4040"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
```

## Verify

Run a Codex session that starts, uses one simple tool, and ends. Confirm that
ATIF was written:

```bash
ls .nemo-relay/atif
```

For a direct endpoint smoke test against a manually started gateway:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-codex","hook_event_name":"sessionStart"}' \
  | NEMO_RELAY_GATEWAY_URL=http://127.0.0.1:4040 nemo-relay hook-forward codex --fail-closed
```

If hooks arrive but LLM spans are missing, confirm Codex was started by
`nemo-relay run` or that the active provider points to the gateway URL.

If LLM spans are present but attached to the top-level agent instead of a
subagent, include `x-nemo-relay-subagent-id` on gateway requests or share
`conversation_id`, `generation_id`, or `request_id` values between hook payloads
and provider requests.
