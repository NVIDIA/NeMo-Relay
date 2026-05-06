<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex Sidecar Guide

Use this guide to observe local Codex CLI sessions and local Codex GUI or app
sessions that honor the same local config and gateway routing. Cloud or remote
Codex tasks are partial or unsupported for local sidecar LLM capture because the
local sidecar cannot observe provider traffic that never reaches the machine.

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- codex
```

The wrapper infers Codex from `codex`, starts a sidecar on a dynamic
`127.0.0.1` port, enables Codex hooks with CLI config overrides, injects hook
commands that use `NEMO_FLOW_SIDECAR_URL`, and sets the active OpenAI provider
`base_url` to the sidecar URL.

Inspect what would be launched without starting Codex:

```bash
nemo-flow-sidecar run \
  --atif-dir .nemo-flow/atif \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run \
  --print \
  -- codex
```

If a launcher hides the command name, pass the agent explicitly:

```bash
nemo-flow-sidecar run --agent codex -- my-codex-wrapper
```

## Shared Config

Create `.nemo-flow/sidecar.toml` for project defaults or
`~/.config/nemo-flow/sidecar.toml` for user defaults:

```toml
[server]
openai_base_url = "https://api.openai.com"

[session]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.codex]
command = "codex"
```

Then run `nemo-flow-sidecar run --agent codex` to use the configured command.
User config takes priority over project and global config.

## Persistent Install

Use persistent hooks only when you want Codex configured outside the wrapper:

```bash
nemo-flow-sidecar install codex \
  --scope user \
  --target both \
  --sidecar-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif
```

Then start the sidecar manually and configure the local Codex provider
`base_url` to `http://127.0.0.1:4040`. Local Codex GUI or app sessions have the
same support level only when they read the same local hook/plugin config and
provider routing. Cloud tasks may still emit some lifecycle hooks, but complete
LLM lifecycle capture requires model traffic to pass through the sidecar.

## Smoke Test

Run a small Codex prompt that starts a session and uses one simple tool. Then
check hook forwarding directly:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-codex","hook_event_name":"sessionStart"}' \
  | NEMO_FLOW_SIDECAR_URL=http://127.0.0.1:4040 nemo-flow-sidecar hook-forward codex --fail-closed
```

The response should match Codex hook semantics. For most lifecycle events it is
an empty JSON object.

## Verify Export

End the Codex session and confirm ATIF exists:

```bash
ls .nemo-flow/atif
```

The sidecar writes `<session-id>.atif.json` on session end. If the file is
missing, confirm `codex_hooks = true`, hook config loading, and `--atif-dir` or
`NEMO_FLOW_ATIF_DIR`.

## Troubleshoot LLM Lifecycle

If agent/tool events exist but LLM spans are missing, the provider `base_url` is
not pointing at the sidecar for the active Codex process. If only GUI sessions
are missing spans, confirm the GUI is using local provider configuration rather
than a remote execution path.
