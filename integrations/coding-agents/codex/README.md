<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Codex Observability

This package contains Codex hook entries that forward canonical Codex hook JSON
to `nemo-flow-sidecar` at `/hooks/codex`.

Codex CLI is fully supported for local sessions. Codex GUI or app sessions are
supported only when they run locally and honor the same hook/plugin config and
provider routing. Cloud or remote Codex tasks are partial or unsupported for
local sidecar LLM capture.

## Files

- `.codex-plugin/plugin.json` describes the Codex plugin package.
- `hooks/hooks.json` contains hook entries that run
  `nemo-flow-sidecar hook-forward codex`.

## Transparent Setup

Build or install the sidecar binary so `nemo-flow-sidecar` is on `PATH`.

Run Codex through the wrapper:

```bash
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- codex
```

The wrapper starts a per-invocation sidecar on a dynamic localhost port,
enables Codex hooks with CLI config overrides, injects hook commands that use
`NEMO_FLOW_SIDECAR_URL`, and sets the active OpenAI provider `base_url` to the
sidecar URL.

Inspect the launch without starting Codex:

```bash
nemo-flow-sidecar run \
  --atif-dir .nemo-flow/atif \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run \
  --print \
  -- codex
```

## Shared Config

Use `.nemo-flow/sidecar.toml` for project defaults or
`~/.config/nemo-flow/sidecar.toml` for user defaults:

```toml
[session]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[agents.codex]
command = "codex"
```

Then run:

```bash
nemo-flow-sidecar run --agent codex
```

## Persistent Setup

Use persistent hooks only when you do not want to launch Codex through the
wrapper:

```bash
nemo-flow-sidecar install codex \
  --scope user \
  --target both \
  --sidecar-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif
```

Then start the sidecar manually and configure the local Codex provider
`base_url` to `http://127.0.0.1:4040`.

## Verify

Run a Codex session that starts, uses one simple tool, and ends. Confirm that
ATIF was written:

```bash
ls .nemo-flow/atif
```

For a direct endpoint smoke test against a manually started sidecar:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-codex","hook_event_name":"sessionStart"}' \
  | NEMO_FLOW_SIDECAR_URL=http://127.0.0.1:4040 nemo-flow-sidecar hook-forward codex --fail-closed
```

If hooks arrive but LLM spans are missing, confirm Codex was started by
`nemo-flow-sidecar run` or that the active provider `base_url` points to the
sidecar URL.
