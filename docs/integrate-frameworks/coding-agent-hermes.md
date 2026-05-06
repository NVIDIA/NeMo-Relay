<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes Sidecar Guide

Use this guide to observe local Hermes Agent sessions with NeMo Flow through
Hermes shell hooks and the `nemo-flow-sidecar` gateway. This sidecar path is
separate from the Hermes third-party patch set under `patches/hermes-agent/`;
use the sidecar when you want hook forwarding without rebuilding a patched
Hermes checkout.

Hermes shell hooks provide session, subagent, tool, and LLM hint lifecycle
events. Complete LLM request and response observability still requires model
traffic to route through the sidecar gateway.

## Transparent Run

Use the wrapper when you want the sidecar lifetime managed for a local Hermes
process:

```bash
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- hermes
```

The wrapper infers Hermes from `hermes` or `hermes-agent`, starts a sidecar on a
dynamic `127.0.0.1` port, and exports `NEMO_FLOW_SIDECAR_URL` for the launched
process. Hermes hook configuration is not temporary in this mode. Install hooks
first, or configure equivalent Hermes shell hooks, so approved hook commands can
discover the dynamic sidecar URL.

Inspect what would be launched without starting Hermes:

```bash
nemo-flow-sidecar run \
  --atif-dir .nemo-flow/atif \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run \
  --print \
  -- hermes
```

If a launcher hides the command name, pass the agent explicitly:

```bash
nemo-flow-sidecar run --agent hermes -- my-hermes-wrapper
```

## Shared Config

Create `.nemo-flow/sidecar.toml` for project defaults or
`~/.config/nemo-flow/sidecar.toml` for user defaults:

```toml
[session]
atif_dir = ".nemo-flow/atif"
metadata = { team = "agent-observability" }

[export.openinference]
endpoint = "http://127.0.0.1:4318/v1/traces"

[agents.hermes]
command = "hermes"
```

Then run `nemo-flow-sidecar run --agent hermes` to use the configured command.
User config takes priority over project and global config.

## Persistent Install

Use persistent hooks to merge NeMo Flow hook commands into
`~/.hermes/config.yaml` or the project `.hermes/config.yaml`:

```bash
nemo-flow-sidecar install hermes \
  --scope user \
  --target cli \
  --sidecar-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif
```

The installer preserves existing YAML config, appends missing NeMo Flow hook
entries, and backs up the file before writing. The generated Hermes hooks cover
`on_session_start`, `on_session_end`, `on_session_finalize`,
`on_session_reset`, `pre_llm_call`, `post_llm_call`, `pre_tool_call`,
`post_tool_call`, `subagent_start`, and `subagent_stop`.

Hermes hook forwarding prefers `NEMO_FLOW_SIDECAR_URL` when it is set, even if
the installed command also includes `--sidecar-url`. This lets persistent hook
config work with `nemo-flow-sidecar run`, where each run uses a dynamic local
port. Without `NEMO_FLOW_SIDECAR_URL`, the installed `--sidecar-url` is used.

Then start the sidecar manually for persistent mode:

```bash
NEMO_FLOW_ATIF_DIR=.nemo-flow/atif nemo-flow-sidecar --bind 127.0.0.1:4040
```

Point Hermes provider traffic at `http://127.0.0.1:4040` for any provider mode
that exposes a local OpenAI-compatible or Anthropic-compatible base URL.

## Smoke Test

Run a small Hermes session that starts, invokes one tool, and exits. Then check
hook forwarding directly:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-hermes","hook_event_name":"on_session_start"}' \
  | NEMO_FLOW_SIDECAR_URL=http://127.0.0.1:4040 nemo-flow-sidecar hook-forward hermes --fail-closed
```

The response should be `{}`. If Hermes prompts for hook consent, approve the
NeMo Flow hook command interactively or through Hermes configuration before
relying on unattended capture.

## Verify Export

End or finalize the Hermes session and confirm ATIF exists:

```bash
ls .nemo-flow/atif
```

The sidecar writes `<session-id>.atif.json` when it receives
`on_session_finalize` or `on_session_reset`. `on_session_end` is treated as a
per-turn mark and does not close the NeMo Flow session by itself.

## Troubleshoot LLM Lifecycle

If hook events appear but LLM spans are missing, Hermes model traffic is not
routed through the sidecar. If LLM spans exist but attach to the top-level agent
instead of a subagent, include shared identifiers in Hermes hook payloads and
gateway requests, such as `conversation_id`, `generation_id`, `request_id`, or
`x-nemo-flow-subagent-id`.
