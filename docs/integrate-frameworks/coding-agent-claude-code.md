<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Code Sidecar Guide

Use this guide to observe Claude Code sessions with NeMo Flow. Claude Code is
the supported integration target. The Claude application, Claude web, and Claude
desktop sessions are unsupported unless they expose the same local hook and
gateway controls as Claude Code.

## Transparent Run

Use the wrapper for no-install local observability:

```bash
nemo-flow-sidecar run --atif-dir .nemo-flow/atif -- claude
```

The wrapper infers Claude Code from `claude`, starts a sidecar on a dynamic
`127.0.0.1` port, creates a temporary Claude plugin directory with NeMo Flow
hooks, passes that plugin with `--plugin-dir`, and sets
`ANTHROPIC_BASE_URL` to the sidecar URL for the launched process.

Inspect what would be launched without starting Claude Code:

```bash
nemo-flow-sidecar run \
  --atif-dir .nemo-flow/atif \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --dry-run \
  --print \
  -- claude
```

If a launcher hides the command name, pass the agent explicitly:

```bash
nemo-flow-sidecar run --agent claude-code -- my-claude-wrapper
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

[agents.claude-code]
command = "claude"
```

Then run `nemo-flow-sidecar run --agent claude-code` to use the configured
command. User config takes priority over project and global config.

## Persistent Install

Use persistent hooks only when you want Claude Code configured outside the
wrapper:

```bash
nemo-flow-sidecar install claude-code \
  --scope user \
  --target cli \
  --sidecar-url http://127.0.0.1:4040 \
  --atif-dir .nemo-flow/atif
```

Then set `ANTHROPIC_BASE_URL` in the Claude Code process environment and start the
sidecar manually in another terminal:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4040
NEMO_FLOW_ATIF_DIR=.nemo-flow/atif nemo-flow-sidecar --bind 127.0.0.1:4040
```

The sidecar forwards Anthropic `/v1/messages`, `/v1/messages/count_tokens`, and
model routes without rewriting provider JSON.

## Smoke Test

Run a small Claude Code prompt that starts a session and uses one simple tool.
Then check that hook forwarding reaches the sidecar:

```bash
curl -f http://127.0.0.1:4040/healthz
printf '{"session_id":"smoke-claude","hook_event_name":"SessionStart"}' \
  | NEMO_FLOW_SIDECAR_URL=http://127.0.0.1:4040 nemo-flow-sidecar hook-forward claude-code --fail-closed
```

The response should be valid Claude Code hook JSON. For most lifecycle events it
is an allow/continue response.

## Verify Export

End the Claude Code session and confirm that session-end closed the NeMo Flow
agent scope and wrote ATIF:

```bash
ls .nemo-flow/atif
```

The sidecar exports `<session-id>.atif.json` on session end. If no file appears,
confirm that `SessionEnd` hooks fire, `--atif-dir` or `NEMO_FLOW_ATIF_DIR` is
set, and the sidecar process can write to the directory.

## Troubleshoot LLM Lifecycle

Missing hooks usually means Claude Code did not load the local hook config or
the `nemo-flow-sidecar` binary is not on `PATH`.

Missing LLM spans with present hook spans means Anthropic traffic is not routed
through the sidecar. Verify `ANTHROPIC_BASE_URL` in the Claude Code process
environment and confirm that requests hit `/v1/messages`.
