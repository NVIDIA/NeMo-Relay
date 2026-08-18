<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay extension for pi

A pi extension that forwards pi's lifecycle to the NeMo Relay CLI gateway and
gates tool calls on the gateway's verdict.

pi has no native hook-configuration file, and its external stream is
observation-only, so hook calls have to originate inside an extension. Unlike
the Codex and Claude Code integrations — which install hook commands the host
runs for them — this extension *is* the hook client. It is deliberately thin:
all policy and all span construction happen in the gateway.

## Status

Proof of concept, tracked under
[RELAY-727](https://linear.app/nvidia/issue/RELAY-727) and
[RELAY-728](https://linear.app/nvidia/issue/RELAY-728). Verified against pi
`v0.84.0`. pi ships breaking changes through *minor* releases and has no
major-release channel, so re-verify hook signatures before relying on them.

## Usage

Start the gateway, then load the extension:

```bash
nemo-relay --bind 127.0.0.1:4040 &
NEMO_RELAY_PI_GATEWAY_URL=http://127.0.0.1:4040 \
  pi -e integrations/pi/index.ts
```

`pi -e` is trust-ungated, loads before discovery, and survives
`--no-extensions`, which makes it the reliable way to load this. For everyday
use, install it with `pi install <source>` or place it in an auto-discovered
directory (`~/.pi/agent/extensions/`, `.pi/extensions/`).

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `NEMO_RELAY_PI_GATEWAY_URL` | `http://127.0.0.1:4040` | Gateway base URL |
| `NEMO_RELAY_PI_TIMEOUT_MS` | `5000` | Per-request timeout |
| `NEMO_RELAY_PI_FAIL` | `open` | `closed` blocks tool calls when the gateway is unreachable |

## How tool gating works

`tool_call` is the only pi hook that can block, and for model-invoked tools it
is the only pre-execution decision point that sees arguments — pi's `--tools`,
`--exclude-tools`, `--no-tools` and runtime `setActiveTools` are all applied at
tool-registry construction, never per call.

The wire contract, pinned from both sides by tests:

| Gateway response | Extension behaviour |
|---|---|
| 2xx | allow |
| 403 with `error.type = "nemo_relay_guardrail_rejected"` | block, using `error.reason` |
| 403 without that marker | fault — an authorization failure is not a policy decision |
| other status, timeout, unreachable | fault, resolved by `NEMO_RELAY_PI_FAIL` |

The block reason reaches the model **verbatim**: pi hands it to
`createErrorToolResult` with no framing. Write guardrail reasons as guidance, not
as error codes — a reason that says what to do instead produces a model that
adapts rather than one that gives up.

## Hook mapping

pi's lifecycle is `session -> agent run -> turn -> message | tool execution`.
Two shapes make a naive mapping wrong.

**Agent-run re-entry.** One prompt can re-enter the agent run several times
(provider retry, post-compaction, queued follow-up), and pi's `turnIndex` resets
to 0 each time. The extension-facing `agent_end` carries no `willRetry` marker,
so a retry cannot be detected there; `agent_settled` is the only event that fires
exactly once per logical run. The gateway's own model is flat
(session -> turn -> tool) and assigns its own monotonic turn index, so the
extension sends `attempt_index` and a session-monotonic `turn_seq` as metadata —
they are the only way to recover which attempt a turn belonged to.

**Concurrent tools.** pi preflights sibling calls sequentially then executes them
concurrently, so `tool_execution_end` arrives out of submission order. All
per-call state is keyed by `toolCallId`, the only correlator pi provides.

| pi hook | Forwarded as | Note |
|---|---|---|
| `session_start` / `session_shutdown` | session boundary | **Not** `agent_start`/`agent_end` — those repeat on re-entry |
| `agent_start` / `agent_end` | attempt markers | Carry `attempt_index`; not a run boundary |
| `agent_settled` | logical run boundary | Fires exactly once, from a `finally` |
| `turn_start` | turn boundary (open) | Carries `turn_index`, `turn_seq` and `attempt_index` |
| `turn_end` | turn boundary (close) | Carries `turn_index` only |
| `tool_call` | tool start, and the gate | The only blocking hook |
| `tool_execution_end` | tool end | For **every** outcome, including blocked |
| `tool_execution_start` | *not forwarded* | Fires before validation and for calls that never execute |

`tool_result` is deliberately unused: it does not fire for blocked calls, and in
the parallel path it fires *before* `tool_execution_end`.

## Development

```bash
npm run typecheck --prefix integrations/pi
node --test integrations/pi/test/*.test.mjs
```

The gateway half of the contract is covered in Rust by
`pi_tool_call_hook_rejects_when_conditional_guardrail_blocks` and
`pi_tool_call_hook_allows_when_no_guardrail_objects` in
`crates/cli/tests/coverage/shared/server_tests.rs`.

`test/fixtures/reentry-driver.ts` forces exactly one agent-run re-entry through
pi's real queued-follow-up path, for reproducing the colliding-turn-index case.

## Related

- CLI agent definition: `crates/cli/src/agents/pi/`
- Hook route: `/hooks/pi` in `crates/cli/src/server/mod.rs`
- Payload classification: `PiPayloadExtractor` in `crates/cli/src/agents/shared/adapters.rs`
