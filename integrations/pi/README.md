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
[RELAY-727](https://linear.app/nvidia/issue/RELAY-727),
[RELAY-728](https://linear.app/nvidia/issue/RELAY-728),
[RELAY-729](https://linear.app/nvidia/issue/RELAY-729) and
[RELAY-730](https://linear.app/nvidia/issue/RELAY-730). Verified against pi
`v0.84.0`. pi ships breaking changes through *minor* releases and has no
major-release channel, so re-verify hook signatures before relying on them.

**Model traffic does not traverse the gateway yet**
([RELAY-732](https://linear.app/nvidia/issue/RELAY-732)). pi has no base-URL
flag and no generic environment override — it resolves `baseUrl` per model from
a generated catalog — so redirection needs this extension to register a
gateway-backed provider, and nothing does that yet. Until it lands there are no
LLM spans and no model-call enforcement, only tool and turn activity.

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
so a retry cannot be detected there — `session_before_compact` is the one hook
that announces one in advance, and only for the compaction case;
`agent_settled` is the only event that fires exactly once per logical run. The
gateway's own model is flat (session -> turn -> tool) and assigns its own
monotonic turn index, so the extension sends `attempt_index` and a
session-monotonic `turn_seq` on every attributable hook — they are the only way
to recover which attempt a turn or tool call belonged to.

Both travel as payload keys, and the gateway's pi extractor promotes them into
each event's **metadata**. That promotion is not cosmetic: mark events record
the raw payload as their `data`, but tool spans are built from the extracted
call id, name, arguments, result and metadata and discard the payload entirely,
so without it the two keys would be accepted on the wire and then dropped. Read
them from `metadata` on scopes and spans, and from either place on marks.

The consequence is worth stating plainly: **re-entry is not nested.** The
gateway model stays flat and gains no attempt level, because that model is
shared with Codex and Claude Code, which have no equivalent concept. Two
attempts of one prompt appear as more turns under one session, distinguished by
`attempt_index` — not as two subtrees.

**Known limitation.** The counters live in the extension factory's closure and
pi re-runs the factory on `/reload` with `moduleCache: false`, while the session
id stays the same. `turn_seq` therefore restarts at 0 and can repeat within one
session — it orders turns within a runtime, not strictly within a session.
Rebuilding it would mean replaying the session or moving the counter into the
gateway; neither is worth it here.

**Concurrent tools.** pi preflights sibling calls sequentially then executes them
concurrently, so `tool_execution_end` arrives out of submission order. All
per-call state is keyed by `toolCallId`, the only correlator pi provides.

| pi hook | Forwarded as | Note |
|---|---|---|
| `session_start` / `session_shutdown` | session boundary | **Not** `agent_start`/`agent_end` — those repeat on re-entry. `session_shutdown` is ignored for `reason: "reload"`, which continues the same session |
| `agent_start` / `agent_end` | run-level marks | Carry `attempt_index`; not a run boundary. Recorded on the session scope, not inside a turn |
| `agent_settled` | run-level mark | Fires exactly once, from a `finally`. Carries `attempts` (the count) and `attempt_index` (the last one) |
| `turn_start` | turn scope **open** | Carries `turn_index`, `turn_seq`, `attempt_index` |
| `turn_end` | turn scope **close** | Carries `turn_index`, `turn_seq`, `attempt_index` |
| `session_before_compact` | mark | Announced, not done, and cancellable by a later extension. Carries `reason`, `will_retry`, `tokens_before` |
| `session_compact` | compaction | The completed compaction, which the runtime treats as proof the context was rebuilt |
| `tool_call` | tool start, and the gate | The only blocking hook. Carries `attempt_index`, `turn_seq` |
| `tool_execution_end` | tool end | For **every** outcome, including blocked. Carries `attempt_index`, `turn_seq` |
| `tool_execution_start` | *not forwarded* | Registered, but only to remember a tool name for the matching end: it fires before validation and for calls that never execute |

`tool_result` is deliberately unused: it does not fire for blocked calls, and in
the parallel path it fires *before* `tool_execution_end`.

Because pi reports both ends of a turn, the gateway never invents one for it. A
mark arriving between turns — the `agent_end` / `agent_settled` tail of a run —
is recorded on the session scope rather than opening an empty turn to hold it.
Codex and Claude Code report only `Stop`, so their turns stay lazily opened by
the first event of the turn.

## What is not represented

**Tool results are truncated at 2000 characters** before they are forwarded, with
the overflow replaced by a `... [truncated N chars]` suffix. The gateway
therefore records what a tool returned, not necessarily all of it — a large file
read or a long command output is cut. This keeps hook payloads bounded; raise
`MAX_CONTENT_CHARS` in `index.ts` if a policy needs to see more.

**Subagents.** pi ships no nested-agent hook of its own — the extension has
nothing to derive a subagent id from — so `subagent_start` / `subagent_end` are
empty for pi and every tool span parents to the turn. The multi-process case is
worth stating separately: a child pi process running this extension resolves its
*own* session id and posts under it, so it does not appear as a subagent of the
parent. It appears as an unrelated session.

**LLM spans**, until model redirection lands. See [Status](#status).

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
