<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Extension for pi

A pi extension that forwards pi's lifecycle to the NeMo Relay CLI gateway and
gates tool calls on the gateway's verdict.

pi has no native hook-configuration file, and its external stream is
observation-only, so hook calls have to originate inside an extension. Unlike
the Codex and Claude Code integrations — which install hook commands the host
runs for them — this extension *is* the hook client. It is deliberately thin:
all policy and all span construction happen in the gateway.

## Status

Proof of concept. Verified against pi `v0.84.0`. pi ships breaking changes
through *minor* releases and has no major-release channel, so re-verify hook
signatures before relying on them — `nemo-relay doctor pi` and the launcher both
report a minor above 0.84.x as unverified rather than supported, and accept it
anyway, because untested is not the same as broken.

**Model traffic is redirected conditionally.** pi has no base-URL flag and no
generic environment override — it resolves `baseUrl` per model from a generated
catalog — so the extension points the active model's provider at the gateway
itself. Under `nemo-relay pi` it also names the endpoint the gateway should
forward to, so an arbitrary provider still produces LLM spans; against a
standalone daemon it redirects only when the gateway already fronts that
endpoint. See [Model Redirection](#model-redirection). Where it cannot redirect,
you get tool and turn activity but no LLM spans.

## Usage

Start the gateway, then load the extension:

```bash
nemo-relay --bind 127.0.0.1:4040 &
NEMO_RELAY_PI_GATEWAY_URL=http://127.0.0.1:4040 \
  pi -e integrations/pi/index.ts
```

`pi -e` is trust-ungated, loads before discovery, and survives
`--no-extensions`, which makes it the reliable way to load this. It is also what
`nemo-relay run --agent pi` uses.

### Where to Install It

**User scope only.** The simplest route is to let Relay do it, which needs no
checkout of this repository -- the `nemo-relay` binary carries a copy of this
extension:

```bash
nemo-relay install pi
```

That writes `~/.pi/agent/extensions/nemo-relay` and records what it wrote, so
`nemo-relay uninstall pi` removes exactly that and leaves anything you edited.
To manage the extension yourself instead, use either of:

```bash
# 1 · file drop
cp -r crates/cli/assets/pi-extension ~/.pi/agent/extensions/nemo-relay

# 2 · pi install, from a LOCAL PATH -- never with --local
pi install /path/to/NeMo-Relay/crates/cli/assets/pi-extension
```

Both copy from `crates/cli/assets/pi-extension` rather than from this directory.
That is where the extension's source actually lives -- see [Layout](#layout) --
and it is the only self-contained copy: this directory reaches the source through
symlinks, which a copy may or may not follow depending on the platform's `cp`.

Install **one** copy. pi de-duplicates its extension set by path rather than by
package, so two copies are two packages: both register hooks and every event is
reported twice. `nemo-relay install pi` refuses when another copy would load
beside it, and the launcher refuses to start.

| Path | Install Here? | Trust-Gated? |
|---|---|---|
| `nemo-relay install pi` | Yes -- writes the first row | No |
| `~/.pi/agent/extensions/` | Yes | No |
| `pi install <local path>` | Yes | No |
| `settings.json` `"extensions": [...]` | Yes | Only the project copy |
| `-e <path>` | Per-invocation; what the launcher uses | No |
| `.pi/extensions/` or `pi install --local` | **No** | **Yes** |
| `pi install <git URL>` | **No** — see below | — |

Either user-scope route is enough for `nemo-relay run --agent pi`: the launcher
resolves the extension from the same places `nemo-relay doctor pi` looks and
passes what it finds to `-e`, so no environment variable is needed. It never
promotes a **project-scoped** install that way — `-e` is not trust-gated, so
doing so would load code pi itself declined to trust.

⚠️ **A git URL does not work, and fails silently.** pi has no
subdirectory syntax for a git source: it clones the repository *root*, then looks
there for a `pi` key in `package.json` or a top-level `extensions/` directory.
This extension lives at `integrations/pi`, and the repository root has neither —
so `pi install https://github.com/NVIDIA/NeMo-Relay` reports success, prints an
install path, and loads no extension. Worse than nothing: the clone's root
`skills/` *is* picked up, so you get NeMo Relay's Codex and Claude skills in pi
and none of the gating.

**This package is deliberately not published to npm.** `nemo-relay install pi`,
the file drop and the local-path install all work and cover user scope;
publishing would add an npm namespace, a build step (the sources are TypeScript
that nothing compiles today), a `files` allowlist and release wiring for one more
route to the same files. It is
`private: true` for that reason, not by oversight — and it carries a
`pi.extensions` manifest key so both working routes resolve explicitly rather
than relying on pi's directory fallback.

⚠️ **A project-scoped install is silently skipped, and nothing tells you.** pi
adds project extensions to its candidate set only when the project is trusted,
and `-p`, `--mode json` and `--mode rpc` never prompt for trust — so under the
default policy the extension is dropped by a bare conditional. It is not an
error path, so pi does not treat it as a failure and never reports it, and the
extension cannot report it either: by construction it is not running. The
symptom is simply that NeMo Relay appears to do nothing.

**`nemo-relay doctor pi` is the check for that.** It reports where the extension
sits, whether that path is trust-gated, and whether the gateway is answering:

```bash
nemo-relay doctor pi
```

Run it first whenever Relay does not seem to be doing anything.

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `NEMO_RELAY_PI_EXTENSION` | unset | The first place the launcher looks for this extension. A path that exists **and** whose `package.json` names `nemo-relay-pi` is used; anything else is ignored and resolution falls through to a user-scope install. Set by the launcher from what it resolved |
| `NEMO_RELAY_PI_GATEWAY_URL` | `http://127.0.0.1:4040` | Gateway base URL |
| `NEMO_RELAY_PI_TIMEOUT_MS` | `5000` | Per-request timeout. Posts are serialized, so a gating hook also waits out everything queued ahead of it — an unresponsive gateway costs this once per queued post, not once |
| `NEMO_RELAY_PI_FAIL` | `open` | `closed` blocks tool calls and inline shell commands when the gateway is unreachable |
| `NEMO_RELAY_PI_REDIRECT` | `match` | `force` redirects without checking the upstream; `off` disables redirection |
| `NEMO_RELAY_PI_OPENAI_UPSTREAM` | unset | What the gateway forwards OpenAI-compatible traffic to. Set by the launcher |
| `NEMO_RELAY_PI_ANTHROPIC_UPSTREAM` | unset | What the gateway forwards Anthropic traffic to. Set by the launcher |
| `NEMO_RELAY_PROXY_CREDENTIAL` | unset | This invocation's proxy credential, set by `nemo-relay run` for every agent it starts. Sent as `x-nemo-relay-proxy-token` on a redirected provider only — a gateway the launcher started rejects a provider call without it. A standalone `nemo-relay --bind` daemon sets none and needs none |

## How Tool Gating Works

For model-invoked tools, `tool_call` is the only pre-execution decision point
that sees arguments — pi's `--tools`, `--exclude-tools`, `--no-tools` and
runtime `setActiveTools` are all applied at tool-registry construction, never
per call. The user's own inline shell takes a different path and is gated
separately; see [Inline Shell](#inline-shell).

The wire contract, pinned from both sides by tests:

| Gateway Response | Extension Behavior |
|---|---|
| 2xx | allow |
| 403 with `error.type = "nemo_relay_guardrail_rejected"` | block, using `error.reason` |
| 403 without that marker | fault — an authorization failure is not a policy decision |
| other status, timeout, unreachable | fault, resolved by `NEMO_RELAY_PI_FAIL` |

The block reason reaches the model **verbatim**: pi hands it to
`createErrorToolResult` with no framing. Write guardrail reasons as guidance, not
as error codes — a reason that says what to do instead produces a model that
adapts rather than one that gives up.

## Argument Transforms

A Relay **request intercept** can rewrite a tool's arguments. The gateway never
runs the tool, so the rewrite comes back in the allow response and the extension
applies it to pi's `event.input` **in place** — which is what pi documents as
the mechanism, and is required: pi hands the same object to the tool and to
later handlers, so replacing the reference would be discarded.

| Response Body | Meaning |
|---|---|
| `{}` | Allow, arguments unchanged. Every non-gated hook, and any `tool_call` no intercept rewrote |
| `{"tool_call": {"tool_call_id": "…", "input": {…}}}` | Allow, but execute *these* arguments |

`tool_call_id` is **echoed, and checked**: the extension applies a transform only
when the id is a string equal to the call it just posted. An envelope carrying
`input` under a missing, non-string, or different id is refused rather than
applied to whichever call happens to be open — and a refusal blocks, the same as
a shape violation does.

⚠️ **The rewrite is constrained, not validated.** pi validates arguments
*before* the `tool_call` hook and never re-validates — its own types say "no
re-validation is performed after mutation" — so a rewrite that violates the
tool's schema would execute.

The extension *could* check it: `pi.getAllTools()` returns every configured
tool — built-ins included — with its TypeBox `parameters` schema. That is
deliberately not used. pi's tool set is per-session mutable (`setActiveTools`,
`registerTool`), so a schema read once can go stale mid-session, and forwarding
it would make the gateway carry pi's tool vocabulary for a check that does not
need it.

So the extension enforces a **shape** invariant instead: a transform may rewrite
the values of existing keys, preserving each value's JSON type, recursively.
Adding a key, removing a key, changing a type, or changing an array's length is
refused. An argument object that satisfied the schema before therefore still has
the required keys of the required types afterwards.

**This is structural, not schema validation.** `pattern`, `enum`, `minimum` and
`format` are not checked — by choice, not because the schema is out of reach. A
transform that rewrites a string to one the schema would reject still executes.

**Nor is it a second policy decision.** The gateway's conditional-execution
guardrails decide on the arguments pi proposed, before the rewrite, and are not
re-run on the result — the same order a managed tool call uses. A request
intercept can therefore rewrite a value a guardrail would have refused. Put the
decision in the guardrail, not in a transform that outruns it.

A refused transform **blocks the call**, with a reason that says the policy could
not be applied rather than that the request was refused. Running the original
arguments instead would silently discard a policy decision, which is the failure
the transform existed to prevent. This is a different axis from
`NEMO_RELAY_PI_FAIL`, which governs an unreachable gateway rather than one that
answered with something unusable.

## Inline Shell

pi's bang prefix runs a command without going through the tool registry:
`!git status` runs it and shows the model the output, `!!git status` runs it and
keeps the output out of the model's context. Neither fires `tool_call`, so none
of the gating above sees them. They reach `user_bash` instead, which this
extension gates the same way — the command is posted to the gateway as a tool
start and the same conditional-execution guardrail chain decides it.

**The tool name is `user_bash`, not `bash`.** A guardrail receives only the tool
name and the arguments, so if both arrived as `bash` a policy could not tell a
command the user typed from one the model proposed — and "the model may not run
shell commands" is a reasonable rule that should not also stop a human typing
`!git status`. The cost is that **a policy which wants to cover both has to name
both**; a rule written only for `bash` does not gate the bang prefix.

| Argument | Value |
|---|---|
| `command` | The command text, exactly as typed after the prefix |
| `cwd` | pi's working directory for the command |
| `exclude_from_context` | `true` for the `!!` form, whose output the model never sees |

### The Refusal Shape

pi gives `user_bash` no block-and-reason contract — there is no `{block,
reason}` here — so a refusal has to be a **synthetic failed command result**,
which pi records exactly as if the command had run:

| Field | Value | Why |
|---|---|---|
| `exitCode` | `126` | The shell convention for "found, but could not be executed". Not `0` (reads as success), not `1` (indistinguishable from the command itself failing), not `127` (sends you hunting for a missing binary) |
| `output` | `NeMo Relay blocked this command.` then a blank line, then the reason verbatim | The attribution line says what declined it; the reason is the guardrail's own words, and for `!cmd` the model reads them |
| `cancelled` | `false` | Nothing was started, so nothing was interrupted |
| `truncated` | `false` | The message is whole |

`NEMO_RELAY_PI_FAIL` governs this path too: a gateway that cannot be reached
allows the command by default, and refuses it under `closed` with a reason that
says explicitly that it is an infrastructure fault rather than a judgment — and
which of four it was: no answer, no answer *in time*, an answer that was not a
decision, or the gate failing before the gateway was asked. A timeout is not an
unreachable gateway, and neither is a 413, so the reader is not sent to debug a
socket that is working.

**A rewritten command is refused, not run.** pi's `user_bash` result can replace
the *result* or the execution backend, but never the command — both call sites
pass the original text straight on to `executeBash` and read nothing back out of
the event. So a request intercept that rewrites an inline command cannot be
honored, and the command is refused rather than run unmodified, on the same
rule the tool path applies to a transform it cannot apply safely.

### Limits Worth Knowing

- **The gate decides; it does not observe.** pi has no completion hook for
  inline shell, so on an allow the span closes immediately: it measures the
  policy round trip, not the command. Taking execution over to fix that would
  mean handing pi custom `operations`, which drops the user's configured shell
  path and pi's process-tree cancellation — the extension does not change how pi
  runs things.
- **`!!` hides the refusal from the model**, not from the user. The terminal
  shows it either way; the model's context does not.
- **The first extension to answer wins**, and there is no priority system. Loaded
  with `pi -e` this extension answers first; installed with `pi install` it loads
  last, and an extension ahead of it that answers `user_bash` unconditionally —
  pi's own `sandbox` and `gondolin` examples do — means the gateway never sees the
  command.
- **Where it fires.** The interactive TUI and RPC mode. Headless `-p` has no
  input loop to type a bang prefix into, so there is nothing to gate there.
- **The prompt looks frozen while the gate is out.** pi builds the terminal
  component *after* the hook resolves, so a slow gateway shows nothing at all
  until `NEMO_RELAY_PI_TIMEOUT_MS` expires. Keep that timeout short.

## Model Redirection

pi resolves a base URL per model from a generated catalog, so there is no flag or
environment variable to point it at the gateway. The extension does it directly:

```ts
pi.registerProvider(providerId, { baseUrl: gatewayUrl });
```

With `baseUrl` and no `models`, pi rewrites the URL of every existing model for
that provider and keeps their API, headers, costs and context windows. That is
much cheaper than pi's own `custom-provider-*` examples, which register a
`streamSimple` and re-implement a provider protocol.

**The gateway has to know where to forward.** It sends each API family to one
statically configured upstream — `--openai-base-url` and `--anthropic-base-url` —
so redirecting a model whose endpoint is not that upstream would reach the wrong
provider. Pointing an NVIDIA model at a gateway configured for `api.openai.com`
does not degrade to "no spans"; it breaks the session.

There are two ways to satisfy that, and which one applies depends on how the
gateway was started:

- **Named per request.** Under `nemo-relay pi` or `run --agent pi`, the extension
  sends `x-nemo-relay-upstream-base-url` on each request, so the gateway forwards
  to a provider it was never configured for. It honors that header only from a
  request carrying this invocation's proxy credential — minted per run and given
  only to the process the launcher starts — and strips it before forwarding.
  A named endpoint receives only the credential the request already carried: the
  gateway never attaches its own configured or environment provider key to a
  destination the client chose, so naming a host is not a way to obtain a key
  for it. Because that credential does travel, a named endpoint must use `https`
  unless it is loopback — a local model server on `127.0.0.1` is allowed, a plain
  `http` host that is reachable from off the machine is refused.
- **Statically configured.** A standalone `nemo-relay --bind` daemon issues no
  credential, so it ignores the header. Redirection there still requires the
  gateway's own upstream to be the model's endpoint.

Each outcome that explains something is recorded as a `model_redirect` mark, so a
trace without LLM spans accounts for itself. Two transient skips are evaluated
but not marked — no model resolved yet, and a provider already pointed at the
gateway — because a mark per `session_start` for either is noise.

| Situation | Launched by `nemo-relay pi` | Standalone daemon |
|---|---|---|
| Gateway upstream equals the model's endpoint | Redirected | Redirected |
| Gateway forwards somewhere else | Redirected, naming the endpoint | Skipped, `upstream-mismatch` |
| Upstream unknown | Redirected, naming the endpoint | Skipped, `unknown-upstream` — set `NEMO_RELAY_PI_REDIRECT=force` to override |
| Model's API has no gateway route (Bedrock, Azure OpenAI Responses, Google, Google Vertex, Mistral, OpenAI Codex, Radius) | Skipped, `unserviceable-api` | Skipped, `unserviceable-api` |
| The provider's models do not share one endpoint | Redirected only if both upstreams already point at it | Redirected only if both upstreams already point at it |

The last row reads the same on both paths because `registerProvider` is
provider-wide: one endpoint is chosen for every model of that provider, so they
have to agree on it. Pointing `--openai-base-url` and `--anthropic-base-url` at
that provider's respective endpoints is what makes them agree — Fireworks
redirects fine once both are set. Naming cannot substitute, because a single
header carries a single endpoint.

The decision is re-evaluated on every `model_select`, so switching to a model the
gateway does not front stops redirecting rather than silently misrouting.

32 of pi's 39 built-in providers speak an API the gateway serves; the seven
above do not. Count from `builtinProviders()` rather than from the 38 files in
`providers/data/`: Radius is a purely dynamic provider with no static catalog
entry, so a file count loses it.

## Hook Mapping

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

| pi Hook | Forwarded As | Note |
|---|---|---|
| `session_start` / `session_shutdown` | session boundary | **Not** `agent_start`/`agent_end` — those repeat on re-entry. `session_shutdown` is ignored for `reason: "reload"`, which continues the same session |
| `agent_start` / `agent_end` | run-level marks | Carry `attempt_index`; not a run boundary. Recorded on the session scope, not inside a turn |
| `agent_settled` | run-level mark | Fires exactly once, from a `finally`. Carries `attempts` (the count) and `attempt_index` (the last one) |
| `turn_start` | turn scope **open** | Carries `turn_index`, `turn_seq`, `attempt_index`. Awaited, so the turn exists before pi's model call arrives |
| `turn_end` | turn scope **close** | Carries `turn_index`, `turn_seq`, `attempt_index`. Awaited, for the same reason |
| `session_start`, then every `model_select` | `model_redirect` mark, for each decision that explains something | Synthesized. Re-evaluates redirection for the newly selected model, so a switch away from a provider the gateway fronts stops redirecting |
| *(after a rewrite)* | `tool_arguments_transformed` mark | Synthesized, so the trace records that the arguments the tool ran were not the ones proposed |
| `session_before_compact` | mark | Announced, not done, and cancellable by a later extension. Carries `reason`, `will_retry`, `tokens_before` |
| `session_compact` | compaction | The completed compaction, which the runtime treats as proof the context was rebuilt |
| `tool_call` | tool start, and the gate | The only blocking hook. Carries `attempt_index`, `turn_seq` |
| `tool_execution_end` | tool end | For **every** outcome, including blocked. Carries `attempt_index`, `turn_seq` |
| `tool_execution_start` | *not forwarded* | Registered, but only to remember a tool name for the matching end: it fires before validation and for calls that never execute |
| `user_bash` | tool start, and the second gate | The bang prefix, which never reaches the tool registry. Gated under the tool name `user_bash` — see [Inline Shell](#inline-shell) |
| *(synthesized)* `user_bash_end` | tool end | pi reports no completion for inline shell, so the extension closes the span itself |

`tool_result` is deliberately unused: it does not fire for blocked calls, and in
the parallel path it fires *before* `tool_execution_end`.

Because pi reports both ends of a turn, the gateway never invents one for it. A
mark arriving between turns — the `agent_end` / `agent_settled` tail of a run —
is recorded on the session scope rather than opening an empty turn to hold it.
Codex and Claude Code report only `Stop`, so their turns stay lazily opened by
the first event of the turn.

## What Is Not Represented

For pi's structured tool results, the extension joins text parts in order and
omits image and other non-text parts. **Tool-result text is truncated at 2000
characters** before it is forwarded. The overflow is replaced by a
`... [truncated N chars]` suffix. The gateway therefore records a bounded text
projection, not the complete result — a large file read or a long command output
is cut. Raise `MAX_CONTENT_CHARS` in `index.ts` if a policy needs to see more.

**Tool arguments are not truncated, and must not be.** The `tool_call` post is
the gated one: a guardrail decides on exactly those arguments, and a request
intercept can send a rewritten copy back for pi to execute. Truncating them would
mean deciding on text the tool will not run — and worse, the rewrite that comes
back would carry the truncation, because the shape invariant checks JSON types
and key sets rather than content, so a shortened `content` is applied verbatim
and a `write` lands on disk cut short. A result has no path back into execution,
which is why only results are bounded.

The ceiling is therefore the gateway's, not the extension's:
`gateway.max_hook_payload_bytes`, 20 MiB by default. A post above it is rejected
with HTTP 413 before any event exists, so the call is decided by
`NEMO_RELAY_PI_FAIL` — under `open` it runs ungated, leaving a span synthesized
from `tool_execution_end` with no arguments. Model-authored arguments do not
approach that ceiling; if some tool ever does, raise the gateway limit rather
than cutting the arguments.

**Subagents.** pi ships no nested-agent hook of its own — the extension has
nothing to derive a subagent id from — so `subagent_start` / `subagent_end` are
empty for pi and every tool span parents to the turn. The multi-process case is
worth stating separately: a child pi process running this extension resolves its
*own* session id and posts under it, so it does not appear as a subagent of the
parent. It appears as an unrelated session.

**An authoritative boundary against *later* extensions.** pi runs every
`tool_call` handler unless one returns `block`, and they all share the same
mutable `input` object with no re-validation afterwards. Loaded with `-e` this
gate runs **first**, which is what stops an earlier extension pre-empting it —
but it also means an extension loaded *after* it can rewrite the arguments once
Relay has authorized them, and those arguments execute unreviewed.

pi offers no ordering API and no post-chain hook, so this cannot be prevented
from inside the extension. **In a mixed extension stack, treat the tool gate as
authoritative over the model, not over the other extensions.** It is sound when
this is the only extension mutating tool arguments, which is the deployment
`nemo-relay run --agent pi` produces.

**Tool-result policy, on either side.** Relay's only middleware that can change
what a tool *returned* is the execution intercept, which wraps the callback and
therefore owns execution. pi never hands the callback over — it runs the tool in
its own process and reports the outcome — and neither does the gateway, which
builds spans from hook posts rather than executing anything.

⚠️ Worth stating in full, because it is not pi-specific and it surprises people:
**a tool execution intercept registered by any plugin never runs under the CLI
gateway.** The registry has exactly one consumer, `tool_call_execute`, and the
gateway does not call it — it uses `tool_call` / `tool_call_end`. Guardrails and
request intercepts do run, because those have standalone runners the gateway
calls directly. There is no response-phase equivalent.

**Anything still queued when the process is interrupted.** pi registers **no
SIGINT handler in any mode** — all three build `["SIGTERM"]` plus SIGHUP off
Windows — and raw mode is set only by the TUI, so under `-p`, `--mode json` and
`--mode rpc` a user's Ctrl+C is a real SIGINT that terminates with teardown
never running. `session_shutdown` never fires, so the queue is never drained.

The loss is bounded, and the bound is what makes this liveable: **every awaited
hook has already reached the gateway.** Both gates (`tool_call`, `user_bash`) and
both turn boundaries block on their round trip, so an interrupt can only drop
observability marks queued since the last awaited one. The gateway keeps
everything already delivered — an interrupted session is left *open*, not lost.
SIGTERM, SIGHUP and `/quit` all run teardown normally; SIGINT, SIGKILL and an
uncaught exception do not.

**LLM spans, when redirection is skipped.** They are present whenever the
gateway fronts the endpoint the active model would otherwise have called, and
absent otherwise — the `model_redirect` mark in the trace names which it was and
why. See [Model Redirection](#model-redirection).

**The outcome of an inline shell command.** pi reports no completion for the bang
prefix, so the gate records the decision, not the command. See
[Inline Shell](#inline-shell).

## Development

```bash
npm run typecheck --prefix integrations/pi
node --test integrations/pi/test/*.test.mjs
```

### Layout

The extension's source lives in `crates/cli/assets/pi-extension`, and
`package.json`, `index.ts` and `src/` here are symlinks to it. There is one copy
of every file, edited from either path.

It lives under the CLI crate for a packaging reason rather than a design one:
`nemo-relay install pi` embeds the extension with `include_str!` so installing
needs no checkout, `nemo-relay-cli` is published to crates.io, and Cargo packages
only files below the crate root. A copy kept in step by a sync step was the
alternative, and it duplicated all seven files.

What is real here is what has no runtime role: this README, `tsconfig.json` and
`test/`. One consequence is worth knowing before you copy this directory
anywhere -- it is not self-contained, so an install copies from
`crates/cli/assets/pi-extension` instead.

The gateway half of the contract is covered in Rust by
`pi_tool_call_hook_rejects_when_conditional_guardrail_blocks`,
`pi_tool_call_hook_allows_when_no_guardrail_objects`,
`pi_user_bash_hook_rejects_when_conditional_guardrail_blocks` and
`pi_user_bash_is_not_gated_by_a_policy_that_names_the_bash_tool` in
`crates/cli/tests/coverage/shared/server_tests.rs`.

`test/fixtures/reentry-driver.ts` forces exactly one agent-run re-entry through
pi's real queued-follow-up path, for reproducing the colliding-turn-index case.

## Related

- CLI agent definition: `crates/cli/src/agents/pi/`
- Hook route: `/hooks/pi` in `crates/cli/src/server/mod.rs`
- Payload classification: `PiPayloadExtractor` in `crates/cli/src/agents/shared/adapters.rs`
