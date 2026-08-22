<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implement The Integration

Implement only after the host assessment identifies an approved shape. Preserve
the host's public behavior when Relay is absent, disabled, or unable to load.

## Use Current Public Contracts

Read these repository guides at the target Relay revision before writing code:

| Need | Guide |
|---|---|
| Select an integration method | `docs/integrate-into-frameworks/about.mdx` |
| Add and propagate scopes | `docs/integrate-into-frameworks/adding-scopes.mdx` |
| Wrap tools and MCP bridges | `docs/integrate-into-frameworks/wrap-tool-calls.mdx` |
| Wrap model calls and streams | `docs/integrate-into-frameworks/wrap-llm-calls.mdx` |
| Keep runtime objects outside Relay values | `docs/integrate-into-frameworks/non-serializable-data.mdx` |
| Use value and provider codecs | `docs/integrate-into-frameworks/using-codecs.mdx` and `provider-codecs.mdx` |
| Use fallback policy helpers | `docs/integrate-into-frameworks/code-examples.mdx` |

Do not copy an API signature from another release. Rust is the semantic source
of truth; Python, Node.js, and maintained integrations show binding-specific
conventions.

When the host overlaps requests, callbacks, streams, or shutdown, also read
[Concurrency And Lifecycle](concurrency-and-lifecycle.md).

## Scope And Identity

- Create one root for each independently executing session or request.
- Create child scopes for the host's real run, turn, tool, and model boundaries;
  do not invent a scope for an index that the host reuses.
- Store handles in request-local or session-local state, not mutable process
  globals.
- Store or reconstruct the owning Relay execution context as well as the
  handle. Passing a parent handle does not activate its scope stack or restore
  scope-local middleware and subscribers.
- Propagate context explicitly across tasks, threads, workers, and processes.
- Fork an isolated stack before dispatch when the integration controls
  concurrent branch creation. When the host exposes only paired callbacks on a
  shared stack, retain out-of-order completions and pop only owned scopes after
  they reach the top.
- Close every accepted boundary exactly once on success, error, cancellation,
  timeout, and shutdown.
- Keep stable host call IDs when available. Do not substitute a parent scope ID
  for a tool call ID.

## Tool Calls

Prefer a managed tool wrapper when the host yields the callback. Relay should
own the wrapper; the host still owns validation, scheduling, retries, and
business behavior unless the approved design changes them.

- Project host arguments into JSON-compatible values.
- Return Relay's canonical tool execution result through the managed boundary,
  then unwrap it only where the host expects its native result type.
- Preserve structured error results such as MCP `isError: true` as tool results,
  not transport exceptions.
- Preserve cancellation and concurrency. Do not serialize parallel tools merely
  to simplify scope storage.
- Protect completion ownership when callbacks arrive on different threads. Do
  not hold a synchronous lock across an await or close a scope owned by another
  handler.
- If the host exposes only a prehook, use standalone conditional execution for
  blocking and standalone request intercepts for mutation. Do not claim
  execution intercepts ran.
- A hook on an MCP bridge covers the bridge invocation. Nested server-side tool
  calls require their own instrumentation and propagated context.

## Model Calls And Provider Codecs

Prefer managed LLM execution when the host yields the provider callback. Attach
the matching provider request codec and response codec when Relay supports the
wire format.

- Preserve the host's provider-native body and fields.
- Treat provider request and response codec guarantees separately. Current
  response codec annotations support observability; they do not make returned
  response mutation a general capability.
- Preserve stream mode and the host's lazy-consumption behavior.
- Keep the Relay operation, callback context, and owning parent alive until a
  stream is exhausted, fails, is cancelled, or is explicitly closed.
- If the host owns the iterator, use explicit lifecycle APIs unless it can hand
  Relay the actual stream callback and returned stream.
- If the integration uses the gateway, document which routes and providers are
  actually compatible. Gateway coverage does not replace host tool or session
  integration.
- Keep SDK-only options, clients, callbacks, streams, and credentials outside
  the provider request projection.

## Hooks And Standalone Middleware

- Call conditional execution only from an awaited hook whose rejection the host
  honors.
- Call request intercepts only when the hook can apply the returned value.
- Validate rewritten tool arguments if the host validated only before the hook.
- Use marks for milestones that do not represent an owned execution boundary.
- Replaying start/end events does not run managed middleware. State that
  limitation in code comments and support documentation where users could
  mistake it for parity.
- Avoid double instrumentation when a hook integration and gateway see the same
  model call. Decide which runtime owns the model span and how correlation is
  propagated.
- Treat a host callback that recursively enters managed Relay execution as a
  separate design case. Prevent double management and do not block the event
  loop that must drive the nested callback or Relay continuation.

## Packaging And Activation

- Prefer a stable public host extension or plugin API over monkeypatching
  internal call sites.
- Match the approved availability contract. An optional integration should
  preserve host operation when Relay is unavailable. A required first-party
  integration should surface an explicit unavailable or failed state. Decide
  loading and readiness behavior separately from policy fail-open or
  fail-closed behavior.
- Cover every shipped distribution separately, including bundled runtimes and
  native addon loading.
- Provide a positive activation signal and a health check. A warning hidden in
  install output is not sufficient when the host may silently skip an
  extension.
- Define configuration ownership, compatibility ranges, logging behavior,
  rollback, flush, and shutdown.
- Keep process-global activation ownership separate from live operation
  ownership. Final teardown must wait for accepted callbacks, streams, and
  delegated work even after the last profile or session releases configuration.
- Distinguish no requested configuration, requested activation failure, and
  pre-existing external activation in health or readiness state.
- Choose adapter release ownership explicitly. A host repository, the Relay
  repository, or a separate package may own it depending on coupling, adoption,
  compatibility, and long-term maintainership. Put reusable Relay behavior
  behind Relay plugin configuration. Do not use a Relay plugin to compensate
  for a missing host callback.

## Implementation Exit Criteria

- [ ] Exact host and Relay revisions are recorded.
- [ ] Each surface uses the method approved in the ownership matrix.
- [ ] Relay-disabled behavior matches the baseline host behavior.
- [ ] Activation is visible and failure semantics are documented.
- [ ] Tool and LLM payload projections are JSON-compatible and codec-correct.
- [ ] Errors, cancellation, retries, streams, subagents, and concurrent calls
      close and correlate correctly.
- [ ] No call is instrumented twice.
- [ ] Unsupported paths and escape hatches are documented without overstating
      coverage.
- [ ] Unit, contract, packaging, and live-smoke responsibilities have owners.
