<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Concurrency And Lifecycle

Use this reference when the host overlaps requests, turns, tools, provider
attempts, streams, subagents, callbacks, or shutdown. Trace the host's real
scheduler and completion paths; do not infer them from the primary language or
an architecture diagram.

## Build The Execution Model

Record these facts for every CLI, server, background-job, core, and plugin
path:

| Question | Why It Matters |
|---|---|
| Can turns overlap within one session? | One mutable scope stack may receive non-LIFO completion. |
| Can tools or provider attempts run in parallel? | Each branch needs explicit context and completion ownership. |
| Which task, thread, or event loop invokes each callback? | Blocking or re-entering the wrong context can deadlock or corrupt ownership. |
| Can a callback start nested tool or model work? | The call can be managed twice or cross event-loop ownership. |
| Is provider execution lazy or streaming? | The operation outlives stream construction. |
| Can subagents outlive the parent turn? | The parent must wait, detach, or propagate ownership explicitly. |
| Can compaction, reset, or shutdown race live work? | Parent and process-global resources cannot close under active children. |

## Audit The Attachment Point

Host attachment and Relay middleware are separate layers. The host adapter
finds a real execution boundary; Relay runs only the middleware reachable from
the API called at that boundary.

| Attachment | Inspect |
|---|---|
| Host core | Canonical dispatchers, optional disabled path, duplicate wrappers, process owner, and release coupling. |
| Host plugin | Instance lifetime, awaited hooks, callback ownership, thread or loop, stream retention, health, and stop behavior. |
| Separate package | Public API and version contract, scheduler ownership, and protection against a second adapter. |
| Gateway or sidecar | Provider-only coverage, transport correlation, streaming cancellation, retry, and backpressure. |

Use a host plugin only for the boundaries its extension API actually exposes.
Move a named boundary into core when extensions cannot own it and first-party
support justifies the coupling. A hybrid is valid, but every managed call and
process-global Relay configuration must still have one owner.

## Choose Scope-Stack Ownership

Relay scope stacks are mutable and strict LIFO. A scope handle identifies a
scope and can establish parentage; it does not activate the stack that owns the
scope or restore scope-local registrations.

| Host work | Stack design |
|---|---|
| Sequential nested work | Reuse the active stack. |
| Sequential task or thread handoff | Bind the stored stack for the callback and restore the worker's previous binding. |
| Concurrent siblings when the adapter controls dispatch | Fork an isolated stack per branch before dispatch. |
| Concurrent siblings exposed only as callbacks on one stack | Record completion and pop only owned completed scopes that reach the top. |
| Independent requests or tenants | Create a fresh isolated stack. |
| Cross-process work in the same trace | Send authenticated propagation context and create an isolated receiving stack. |

Forking preserves Relay event parentage but does not transfer scope-local
middleware or subscribers. Install required branch-local behavior explicitly
or choose a serialized design when policy must remain local to the parent
stack.

When callbacks force overlapping siblings onto one stack, never pop through a
live scope. Snapshot output, status, error, and the real end time when the host
signals completion. Under one re-entrant synchronization boundary, drain
completed scopes from the current top until reaching a live or unowned scope.
Remove a completion only after Relay accepts the pop so a transient failure can
be retried. Degrade invalid output without abandoning the scope. Bound and
clean retained entries for scopes closed outside the adapter.

Relay's LangChain callback handler and its real-stack tests are a source example
of deferred close. Recheck the selected Relay revision rather than copying its
private implementation blindly.

## Preserve Callback And Stream Ownership

- Invoke the host callback once unless a selected Relay execution intercept
  deliberately invokes its continuation more than once.
- Apply Relay's rewritten request before calling the host.
- Preserve the host result or error when adapter cleanup fails afterward.
- Do not hold a synchronous lock across an await.
- Copy context values for overlapping callbacks; do not concurrently enter one
  mutable Python `Context` or leave a Relay stack bound to a pooled thread.
- Treat recursive managed entry as an explicit design. Keep the owning event
  loop running or choose one boundary to remain unmanaged.

A stream remains live until it is exhausted, fails, is cancelled, or is
explicitly closed. Keep the Relay runtime, operation lease, callback context,
and owning parent scope alive for that entire interval. Close provider and
Relay streams exactly once before closing their parent. Test lazy callbacks,
prefetched first chunks, partial consumption, never-iterated streams, and a
complete response returned from a nominal streaming API.

## Own Process-Global Lifecycle Once

Relay plugin activation, global middleware, and global subscribers are
process-wide even when the host creates profile- or session-scoped adapters.
Track configuration owners separately from callbacks, streams, and deferred
children that still use Relay. Scope depth is not a live-operation count.

Use this shutdown order:

1. Stop accepting new managed operations.
2. Wait for or cancel accepted callbacks, streams, and delegated work.
3. Close child, turn, session, and profile scopes in ownership order.
4. Drain subscriber and exporter publication once at the process or activation
   boundary.
5. Clear middleware or close activation.

Do not flush process-global subscribers as per-session cleanup. Keep activation
state truthful: no requested configuration, requested configuration failure,
successful activation, and pre-existing external activation are distinct
states.

## Qualification Checklist

Use the real host scheduler and selected Relay binding. Test:

- two turns and parallel tools finishing in both possible orders;
- isolated branch stacks and callback-based deferred close;
- two handlers sharing a stack without closing each other's scopes;
- task, thread, worker-pool, and process context boundaries;
- nested managed tool or model work;
- callback failure, policy rejection, timeout, cancellation, and retry;
- stream exhaustion, error, cancellation, explicit close, and no iteration;
- subagents finishing before, after, and during parent cancellation;
- compaction or session rotation during a live turn;
- the last host owner closing while an operation remains active;
- shutdown racing live work and queued publication;
- multiple host profiles sharing process policy without sharing request scopes.

Assert host-visible behavior, callback and middleware counts, parentage, close
order, actual completion timestamps, activation state, drain behavior, and the
absence of leaked scopes or tasks.
