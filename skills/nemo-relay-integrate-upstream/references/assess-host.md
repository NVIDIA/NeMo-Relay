<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Assess The Host

Use this workflow to answer "Can Relay integrate with this framework?" without
changing either repository.

## 1. Pin The Host And Relay Releases

Record the host repository, exact commit or release, and every distribution in
scope.

Establish the Relay baseline independently:

1. Check GitHub Releases for the latest stable and prerelease versions.
2. Read `docs/about-nemo-relay/release-notes/index.mdx` at the candidate Relay
   revision.
3. Read the matching support matrix and migration guide.
4. Inspect package manifests and release workflows for the binding or binary
   under consideration.
5. Choose one target: a stable release, a named prerelease, or an exact `main`
   commit. State why that target is appropriate.

Do not describe a repository version or nightly tag as the latest stable
release. Do not use `main` API semantics while proposing an integration pinned
to an older published package.

## 2. Build The Host And Relay Packaging Matrix

For each host distribution, record:

- whether it is a library, CLI, desktop app, server, or hosted runtime;
- OS, architecture, and libc where relevant;
- package, installer, container, bundle, or standalone-binary format;
- host runtime and minimum supported version;
- the language and runtime available at each extension or core boundary;
- whether the path is public, experimental, private, or unavailable as source.

Treat separate distributions independently. A native addon loading in an npm
package does not prove that it loads in a bundled binary.

Audit the artifacts actually published for the selected Relay release, then
complete this matrix:

| Host distribution | Host and extension runtime | OS / architecture / libc | Relay surface and package | Published and version-compatible | Install and load evidence | Gap or decision |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

Use this Relay mapping only as a starting point:

| Host execution environment | Relay surface to evaluate | Boundary |
|---|---|---|
| Python process | `nemo-relay` Python binding | In-process scopes, managed calls, lifecycle APIs, middleware, and plugin activation. |
| Node.js-compatible process | `nemo-relay-node` binding | In-process scopes, managed calls, lifecycle APIs, middleware, and plugin activation. |
| Rust process | `nemo-relay` Rust crate | Native in-process access to Relay's public Rust APIs. |
| Any host with compatible provider base URLs | Relay gateway | Provider HTTP traffic only; not host tools, sessions, or subagents. |
| Unsupported in-process language or runtime | Raw C FFI or a sidecar, after a focused spike | Advanced integration with explicit ABI, serialization, threading, lifetime, and shutdown ownership. |

Confirm the API surface against the target Relay revision. Then test that the
actual package or native artifact loads in each host distribution. Do not infer
Node.js addon compatibility for Bun or a bundled executable, Python wheel
availability for an unsupported platform, or raw FFI safety from language
compatibility alone.

Distinguish all of these outcomes:

- source code supports the target;
- CI builds or tests the target;
- the release publishes the required artifact;
- the host package manager installs that artifact;
- the host's actual runtime or bundle loads it successfully.

Only the final result proves that the proposed in-process attachment works for
that distribution.

## 3. Trace The Host From Source

Search for the code that owns:

- session, run, turn, and compaction start/end;
- tool registration, validation, dispatch, execution, and result handling;
- LLM request construction, provider client creation, retries, response
  parsing, and stream consumption;
- subagent scheduling, task/thread/process boundaries, cancellation, and
  shutdown;
- extension, middleware, callback, hook, event, and plugin registration;
- base URL, proxy, provider, and credential configuration;
- health, readiness, install, activation, logging, flush, and cleanup.

Read the implementation and the tests. A hook name such as `before_tool` does
not prove that the host awaits it or honors its return value.

## 4. Classify Each Extension Point

For every candidate hook, answer:

| Question | Why it matters |
|---|---|
| Does the host pass the actual callback? | Required for managed execution and execution intercepts. |
| Does the host await the hook? | Required for an in-process decision to block reliably. |
| Is the return value applied? | Required for request or result mutation. |
| Are both start and end emitted? | Required for balanced manual lifecycle events. |
| Are errors and cancellation represented? | Required to close scopes and calls truthfully. |
| Are retries visible individually? | Required for provider-attempt coverage. |
| Is a stream exposed before consumption? | Determines whether Relay can own or only observe streaming. |
| Are stable IDs and parent context available? | Required for correlation under concurrency. |
| Which thread, event loop, or task invokes each hook? | Determines how Relay context and async callbacks can be entered without blocking or re-entering one mutable context. |
| Can sibling runs finish out of start order? | Determines whether branches need isolated stacks or callback-safe deferred completion. |
| Is the extension instance process-, profile-, session-, or request-scoped? | Determines who may initialize, share, and tear down process-global Relay state. |
| Does it cover built-ins, extensions, MCP, and subagents? | Prevents a narrow hook from being called universal. |

## 5. Choose The Attachment Shape

Choose the combination that provides the required coverage and fits the host's
architecture, intended product relationship, and support model:

- **Managed native integration**: the host yields real tool and model callbacks.
- **Hook integration**: the host exposes usable pre/post extension points but
  retains execution ownership.
- **Gateway integration**: provider traffic can be routed through Relay by
  changing a compatible base URL.
- **Hybrid integration**: hooks cover tools and host lifecycle while the
  gateway owns model HTTP calls.
- **Manual instrumentation**: the application owns a small number of direct
  callbacks and does not need a framework adapter.
- **Not currently supportable**: the host exposes neither usable boundaries nor
  compatible provider routing for the requested capability.

Do not choose FFI merely because the host is not written in Python, Node.js, or
Rust. First check whether an upstream extension can call a supported binding or
whether gateway plus hooks meets the need. Use raw C FFI only after proving the
runtime, packaging, and lifetime model.

## 6. Audit False Positives And False Negatives

Check for false positives:

- A successful import is not proof that hooks registered.
- A start event is not proof that the matching end event exists.
- An LLM base URL is not tool or subagent coverage.
- A computed rejection is not enforcement unless the host stops.
- A returned mutation is not effective unless the host applies it.
- One main-loop request is not retry, compaction, summarization, or extension
  provider coverage.
- One sequential request is not proof that scopes close correctly when sibling
  callbacks finish out of order.
- A returned stream object is not proof that its managed operation stays alive
  through exhaustion, cancellation, failure, and explicit close.
- A tool name and arguments do not reveal nested shell, network, or MCP work.
- A scripted provider proves protocol mechanics, not live-service readiness.

Check for false negatives:

- A host may allow a hybrid even when no single hook provides full coverage.
- Paired events can still provide valuable correlation without managed
  middleware.
- A provider gateway may cover hidden model calls that payload hooks miss.
- A generic plugin or callback registry may be sufficient even when Relay is
  not named by the host.
- Missing host ownership may require a stable host extension or core change
  rather than a new Relay API.

## Assessment Output

Use one row for each independently owned path. Split broad surfaces when main
loop, retry, compaction, subagent, extension, gateway, or nested calls have
different owners or hook coverage.

| Surface and dispatch family | Host owner and Relay attachment point | Logical/physical cardinality and identity | Relay method and proven capability | Covered paths and bypasses | Close/error owner | Evidence |
|---|---|---|---|---|---|---|
| Session/run |  |  |  |  |  |  |
| Tool request/execution/result |  |  |  |  |  |  |
| LLM request/attempt/response |  |  |  |  |  |  |
| Streaming |  |  |  |  |  |  |
| Subagent/context |  |  |  |  |  |  |
| Approval/HITL |  |  |  |  |  |  |
| Configuration/activation |  |  |  |  |  |  |
| Flush/shutdown |  |  |  |  |  |  |

End with one recommended shape, the bounded spikes needed to resolve unknowns,
and explicit non-goals. Do not estimate implementation work until the ownership
and distribution questions are answered.

For hosts with overlapping work, complete the execution-model and ownership
checklist in [Concurrency And Lifecycle](concurrency-and-lifecycle.md) before
recommending core, plugin, package, gateway, or hybrid attachment.
