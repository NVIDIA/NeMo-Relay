---
name: nemo-relay-integrate-upstream
description: Use this skill when assessing, extending, or implementing NeMo Relay support in an agent harness or agent framework, including coding agents and orchestration runtimes, when the host lacks Relay support or an existing integration needs deeper coverage. It identifies the host's execution boundaries and extension points, selects an appropriate attachment method for each boundary, and verifies the resulting coverage.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Integrate Relay Into An Agent Harness

Use this skill to determine the right NeMo Relay integration for an agent
framework or harness that lacks support or needs deeper coverage. First
identify how the host runs tools, model calls, streams, subagents, and lifecycle
events and what its extension points can control. Then choose the integration
depth and Relay attachment methods that meet the required coverage and fit the
host's architecture, support model, and lifecycle, using an extension, separate
adapter, core wiring, or hybrid as appropriate.

## Choose The Work Mode

Choose one mode from the user's request. Do not silently move from assessment
to implementation.

1. **Assess**: Determine whether and how Relay can integrate. Inspect only; do
   not edit the host, Relay, issues, or pull requests. Read
   [Assess The Host](references/assess-host.md).
2. **Implement**: Make an already approved integration change. Reconfirm the
   exact boundary and read [Implement The Integration](references/implement-integration.md).
3. **Qualify**: Test an existing integration at an exact host and Relay
   revision. Treat this as read-only unless the user also requests fixes. Read
   [Qualify The Integration](references/qualify-integration.md).

If the host already has a maintained Relay integration, treat it as the
starting point and verify its coverage against the requested outcome. Use it
when it owns the required boundaries. When it does not, identify the missing
ownership and evaluate whether to extend the existing integration, combine it
with another attachment method, or move a named boundary deeper into host core.
Do not create a competing adapter without first proving that gap. Moving from a
host plugin to core may be appropriate when requirements need deeper execution
ownership or when the intended first-party support, lifecycle, performance, and
release model call for core integration. Hermes uses core to own real callbacks
and lifecycle rather than relying on its former observability plugin.

If an application directly owns only a few tool or model callbacks, hand off to
`nemo-relay-instrument-calls`.

## Establish The Evidence Baseline

Before recommending a shape:

1. Pin the host: record its exact revision or release and every distribution
   form the integration must support.
2. Establish the Relay release baseline separately. Check the latest stable and
   prerelease GitHub releases, then read the matching Relay release notes,
   support matrix, migration guide, package metadata, and relevant API docs.
   Distinguish a published stable release from a release candidate, nightly or
   alpha artifact, and current `main`. Choose and record one Relay target with
   a reason; do not silently design against whichever source tree is open.
3. Build a host-to-Relay packaging matrix. For each host distribution, record
   its runtime and extension language, OS, architecture, libc, and package or
   bundle format. Compare it with the Python, Node.js, Rust, gateway, or
   advanced FFI or sidecar artifacts actually published for the selected Relay
   release. Verify compatible versions and test installation and loading in
   the real host distribution; source support or a CI build is not proof that
   a compatible artifact was published or can load in a bundled host.
4. Determine which host integration shapes exist: built-in Relay support, an
   optional plugin or extension system, public middleware or adapter APIs,
   direct core wiring, and configurable provider base URLs.
5. Inspect those surfaces in source and tests. Trace representative tool and
   model calls through retries, streaming, errors, cancellation, and subagents;
   classify whether each surface owns a callback, honors a decision or
   mutation, exposes paired lifecycle events, or only observes.
6. Map every required boundary to an exact public API in the selected Relay
   release. Then verify configuration, activation, health, cleanup, and
   duplicate-instrumentation behavior. Binding availability alone does not
   prove that the required API or semantics exist, and an integration that can
   be silently skipped needs an explicit verification story.

Use source and targeted tests as evidence. When only a closed binary or public
documentation is available, state that limitation and do not claim complete
coverage.

## Build An Ownership Matrix

Evaluate these surfaces independently:

- process, session, run, turn, and compaction lifecycle;
- tool request, execution, result, and MCP bridge behavior;
- LLM request, provider attempt, response, and streaming finalization;
- subagent creation and cross-task, thread, or process context propagation;
- approvals, human-in-the-loop decisions, retries, and queued work;
- configuration, activation, health, flush, and shutdown.

Split a broad surface into separate rows when its paths have different owners
or hooks. For example, do not combine a main-loop model call, provider retry,
compaction call, and extension-owned provider call into one LLM row.

For each independently owned path, record:

- the exact call path, Relay attachment point, covered dispatch families, and
  bypasses;
- logical-versus-physical cardinality across retries, streams, re-entry, and
  concurrency;
- the owning scope stack or execution context, callback thread or event loop,
  and whether siblings can finish out of order;
- callback or network ownership, hook ordering, honored returns, and whether
  the host can block, mutate, replace, or only observe;
- stable identity, parent context, JSON projection, and streaming behavior;
- success, failure, cancellation, retry, stream, and shutdown closure
  ownership, including which process component owns Relay activation and
  final drain.

Do not summarize a host as simply "hooked" or "not hooked." One useful hook
does not prove the other surfaces or every dispatch path within that surface.

## Classify The Host Attachment Contract

Read [Host Attachment And Hook Patterns](references/host-attachment-patterns.md).
Core adapters and wrapper middleware can support managed execution only when
they receive the real callback. Awaited prehooks can support blocking or
mutation only when the host applies their return value. Paired events support
manual lifecycle observation, while session and subagent events support causal
structure. A compatible base URL covers provider traffic only. Plugin lifecycle
hooks must make activation, health, drain, and cleanup visible.

## Audit Concurrency And Lifecycle

Read [Concurrency And Lifecycle](references/concurrency-and-lifecycle.md) when
the host can overlap requests, turns, tools, model attempts, streams,
subagents, callbacks, or shutdown. Determine whether the integration controls
task creation, receives only paired callbacks after scheduling, crosses threads
or event loops, or shares process-global Relay state across multiple host
objects.

A scope handle establishes identity and parentage; it is not the execution
context that owns the mutable scope stack or makes scope-local middleware
visible. Give every managed call one adapter owner, preserve the callback and
stream lifetime, and keep host-instance ownership separate from live Relay
operations and process-global plugin activation.

## Choose The Host Integration Depth

Use the evidence and ownership matrix to decide where Relay should attach.
Choose based on required coverage, host architecture, adoption model, support
and release ownership, operational lifecycle, and performance.

| Host integration shape | Choose it when | Main trade-off |
|---|---|---|
| Existing maintained integration | Relay already supports the host and the integration covers the required boundaries. | Do not create a competing adapter or duplicate instrumentation. |
| Optional host plugin or extension | A public extension API exposes enough awaited, mutable, and paired hooks for the required behavior. | Easy to adopt and remove, but limited to the ownership the host exposes. |
| Separate integration package | Public framework callbacks are sufficient, but the host has no suitable plugin registry or should not depend directly on Relay. | Keeps release ownership separate, but compatibility must be maintained across both projects. |
| Direct host-core integration | Required tool, model, stream, subagent, or lifecycle ownership is unavailable to extensions and Relay is intended to be a first-party capability. | Provides the deepest coverage but couples Relay to the host's runtime and release lifecycle. |
| Hybrid | Different surfaces have different owners, such as a host extension for tools and the Relay gateway for model traffic. | Often the honest shape, but activation, correlation, and duplicate instrumentation require explicit design. |

Do not confuse a host plugin with a Relay plugin. A host plugin or extension
attaches Relay to the harness. Relay plugins configure middleware, policy, and
exporters after the host has attached Relay to its execution boundaries.

Do not treat an extension or core integration as the default. Justify the
chosen depth from named requirements and the intended product relationship.

## Select A Relay Method Per Surface

| Host boundary | Preferred Relay method | Honest capability |
|---|---|---|
| Host yields the actual tool or LLM callback | Managed execution wrapper | Relay owns the full middleware and lifecycle boundary. |
| Host exposes paired start/end events only | Explicit lifecycle APIs | Observation and correlation; middleware does not run automatically. |
| Host awaits an allow-or-block prehook | Standalone conditional execution | Policy can reject before the host continues. |
| Host honors a mutable request prehook | Standalone request intercepts | Relay can return a rewritten request for the host to apply. |
| Host supports an OpenAI- or Anthropic-compatible base URL | Relay gateway | Managed provider traffic only; this does not cover tools or host lifecycle. |
| Host exposes a milestone without a call boundary | Mark | A correlated event, not managed execution. |
| Work crosses a task, thread, or process boundary | Explicit context propagation | Parentage only when the receiving side restores the context correctly. |

Choose these independently. For example, an extension may gate tools, replay
lifecycle events, and route model HTTP traffic through the gateway. Do not call
that a single managed integration.

## Keep Capability Claims Separate

Use these terms precisely:

- **Observable**: Relay receives enough data to emit an event.
- **Correlated**: Events have stable parent, call, and run identity.
- **Evaluable**: Relay can compute a policy decision.
- **Enforceable**: The host honors a reject or block before execution.
- **Mutable**: The host applies Relay's returned request or result.
- **Managed**: Relay owns the callback and executes its complete managed
  middleware and lifecycle sequence.
- **Complete**: Every in-scope path is covered, including retries, streams,
  subagents, errors, and alternate providers.

Never infer enforcement from evaluation, mutation from observation, managed
behavior from paired events, or tool coverage from a provider gateway. A tool
hook sees the declared tool call, not necessarily nested commands or other work
performed inside the tool.

## Produce A Decision Record

For an assessment or review, report:

1. exact revisions and distribution targets;
2. the ownership matrix with source or test evidence;
3. the recommended Relay method and proven capability for each surface;
4. known gaps and escape paths;
5. the required upstream, Relay, and deployment changes and their owners;
6. a deterministic qualification plan and unresolved ownership decisions.

Separate facts proven at the exact revision from hypotheses that require a
spike. Do not turn every missing capability into a Relay API proposal. When the
missing ownership belongs to the host, determine whether a stable extension
point or first-party core change should provide it.

## Handoffs

- First-time Relay setup -> `nemo-relay-install`, then
  `nemo-relay-get-started`.
- A few application-owned callbacks -> `nemo-relay-instrument-calls`.
- Scope leakage across concurrent work ->
  `nemo-relay-instrument-context-isolation`.
- Provider or typed value projection ->
  `nemo-relay-instrument-typed-wrappers`.
- Reusable configuration-driven Relay behavior -> `nemo-relay-plugin-build`.
- Missing events or activation failures ->
  `nemo-relay-debug-runtime-integration`.
