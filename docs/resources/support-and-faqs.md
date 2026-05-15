<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Support and FAQs

Use this page to decide where to start, which runtime surface to use, and where
to look when a NeMo Flow workflow does not behave as expected.

## Library Positioning

Use these questions to understand what NeMo Flow is, what it is not, and how it
fits into the agent and NVIDIA NeMo ecosystem.

### What Is NeMo Flow Responsible For?

NeMo Flow provides shared runtime instrumentation for scopes, tool calls, LLM
calls, middleware, lifecycle events, subscribers, plugins, and adaptive
optimization. It gives applications and framework integrations a consistent
execution model across supported bindings.

NeMo Flow sits inside an application, framework, or integration and makes
runtime behavior observable, policy-aware, and reusable.

### Is This An Agent Framework?

NeMo Flow is an agent runtime framework, not a full agent application
framework. It does not decide which agent pattern to use, choose a planner, own
memory, provide a hosted workbench, or replace an existing harness.

Use NeMo Flow when you want the tool calls, LLM calls, scopes, middleware,
plugins, and observability inside an agent system to follow one runtime model.

### What Is NeMo Flow Not?

NeMo Flow is not:

- A model provider
- A vector database
- A hosted tracing service
- A prompt authoring environment
- A full agent framework or agent workbench
- A replacement for application, framework, or provider SDK code

It is the runtime layer that gives those systems shared execution scopes,
middleware, lifecycle events, subscribers, plugins, and adaptive behavior.

### How Does This Differ From NeMo Agent Toolkit?

NeMo Flow is the lower-level runtime layer for scopes, middleware, events,
plugins, and observability around tool and LLM execution. NeMo Agent Toolkit is
a higher-level agent toolkit in the NVIDIA NeMo ecosystem.

Use NeMo Agent Toolkit to build, optimize, and run agent workflows. Use NeMo Flow
inside an application, harness, framework integration, or plugin when you need
consistent runtime instrumentation and policy behavior around the actual tool
and model calls.

### How Does NeMo Flow Relate To Other NVIDIA NeMo Products?

NeMo Flow belongs in the NVIDIA NeMo ecosystem, but it has a specific role: it
is a runtime instrumentation and policy layer for agent execution. It does not
replace model training, model serving, guardrail authoring, data pipelines, or
agent application frameworks provided by other NeMo projects.

When another NeMo product or framework owns the high-level workflow, NeMo Flow
can be used at the execution boundaries where scopes, lifecycle events,
middleware, subscribers, exporters, or adaptive plugins are needed.

### Does NeMo Flow Orchestrate Agents?

No. NeMo Flow does not choose the next step, schedule a multi-agent workflow,
own a planner, or decide which tool an agent should call. That remains the
responsibility of the application, framework, or agent harness.

NeMo Flow observes and controls the execution boundaries that the orchestrator
uses: scopes, tool calls, LLM calls, middleware, events, subscribers, and
plugins.

### Why The Name "NeMo Flow"?

"NeMo" places the project in the NVIDIA NeMo ecosystem. "Flow" refers to the
runtime flow of agent work through scopes, middleware, events, subscribers,
plugins, and exporters.

The name is about the execution path NeMo Flow makes visible and controllable;
it is not a claim that NeMo Flow owns the full agent workflow or orchestration
layer.

## Technology And Bindings

Use these questions to choose a supported binding and understand the runtime
implementation model.

### What Are The Main Technologies Used?

The core runtime is written in Rust and uses JSON-compatible payloads for the
shared runtime boundary. The primary documented bindings are Rust, Python, and
Node.js. Python uses a native PyO3 extension, Node.js uses a NAPI binding, Go
uses the raw C FFI, and WebAssembly uses a `wasm-bindgen` binding.

The documentation and examples also cover integration with ATIF trajectory
export, OpenTelemetry traces, OpenInference-compatible data, and third-party
agent framework patch sets.

### Why Is NeMo Flow's Core Written In Rust?

Rust gives NeMo Flow one native source of truth for runtime behavior while
keeping overhead low at hot tool and LLM boundaries. It also gives the project
strong ownership, error, and async primitives for scope stacks, middleware
registries, callbacks, subscribers, and binding-facing FFI layers.

The language bindings expose that shared model to application code without
requiring each binding to reimplement the runtime semantics independently.

### Which Bindings Should I Start With?

Start with Python, Node.js, or Rust. These are the primary documented bindings
and have the broadest getting-started, concept, guide, and generated API
coverage.

- Use [Python Quick Start](../getting-started/python/index.md) when you are adding
  NeMo Flow to Python application code or agent harnesses.
- Use [Node.js Quick Start](../getting-started/nodejs.md) when your application,
  framework integration, or plugin-facing code runs in Node.js.
- Use [Rust Quick Start](../getting-started/rust.md) when you want the native
  runtime surface or are building lower-level integrations.

### Are Go, WebAssembly, And FFI Supported?

Go, WebAssembly, and the raw C FFI surface are experimental and source-first.
Use the repository source tree and binding-specific tests when you need those
surfaces.

### Does Middleware Registered In One Language Work In Another Language?

Middleware registrations are runtime-local and process-local. A Python callback
registered as middleware does not automatically execute inside a separate
Node.js, Rust, Go, or WebAssembly process.

The runtime semantics are shared across bindings, so the same policy can be
implemented or packaged consistently in each language. For cross-language
behavior, install equivalent middleware or plugin behavior in each process, or
put the shared implementation behind a service or native component that each
binding can call.

## Getting Started

Use these questions to find the right first guide for your task.

### What Should I Read First?

Use the reading path that matches your task:

| Task | Start With |
|---|---|
| Run a minimal example | [Quick Start](../getting-started/quick-start.md) |
| Install packages or local source | [Installation](../getting-started/installation.md) |
| Understand the runtime model | [Concepts](../about/concepts/index.md) |
| Instrument an application | [Instrument Applications](../instrument-applications/about.md) |
| Integrate a framework | [Integrate into Frameworks](../integrate-frameworks/about.md) |
| Observe a local coding-agent CLI | [NeMo Flow CLI](../nemo-flow-cli/about.md) |
| Package reusable behavior | [Build Plugins](../build-plugins/about.md) |
| Export traces or trajectories | [Observability](../plugins/observability/about.md) |
| Configure adaptive behavior | [Adaptive](../plugins/adaptive/about.md) |
| Look up symbols | [API](../reference/api/index.md) |

## Runtime Model And Execution

Use these questions when deciding where runtime work belongs and which
execution API to call.

### When Should I Add Scopes?

Add a scope around each meaningful unit of work: an agent run, user request,
workflow step, tool call group, LLM call group, evaluator, retriever, or custom
operation that needs ownership in the event stream.

Scopes establish parent-child relationships for events and define the lifetime
for scope-local middleware and subscribers. Refer to [Scopes](../about/concepts/scopes.md)
and [Adding Scopes and Marks](../instrument-applications/adding-scopes-and-marks.md).

### How Does NeMo Flow Handle Multiple Concurrent Agents Or Requests?

Use an isolated scope stack for each concurrent request, tenant, worker, or
agent run. The root scope identifies the run, and emitted events include root
identity so subscribers and exporters can separate overlapping work.

Refer to [Scopes](../about/concepts/scopes.md#context-isolation) and the
[scope and context helper examples](../instrument-applications/code-examples.md#scope-and-context-helpers).

### When Should I Emit A Mark Instead Of A Scope?

Use a scope when the work has a start and an end. Use a mark for a
point-in-time milestone that should appear in the event stream but does not
represent a full lifecycle span, such as "routing complete" or "selected
candidate plan."

Refer to [Events](../about/concepts/events.md) and
[Adding Scopes and Marks](../instrument-applications/adding-scopes-and-marks.md#when-to-add-marks).

### Should I Use Managed Execution Helpers Or Manual Lifecycle APIs?

Use managed execution helpers when the application or framework exposes a
stable callable boundary. Managed helpers preserve middleware order, emit start
and end events, run execution intercepts around the real callback, and keep
subscriber payloads consistent.

Use manual lifecycle APIs only when the framework owns the invocation internally
and you only have start and finish hooks. In that case, preserve the start
handle, emit the matching end event, and handle error paths deliberately. See
[Invocation API Selection](../instrument-applications/code-examples.md#invocation-api-selection)
and [Preferred Integration Order](../integrate-frameworks/code-examples.md#preferred-integration-order).

### How Are Tool Calls And LLM Calls Different?

Both use the same scope, middleware, and event model. Tool calls represent
structured function or tool execution, while LLM calls represent model-provider
requests and responses. LLM calls also carry model-oriented metadata, and
streaming LLM flows can use stream-specific execution behavior.

Start with [Instrument a Tool Call](../instrument-applications/instrument-tool-call.md)
or [Instrument an LLM Call](../instrument-applications/instrument-llm-call.md).

### Does NeMo Flow Support Streaming LLM Responses?

Yes. Streaming LLM workflows have a stream execution path so middleware can run
around chunk delivery and finalization, not only around a single response
object. Use the streaming helpers when callers need incremental output or when
subscribers should observe streaming lifecycle activity.

Refer to [Streaming LLM Execution](../instrument-applications/code-examples.md#streaming-llm-execution)
and [Wrap LLM Calls](../integrate-frameworks/wrap-llm-calls.md#streaming-providers).

## Middleware, Policy, And Data

Use these questions when adding policy, request rewriting, execution wrappers,
or payload handling.

### Which Middleware Type Should I Use?

Choose the surface based on what must change:

| Need | Use |
|---|---|
| Allow or reject work before execution | Conditional-execution guardrail |
| Rewrite the real request before execution | Request intercept |
| Wrap, route, retry, or time the real callback | Execution intercept |
| Rewrite only emitted request or response payloads | Sanitize guardrail |
| Wrap streaming chunk delivery and finalization | Stream execution intercept |

Refer to [Middleware](../about/concepts/middleware.md) and
[Add Middleware](../instrument-applications/advanced-guide.md).

### What Is The Middleware Pipeline?

The middleware pipeline is the ordered runtime path that a managed tool or LLM
call follows before, during, and after the real callback. It is how NeMo Flow
applies policy, request transformation, execution wrapping, and observability
sanitization without moving that logic into every call site.

For managed execution, the pipeline runs:
- Conditional guardrails
- Request intercepts
- Request sanitization for emitted start events
- Execution intercepts
- The original callback
- Response sanitization for emitted end events

### What Is The Difference Between Guardrails And Intercepts?

Intercepts affect the real execution path. Request intercepts can change the
input passed to the callback, and execution intercepts can wrap or replace the
callback invocation.

Guardrails enforce policy or sanitize observability payloads. Conditional
guardrails can reject work. Sanitize guardrails change what subscribers and
exporters see, but they do not change the actual callback input or returned
value.

### Should Middleware Be Global Or Scope-Local?

Use global registrations for process-wide defaults. Use scope-local
registrations when a policy, intercept, or subscriber should apply only to one
request, workflow, or nested unit of work and disappear automatically when that
scope closes.

Use plugins when the behavior should be reusable and installed from
configuration. Refer to [Registration Levels](../about/concepts/middleware.md#registration-levels)
and [Middleware Registration Families](../instrument-applications/advanced-guide.md#middleware-registration-families).

### How Does Middleware Ordering Work?

Managed execution applies conditional guardrails first, then request intercepts,
then request sanitization for emitted start events, then execution intercepts
and the real callback, then response sanitization for emitted end events.

Registries are priority ordered. When scope-local behavior is present, NeMo Flow
combines applicable global and ancestor scope-local entries into the execution
chain. Refer to [Managed Execution Order](../about/concepts/middleware.md#managed-execution-order).

### What Data Can Middleware And Events Carry?

Middleware and event payloads should be JSON-compatible. Keep SDK clients,
streams, sockets, callbacks, file handles, and framework-specific object
instances outside NeMo Flow payloads.

When a framework exposes non-serializable objects, pass stable IDs or summarized
metadata through NeMo Flow and keep the original objects in framework-owned
storage. Refer to [Handle Non-Serializable Data](../integrate-frameworks/non-serializable-data.md).

### How Do I Keep Sensitive Data Out Of Observability?

Use sanitize-request and sanitize-response guardrails before production
subscribers or exporters receive payloads. Keep credentials out of plugin
configuration and source code, and emit summarized metadata when full request or
response bodies are not needed.

Refer to [Middleware](../about/concepts/middleware.md#guardrails) and
[Observability](../plugins/observability/about.md).

## Observability And Export

Use these questions when inspecting emitted events or exporting runtime data.

### How Are Events Observed?

Subscribers consume lifecycle events in process. Start with a local subscriber
when validating instrumentation, then add exporter-oriented subscribers for
operational tracing, trajectory export, or analytics.

Refer to [Subscribers](../about/concepts/subscribers.md),
[Events](../about/concepts/events.md), and
[Subscribers](../about/concepts/subscribers.md).

### Which Exporter Should I Use?

Use a local subscriber for debugging and development. Use OpenTelemetry when you
want OTLP-compatible traces in existing observability infrastructure. Use
OpenInference when your tracing stack expects OpenInference-style agent and LLM
semantics. Use ATIF when you need trajectory artifacts for analysis, replay, or
evaluation workflows.

Refer to [Exporter Selection](../plugins/observability/about.md#exporter-selection),
[OpenTelemetry](../plugins/observability/opentelemetry.md),
[OpenInference](../plugins/observability/openinference.md), and
[ATIF](../plugins/observability/atif.md).

### Can I Use NeMo Flow Just For Observability Without Adaptive Optimization Or Middleware?

Yes. Adaptive optimization and custom middleware are optional. You can start by
adding scopes, routing tool or LLM calls through managed execution helpers or
manual lifecycle APIs, and registering subscribers or exporters.

If no middleware is registered, managed execution still emits consistent start,
end, and mark events and keeps calls attached to the active scope. Add middleware
or adaptive plugins later only when you need policy, request rewriting, execution
wrapping, or adaptive behavior.

## Plugins And Adaptive Behavior

Use these questions when packaging reusable behavior or enabling adaptive
runtime components.

### What Are Plugins For?

Plugins package reusable runtime behavior behind configuration. A plugin can
validate configuration, initialize components, and register middleware,
subscribers, or related runtime helpers without requiring every application to
hand-register the same behavior.

Plugins install behavior into the same runtime model; they do not replace
scopes, middleware, or subscribers. Refer to [Plugins](../about/concepts/plugins.md)
and [Build Plugins](../build-plugins/about.md).

### When Should I Build A Plugin Instead Of Application Code?

Keep behavior in application code when it is specific to one service, request
path, or experiment. Build a plugin when the behavior should be reused across
applications, activated through configuration, validated before startup, or
reported through structured activation diagnostics.

Refer to [Define a Plugin](../build-plugins/basic-guide.md) and
[Register Plugin Behavior](../build-plugins/register-behavior.md).

### How Is Adaptive Optimization Enabled?

Adaptive optimization is activated through the plugin system. Start with
telemetry and in-memory state so the runtime can observe representative
workflows before changing behavior. Enable active behavior one area at a time,
such as adaptive hints, tool parallelism, or cache-governor behavior.

Refer to [Adaptive](../plugins/adaptive/about.md),
[Adaptive Configuration](../plugins/adaptive/configuration.md),
[ACG](../plugins/adaptive/acg.md), and
[Adaptive Hints](../plugins/adaptive/adaptive-hints.md).

## Framework Integration And APIs

Use these questions when integrating framework-owned call paths, provider
payloads, generated references, or third-party patch sets.

### How Should Framework Integrations Be Designed?

Prefer managed execution wrappers when the framework exposes a callable
boundary. Fall back to explicit start and end lifecycle calls when the framework
owns execution. Use standalone conditional-execution or request-intercept
helpers only when the framework needs those decisions before it invokes its own
downstream code.

Refer to [Framework Integrations](../about/concepts/framework-integrations.md) and
[Integrate into Frameworks](../integrate-frameworks/about.md).

### How Does NeMo Flow Connect To My Favorite Agent Harness Or Framework?

Connect NeMo Flow at the stable boundaries where the harness or framework
starts a run, invokes a tool, calls an LLM provider, streams model output, or
emits lifecycle milestones.

Use managed execution wrappers when the framework can expose the real callback
to NeMo Flow. Use explicit start and end lifecycle APIs when the framework owns
the invocation internally. Use codecs when the framework uses typed or
provider-specific payloads but middleware and events need JSON-compatible data.

### When Do I Need Codecs?

Use typed value codecs when application types need stable conversion at a tool
or LLM boundary. Use provider codecs when a provider payload needs to be decoded
into a normalized request shape for middleware and encoded back before the real
provider call. Use provider response codecs when provider responses need a
normalized annotated shape for events or downstream consumers.

Refer to [Using Codecs](../integrate-frameworks/using-codecs.md),
[Provider Codecs](../integrate-frameworks/provider-codecs.md), and
[Provider Response Codecs](../integrate-frameworks/provider-response-codecs.md).

### How Are Third-Party Integrations Maintained?

Some framework integrations are maintained as patch sets against pinned
upstream source checkouts under `third_party/`. Use the repository wrapper
scripts when checking or applying those patches, and regenerate the patch after
changing files inside a submodule.

Refer to [Framework Integration Code Examples](../integrate-frameworks/code-examples.md#quickstart-apply-maintained-patches)
and the repository root `third_party/README.md`.

### Where Are The API Docs?

Use [API](../reference/api/index.md) for generated symbol-level documentation.
The primary generated entry points are [Python API](../reference/api/python/index.md),
[Node.js API](../reference/api/nodejs/index.md), and
[Rust API](../reference/api/rust/index.md).

Go, WebAssembly, and raw FFI are experimental and source-first; use their source
directories and tests when you need exact behavior.

### What Should I Know About Performance?

Keep middleware focused, avoid unnecessary payload copies, and prefer
scope-local behavior when policy should apply only to one request or workflow.
Sanitize or summarize large payloads before production subscribers and
exporters consume them.

Refer to [Performance](../reference/performance.md) for runtime-model guidance and
related topics.

## Troubleshooting, Releases, And Contributing

Use these questions when something fails, when you need project status, or when
you are preparing a contribution.

### What Should I Check When Something Fails?

Start with [Troubleshooting Guide](../troubleshooting/troubleshooting-guide.md).
Common checks include:

- Confirm the environment matches [Prerequisites](../getting-started/prerequisites.md).
- Verify work is running inside the intended scope stack.
- Check whether middleware was registered globally or scope-locally.
- Confirm manual lifecycle calls emit matching start and end events.
- Inspect emitted events before adding production exporters.
- Ensure middleware and event payloads are JSON-compatible.

### Where Are Release Notes?

Use [Release Notes](../about/release-notes/index.md) for
documentation-visible release status. Complete release history is published
through the repository release page when that page is available.

### Where Do I Report Issues?

Use the repository issue tracker when it is available for the project. Include
the binding, version, operating system, reproduction steps, and any relevant
logs or emitted events.

### Where Should Contributors Start?

Start with [Contribute](../contribute/about.md) and the repository root
`CONTRIBUTING.md`, then use
[Development Setup](../contribute/development-setup.md),
[Workflow And Reviews](../contribute/workflow-and-reviews.md), and
[Testing And Documentation](../contribute/testing-and-docs.md).

### How Will AI Coding Assistants Find NeMo Flow?

AI coding assistants should start from the repository root `AGENTS.md`,
`README.md`, and the documentation index. Those entry points describe the
runtime model, repository layout, supported bindings, validation commands, and
contribution workflow.

For symbol-level work, assistants should use the generated Rust, Python, and
Node.js API references. For repository-specific automation, use the NeMo Flow
agent skills under `skills/` and keep examples aligned with the public docs.

### Which Tests Should I Run For A Change?

Choose the smallest validation set that covers the touched surface:

- Rust core or adaptive changes: `cargo test --workspace` or focused crate tests.
- Python binding changes: `uv run pytest`.
- Node.js binding changes: `npm test --workspace=nemo-flow-node`.
- Go binding changes: build the release FFI library first, then run Go tests under `go/nemo_flow`.
- WebAssembly changes: run `just test-wasm` and the WebAssembly crate tests
  (`cargo test -p nemo-flow-wasm`) when integration behavior changed. For
  focused debugging, you can run `wasm-pack test --node crates/wasm` directly.
- Documentation changes: run `./scripts/build-docs.sh html`.

Refer to [Testing and Documentation](../contribute/testing-and-docs.md) for the
current contribution workflow.

### What License Applies?

NeMo Flow is licensed under Apache-2.0. Source and documentation files use SPDX
headers. Refer to [Legal](legal/index.md) and the repository root `LICENSE`.
