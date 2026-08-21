<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host Attachment And Hook Patterns

Use these shapes to classify a host API. They are illustrative contracts, not
API names to copy. Verify the exact host and Relay revisions in source and
tests before implementing against them.

## Separate Host Attachment From Relay Middleware

For an upstream integration, Relay middleware is usually a coverage
requirement, not the mechanism that attaches Relay to the host. First identify
a host-owned tool, provider, stream, or lifecycle boundary. Then call the Relay
API that makes the required middleware reachable.

The adapter may live in host core, a host plugin, or a separate package. That
location does not prove coverage:

- A callback-owning adapter can use managed execution and reach the complete
  managed middleware sequence.
- An awaited decision hook can reach standalone conditional execution only
  when the host honors rejection.
- A mutable prehook can reach request intercepts only when the host applies the
  returned request.
- Paired notifications can replay lifecycle events, but they do not run
  managed conditional, request, or execution middleware automatically.
- Observation-only hooks can emit events but cannot control the host operation.

Do not implement the upstream adapter as a Relay execution intercept merely to
attach Relay to the host. Use execution intercepts for reusable Relay behavior
inside calls that already enter managed execution. Do not confuse a host plugin
with a Relay plugin: the host plugin attaches Relay; Relay plugins install
policy, middleware, subscribers, and exporters after attachment.

## Core-Owned Managed Adapter

A first-party host can place a small Relay adapter at its canonical tool and
provider dispatchers instead of exposing Relay through a host plugin:

```python
async def execute_provider(request, provider_callback, context):
    return await relay.llm.execute(
        context.operation,
        project_request(request),
        lambda rewritten: provider_callback(apply_request(request, rewritten)),
        handle=context.parent,
        codec=context.request_codec,
        response_codec=context.response_codec,
    )
```

Use this shape when the host intends Relay to be a first-party capability and
public extensions cannot provide the required callback, streaming, subagent,
or lifecycle ownership.

Keep the core integration bounded:

- centralize Relay loading, scope ownership, tool wrapping, and provider
  wrapping in small adapter modules;
- call those adapters from the host's canonical dispatchers rather than every
  provider and tool implementation;
- preserve a Relay-disabled or unsupported-platform path;
- distinguish logical model calls from physical provider attempts and retries;
- propagate context through the host's real task and subagent boundaries;
- preserve host-native request, response, stream, error, and cancellation
  behavior;
- enumerate dispatchers that intentionally bypass managed Relay execution;
- keep Relay runtime plugin activation separate from the host attachment.

Hermes uses this shape. Its core Relay runtime owns profile/session/turn
context and plugin lifetime, while focused tool and model adapters pass the
real callbacks into Relay managed execution. Relay `plugins.toml` remains an
optional process-level policy and exporter layer. The removed Hermes
observability plugin is not the current attachment mechanism.

This is not automatically preferable to a public host extension. Choose it
when required coverage, first-party product intent, support and release
ownership, lifecycle, or performance justify the additional coupling. Do not
choose core or an extension by default.

## Continuation Or Execution Wrapper

This is the strongest host extension point because it yields the real callback:

```python
async def around_tool(request, next_call):
    return await relay_managed_tool(
        request,
        lambda rewritten: next_call(request.with_args(rewritten)),
    )
```

The host must:

- await the middleware;
- provide the actual downstream callback or stream factory;
- apply the returned result;
- allow the callback to receive a rewritten request;
- preserve errors, cancellation, and stream finalization;
- prevent accidental multiple calls to a single-use continuation.

This shape can support Relay managed execution. LangChain's public
`wrap_model_call` / `awrap_model_call` and `wrap_tool_call` /
`awrap_tool_call` hooks are concrete examples: each receives a `handler` that
the Relay integration invokes inside the managed wrapper. Hermes core has the
same semantic shape in its execution-middleware chain through `next_call`, and
its first-party Relay adapter wraps actual tool and provider callbacks.

Do not assume every function named `middleware` has this contract. Some
middleware registries expose only a notification or mutable request.

## Awaited Pre-Execution Decision

This hook runs before execution but does not yield the callback:

```typescript
host.on("tool_call", async (event) => {
  const decision = await evaluate(event.toolName, event.input);
  if (!decision.allowed) {
    return { block: true, reason: decision.reason };
  }
});
```

It supports enforcement only when the host:

- awaits all relevant handlers before dispatch;
- interprets the returned block result;
- guarantees the callback will not run after rejection;
- defines ordering when multiple extensions participate;
- defines whether handler errors fail open or fail closed.

This can call Relay standalone conditional execution. It does not run Relay's
full managed pipeline because the integration never receives the callback.
Pi's `tool_call` extension event is an example of this shape. Its precise
blocking and ordering semantics must be checked at the selected Pi revision.

## Mutable Request Hook

This hook allows a request to be replaced before execution:

```python
def before_tool(name, args):
    return {"args": rewrite(args)}
```

or:

```typescript
host.on("before_provider_request", (event) => {
  return rewriteProviderPayload(event.payload);
});
```

Verify:

- whether mutation is in place or returned;
- whether the host applies the value;
- whether later handlers see earlier changes;
- whether the host revalidates tool arguments after mutation;
- whether headers and body are available together;
- whether the hook covers every provider and background call.

This can call Relay standalone request intercepts when the Relay request and
host payload can be projected without loss. Hermes request middleware applies
returned `args` or `request` objects. Pi's provider-request event can replace
the provider body, while its header hook has different mutation semantics.

## Mutable Result Hook

A post-execution hook may allow result replacement:

```typescript
host.on("tool_result", async (event) => {
  return { content: sanitize(event.content), isError: event.isError };
});
```

This proves the host can apply a replacement result. It does not prove that
Relay exposes a compatible standalone result-policy API in the selected
release. Record the host capability and Relay capability separately.

Pi's current extension API exposes a result hook that can replace structured
tool-result fields. Earlier pinned Pi audits reached different conclusions, so
this is a useful example of why host revision pinning matters.

## Paired Lifecycle Events

Paired events provide observation without callback ownership:

```typescript
host.on("tool_start", ({ id, name, args }) => openTool(id, name, args));
host.on("tool_end", ({ id, result, error }) => closeTool(id, result, error));
```

Verify that:

- one stable ID appears on both events;
- every success, failure, rejection, and cancellation produces a terminal
  event;
- events remain paired under parallel execution and retries;
- start and end describe the same logical or physical operation;
- shutdown exposes enough information to close abandoned work.

Use explicit Relay lifecycle APIs for this shape. Do not claim managed
middleware, blocking, or mutation. OpenClaw's lifecycle, LLM input/output,
model timing, tool completion, and subagent hooks demonstrate this replay
pattern.

## Session, Turn, And Subagent Hooks

Lifecycle hooks can create causal structure:

```typescript
host.on("session_start", openSession);
host.on("turn_start", openTurn);
host.on("subagent_spawned", attachChild);
host.on("subagent_ended", closeChild);
host.on("turn_end", closeTurn);
host.on("session_end", closeSession);
```

Check whether IDs survive retries, compaction, session rotation, queued
continuations, and process boundaries. An index that resets when an agent loop
re-enters is not a stable run identity. A child process requires explicit
context export and import; a subagent label alone is not causal linkage.

Pi's `agent_settled` event is more useful for closing a logical run than a
single `agent_end` when automatic retry, compaction, or queued continuation may
re-enter the loop. Hermes instead owns session and turn state in core and can
carry Relay context directly through its task execution paths.

## Provider Transport And Gateway Hooks

A provider wrapper that receives `next_call` can support managed LLM execution:

```python
async def around_provider(request, next_call):
    return await relay_managed_llm(request, next_call)
```

A configurable compatible base URL is not a callback hook. It can route model
HTTP traffic through the Relay gateway, but it does not cover tools, sessions,
subagents, or provider calls that ignore that configuration.

For both shapes, verify unary and streaming calls, headers, retries, background
compaction or summarization, provider switching, and exact request/response
formats.

## Plugin Lifecycle Hooks

An optional integration needs a lifecycle contract as well as execution hooks:

```typescript
plugin.register(api);
await plugin.start(runtime);
plugin.status();
await plugin.stop();
```

Verify:

- registration order and priority;
- whether loading can be silently skipped;
- positive activation and health evidence;
- ownership across multiple profiles or sessions;
- whether stop rejects new work and drains accepted work;
- cleanup after partial startup failure;
- behavior when the binding or native artifact cannot load.

Give process-global Relay configuration one owner even when the host creates
multiple plugin, profile, or session objects. Track configured owners
separately from callbacks, streams, and deferred children that still use Relay.
Final teardown must stop new work, wait for accepted operations, close owned
scopes, drain publication once, and only then clear middleware or activation.
Do not flush process-global subscribers during every session close.

OpenClaw provides plugin registration plus service and runtime cleanup hooks.
Hermes owns Relay process-wide in core and activates Relay runtime plugins
separately. Those are different integration shapes even though both eventually
run Relay middleware and exporters.

## Classification Rule

For every hook, reduce the contract to these questions:

1. Does it provide the real callback or only data?
2. Does the host await it?
3. Does the host apply its return value?
4. Can it reject, mutate the request, replace the result, or only observe?
5. Which dispatch paths invoke it?
6. Which identity, retry, stream, error, cancellation, and shutdown semantics
   are guaranteed?

Only then choose a Relay API.

For task, thread, stream, callback, and shutdown ownership, continue with
[Concurrency And Lifecycle](concurrency-and-lifecycle.md).
