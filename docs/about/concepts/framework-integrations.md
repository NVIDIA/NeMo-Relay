<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Framework Integrations

This page explains how framework integrations should attach existing application work to
NeMo Relay runtime semantics.

## Why Framework Integrations Are Different

Application code can usually call the managed NeMo Relay helpers directly.
Framework integrations often cannot.

A framework may already own:

- The real invocation boundary
- The scheduling model
- The retry loop
- The callback signature
- The provider payload shape

That means framework integrations must choose the best instrumentation boundary
available rather than assuming direct runtime ownership.

## Preferred Integration Order

When integrating NeMo Relay into an existing framework, prefer these choices in
order:

1. Execution wrappers through managed execute helpers
2. Explicit API calls for lifecycle emission, conditional execution, or request intercepts
3. Mark events only

This order preserves the most runtime semantics with the least distortion.

## First Choice: Execution Wrappers

Execution wrappers are the preferred integration boundary when a framework exposes a
real callback or handler.

### Managed Execute Helpers

Use the managed execute helpers when the framework exposes a stable callable
boundary that NeMo Relay can wrap.

### Why This Is Preferred

This is the best integration shape because it preserves:

- Correct lifecycle ordering
- The full middleware pipeline
- Natural parent-child scope relationships
- The cleanest wrapper point for retries, routing, and timing

Execution wrappers are also the natural place to align framework semantics with
NeMo Relay execution intercepts.

## Fallback: Explicit API Calls

Use explicit API calls when the framework owns part of the invocation lifecycle
and cannot hand NeMo Relay a stable callback to wrap. Explicit calls let the
framework keep its own scheduler, retry loop, callback signature, or provider
client while still using selected NeMo Relay runtime behavior.

### What You Lose From Managed Execution Wrappers

Explicit API calls are useful, but they are narrower than managed execution
wrappers. Depending on which explicit APIs you call, you can lose:

- Automatic start-to-end lifecycle pairing
- Automatic execution-intercept chaining around the real callback
- Automatic request and response guardrail placement
- One canonical parent-child relationship for the wrapped span
- One call site that applies the full middleware pipeline

Use explicit APIs when they match the framework boundary. Prefer managed
execution wrappers whenever the framework can expose the real callback.

### Explicit Start, End, and Mark Emission

Use explicit start and end emission when the framework gives reliable lifecycle
hooks but does not let NeMo Relay wrap the real invocation.

1. Call the explicit start API as early as the framework can identify the work.
2. Retain the returned handle.
3. Call the matching end API when the work succeeds or fails.
4. Emit mark events for milestones that are important but are not full tool or
   LLM calls.

This fallback preserves lifecycle visibility, but the framework must pair start
and end calls correctly.

### Conditional Execution

Use standalone conditional-execution helpers when the framework only needs an
allow-or-block decision before continuing its own invocation path.

This is the preferred explicit API when the framework can ask NeMo Relay for a
policy decision but must still execute the real tool or provider call itself.
The helper returns the guardrail decision; it does not emit a full managed
lifecycle span by itself.

### Request Intercepts

Use standalone request-intercept helpers when the framework needs NeMo Relay to
rewrite the request before the framework continues execution on its own.

This is the preferred explicit API when the framework owns execution but can
accept a rewritten JSON-compatible request before it calls the underlying tool
or provider. Request-intercept helpers apply request transformation without
owning callback execution.

Use mark events when the framework exposes important milestones but not a clean
start/end lifecycle boundary.

Mark events are useful for:

- Retries
- Queue transitions
- Scheduler milestones
- State changes
- Debugging checkpoints

They provide visibility, but they are not a replacement for full lifecycle
instrumentation.

## Choosing the Right Integration Boundary

Use these rules to decide where NeMo Relay should wrap framework behavior.

- If you can wrap the real callback, use managed execute helpers.
- If you cannot wrap the callback but you do have reliable start and end hooks,
  use explicit lifecycle APIs.
- If you only need a block/allow decision, use conditional-execution helpers.
- If you only need request transformation, use request-intercept helpers.
- If you only have milestone visibility, emit mark events.

## Practical Guidance

Use these practices when applying the concept in application or integration code.

- Prefer execution wrappers over explicit helper calls whenever the framework
  allows it.
- Treat explicit lifecycle calls as the main fallback for framework-owned invocation.
- Use conditional-execution functions and request-intercept helpers before
  continuing framework-owned execution when you need policy or transformation
  without managed callback wrapping.
- Use mark events to fill visibility gaps rather than to model full execution
  spans.
- Keep binding-level API details in the [API Reference](../../reference/api/index.md) and
  deeper integration patterns in [Integrate into Frameworks](../../integrate-frameworks/about.md).
