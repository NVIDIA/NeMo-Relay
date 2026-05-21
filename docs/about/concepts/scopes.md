<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scopes

This page explains how scope stacks establish ownership, parentage, cleanup, and
isolation.

## Why Scopes Exist

Scopes are the ownership backbone of NeMo Relay. Every tool call, LLM call, and
mark event attaches to a scope hierarchy.

That hierarchy lets the runtime:

- Model nested agent work
- Preserve parent-child relationships
- Expose scope-local middleware and subscribers
- Clean up scope-owned runtime state automatically
- Isolate concurrent work

## What a Scope Represents

A scope represents a logical unit of work such as:

- An agent run
- A request
- A workflow step
- A background task
- A nested function or tool orchestration boundary

Scopes are not just labels. They define ownership and visibility for other
runtime behavior.

## Scope Hierarchy and Ownership

Scopes form a tree. A child scope inherits the active execution context from its
parent and contributes new nested work beneath it.

That hierarchy determines:

- Event parentage
- Lifetime boundaries
- Scope-local middleware visibility
- Scope-local subscriber visibility

## Scope Types

NeMo Relay includes standard scope types for common runtime semantics, including:

- `Agent`
- `Function`
- `Tool`
- `Llm`
- `Retriever`
- `Embedder`
- `Reranker`
- `Guardrail`
- `Evaluator`
- `Custom`
- `Unknown`

The specific type helps subscribers and downstream tracing systems understand
what the scope represents semantically.

## Scope Behavior

These scope behaviors define how root, child, and scope-local runtime state interact.

### Root Scope

A root scope is always present. Other scopes are pushed beneath that root as
work becomes more specific.

### Parent-Child Relationships

Nested scopes create the ownership tree used by emitted events. Tool and LLM
calls then attach beneath the active scope.

### Scope Lifetimes

Scopes have explicit lifetime boundaries. A scope starts when it becomes active
and ends when it is popped or closed.

### Scope-Local Cleanup

Scope-local middleware and subscribers are tied to the owning scope lifecycle.
When the scope closes, those registrations disappear automatically.

## Semantic Payloads

Scopes may expose semantic `input` and `output` payloads on their emitted start
and end events.

### Scope Input

Use scope `input` when the scope itself represents a request-style or task-style
unit of work whose starting payload matters semantically.

### Scope Output

Use scope `output` when the scope itself produces a meaningful semantic result.

Those payloads live on the emitted events rather than on the scope handle
itself.

## Context Isolation

Context isolation keeps concurrent requests, tenants, and agents from sharing scope-
local state accidentally.

### Why Isolation Matters

Concurrent requests must not share the same active scope stack accidentally.
Otherwise:

- Unrelated work can appear under the wrong parent
- Scope-local middleware can leak across requests
- Scope-local subscribers can observe the wrong execution tree

## Reuse an Existing Logical Trace

Reuse or propagate the active scope stack when detached work should continue the
same logical request or agent trace.

Use this when:

- Worker events should appear under the same parent request
- Scope-local middleware from the parent should still apply
- Subscribers should observe one continuous execution tree

## Start a Fresh Isolated Context

Create and bind a fresh stack when detached work should be independent.

Use this when:

- The worker is a separate job rather than part of the parent trace
- The boundary cannot safely carry a native stack handle
- You want a clean root scope with isolated scope-local registrations

## Practical Guidance

Use these practices when applying the concept in application or integration code.

- Push a top-level scope at the entry point of a request, workflow, or agent
  run.
- Let nested helpers attach work beneath that scope whenever possible.
- Use scope-local registrations when the behavior should disappear with the
  owning scope.
- Emit mark events for retries, checkpoints, interrupts, or state transitions
  that are important for debugging but are not full spans.
- Prefer explicit isolation decisions when work crosses thread, task, or worker
  boundaries.
