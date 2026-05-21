<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Runtime Primer

Use this page before the Quick Start if you know agent applications but want the
NeMo Flow runtime model first.

Agent applications usually spread execution across framework callbacks, model
provider SDKs, tool wrappers, retries, policy checks, traces, and exporters.
That makes it hard to answer basic production questions: which request owns this
tool call, which policy ran, what changed the live request, what was only
sanitized for telemetry, and where did the evidence go?

NeMo Flow gives those execution boundaries a shared runtime layer. It does not
decide what the agent should do next. It records what happens in scopes and
applies middleware around managed tool and LLM calls.

## What NeMo Flow Adds

The runtime model has a small set of pieces:

| Runtime Piece | What It Answers |
|---|---|
| Scopes | Where work belongs, which parent event owns it, and which request-local behavior is visible. |
| Managed tool and LLM calls | Where execution crosses a NeMo Flow boundary and emits lifecycle events. |
| Middleware | What can block execution, transform live requests, wrap callbacks, or sanitize emitted telemetry. |
| Events | What happened during scopes, marks, tool calls, LLM calls, and runtime checkpoints. |
| Subscribers and exporters | Where the event stream goes for logs, traces, trajectories, diagnostics, or analysis. |
| Plugins | How reusable middleware, subscribers, exporters, and related runtime behavior are packaged from configuration. |

These pieces are shared across the primary Rust, Python, and Node.js surfaces.
The binding can change, but the runtime questions stay the same.

## What NeMo Flow Does Not Replace

NeMo Flow is not an agent orchestration framework. It does not replace:

- Agent framework logic such as planning, memory, scheduling, retries, or tool
  discovery.
- NeMo Agent Toolkit or other framework-level systems that build and run agent
  workflows.
- Model providers, provider SDKs, authentication, transport, or deployment
  infrastructure.
- Application business logic or user session state.
- Observability backends that store traces, trajectories, dashboards, or alerts.

Instead, NeMo Flow sits underneath those systems as the execution runtime
contract they can share.

## Tiny Mental Model

Read a NeMo Flow integration as this path:

```text
app or framework boundary
  -> NeMo Flow scope
  -> managed tool or LLM call
  -> middleware
  -> lifecycle event
  -> subscriber or exporter
```

If your application owns the tool or LLM call site, place NeMo Flow directly at
that boundary. If a framework owns the call site, use a framework integration or
add NeMo Flow at the framework hook that sees the same lifecycle.

## Where To Go Next

Choose the next page based on your first task:

| Task | Next Page |
|---|---|
| Run a minimal example with a primary binding | [Quick Start](quick-start.md) |
| Add scopes, tool events, or LLM events to application code | [Instrument Applications](../instrument-applications/about.md) |
| Add NeMo Flow under a framework that owns the call site | [Integrate into Frameworks](../integrate-frameworks/about.md) |
| Export traces, events, or trajectories | [Observability Plugin](../plugins/observability/about.md) |
| Package reusable middleware or exporters from configuration | [Build Plugins](../build-plugins/about.md) |
| Understand each runtime piece in more detail | [Concepts](../about/concepts/index.md) |
