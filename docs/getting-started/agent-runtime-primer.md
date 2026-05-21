<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Runtime Primer

NeMo Relay is a portable runtime layer for agent systems that already have an
application, framework, or model provider. Use this primer when you need to
understand what NeMo Relay adds before running [Quick Start](quick-start.md).

Agent applications usually cross several boundaries in one request: an entry
point starts work, the agent calls a model, the model asks for tools, tools call
services, and tracing or policy systems need to understand the result. Without a
shared runtime layer, each boundary tends to grow its own wrappers, callback
shape, trace vocabulary, and cleanup rules.

NeMo Relay gives those boundaries one execution model.

## What NeMo Relay Adds

NeMo Relay does not decide what your agent should do. It describes and manages
what happens when your agent crosses runtime boundaries.

The core runtime model has five parts:

- **Scopes** describe where work belongs. They preserve parent-child
  relationships across requests, agent runs, tools, LLM calls, background work,
  and nested functions.
- **Managed tool and LLM calls** attach work to the active scope, run middleware
  in a consistent order, and emit lifecycle events. The application result is
  preserved unless registered intercepts or guardrails intentionally change the
  execution path.
- **Middleware** runs around managed execution. Intercepts can transform or wrap
  real calls. Guardrails can block execution or sanitize emitted observability
  payloads.
- **Events** record what happened. NeMo Relay emits Agent Trajectory
  Observability Format (ATOF) lifecycle records that subscribers and exporters
  can consume.
- **Plugins** package reusable runtime behavior so teams can install middleware,
  subscribers, exporters, or adaptive behavior from configuration instead of
  repeating setup code in every application.

The simplest mental model is:

```text
app or framework boundary
  -> NeMo Relay scope
  -> managed tool or LLM call
  -> middleware
  -> lifecycle event
  -> subscriber or exporter
```

## What NeMo Relay Does Not Replace

NeMo Relay sits below the choices your application already makes.

It does not replace:

- your agent framework or orchestration logic
- your model provider or provider SDK
- your application business logic
- your production observability backend
- NeMo Agent Toolkit

Instead, it gives those systems a shared runtime contract for call boundaries,
policy hooks, event emission, and export.

## Choose The Boundary You Own

Where you start depends on who owns the call boundary.

If your application directly calls tools or model providers, start by
instrumenting the application boundary. Add scopes first, then wrap the tool and
LLM calls your code owns.

If a framework owns scheduling, retries, callbacks, or provider payloads, use a
framework integration. The integration should preserve framework behavior while
adding NeMo Relay scopes, managed calls, codecs, middleware, and events at stable
framework boundaries.

If you need the same behavior across multiple services or teams, package it as a
plugin. Plugins are the configuration-driven path for reusable middleware,
subscribers, exporters, and adaptive components.

## Read Next

The following pages help you choose the next step for your integration.

- Use [Quick Start](quick-start.md) for the smallest binding-specific example.
- Use [Instrument Applications](../instrument-applications/about.md) when you
  own the tool or LLM call site.
- Use [Integrate into Frameworks](../integrate-frameworks/about.md) when a
  framework owns invocation, scheduling, retries, callbacks, or provider
  payloads.
- Use [Build Plugins](../build-plugins/about.md) when behavior should be
  reusable and activated from configuration.
- Use [Observability](../plugins/observability/about.md) when you need to export
  runtime events to ATIF, OpenTelemetry, or OpenInference.
- Use [Adaptive](../plugins/adaptive/about.md) after baseline instrumentation is
  working and you want to tune behavior from observed runtime signals.
