<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Subscribers

This page explains how subscribers consume lifecycle events without changing runtime
execution.

## What Subscribers Are

Subscribers are consumers of the NeMo Flow event stream. They receive emitted
lifecycle events and use them for observation, forwarding, export, or analysis.

## How Subscribers Relate to Events

Events describe what happened. Subscribers are the components that watch those
events.

That separation matters:

- The runtime can emit one canonical event stream
- Many subscribers can consume that same stream
- Observability behavior stays downstream from execution semantics

## Registration Levels

Middleware and subscribers can be registered at different levels depending on their
lifetime and visibility.

### Global Subscribers

Global subscribers remain active process-wide until they are removed.

### Scope-Local Subscribers

Scope-local subscribers are owned by one active scope and disappear when that
scope closes.

### Plugin-Installed Subscribers

Plugins can install subscribers as reusable, configuration-driven runtime
components.

## What Subscribers Consume

Subscribers consume the canonical event stream. They do not define the event
model. They react to it.

This lets plain subscribers, exporters, and tracing adapters share one runtime
source of truth.

## Common Subscriber Roles

Subscribers are commonly used for in-process observation, counters, debugging, and
exporter handoff.

### In-Process Observation

Some subscribers stay inside the process and power custom logging, analytics, or
debugging logic.

#### Host Integration Event JSON

For host integrations that need a serialized event payload, use the event
object's canonical JSON helpers instead of reconstructing payloads from native
attributes. Python subscribers can call `event.to_dict()` or `event.to_json()`
from the callback while still using the normal subscriber registration API.

This pattern is useful when an agent runtime, framework adapter, or plugin host
already has its own lifecycle hooks but wants NeMo Flow to be the shared
telemetry representation. The host integration maps those hooks into NeMo Flow
scopes, LLM calls, tool calls, or marks. NeMo Flow emits the canonical ATOF event
stream, and each subscriber chooses whether to consume the native event object,
the canonical JSON helper, or an exporter-specific translation.

```{mermaid}
flowchart
    Host[Host Integration]

    subgraph NeMoFlow[NeMo Flow]
        direction TB
        Binding[Binding API]
        Core[Rust Core Runtime]
        Events[Canonical ATOF Event Stream]
        Observer[In-Process Subscriber]
        Json[Canonical Event JSON]
        Exporters[Exporter Subscribers]
        Backends[JSONL / ATIF / OTLP]

        Binding -->|emits scopes, tools, LLMs, marks| Core
        Core --> Events
        Events --> Observer
        Observer -->|to_dict / to_json / JSON| Json
        Events --> Exporters
        Exporters --> Backends
    end

    Host -->|maps lifecycle hooks| Binding
    Json -. host consumes canonical telemetry .-> Host
```

The important boundary is that subscribers do not define the event schema. They
receive the runtime event and may serialize it through the binding helper when
they need a stable JSON payload. Exporter subscribers, such as the ATOF JSONL
exporter, consume the same event stream and serialize the same canonical event
shape for their target backend.

### Forwarding and Export

Some subscribers translate the event stream into external formats or transport
it to another system.

### Analytics and Diagnostics

Some subscribers derive measurements, trajectories, or diagnostics from the
event stream without affecting execution behavior.

## Built-In Subscriber Examples

These examples show how built-in subscriber patterns relate to custom subscribers and
exporters.

### Custom Subscribers

A plain custom subscriber is the right choice when you want in-process handling
of the canonical event stream.

### Agent Trajectory Interchange Format (ATIF) Exporter

The [Agent Trajectory Interchange Format (ATIF) exporter](../../plugins/observability/atif.md)
collects lifecycle events and emits trajectory artifacts for offline analysis,
replay, or debugging.

### Agent Trajectory Observability Format (ATOF) JSONL Exporter

The [Agent Trajectory Observability Format (ATOF) JSONL exporter](../../plugins/observability/atof.md)
writes the canonical event stream to a native filesystem path as one raw ATOF
event per line.

### OpenTelemetry Subscriber

The OpenTelemetry subscriber maps runtime events into OTLP traces for tracing
backends.

### OpenInference Subscriber

The OpenInference subscriber maps runtime events into OTLP traces using
OpenInference semantics for model-centric observability.

Detailed setup, configuration, and API shape for these subscribers belongs in
[Observability](../../plugins/observability/about.md).
For configuration-driven setup, use the built-in
[`observability` plugin](../../plugins/observability/configuration.md)
to install ATOF, ATIF, OpenTelemetry, and OpenInference subscribers from one
plugin component.

## Practical Guidance

Use these practices when applying the concept in application or integration code.

- Use a plain subscriber when you want in-process custom behavior.
- Use `event.to_dict()` or `event.to_json()` when a host runtime or exporter
  needs the canonical event JSON shape in-process.
- Use a scope-local subscriber when the observation should disappear with the
  owning scope.
- Use a plugin-installed subscriber when the behavior should be reusable and
  config-driven.
- Use an exporter-oriented subscriber when the event stream should leave the
  process.
