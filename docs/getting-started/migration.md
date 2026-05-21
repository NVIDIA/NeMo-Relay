<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrate Existing Agent Instrumentation to NeMo Flow

Use this page as a routing guide when your agent application already has
callbacks, traces, or custom logging and you want to adopt NeMo Flow without
rewriting the application first.

This is an adoption on-ramp, not a drop-in migration contract. Start by adding
NeMo Flow at the execution boundaries you can observe, validate that application
behavior stays the same, then add policy, adaptive behavior, or production
exporters after the basic event stream is correct.

## Who This Guide Is For

Use this guide when you are starting from one of these systems:

- LangChain, LangGraph, Deep Agents, or another framework callback surface.
- Existing OpenTelemetry or OpenInference traces.
- A custom agent harness that emits JSON traces, run records, or callback logs.

If you are building a new application without existing instrumentation, start
with the [Quick Start](quick-start.md) instead.

## Migration Decision Tree

Use the table below to choose the first NeMo Flow path. Pick one path first,
validate it, then add additional exporters or middleware.

| Starting Point | First NeMo Flow Path |
|---|---|
| You own the direct tool or LLM call site | Use [Instrument Applications](../instrument-applications/about.md). |
| A framework owns the tool or LLM call site | Use [Supported Integrations](../integrations/about.md) if one exists; otherwise use [Integrate into Frameworks](../integrate-frameworks/about.md). |
| Your backend expects generic OTLP traces | Use the [OpenTelemetry exporter](../plugins/observability/opentelemetry.md). |
| Your backend expects OpenInference agent or LLM spans | Use the [OpenInference exporter](../plugins/observability/openinference.md). |
| You need portable offline trajectories for replay, analysis, or evaluation | Use [ATIF export](../plugins/observability/atif.md). |
| You need raw lifecycle events for a custom pipeline | Use [ATOF export](../plugins/observability/atof.md) or a direct subscriber. |

## Preserve Behavior First

Adopt NeMo Flow in this order:

1. Add scopes around the request, run, or agent boundary.
2. Add a lightweight subscriber or local exporter so you can inspect emitted
   events.
3. Instrument one tool or LLM boundary at a time.
4. Compare application outputs, framework callbacks, and existing traces before
   enabling guardrails, intercepts, adaptive tuning, or production exporters.

The first milestone is not performance tuning or policy coverage. The first
milestone is the same application behavior with a correct NeMo Flow event
stream.

## Map Existing Concepts

Use this mapping to translate current instrumentation concepts into NeMo Flow
terms before changing code.

| Existing Concept | NeMo Flow Concept |
|---|---|
| LangChain callback run | Scope plus lifecycle event stream. |
| OpenInference span | OpenInference subscriber output from NeMo Flow events. |
| OpenTelemetry span | OpenTelemetry subscriber output from NeMo Flow events. |
| Custom trace object | Subscriber output, ATOF JSONL, or ATIF trajectory export. |
| Request-local state | Scope stack and context isolation. |
| Callback middleware | NeMo Flow middleware around managed tool and LLM calls. |
| Trace correlation ID | Root scope UUID and parent-child event IDs. |

## Common Starting Scenarios

If you already use LangChain callbacks, keep the framework behavior in place and
start with the [LangChain integration](../integrations/langchain.md). Validate
that the agent result and existing callback behavior do not change before adding
new middleware.

If you already export OpenInference spans to Phoenix or another compatible
backend, start with the [OpenInference exporter](../plugins/observability/openinference.md).
NeMo Flow becomes the event source for those spans; the backend setup can stay
backend-specific.

If your custom harness emits JSON traces, decide whether you need raw lifecycle
events, portable trajectories, or direct in-process handling. Use [ATOF export](../plugins/observability/atof.md)
for raw events, [ATIF export](../plugins/observability/atif.md) for trajectories,
or a direct subscriber when your application should own the output shape.

## Next Steps

After you pick the first path, use these pages for implementation details:

- [Quick Start](quick-start.md)
- [Instrument a Tool Call](../instrument-applications/instrument-tool-call.md)
- [Instrument an LLM Call](../instrument-applications/instrument-llm-call.md)
- [Integrate into Frameworks](../integrate-frameworks/about.md)
- [Observability Plugin](../plugins/observability/about.md)
