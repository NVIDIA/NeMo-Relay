<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Observability

Use the Observability plugin when you need to inspect NeMo Flow lifecycle events
in process or export agent activity to tracing, trajectory, or analysis
systems from one plugin configuration document.

Observability in NeMo Flow starts with events. Scopes, marks, managed tool
calls, managed LLM calls, middleware, and manual lifecycle APIs emit the
canonical Agent Trajectory Observability Format (ATOF) event stream.
Subscribers consume that stream in process, and exporter-oriented subscribers
write raw ATOF JSONL or translate events into Agent Trajectory Interchange
Format (ATIF), OpenTelemetry, or OpenInference.

The first-party plugin component has kind `observability`. It can install:

- Agent Trajectory Observability Format (ATOF) JSONL export for raw lifecycle events.
- Agent Trajectory Interchange Format (ATIF) trajectory export for each top-level agent scope.
- OpenTelemetry OTLP trace export.
- OpenInference-oriented OTLP trace export.

## Plugin-Managed Versus Manual Export

Use the Observability plugin for process-level exporter setup that should be
activated from config, `plugins.toml`, or a shared plugin document.

Use manual subscriber or exporter APIs when a test, script, or application
needs direct control over registration names, collection windows, explicit
flush timing, or per-run exporter objects. The plugin owns subscriber names and
teardown for the sections it enables.

## Use Observability When

Start here when you need to:

- Verify that instrumentation is attached to the right scope.
- Inspect tool and LLM inputs and outputs after sanitization.
- Correlate concurrent agent runs by root scope.
- Export traces to OTLP-compatible infrastructure.
- Produce trajectory data for analysis, replay, or evaluation workflows.

If you have not instrumented scopes, tools, or LLM calls yet, start with
[Instrument Applications](../../instrument-applications/about.md).

## Exporter Selection

Choose the exporter based on the downstream system:

| Need | Use |
|---|---|
| Raw canonical event stream | [Agent Trajectory Observability Format (ATOF)](atof.md) |
| Offline analysis, replay, or evaluation trajectories | [Agent Trajectory Interchange Format (ATIF)](atif.md) |
| Generic OTLP traces | [OpenTelemetry](opentelemetry.md) |
| OpenInference-oriented agent and LLM spans | [OpenInference](openinference.md) |

Start with local event inspection before production export. Add sanitize
guardrails before exporters receive sensitive payloads.

## Pages

- [Observability Configuration](configuration.md) documents the whole plugin
  component shape, activation, validation, and teardown.
- [Agent Trajectory Observability Format (ATOF)](atof.md) covers raw JSONL event stream export.
- [Agent Trajectory Interchange Format (ATIF)](atif.md) covers per-agent trajectory export.
- [OpenTelemetry](opentelemetry.md) covers generic OTLP trace export.
- [OpenInference](openinference.md) covers OpenInference-oriented OTLP trace
  export.
