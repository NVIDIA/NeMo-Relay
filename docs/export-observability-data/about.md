<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when you need to inspect NeMo Flow lifecycle events in process
or export agent activity to tracing, trajectory, or analysis systems.

Observability in NeMo Flow starts with events. Scopes, marks, managed tool
calls, managed LLM calls, middleware, and manual lifecycle APIs emit a canonical
event stream. Subscribers consume that stream inside the process, and
exporter-oriented subscribers translate it into formats such as ATIF,
OpenTelemetry, and OpenInference.

Use these guides to confirm what ran, where it belonged, which model or tool was
involved, and what sanitized payload was observed across Rust, Python, and
Node.js.

## Start Here When

Start here when you need to perform one of the following checks:

- Verify that instrumentation is attached to the right scope
- Inspect tool and LLM inputs and outputs after sanitization
- Correlate concurrent agent runs by root scope
- Export traces to existing OTLP-compatible infrastructure
- Produce trajectory data for analysis, replay, or evaluation workflows

If you have not instrumented any scopes, tools, or LLM calls yet, start with [Instrument Applications](../instrument-applications/about.md).

## Guides

The following guides describe available tutorials and exporters:

- [Basic Guide: Register a Subscriber](basic-guide.md) shows a simple subscriber lifecycle and validation workflow.
- [Advanced Guide: Export OpenTelemetry Data](opentelemetry.md) shows how to export generic OTLP spans.
- [Advanced Guide: Export OpenInference Data](advanced-guide.md) shows how to configure and operate the OpenInference exporter.
- [Advanced Guide: Export ATIF](atif.md) shows how to collect and export trajectory artifacts.
- [Code Examples](code-examples.md) shows event shape, scope-local subscribers, ATIF export, OpenTelemetry export, and exporter selection snippets.

Begin with a local subscriber so you can confirm the application emits the
expected scope, tool, LLM, and mark events. Add exporters only after the event
stream is correct and sensitive payloads are sanitized.

For production export, register subscribers before the first instrumented
request, use stable service identity fields, keep credentials outside source
code, flush during graceful shutdown, and filter by `root_uuid` when analyzing
concurrent agent runs.
