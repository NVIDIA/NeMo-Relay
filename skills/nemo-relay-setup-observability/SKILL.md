---
name: nemo-relay-setup-observability
description: Choose and set up the right NeMo Relay observability path for an application
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

<!-- Non-semantic CI refresh marker: 2026-06-06. -->

# Set Up Observability

Use this skill when an application developer wants visibility into NeMo Relay
activity but has not yet decided which output they need.

## Choose The Output

- **Console or custom event handling**
  Use subscribers.
- **Portable execution trajectories**
  Use `AtifExporter`.
- **General OTLP tracing**
  Use the OpenTelemetry subscriber.
- **OpenInference-aware backends**
  Use the OpenInference subscriber.

## Embedded Event And Subscriber Model

- NeMo Relay emits one canonical event stream from scopes, marks, managed tool
  calls, managed LLM calls, middleware, and manual lifecycle APIs.
- Subscribers consume events without defining the event model. Multiple
  subscribers can observe the same stream for logging, export, analytics, or
  diagnostics.
- Global subscribers remain active process-wide until removed.
- Scope-local subscribers are owned by one active scope and disappear when that
  scope closes.
- Plugin-installed subscribers are reusable, configuration-driven runtime
  components.
- Exporter-oriented subscribers translate the event stream into ATIF,
  OpenTelemetry, or OpenInference output.
- Event payloads reflect sanitized post-guardrail input and output when calls use
  managed helpers or manual lifecycle params provide those fields.
- Event fields include semantic input/output through the ATOF `data` field,
  typed profile data such as `model_name` and `tool_call_id`, and codec-provided
  annotated LLM request/response data for in-process subscribers and exporters.

## Shared Lifecycle

1. Create the exporter or subscriber.
2. Register it with a unique name before the relevant scoped work.
3. Run NeMo Relay-instrumented work inside scopes.
4. Deregister it.
5. Flush or shut down if the binding supports it and deterministic delivery is needed.

## Binding Names

- Python: `nemo_relay.subscribers.register(...)`,
  `AtifExporter`, `OpenTelemetrySubscriber`, and `OpenInferenceSubscriber`
- Node.js: root exports `registerSubscriber(...)`, `AtifExporter`,
  `OpenTelemetrySubscriber`, and `OpenInferenceSubscriber`
- Rust: `nemo_relay::api::subscriber` and `nemo_relay::observability::*`
- Go and WebAssembly: source-first wrappers expose equivalent register, exporter, and
  subscriber lifecycle methods

## Use Another Skill When

- You already know you need ATIF -> `nemo-relay-export-atif-trajectories`
- You already know you need OTEL -> `nemo-relay-export-otel`
- You already know you need OpenInference -> `nemo-relay-export-openinference`
- You need to package subscriber-based export behavior as a reusable plugin ->
  `nemo-relay-build-plugin`
- You are debugging missing telemetry -> `nemo-relay-debug-runtime-integration`

## Related Skills

- `nemo-relay-export-atif-trajectories`
- `nemo-relay-export-otel`
- `nemo-relay-export-openinference`
- `nemo-relay-build-plugin`
