---
name: nemo-flow-export-otel
description: Configure and use NeMo Flow OpenTelemetry export for OTLP-compatible tracing backends
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Export OpenTelemetry Traces

Use this skill when the destination is an OTLP/OpenTelemetry backend such as an
OpenTelemetry Collector, Jaeger, Tempo, or Honeycomb.

## Default Path

- Build the binding-specific `OpenTelemetryConfig`
- Set endpoint, service name, and any required headers
- Construct the subscriber
- Register it before running scoped work
- Deregister, flush, and shut down when the process or subsystem is done

## Embedded OpenTelemetry Semantics

- OpenTelemetry export maps NeMo Flow runtime events into OTLP traces for
  tracing backends and collectors.
- Configure `transport`, `endpoint`, `service_name`, optional namespace and
  version, instrumentation scope, headers, resource attributes, and timeout.
- Start with `http_binary` transport and an OTLP traces endpoint such as a local
  collector on port `4318` unless deployment requirements differ.
- `grpc` transport is available on native targets when a Tokio runtime is
  active. WebAssembly supports `http_binary` and rejects `grpc`.
- Use explicit config objects in application code; environment variables may be
  honored by the underlying exporter but should not be the only source of
  application behavior.
- Register before the first instrumented request, use stable service identity,
  keep auth and endpoints out of source code, flush during graceful shutdown,
  and redact sensitive payloads before production export.
- Validate export by checking subscriber construction, collector requests,
  backend spans for scopes/tools/LLMs, and span grouping by root scope.

## Things To Confirm

- Transport: `http_binary` vs `grpc`
- Endpoint and auth headers
- Service naming and resource attributes
- Whether deterministic flush-before-exit is required
- Whether the chosen binding and target support the desired transport

## Troubleshooting Focus

- No spans visible
- Wrong endpoint or auth headers
- Events emitted outside active scopes
- `grpc` selected without a native Tokio runtime or on a WebAssembly target
- Forgetting register/deregister or flush/shutdown steps

## Related Skills

- `nemo-flow-setup-observability`
- `nemo-flow-export-openinference`
- `nemo-flow-debug-runtime-integration`
