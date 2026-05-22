---
name: nemo-relay-export-openinference
description: Configure and use NeMo Relay OpenInference export for OTLP backends that understand OpenInference semantics
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Export OpenInference Traces

Use this skill when the destination expects OpenInference semantic conventions,
for example Arize Phoenix or another OpenInference-aware OTLP backend.

## Default Path

- Build the binding-specific `OpenInferenceConfig`
- Set endpoint, transport, service metadata, and headers
- Construct and register the subscriber
- Run instrumented scoped work
- Deregister, flush, and shut down when done

## Embedded OpenInference Semantics

- OpenInference export is for OTLP backends that understand model-centric
  OpenInference semantic conventions.
- Configure `transport`, `endpoint`, `service_name`, optional namespace and
  version, instrumentation scope, headers, resource attributes, and timeout.
- Start with `http_binary` transport and an OTLP/HTTP traces endpoint. Native
  targets can use `grpc` only when a Tokio runtime is active; WebAssembly rejects
  `grpc`.
- Scope, tool, and LLM start inputs become `input.value`.
- Scope, tool, and LLM end outputs become `output.value`.
- LLM usage metadata maps token counters when provider responses include usage.
- Use explicit config fields for endpoint, headers, resource attributes, and
  service identity in application code.
- Validate export by checking construction logs, collector traffic, and spans
  from the same `root_uuid` in the tracing backend.

## Important Semantics

- Spans include OpenInference semantic attributes
- LLM spans derive `input.value` from request content, not request headers
- Scope types map to OpenInference span kinds
- Orphan mark events still export as zero-duration spans

## Troubleshooting Focus

- No spans in the OpenInference-aware backend
- Expected semantic attributes missing
- Wrong scope types or no active scope
- Wrong OTLP transport for the chosen binding or target

## Related Skills

- `nemo-relay-setup-observability`
- `nemo-relay-export-otel`
- `nemo-relay-typed-wrappers-codecs`
