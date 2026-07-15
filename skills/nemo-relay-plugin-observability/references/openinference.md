<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export OpenInference Traces

Use this reference when the destination expects OpenInference semantic conventions,
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
- Start with `http_binary` transport and an OTLP/HTTP traces endpoint. Use
  `grpc` only when a Tokio runtime is active.
- Scope, tool, and LLM start inputs become `input.value`.
- Scope, tool, and LLM end outputs become `output.value`.
- Each owning agent scope starts fresh, and a `compaction` mark refreshes it.
  The annotated input for the first subsequent LLM start retains complete
  history. Later starts retain system instructions, the latest user message,
  and every following assistant or tool message. With a request codec, the
  event's provider-shaped input uses the same projection. Provider execution
  remains unchanged.
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

- `nemo-relay-plugin-observability`
- `nemo-relay-instrument-typed-wrappers`
