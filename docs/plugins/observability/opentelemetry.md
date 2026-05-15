<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenTelemetry

Use the `opentelemetry` section when you want NeMo Flow lifecycle events
exported as generic OpenTelemetry Protocol (OTLP) trace spans.

OpenTelemetry export is a good fit when your tracing backend already expects
OTLP spans and you want NeMo Flow scopes, tool calls, LLM calls, and marks to
appear in the same tracing pipeline as the rest of the application.

## `plugins.toml` Example

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.opentelemetry]
enabled = true
transport = "http_binary"
endpoint = "http://localhost:4318/v1/traces"
service_name = "agent-service"
service_namespace = "nemo"
service_version = "1.0.0"
instrumentation_scope = "nemo-flow-otel"
timeout_millis = 3000

[components.config.opentelemetry.headers]
authorization = "Bearer <token>"

[components.config.opentelemetry.resource_attributes]
"deployment.environment" = "dev"
```

This configuration registers a plugin-owned OpenTelemetry subscriber and sends
NeMo Flow trace spans to the configured OTLP endpoint.

## Fields

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to construct and register the subscriber. |
| `transport` | `http_binary` | `http_binary` or `grpc`. |
| `endpoint` | Exporter default | OTLP endpoint. |
| `headers` | `{}` | String-to-string exporter headers. |
| `resource_attributes` | `{}` | String-to-string OTLP resource attributes. |
| `service_name` | `nemo-flow` | `service.name` resource attribute. |
| `service_namespace` | Omitted | Optional `service.namespace`. |
| `service_version` | Omitted | Optional `service.version`. |
| `instrumentation_scope` | Omitted | Optional instrumentation scope name. |
| `timeout_millis` | `3000` | Export timeout. |

## Expected Output

The collector should receive OTLP trace export requests. The tracing backend
should show spans for NeMo Flow scopes, tools, LLM calls, and marks grouped by
root scope.

Register the plugin before the first instrumented request, use stable service
identity fields, keep credentials outside source code, and flush during
graceful shutdown.

## Manual API

Use the manual subscriber API when you need an explicit subscriber name or
direct `force_flush` control.

```python
from nemo_flow import OpenTelemetryConfig, OpenTelemetrySubscriber

config = OpenTelemetryConfig()
config.transport = "http_binary"
config.endpoint = "http://localhost:4318/v1/traces"
config.service_name = "agent-service"

subscriber = OpenTelemetrySubscriber(config)
subscriber.register("otel-exporter")

# Run instrumented application work here.

subscriber.force_flush()
subscriber.deregister("otel-exporter")
subscriber.shutdown()
```

## Common Validation Failures

- `transport` is not `http_binary` or `grpc`.
- Headers or resource attributes are not string-to-string maps.
- The exporter feature is unavailable in the current build or target.
- The endpoint is unreachable at runtime.
