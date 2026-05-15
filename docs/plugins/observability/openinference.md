<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenInference

Use the `openinference` section when you want NeMo Flow lifecycle events
exported as OTLP trace spans with OpenInference-oriented semantics.

OpenInference export maps model-centric payloads directly into trace
attributes. Scope, tool, and LLM start inputs become `input.value`; end outputs
become `output.value`; LLM usage metadata maps to token-count attributes when
the provider response includes usage information.

## `plugins.toml` Example

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.openinference]
enabled = true
transport = "http_binary"
endpoint = "http://localhost:6006/v1/traces"
service_name = "agent-service"
service_namespace = "nemo"
service_version = "1.0.0"
instrumentation_scope = "nemo-flow-openinference"
timeout_millis = 3000

[components.config.openinference.headers]
authorization = "Bearer <token>"

[components.config.openinference.resource_attributes]
"deployment.environment" = "dev"
```

This configuration registers a plugin-owned OpenInference subscriber and sends
OpenInference-style OTLP spans to Phoenix or another compatible backend.

## Fields

OpenInference uses the same OTLP section shape as
[OpenTelemetry](opentelemetry.md):

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

The backend should show OpenInference-oriented spans for scopes, tools, and LLM
calls from the same `root_uuid`. LLM usage metadata appears as token counters
when provider responses include usage information.

Redact sensitive event payloads with sanitize guardrails before production
export.

## Manual API

Use the manual subscriber API when you need an explicit subscriber name or
direct `force_flush` control.

```python
from nemo_flow import OpenInferenceConfig, OpenInferenceSubscriber

config = OpenInferenceConfig()
config.transport = "http_binary"
config.endpoint = "http://localhost:6006/v1/traces"
config.service_name = "agent-service"

subscriber = OpenInferenceSubscriber(config)
subscriber.register("openinference-exporter")

# Run instrumented application work here.

subscriber.force_flush()
subscriber.deregister("openinference-exporter")
subscriber.shutdown()
```

## Common Validation Failures

- `transport` is not `http_binary` or `grpc`.
- Headers or resource attributes are not string-to-string maps.
- The OpenInference feature is unavailable in the current build or target.
- Tool and LLM calls do not use managed helpers, so spans contain only scope
  lifecycle data.
