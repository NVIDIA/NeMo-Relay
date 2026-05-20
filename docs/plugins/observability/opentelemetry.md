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

Each lifecycle span includes `nemo_flow.uuid` and `nemo_flow.parent_uuid`
attributes. These values match ATIF `step.extra.ancestry.function_id` and
`step.extra.ancestry.parent_id` for the same events. For plugin-managed ATIF,
the root agent span's `nemo_flow.uuid` also matches the ATIF `session_id`.
Backend-native `trace_id` and `span_id` values are not written into ATIF.

Register the plugin before the first instrumented request, use stable service
identity fields, keep credentials outside source code, and flush during
graceful shutdown.

## Plugin Configuration

Use plugin configuration when the application should let NeMo Flow own the
OpenTelemetry subscriber lifecycle.

:::::{tab-set}
:sync-group: language

::::{tab-item} Python
:sync: python

```python
from nemo_flow import plugin
from nemo_flow.observability import ComponentSpec, ObservabilityConfig, OtlpConfig

config = plugin.PluginConfig(
    components=[
        ComponentSpec(
            ObservabilityConfig(
                opentelemetry=OtlpConfig(
                    enabled=True,
                    transport="http_binary",
                    endpoint="http://localhost:4318/v1/traces",
                    service_name="agent-service",
                    service_namespace="nemo",
                    service_version="1.0.0",
                    instrumentation_scope="nemo-flow-otel",
                    resource_attributes={"deployment.environment": "dev"},
                    headers={"authorization": "Bearer <token>"},
                )
            )
        )
    ]
)

report = plugin.validate(config)
if any(diagnostic["level"] == "error" for diagnostic in report["diagnostics"]):
    raise RuntimeError(report["diagnostics"])

await plugin.initialize(config)
try:
    # Run instrumented application work here.
    pass
finally:
    plugin.clear()
```

::::

::::{tab-item} Node.js
:sync: node

```js
const plugin = require("nemo-flow-node/plugin");
const observability = require("nemo-flow-node/observability");

await plugin.initialize({
  version: 1,
  components: [
    observability.ComponentSpec({
      version: 1,
      opentelemetry: observability.otlpConfig({
        enabled: true,
        transport: "http_binary",
        endpoint: "http://localhost:4318/v1/traces",
        service_name: "agent-service",
        service_namespace: "nemo",
        service_version: "1.0.0",
        instrumentation_scope: "nemo-flow-otel",
        resource_attributes: {
          "deployment.environment": "dev",
        },
        headers: {
          authorization: "Bearer <token>",
        },
      }),
    }),
  ],
});

try {
  // Run instrumented application work here.
} finally {
  plugin.clear();
}
```

::::

::::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::observability::plugin_component::{
    ComponentSpec, ObservabilityConfig, OtlpSectionConfig,
};
use nemo_flow::plugin::{initialize_plugins, validate_plugin_config, PluginConfig};

let component = ComponentSpec::new(ObservabilityConfig {
    opentelemetry: Some(OtlpSectionConfig {
        enabled: true,
        transport: "http_binary".into(),
        endpoint: Some("http://localhost:4318/v1/traces".into()),
        service_name: "agent-service".into(),
        service_namespace: Some("nemo".into()),
        service_version: Some("1.0.0".into()),
        instrumentation_scope: Some("nemo-flow-otel".into()),
        resource_attributes: [("deployment.environment".into(), "dev".into())].into(),
        headers: [("authorization".into(), "Bearer <token>".into())].into(),
        ..OtlpSectionConfig::default()
    }),
    ..ObservabilityConfig::default()
});

let config = PluginConfig {
    version: 1,
    components: vec![component.into()],
    policy: Default::default(),
};

let report = validate_plugin_config(&config);
assert!(!report.has_errors());

let active = initialize_plugins(config).await?;
```

::::

:::::

## Manual API

Use the manual subscriber API when you need an explicit subscriber name or
direct `force_flush` control.

:::::{tab-set}
:sync-group: language

::::{tab-item} Python
:sync: python

```python
from nemo_flow import OpenTelemetryConfig, OpenTelemetrySubscriber

config = OpenTelemetryConfig()
config.transport = "http_binary"
config.endpoint = "http://localhost:4318/v1/traces"
config.service_name = "agent-service"
config.set_resource_attribute("deployment.environment", "dev")

subscriber = OpenTelemetrySubscriber(config)
subscriber.register("otel-exporter")

# Run instrumented application work here.

subscriber.force_flush()
subscriber.deregister("otel-exporter")
subscriber.shutdown()
```

::::

::::{tab-item} Node.js
:sync: node

```js
const { OpenTelemetrySubscriber } = require("nemo-flow-node");

const subscriber = new OpenTelemetrySubscriber({
  transport: "http_binary",
  endpoint: "http://localhost:4318/v1/traces",
  serviceName: "agent-service",
  resourceAttributes: {
    "deployment.environment": "dev",
  },
});
subscriber.register("otel-exporter");

try {
  // Run instrumented application work here.

  subscriber.forceFlush();
} finally {
  subscriber.deregister("otel-exporter");
  subscriber.shutdown();
}
```

::::

::::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::observability::otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};

let config = OpenTelemetryConfig::http_binary("agent-service")
    .with_endpoint("http://localhost:4318/v1/traces")
    .with_resource_attribute("deployment.environment", "dev");
let subscriber = OpenTelemetrySubscriber::new(config)?;
subscriber.register("otel-exporter")?;

// Run instrumented application work here.

subscriber.force_flush()?;
let _ = subscriber.deregister("otel-exporter")?;
subscriber.shutdown()?;
```

::::

:::::

## Common Validation Failures

- `transport` is not `http_binary` or `grpc`.
- Headers or resource attributes are not string-to-string maps.
- The exporter feature is unavailable in the current build or target.
- The endpoint is unreachable at runtime.
