<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Guide: Export OpenInference Data

Use this guide when you want NeMo Flow lifecycle events exported as OTLP trace spans with OpenInference-oriented semantics.

## What You Build

You will configure the OpenInference subscriber, register it before scoped work starts, run instrumented work, flush spans, and shut down the exporter.

The OpenInference subscriber maps NeMo Flow lifecycle payloads to trace attributes:

- Scope, tool, and LLM start inputs become `input.value`.
- Scope, tool, and LLM end outputs become `output.value`.
- LLM usage metadata maps to OpenInference token-count attributes when the provider response includes usage information.

## Before You Start

Complete these steps:

1. Instrument at least one scope, tool call, or LLM call.
2. Decide where OTLP traces should be sent.
3. Start an OTLP/HTTP collector or tracing backend.
4. Redact sensitive event payloads with sanitize guardrails before production export.

## Configure the Exporter

Set these fields first:

| Field | Purpose |
|---|---|
| `endpoint` | OTLP trace endpoint, such as `http://localhost:4318/v1/traces` |
| `service_name` | Logical service name shown in traces |
| `service_namespace` | Optional namespace for grouping services |
| `service_version` | Optional application or package version |
| `headers` | Optional auth or routing headers |
| `resource_attributes` | Deployment metadata such as environment or region |
| `timeout_millis` | Export timeout in milliseconds |

OpenInference exports semantic lifecycle payloads directly:

- Scope, tool, and LLM start inputs become `input.value`
- Scope, tool, and LLM end outputs become `output.value`
- LLM usage metadata maps token counters when the provider response includes usage information

`OTEL_*` variables may be used by the underlying OpenTelemetry exporter when values are not set directly in config. Prefer explicit config fields for endpoint, headers, resource attributes, and service identity in application code.

## Register and Export

The examples below show how to register the exporter and emit data from instrumented
work.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nemo_flow import OpenInferenceConfig, OpenInferenceSubscriber

config = OpenInferenceConfig()
config.transport = "http_binary"
config.endpoint = "http://localhost:4318/v1/traces"
config.service_name = "agent-service"
config.service_namespace = "nemo"
config.service_version = "1.0.0"
config.instrumentation_scope = "nemo-flow-openinference"
config.timeout_millis = 3000
config.headers = {"authorization": "Bearer token"}
config.resource_attributes = {"deployment.environment": "dev"}

subscriber = OpenInferenceSubscriber(config)
subscriber.register("openinference-exporter")

# Run instrumented application work here.

subscriber.force_flush()
subscriber.deregister("openinference-exporter")
subscriber.shutdown()
```
:::

:::{tab-item} Node.js
:sync: node

```js
const { OpenInferenceSubscriber } = require("nemo-flow-node");

const subscriber = new OpenInferenceSubscriber({
  transport: "http_binary",
  endpoint: "http://localhost:4318/v1/traces",
  serviceName: "agent-service",
  serviceNamespace: "nemo",
  serviceVersion: "1.0.0",
  instrumentationScope: "nemo-flow-openinference",
  timeoutMillis: 3000,
  headers: { authorization: "Bearer token" },
  resourceAttributes: { "deployment.environment": "dev" },
});

subscriber.register("openinference-exporter");

// Run instrumented application work here.

subscriber.forceFlush();
subscriber.deregister("openinference-exporter");
subscriber.shutdown();
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::observability::openinference::{OpenInferenceConfig, OpenInferenceSubscriber};

let subscriber = OpenInferenceSubscriber::new(
    OpenInferenceConfig::new()
        .with_service_name("agent-service")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_header("authorization", "Bearer token")
        .with_resource_attribute("deployment.environment", "dev")
        .with_service_namespace("nemo")
        .with_service_version("1.0.0")
        .with_instrumentation_scope("nemo-flow-openinference"),
)?;

subscriber.register("openinference-exporter")?;

// Run instrumented application work here.

subscriber.force_flush()?;
subscriber.deregister("openinference-exporter")?;
subscriber.shutdown()?;
```
:::

::::

## Validate the Export

Check the export in three places:

1. Application logs should not show exporter construction or transport errors.
2. The collector should receive OTLP/HTTP trace export requests.
3. The tracing backend should show spans for scopes, tools, and LLM calls from the same `root_uuid`.

If spans arrive without useful payloads, check that tool and LLM calls pass JSON-compatible `input` and `output` values. If payloads contain sensitive fields, add sanitize guardrails before exporting.

## Production Checklist

Use this checklist before running the pattern in production traffic.

- Register the exporter before the first instrumented request.
- Flush during graceful shutdown.
- Use stable service identity fields across deployments.
- Keep headers and endpoint configuration outside source code.
- Filter or redact sensitive payloads before export.
- Use `root_uuid` to isolate concurrent agent runs in trace analysis.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **No spans appear**: Confirm that the OTLP endpoint is reachable and that application work emits events.
- **Exporter fails on startup**: Confirm the transport is `http_binary`.
- **Only scope spans appear**: Route tools and LLM calls through the managed execute helpers.
- **Sensitive data appears in the backend**: Add sanitize guardrails for event payloads.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Add custom redaction with [Advanced Guide: Add Middleware](../instrument-applications/advanced-guide.md).
- Compare generic OTLP export with [Advanced Guide: Export OpenTelemetry Data](opentelemetry.md).
- Export trajectory artifacts with [Advanced Guide: Export ATIF](atif.md).
- Review [Code Examples](code-examples.md) for event shape, ATIF, OpenTelemetry, and exporter selection snippets.
