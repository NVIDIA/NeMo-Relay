<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-otel

OpenTelemetry OTLP subscriber support for NeMo Flow.

For the repository-level guide covering event mapping, binding-specific config
objects, and runtime constraints, see
[`docs/observability-with-opentelemetry.md`](../../docs/observability-with-opentelemetry.md).

## Overview

This crate keeps OpenTelemetry support out of the core runtime while still
integrating cleanly with NeMo Flow's existing event subscriber model.

- NeMo Flow `Start` events become OpenTelemetry spans
- NeMo Flow `End` events close those spans
- NeMo Flow `Mark` events become span events when a parent span is active
- OTLP/HTTP and OTLP/gRPC exporters are supported

## Example

```rust
use nemo_flow_otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenTelemetryConfig::http_binary("demo-agent")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_service_version("0.1.0");

    let subscriber = OpenTelemetrySubscriber::new(config)?;
    subscriber.register("otel")?;

    // ... run NeMo Flow-instrumented work here ...

    subscriber.deregister("otel")?;
    subscriber.force_flush()?;
    subscriber.shutdown()?;
    Ok(())
}
```

## Build

```bash
cargo build -p nemo-flow-otel
cargo test -p nemo-flow-otel
```
