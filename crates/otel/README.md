<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-otel

OpenTelemetry OTLP subscriber support for NeMo Agent Toolkit Nexus.

For the repository-level guide covering event mapping, binding-specific config
objects, and runtime constraints, see
[`docs/observability-with-opentelemetry.md`](../../docs/observability-with-opentelemetry.md).

## Overview

This crate keeps OpenTelemetry support out of the core runtime while still
integrating cleanly with Nexus's existing event subscriber model.

- Nexus `Start` events become OpenTelemetry spans
- Nexus `End` events close those spans
- Nexus `Mark` events become span events when a parent span is active
- OTLP/HTTP and OTLP/gRPC exporters are supported

## Example

```rust
use nvidia_nat_nexus_otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenTelemetryConfig::http_binary("demo-agent")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_service_version("0.1.0");

    let subscriber = OpenTelemetrySubscriber::new(config)?;
    subscriber.register("otel")?;

    // ... run Nexus-instrumented work here ...

    subscriber.deregister("otel")?;
    subscriber.force_flush()?;
    subscriber.shutdown()?;
    Ok(())
}
```

## Build

```bash
cargo build -p nvidia-nat-nexus-otel
cargo test -p nvidia-nat-nexus-otel
```
