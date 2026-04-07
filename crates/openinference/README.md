<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-nat-nexus-openinference

OpenInference OTLP subscriber support for NeMo Agent Toolkit Nexus.

For the repository-level guide covering event mapping, binding-specific config
objects, and runtime constraints, see
[`docs/observability-with-openinference.md`](../../docs/observability-with-openinference.md).

## Overview

This crate keeps OpenInference support out of the core runtime while still
integrating cleanly with Nexus's existing event subscriber model.

- Nexus `Start` events become OpenInference spans
- Nexus `End` events close those spans
- Nexus `Mark` events become span events when a parent span is active
- OTLP/HTTP and OTLP/gRPC exporters are supported

## Example

```rust
use std::time::Duration;

use nvidia_nat_nexus_openinference::{OpenInferenceConfig, OpenInferenceSubscriber, OtlpTransport};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenInferenceConfig::new()
        .with_transport(OtlpTransport::HttpBinary)
        .with_service_name("demo-agent")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_service_version("0.1.0")
        .with_timeout(Duration::from_secs(3));

    let subscriber = OpenInferenceSubscriber::new(config)?;
    subscriber.register("openinference")?;

    // ... run Nexus-instrumented work here ...

    subscriber.deregister("openinference")?;
    subscriber.force_flush()?;
    subscriber.shutdown()?;
    Ok(())
}
```

## Build

```bash
cargo build -p nvidia-nat-nexus-openinference
cargo test -p nvidia-nat-nexus-openinference
```
