<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Observability with OpenTelemetry

Nexus can export scope, tool, LLM, and mark events as OpenTelemetry traces
through an OTLP-backed subscriber. The exporter implementation lives in the
separate Rust crate `nvidia-nat-nexus-otel`, and the Python, Node.js, Go, and
WASM bindings expose binding-native config objects on top of the same
subscriber.

Use this when you want Nexus activity to appear in an OpenTelemetry Collector,
Jaeger, Tempo, Honeycomb, or another OTLP-compatible backend.

## What Gets Emitted

The subscriber maps Nexus lifecycle events to OpenTelemetry spans like this:

- Nexus `Start` events open spans.
- Nexus `End` events close spans.
- Nexus `Mark` events become span events when a parent span is active.
- Orphan `Mark` events fall back to zero-duration spans so they still export.

That means `scope.push(...)` / `scope.pop(...)`, managed tool execution, managed
LLM execution, and explicit `event(...)` calls all contribute to traces once the
subscriber is registered.

## Configuration Model

Each binding uses a mutable config object before subscriber construction:

- Rust: `OpenTelemetryConfig::{http_binary, grpc}(service_name)` builder.
- Python: `OpenTelemetryConfig()` with mutable fields.
- Node.js: plain object passed to `new OpenTelemetrySubscriber(config)`.
- Go: `NewOpenTelemetryConfig()` returns a mutable struct.
- WASM: `defaultOpenTelemetryConfig()` returns a mutable JS object.

Common fields:

- `endpoint`: OTLP endpoint such as `http://127.0.0.1:4318/v1/traces`
- `service_name`: `service.name` resource attribute
- `service_namespace`: optional `service.namespace`
- `service_version`: optional `service.version`
- `instrumentation_scope`: OpenTelemetry instrumentation scope name
- `headers`: OTLP HTTP headers or gRPC metadata
- `resource_attributes`: extra string resource attributes
- `timeout` or `timeout_millis`: export timeout
- `transport`: `http_binary` or `grpc` where supported

## Authentication

The OpenTelemetry subscriber supports header-based authentication by forwarding
configured `headers` to the underlying OTLP exporter:

- OTLP/HTTP uses them as normal HTTP headers.
- OTLP/gRPC uses them as request metadata.

Typical values include `authorization: Bearer <token>` and vendor API-key
headers such as `x-honeycomb-team: <key>`.

### Rust

HTTP:

```rust
use nvidia_nat_nexus_otel::OpenTelemetryConfig;

let config = OpenTelemetryConfig::http_binary("demo-agent")
    .with_endpoint("http://127.0.0.1:4318/v1/traces")
    .with_header("authorization", "Bearer <token>");
```

gRPC metadata:

```rust
use nvidia_nat_nexus_otel::OpenTelemetryConfig;

let config = OpenTelemetryConfig::grpc("demo-agent")
    .with_endpoint("http://127.0.0.1:4317")
    .with_header("authorization", "Bearer <token>")
    .with_header("x-tenant-id", "tenant-a");
```

### Python

HTTP:

```python
config = nat_nexus.OpenTelemetryConfig()
config.transport = "http_binary"
config.endpoint = "http://127.0.0.1:4318/v1/traces"
config.headers = {
    "authorization": "Bearer <token>",
}
```

gRPC metadata:

```python
config = nat_nexus.OpenTelemetryConfig()
config.transport = "grpc"
config.endpoint = "http://127.0.0.1:4317"
config.headers = {
    "authorization": "Bearer <token>",
    "x-tenant-id": "tenant-a",
}
```

### Node.js

HTTP:

```javascript
const config = {
  transport: "http_binary",
  endpoint: "http://127.0.0.1:4318/v1/traces",
  headers: {
    authorization: "Bearer <token>",
  },
};
```

gRPC metadata:

```javascript
const config = {
  transport: "grpc",
  endpoint: "http://127.0.0.1:4317",
  headers: {
    authorization: "Bearer <token>",
    "x-tenant-id": "tenant-a",
  },
};
```

### Go

HTTP:

```go
config := nat_nexus.NewOpenTelemetryConfig()
config.Transport = nat_nexus.OpenTelemetryTransportHTTPBinary
config.Endpoint = "http://127.0.0.1:4318/v1/traces"
config.Headers["authorization"] = "Bearer <token>"
```

gRPC metadata:

```go
config := nat_nexus.NewOpenTelemetryConfig()
config.Transport = nat_nexus.OpenTelemetryTransportGrpc
config.Endpoint = "http://127.0.0.1:4317"
config.Headers["authorization"] = "Bearer <token>"
config.Headers["x-tenant-id"] = "tenant-a"
```

### WebAssembly

WASM currently supports OTLP/HTTP only:

```javascript
const config = defaultOpenTelemetryConfig();
config.transport = "http_binary";
config.endpoint = "http://127.0.0.1:4318/v1/traces";
config.headers = {
  authorization: "Bearer <token>",
};
```

For OTLP/gRPC, those key/value pairs are sent as request metadata on each
export call.

This repo does not currently expose first-class knobs for mTLS certificates or
custom transport auth flows. When those are required, they need additional
exporter-level support beyond the current Nexus config surface.

## Lifecycle

The intended lifecycle is the same across languages:

1. Create and populate the config object.
2. Construct the `OpenTelemetrySubscriber`.
3. Register it with a unique name.
4. Run Nexus-instrumented work.
5. Deregister it.
6. `force_flush()` / `forceFlush()` if you need deterministic delivery before exit.
7. `shutdown()` when the process or subsystem is done with the exporter.

## Rust

```rust
use nvidia_nat_nexus_core::{ScopeAttributes, ScopeType, nat_nexus_pop_scope, nat_nexus_push_scope};
use nvidia_nat_nexus_otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenTelemetryConfig::http_binary("demo-agent")
        .with_endpoint("http://127.0.0.1:4318/v1/traces")
        .with_service_namespace("agents")
        .with_service_version("0.1.0")
        .with_resource_attribute("deployment.environment", "dev");

    let subscriber = OpenTelemetrySubscriber::new(config)?;
    subscriber.register("otel")?;

    let handle = nat_nexus_push_scope(
        "agent-turn",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )?;

    // ... tool / LLM / mark events here ...

    nat_nexus_pop_scope(&handle.uuid)?;

    subscriber.deregister("otel")?;
    subscriber.force_flush()?;
    subscriber.shutdown()?;
    Ok(())
}
```

For direct Rust integration, the crate-specific overview also lives in
[`crates/otel/README.md`](../crates/otel/README.md).

## Python

```python
import nat_nexus

config = nat_nexus.OpenTelemetryConfig()
config.endpoint = "http://127.0.0.1:4318/v1/traces"
config.service_name = "demo-agent"
config.service_namespace = "agents"
config.service_version = "0.1.0"
config.resource_attributes = {"deployment.environment": "dev"}

subscriber = nat_nexus.OpenTelemetrySubscriber(config)
subscriber.register("otel")

try:
    with nat_nexus.scope.scope("agent-turn", nat_nexus.ScopeType.Agent) as handle:
        nat_nexus.scope.event(
            "tool-selection",
            handle=handle,
            data={"tool": "search"},
        )
        # ... Nexus-managed tool / LLM execution here ...
finally:
    subscriber.deregister("otel")
    subscriber.force_flush()
    subscriber.shutdown()
```

## Node.js

```javascript
import {
  OpenTelemetrySubscriber,
  ScopeType,
  pushScope,
  popScope,
  event,
} from "@nvidia/nat-nexus-node";

const config = {
  endpoint: "http://127.0.0.1:4318/v1/traces",
  serviceName: "demo-agent",
  serviceNamespace: "agents",
  serviceVersion: "0.1.0",
  resourceAttributes: {
    "deployment.environment": "dev",
  },
};

const subscriber = new OpenTelemetrySubscriber(config);
subscriber.register("otel");

try {
  const scope = pushScope("agent-turn", ScopeType.Agent);
  event("tool-selection", scope, { tool: "search" }, null);
  // ... Nexus-managed tool / LLM execution here ...
  popScope(scope);
} finally {
  subscriber.deregister("otel");
  subscriber.forceFlush();
  subscriber.shutdown();
}
```

## Go

```go
config := nat_nexus.NewOpenTelemetryConfig()
config.Endpoint = "http://127.0.0.1:4318/v1/traces"
config.ServiceName = "demo-agent"
config.ServiceNamespace = "agents"
config.ServiceVersion = "0.1.0"
config.ResourceAttributes["deployment.environment"] = "dev"

subscriber, err := nat_nexus.NewOpenTelemetrySubscriber(config)
if err != nil {
    return err
}
defer subscriber.Close()

if err := subscriber.Register("otel"); err != nil {
    return err
}
defer subscriber.Deregister("otel")

handle, err := nat_nexus.PushScope("agent-turn", nat_nexus.ScopeTypeAgent)
if err != nil {
    return err
}
if err := nat_nexus.EmitEvent("tool-selection", nat_nexus.WithEventParent(handle)); err != nil {
    return err
}
if err := nat_nexus.PopScope(handle); err != nil {
    return err
}

if err := subscriber.ForceFlush(); err != nil {
    return err
}
if err := subscriber.Shutdown(); err != nil {
    return err
}
```

## WebAssembly

```javascript
import init, {
  defaultOpenTelemetryConfig,
  OpenTelemetrySubscriber,
  ScopeType,
  pushScope,
  popScope,
  event,
} from "./pkg/nvidia_nat_nexus_wasm.js";

await init();

const config = defaultOpenTelemetryConfig();
config.endpoint = "http://127.0.0.1:4318/v1/traces";
config.service_name = "demo-agent";
config.resource_attributes = {
  "deployment.environment": "dev",
};

const subscriber = new OpenTelemetrySubscriber(config);
subscriber.register("otel");

try {
  const scope = pushScope("agent-turn", ScopeType.Agent, null, null, null, null);
  event("tool-selection", scope, { tool: "search" }, null);
  popScope(scope);
} finally {
  subscriber.deregister("otel");
  subscriber.forceFlush();
  subscriber.shutdown();
}
```

## Constraints

- Native OTLP/gRPC requires an active Tokio runtime in the Rust process.
- WASM currently supports OTLP/HTTP only, not gRPC.
- On WASM, OTLP/HTTP dispatch is asynchronous so request delivery is not tied to
  the synchronous `popScope(...)` call stack.
- Subscriber callbacks are still attached to the Nexus request path, so use
  `force_flush` only when you need deterministic delivery boundaries such as
  tests, shutdown, or short-lived CLI programs.

## Verification

The repository includes end-to-end tests that register the OpenTelemetry
subscriber, emit scope and mark events, and assert OTLP requests across Rust,
Python, Node.js, Go, and the generated WASM package.

## Related Docs

- [Language Bindings](language-bindings.md)
- [Recipes](recipes.md)
- [Testing](testing.md)
