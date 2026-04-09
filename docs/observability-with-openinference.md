<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Observability with OpenInference

NeMo Flow can export scope, tool, LLM, and mark events as OTLP traces annotated
with OpenInference semantic conventions. The exporter implementation lives in
the separate Rust crate `nemo-flow-openinference`, and the Python,
Node.js, Go, and WASM bindings expose language-native config objects on top of
the same subscriber.

Use this when you want NeMo Flow activity to show up in Arize Phoenix or another
backend that understands OpenInference-style semantic attributes over OTLP.

## What Gets Emitted

The subscriber maps NeMo Flow lifecycle events like this:

- NeMo Flow `Start` events open spans.
- NeMo Flow `End` events close spans.
- NeMo Flow `Mark` events become span events when a parent span is active.
- Orphan `Mark` events fall back to zero-duration spans so they still export.

In addition to the base NeMo Flow JSON attributes, spans are annotated with
OpenInference fields such as:

- `openinference.span.kind`
- `input.value` and `input.mime_type`
- `output.value` and `output.mime_type`
- `metadata`
- `llm.model_name`
- `tool.name`, `tool.parameters`, `tool_call.id`

For LLM spans, `input.value` is derived from the request content only. NeMo Flow
does not copy request headers into that OpenInference field.

Span kind mapping:

- `Agent` scopes -> `AGENT`
- `Tool` scopes -> `TOOL`
- `Llm` scopes -> `LLM`
- `Retriever` scopes -> `RETRIEVER`
- `Embedder` scopes -> `EMBEDDING`
- `Reranker` scopes -> `RERANKER`
- `Guardrail` scopes -> `GUARDRAIL`
- `Evaluator` scopes -> `EVALUATOR`
- `Function`, `Custom`, and unknown scopes -> `CHAIN`

## Configuration Model

Each binding uses a mutable config object before subscriber construction:

- Rust: `OpenInferenceConfig::new()` plus chained setters
- Python: `OpenInferenceConfig()` with mutable fields
- Node.js: plain object passed to `new OpenInferenceSubscriber(config)`
- Go: `NewOpenInferenceConfig()` returns a mutable struct
- WASM: `defaultOpenInferenceConfig()` returns a mutable JS object

Common fields:

- `transport`: `http_binary` or `grpc` where supported
- `endpoint`: OTLP endpoint such as `http://127.0.0.1:4318/v1/traces`
- `service_name`: `service.name` resource attribute
- `service_namespace`: optional `service.namespace`
- `service_version`: optional `service.version`
- `instrumentation_scope`: instrumentation scope name
- `headers`: OTLP HTTP headers or gRPC metadata
- `resource_attributes`: extra string resource attributes
- `timeout` or `timeout_millis`: export timeout

## Authentication

The OpenInference subscriber forwards configured `headers` to the OTLP
exporter:

- OTLP/HTTP uses them as request headers
- OTLP/gRPC uses them as request metadata

Typical values include `authorization: Bearer <token>` and vendor-specific API
key headers.

### Rust

```rust
use std::time::Duration;

use nemo_flow_openinference::{OpenInferenceConfig, OtlpTransport};

let config = OpenInferenceConfig::new()
    .with_transport(OtlpTransport::Grpc)
    .with_service_name("demo-agent")
    .with_endpoint("http://127.0.0.1:4317")
    .with_header("authorization", "Bearer <token>")
    .with_header("x-tenant-id", "tenant-a")
    .with_timeout(Duration::from_secs(3));
```

### Python

```python
config = nemo_flow.OpenInferenceConfig()
config.transport = "grpc"
config.endpoint = "http://127.0.0.1:4317"
config.headers = {
    "authorization": "Bearer <token>",
    "x-tenant-id": "tenant-a",
}
```

### Node.js

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

```go
config := nemo_flow.NewOpenInferenceConfig()
config.Transport = nemo_flow.OpenInferenceTransportGrpc
config.Endpoint = "http://127.0.0.1:4317"
config.Headers["authorization"] = "Bearer <token>"
config.Headers["x-tenant-id"] = "tenant-a"
```

### WebAssembly

WASM currently supports OTLP/HTTP only:

```javascript
const config = defaultOpenInferenceConfig();
config.transport = "http_binary";
config.endpoint = "http://127.0.0.1:4318/v1/traces";
config.headers = {
  authorization: "Bearer <token>",
};
```

## Lifecycle

The intended lifecycle is the same across languages:

1. Create and populate the config object.
2. Construct the `OpenInferenceSubscriber`.
3. Register it with a unique name.
4. Run NeMo Flow-instrumented work.
5. Deregister it.
6. `force_flush()` / `forceFlush()` if you need deterministic delivery before exit.
7. `shutdown()` when the exporter is no longer needed.

## Rust

```rust
use std::time::Duration;

use nemo_flow_core::{ScopeAttributes, ScopeType, nemo_flow_event, nemo_flow_pop_scope, nemo_flow_push_scope};
use nemo_flow_openinference::{OpenInferenceConfig, OpenInferenceSubscriber, OtlpTransport};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenInferenceConfig::new()
        .with_transport(OtlpTransport::HttpBinary)
        .with_service_name("demo-agent")
        .with_endpoint("http://127.0.0.1:4318/v1/traces")
        .with_service_namespace("agents")
        .with_service_version("0.1.0")
        .with_instrumentation_scope("demo-openinference")
        .with_resource_attribute("deployment.environment", "dev")
        .with_timeout(Duration::from_secs(3));

    let subscriber = OpenInferenceSubscriber::new(config)?;
    subscriber.register("openinference")?;

    let handle = nemo_flow_push_scope(
        "agent-turn",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        Some(serde_json::json!({"scope": true})),
        Some(serde_json::json!({"phase": "start"})),
    )?;

    nemo_flow_event(
        "tool-selection",
        Some(&handle),
        Some(serde_json::json!({"tool": "search"})),
        Some(serde_json::json!({"source": "rust"})),
    )?;
    nemo_flow_pop_scope(&handle.uuid)?;

    subscriber.deregister("openinference")?;
    subscriber.force_flush()?;
    subscriber.shutdown()?;
    Ok(())
}
```

For direct Rust integration, the crate-specific overview also lives in
[`crates/openinference/README.md`](../crates/openinference/README.md).

## Python

```python
import nemo_flow

config = nemo_flow.OpenInferenceConfig()
config.endpoint = "http://127.0.0.1:4318/v1/traces"
config.service_name = "demo-agent"
config.service_namespace = "agents"
config.service_version = "0.1.0"
config.instrumentation_scope = "demo-openinference"
config.resource_attributes = {"deployment.environment": "dev"}

subscriber = nemo_flow.OpenInferenceSubscriber(config)
subscriber.register("openinference")

try:
    with nemo_flow.scope.scope("agent-turn", nemo_flow.ScopeType.Agent) as handle:
        nemo_flow.scope.event(
            "tool-selection",
            handle=handle,
            data={"tool": "search"},
            metadata={"source": "python"},
        )
finally:
    subscriber.deregister("openinference")
    subscriber.force_flush()
    subscriber.shutdown()
```

## Node.js

```javascript
import {
  OpenInferenceSubscriber,
  ScopeType,
  event,
  popScope,
  pushScope,
} from "@nvidia/nemo-flow-node";

const config = {
  endpoint: "http://127.0.0.1:4318/v1/traces",
  serviceName: "demo-agent",
  serviceNamespace: "agents",
  serviceVersion: "0.1.0",
  instrumentationScope: "demo-openinference",
  resourceAttributes: {
    "deployment.environment": "dev",
  },
};

const subscriber = new OpenInferenceSubscriber(config);
subscriber.register("openinference");

try {
  const handle = pushScope("agent-turn", ScopeType.Agent, null, null, { scope: true }, null);
  event("tool-selection", handle, { tool: "search" }, { source: "node" });
  popScope(handle);
} finally {
  subscriber.deregister("openinference");
  subscriber.forceFlush();
  subscriber.shutdown();
}
```

## Go

```go
config := nemo_flow.NewOpenInferenceConfig()
config.Endpoint = "http://127.0.0.1:4318/v1/traces"
config.ServiceName = "demo-agent"
config.ServiceNamespace = "agents"
config.ServiceVersion = "0.1.0"
config.InstrumentationScope = "demo-openinference"
config.ResourceAttributes["deployment.environment"] = "dev"

subscriber, err := nemo_flow.NewOpenInferenceSubscriber(config)
if err != nil {
    panic(err)
}
defer subscriber.Close()

if err := subscriber.Register("openinference"); err != nil {
    panic(err)
}
defer subscriber.Deregister("openinference")
defer subscriber.Shutdown()
```

## WebAssembly

```javascript
import init, {
  defaultOpenInferenceConfig,
  OpenInferenceSubscriber,
  ScopeType,
  event,
  popScope,
  pushScope,
} from "./pkg/nemo_flow_wasm.js";

await init();

const config = defaultOpenInferenceConfig();
config.endpoint = "http://127.0.0.1:4318/v1/traces";
config.service_name = "demo-agent";
config.instrumentation_scope = "demo-openinference";
config.resource_attributes = {
  "deployment.environment": "dev",
};

const subscriber = new OpenInferenceSubscriber(config);
subscriber.register("openinference");

try {
  const handle = pushScope("agent-turn", ScopeType.Agent, null, 0, { scope: true }, null);
  event("tool-selection", handle, { tool: "search" }, { source: "wasm" });
  popScope(handle);
} finally {
  subscriber.deregister("openinference");
  subscriber.forceFlush();
  subscriber.shutdown();
}
```

## Runtime Constraints

- Native OTLP/gRPC requires an active Tokio runtime.
- WASM supports OTLP/HTTP only.
- WASM export is dispatched asynchronously with `fetch`; request dispatch is
  best-effort and browser or runtime errors surface through console warnings.
