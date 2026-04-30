<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Examples

Use these examples when you need to inspect or export the lifecycle event stream.

## Event Shape

NeMo Flow emits Agent Trajectory Observability Format (ATOF) `0.1` events. The
wire format has two event kinds:

- `scope`: start or end lifecycle events for agent, function, tool, LLM, retrieval, embedding, reranking, guardrail, evaluator, custom, or unknown work
- `mark`: point-in-time checkpoints that do not represent a full lifecycle span

Every event includes a shared envelope:

- `kind`
- `atof_version`
- `parent_uuid`
- `uuid`
- `timestamp`
- `name`
- `data`
- `data_schema`
- `metadata`

Scope events add:

- `scope_category`: `start` or `end`
- `category`: semantic work category, such as `agent`, `function`, `tool`, or `llm`
- `attributes`: behavioral flags, such as `remote`, `stateful`, `streaming`, `parallel`, or `relocatable`
- `category_profile`: fields such as `model_name`, `tool_call_id`, or `subtype`

Start and end events for the same lifecycle use the same `uuid`. The `data` field is the semantic input on start events and the semantic output on end events.

## Inspect Events

The examples below show how to inspect emitted event payloads from each binding.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nemo_flow import Event


def _profile_value(profile, field: str):
    if isinstance(profile, dict):
        return profile.get(field)
    return getattr(profile, field, None)


def on_event(event: Event) -> None:
    print(event.kind, event.name, getattr(event, "uuid", None))

    if event.kind == "scope":
        print(event.scope_category, event.category, event.data)
        print(_profile_value(event.category_profile, "model_name"))
        print(_profile_value(event.category_profile, "tool_call_id"))
    elif event.kind == "mark":
        print(event.data, event.metadata)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { registerSubscriber } from 'nemo-flow-node';

registerSubscriber('logger', (event) => {
  console.log(event.kind, event.name, event.uuid);

  if (event.kind === 'scope') {
    console.log(event.scope_category, event.category, event.data);
    console.log(event.category_profile?.model_name, event.category_profile?.tool_call_id);
  } else if (event.kind === 'mark') {
    console.log(event.data, event.metadata);
  }
});
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::event::{Event, ScopeCategory};

match event {
    Event::Scope(e) => {
        let input = (e.scope_category == ScopeCategory::Start).then_some(&e.base.data);
        let output = (e.scope_category == ScopeCategory::End).then_some(&e.base.data);
        let _ = (&e.base.uuid, &e.category, &e.attributes, input, output);
    }
    Event::Mark(e) => {
        let _ = (&e.base.uuid, &e.base.data, &e.category, &e.category_profile);
    }
}
```
:::

::::

## Scope-Local Subscriber

Use scope-local subscribers when observation should be owned by one request and removed when that scope closes.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow

scope = nemo_flow.scope.push("request", nemo_flow.ScopeType.Agent)
nemo_flow.scope_local.register_subscriber(
    scope,
    "scoped-logger",
    lambda event: print(event.kind, event.name),
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { ScopeType, pushScope, scopeRegisterSubscriber } from 'nemo-flow-node';

const scope = pushScope('request', ScopeType.Agent, null, null, null, null, null);
scopeRegisterSubscriber(scope.uuid, 'scoped-logger', (event) => {
  console.log(event.kind, event.name);
});
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::scope::{push_scope, PushScopeParams, ScopeAttributes, ScopeType};
use nemo_flow::api::subscriber::scope_register_subscriber;
use std::sync::Arc;

let scope = push_scope(
    PushScopeParams::builder()
        .name("request")
        .scope_type(ScopeType::Agent)
        .attributes(ScopeAttributes::empty())
        .build(),
)?;

scope_register_subscriber(&scope.uuid, "scoped-logger", Arc::new(|event| {
    let _ = (event.kind(), event.name());
}))?;
```
:::

::::

## ATIF Export

The ATIF exporter collects lifecycle events and exports an ATIF trajectory for offline analysis, replay, or debugging.

For operational guidance, see [Advanced Guide: Export ATIF](atif.md).

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nemo_flow import AtifExporter

exporter = AtifExporter("session-1", "agent", "1.0.0", model_name="demo-model")
exporter.register("atif-exporter")

# Run instrumented application work here.

trajectory = exporter.export()
trajectory_json = exporter.export_json()
exporter.deregister("atif-exporter")
exporter.clear()
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { AtifExporter } from 'nemo-flow-node';

const exporter = new AtifExporter('session-1', 'agent', '1.0.0', 'demo-model');
exporter.register('atif-exporter');

// Run instrumented application work here.

const trajectory = JSON.parse(exporter.exportJson());
exporter.deregister('atif-exporter');
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_flow::observability::atif::{AtifAgentInfo, AtifExporter};

let exporter = AtifExporter::new(
    "session-1".into(),
    AtifAgentInfo {
        name: "agent".into(),
        version: "1.0.0".into(),
        model_name: Some("demo-model".into()),
        tool_definitions: None,
        extra: None,
    },
);

let subscriber = exporter.subscriber();
register_subscriber("atif-exporter", subscriber.clone())?;

// Run instrumented application work here.

let trajectory = exporter.export();
let removed = deregister_subscriber("atif-exporter")?;
exporter.clear();
```
:::

::::

## OpenTelemetry Export

Use the OpenTelemetry subscriber when you want generic OTLP spans from NeMo Flow lifecycle events.

For setup and validation guidance, see [Advanced Guide: Export OpenTelemetry Data](opentelemetry.md).

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nemo_flow import OpenTelemetryConfig, OpenTelemetrySubscriber

config = OpenTelemetryConfig()
config.transport = "http_binary"
config.endpoint = "http://localhost:4318/v1/traces"
config.service_name = "agent"
config.service_namespace = "nemo"
config.service_version = "1.0.0"
config.instrumentation_scope = "nemo-flow-otel"
config.timeout_millis = 3000
config.headers = {"authorization": "Bearer token"}
config.resource_attributes = {"deployment.environment": "dev"}

subscriber = OpenTelemetrySubscriber(config)
subscriber.register("otel-exporter")
subscriber.force_flush()
subscriber.deregister("otel-exporter")
subscriber.shutdown()
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import { OpenTelemetrySubscriber } from 'nemo-flow-node';

const subscriber = new OpenTelemetrySubscriber({
  transport: 'http_binary',
  endpoint: 'http://localhost:4318/v1/traces',
  serviceName: 'agent',
  serviceNamespace: 'nemo',
  serviceVersion: '1.0.0',
  instrumentationScope: 'nemo-flow-otel',
  timeoutMillis: 3000,
  headers: { authorization: 'Bearer token' },
  resourceAttributes: { 'deployment.environment': 'dev' },
});

subscriber.register('otel-exporter');
subscriber.forceFlush();
subscriber.deregister('otel-exporter');
subscriber.shutdown();
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::observability::otel::{OpenTelemetryConfig, OpenTelemetrySubscriber};

let subscriber = OpenTelemetrySubscriber::new(
    OpenTelemetryConfig::http_binary("agent")
        .with_endpoint("http://localhost:4318/v1/traces")
        .with_header("authorization", "Bearer token")
        .with_resource_attribute("deployment.environment", "dev")
        .with_service_namespace("nemo")
        .with_service_version("1.0.0")
        .with_instrumentation_scope("nemo-flow-otel"),
)?;

subscriber.register("otel-exporter")?;
subscriber.force_flush()?;
subscriber.deregister("otel-exporter")?;
subscriber.shutdown()?;
```
:::

::::

## Exporter Selection

The table below summarizes which exporter or subscriber to start with for each goal.

| Subscriber / Exporter | Purpose |
|---|---|
| Custom subscriber | Consume events in process. |
| ATIF exporter | Collect events and export ATIF v1.6 trajectories. |
| OpenTelemetry subscriber | Export lifecycle events as OTLP spans. |
| OpenInference subscriber | Export lifecycle events as OTLP spans with OpenInference-oriented semantics. |

OpenInference maps lifecycle payloads directly:

- Start inputs become `input.value`
- End outputs become `output.value`
- LLM usage metadata maps to OpenInference token-count attributes when the response includes usage

`OTEL_*` variables may be used by the underlying OpenTelemetry exporter when endpoint settings are not passed directly in config, but prefer explicit config objects for application code.
