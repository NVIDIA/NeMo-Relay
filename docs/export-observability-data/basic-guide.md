<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Basic Guide: Register a Subscriber

Use this guide when you want to observe NeMo Flow lifecycle events inside the same process. Subscribers are the simplest way to verify instrumentation, collect debug logs, update counters, or forward events to another exporter.

## What You Build

You will register a subscriber, run scoped work, inspect emitted events, and deregister the subscriber when it is no longer needed.

Subscribers receive events from instrumented scopes, tool calls, LLM calls, middleware, and mark events. They do not change runtime behavior. Use middleware when you need to transform or block execution.

## Before You Start

Instrument at least one scope, tool call, or LLM call. A subscriber only sees events that the runtime emits.

Good starting points:

- [Basic Guide: Adding Scopes and Marks](../instrument-applications/adding-scopes-and-marks.md)
- [Basic Guide: Instrument a Tool Call](../instrument-applications/instrument-tool-call.md)
- [Basic Guide: Instrument an LLM Call](../instrument-applications/instrument-llm-call.md)
- [Python Quick Start](../getting-started/python/index.md)
- [Node.js Quick Start](../getting-started/nodejs.md)
- [Rust Quick Start](../getting-started/rust.md)

## Subscriber Lifecycle

Follow this lifecycle to register, use, and clean up a subscriber safely.

1. Define a callback that accepts one event.
2. Register the callback under a stable subscriber name.
3. Run scoped application work.
4. Inspect or export the event fields you need.
5. Deregister the subscriber during teardown or test cleanup.

Use global subscribers for process-level export. Use scope-local subscribers when only one request or tenant should be observed.

## Register a Global Subscriber

The examples below register one process-wide subscriber and remove it during cleanup.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow


def log_event(event) -> None:
    print({
        "kind": event.kind,
        "name": event.name,
        "root_uuid": getattr(event, "root_uuid", None),
    })


nemo_flow.subscribers.register("debug-logger", log_event)

# Run instrumented application work here.

removed = nemo_flow.subscribers.deregister("debug-logger")
```
:::

:::{tab-item} Node.js
:sync: node

```js
const { deregisterSubscriber, registerSubscriber } = require("nemo-flow-node");

registerSubscriber("debug-logger", (event) => {
  console.log({
    kind: event.kind,
    name: event.name,
    root_uuid: event.root_uuid,
  });
});

// Run instrumented application work here.

const removed = deregisterSubscriber("debug-logger");
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use std::sync::Arc;

register_subscriber(
    "debug-logger",
    Arc::new(|event| {
        println!("{} {}", event.kind(), event.name());
    }),
)?;

// Run instrumented application work here.

let removed = deregister_subscriber("debug-logger")?;
```
:::

::::

## Decide What to Capture

Start with a small event record:

- `kind`: The lifecycle event type.
- `name`: The scope, tool, LLM provider, subscriber, or mark name.
- `root_uuid`: The root scope identifier for isolating concurrent agents.
- `input`: The post-guardrail input payload for start events.
- `output`: The post-guardrail output payload for end events.
- `model_name`: The model name for LLM events when provided.
- `tool_call_id`: The tool call identifier for tool events when provided.

Do not serialize full payloads by default in production. Use sanitize guardrails to redact sensitive fields before subscribers or exporters receive events.

## Subscriber Options

The table below compares subscriber and exporter options for common observability needs.

| Subscriber / Exporter | Purpose |
|---|---|
| Custom subscriber | Consume events in process. |
| Scope-local subscriber | Observe one request or tenant and clean up when its scope closes. |
| ATOF JSONL exporter | Write raw ATOF events as one JSON object per line. |
| ATIF exporter | Collect events and export ATIF v1.6 trajectories. |
| OpenTelemetry subscriber | Export lifecycle events as OTLP spans. |
| OpenInference subscriber | Export lifecycle events as OTLP spans with OpenInference-oriented semantics. |
| Observability plugin | Configure ATOF, per-agent ATIF, OpenTelemetry, and OpenInference from one built-in plugin component. |

## Validate the Subscriber

Run one known instrumented path and check for the expected lifecycle sequence:

1. A scope start event for the active agent, request, or workflow.
2. Tool or LLM start events for managed calls.
3. Tool or LLM end events after callbacks return.
4. A scope end event when the scope closes.

If the subscriber receives no events, verify that the work uses `tools.execute`, `toolCallExecute`, `tool_call_execute`, `llm.execute`, `llmCallExecute`, or `llm_call_execute`.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **Duplicate subscriber name**: Subscriber names are registry keys. Reusing a name replaces or conflicts with the existing registration depending on the binding behavior.
- **Events from unrelated requests appear**: Filter by `root_uuid` or use scope-local subscribers.
- **Sensitive payloads appear in logs**: Add sanitize guardrails before registering production exporters.
- **Tests leak subscribers**: Deregister test subscribers in teardown or a `finally` block.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Export generic OTLP spans with [Advanced Guide: Export OpenTelemetry Data](opentelemetry.md).
- Export traces with [Advanced Guide: Export OpenInference Data](advanced-guide.md).
- Export trajectory artifacts with [Advanced Guide: Export ATIF](atif.md).
- Configure standard exporters with [Basic Guide: Configure the Observability Plugin](observability-plugin.md).
- Use [Code Examples](code-examples.md) for event shape, scope-local subscribers, ATIF, and OpenTelemetry snippets.
- Add redaction with [Advanced Guide: Add Middleware](../instrument-applications/advanced-guide.md).
