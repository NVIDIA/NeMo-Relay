<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Guide: Export ATIF

Use this guide when you want to collect NeMo Flow lifecycle events and export them as an Agent Trajectory Interchange Format (ATIF) trajectory for offline analysis, replay, or evaluation workflows.

## What You Build

You will create an ATIF exporter, register it as a subscriber, run instrumented work, export the collected trajectory, and clear or deregister the exporter when the collection window ends.

Unlike OpenTelemetry and OpenInference export, ATIF export is in-process and buffered. The exporter collects events until you call `export`, `export_json`, or `clear`.

For automatic per-agent file export, use the built-in
[Observability plugin](observability-plugin.md). Its `atif` section creates one
scope-local exporter for each top-level agent scope and writes each trajectory
when that agent scope ends.

## Before You Start

Complete these steps:

1. Instrument scope, tool, or LLM work so the runtime emits events.
2. Choose a stable `session_id` for the trajectory.
3. Set agent metadata such as name, version, and optional model name.
4. Decide when the collection window starts and ends.
5. Sanitize sensitive event payloads before exporting trajectories outside the process.

## How Events Map to ATIF

The exporter translates NeMo Flow events into ATIF v1.6 trajectory data:

| NeMo Flow event | ATIF output |
|---|---|
| LLM start and end | Agent steps and model metadata. |
| Tool start and end | Tool calls and observations. |
| Scope nesting | Parent-child lineage in trajectory metadata. |
| Event payloads | Step input, output, tool call, or observation content. |

The exporter preserves the collected event order and uses lifecycle pairing to reconstruct the trajectory. Use `root_uuid` or separate collection windows when concurrent agent runs should produce separate trajectories.

## Register and Export

The examples below show how to register the exporter and emit data from instrumented
work.

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

```js
const { AtifExporter } = require("nemo-flow-node");

const exporter = new AtifExporter("session-1", "agent", "1.0.0", "demo-model");
exporter.register("atif-exporter");

// Run instrumented application work here.

const trajectory = JSON.parse(exporter.exportJson());
exporter.deregister("atif-exporter");
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

## Validate the Trajectory

Inspect the exported trajectory before using it in evaluation workflows:

1. Confirm `schema_version` is `ATIF-v1.6`.
2. Confirm agent metadata matches the intended workflow.
3. Confirm the expected LLM and tool steps are present.
4. Confirm tool observations appear after their tool calls.
5. Confirm sensitive fields are absent or sanitized.

If a trajectory is missing tool or LLM steps, verify that those calls use managed execution helpers or explicit lifecycle APIs.

## Collection Boundaries

Choose one of these patterns:

| Pattern | Use When |
|---|---|
| One exporter per run | Each agent run should produce one trajectory. |
| Long-lived exporter with `clear` | A test or local tool exports multiple trajectories in one process. |
| Filtered analysis by root scope | Concurrent runs share one process but can be separated later. |
| Observability plugin ATIF section | Each direct child agent scope should write its own trajectory file automatically. |

For production services, prefer bounded collection windows. Long-lived unbounded exporters can accumulate more event data than expected.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **Trajectory has only scope events**: Route tool and LLM calls through managed execute helpers.
- **Multiple runs appear in one trajectory**: Use one exporter per run or clear the exporter between runs.
- **Payloads are too large**: Sanitize or summarize event payloads before export.
- **Model name is missing**: Pass model metadata to managed LLM calls or set exporter agent metadata when constructing the exporter.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Review event fields in [Code Examples](code-examples.md#event-shape).
- Export trace spans with [Advanced Guide: Export OpenTelemetry Data](opentelemetry.md).
- Add redaction with [Advanced Guide: Add Middleware](../instrument-applications/advanced-guide.md).
