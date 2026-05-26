<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Trajectory Interchange Format (ATIF)

Use the `atif` section when you want one Agent Trajectory Interchange Format
(ATIF) trajectory artifact per top-level agent run.

The plugin-managed ATIF dispatcher watches for direct child scopes with category
`agent`, creates a scope-local exporter for each one, and writes the trajectory
when that agent scope ends. Nested agent scopes remain in the parent
trajectory.

## `plugins.toml` Example

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atif]
enabled = true
agent_name = "Planner"
agent_version = "1.0.0"
model_name = "unknown"
output_directory = "logs"
filename_template = "trajectory-{session_id}.json"
```

This configuration writes a trajectory file such as
`logs/trajectory-<scope-uuid>.json` for each top-level agent scope.

## Fields

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to write trajectories. |
| `agent_name` | `NeMo Relay` | Agent metadata written into the trajectory. |
| `agent_version` | NeMo Relay crate version | Agent version metadata. |
| `model_name` | `unknown` | Default model metadata when no call-level model is present. |
| `tool_definitions` | Omitted | Optional ATIF tool metadata. |
| `extra` | Omitted | Optional ATIF agent metadata. |
| `output_directory` | Current working directory | Directory containing trajectory files. |
| `filename_template` | `nemo-relay-atif-{session_id}.json` | Must contain `{session_id}`. |
| `subagent_export_mode` | `embedded` | `embedded` writes child trajectories in `subagent_trajectories`; `file_ref` writes parent references without child bodies. |

## Expected Output

The exporter translates NeMo Relay lifecycle events into ATIF v1.7 trajectory
data. LLM start and end events become model steps, tool events become tool
calls and observations, and scope nesting contributes lineage metadata.
Nested agent scopes are embedded in the parent file as `subagent_trajectories`
and referenced from parent observation results with `trajectory_id`.
Set `subagent_export_mode = "file_ref"` when the host writes child trajectories
as separate artifacts and wants the parent to reference them by path instead.

The plugin writes each trajectory when the top-level agent scope closes. If the
plugin is cleared while an agent is still open, teardown flushes the partial
trajectory.

To correlate ATIF with OpenTelemetry or OpenInference traces from the same run,
join on NeMo Relay UUIDs. The plugin-managed ATIF `session_id` is the top-level
agent scope UUID. Each step's `extra.ancestry.function_id` is the event UUID,
and `extra.ancestry.parent_id` is the parent event UUID. Trace spans expose the
same values as `nemo_relay.uuid` and `nemo_relay.parent_uuid` attributes.

## Plugin Configuration

Use plugin configuration when the application should let NeMo Relay own the ATIF
dispatcher lifecycle.

:::::{tab-set}
:sync-group: language

::::{tab-item} Python
:sync: python

```python
from nemo_relay import plugin
from nemo_relay.observability import AtifConfig, ComponentSpec, ObservabilityConfig

config = plugin.PluginConfig(
    components=[
        ComponentSpec(
            ObservabilityConfig(
                atif=AtifConfig(
                    enabled=True,
                    agent_name="Planner",
                    agent_version="1.0.0",
                    model_name="unknown",
                    output_directory="logs",
                    filename_template="trajectory-{session_id}.json",
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
const plugin = require("nemo-relay-node/plugin");
const observability = require("nemo-relay-node/observability");

await plugin.initialize({
  version: 1,
  components: [
    observability.ComponentSpec({
      version: 1,
      atif: observability.atifConfig({
        enabled: true,
        agent_name: "Planner",
        agent_version: "1.0.0",
        model_name: "unknown",
        output_directory: "logs",
        filename_template: "trajectory-{session_id}.json",
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
use nemo_relay::observability::plugin_component::{
    AtifSectionConfig, ComponentSpec, ObservabilityConfig,
};
use nemo_relay::plugin::{initialize_plugins, validate_plugin_config, PluginConfig};

let component = ComponentSpec::new(ObservabilityConfig {
    atif: Some(AtifSectionConfig {
        enabled: true,
        agent_name: "Planner".into(),
        agent_version: "1.0.0".into(),
        model_name: "unknown".into(),
        output_directory: Some("logs".into()),
        filename_template: "trajectory-{session_id}.json".into(),
        ..AtifSectionConfig::default()
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

Use the manual `AtifExporter` API when you need explicit collection boundaries
or one exporter object per run.

:::::{tab-set}
:sync-group: language

::::{tab-item} Python
:sync: python

```python
from nemo_relay import AtifExporter

exporter = AtifExporter("session-1", "agent", "1.0.0", model_name="demo-model")
exporter.register("atif-exporter")

# Run instrumented application work here.

trajectory = exporter.export()
exporter.deregister("atif-exporter")
exporter.clear()
```

::::

::::{tab-item} Node.js
:sync: node

```js
const { AtifExporter } = require("nemo-relay-node");

const exporter = new AtifExporter("session-1", "agent", "1.0.0", "demo-model");
exporter.register("atif-exporter");

try {
  // Run instrumented application work here.

  const trajectoryJson = exporter.exportJson();
  console.log(trajectoryJson);
} finally {
  exporter.deregister("atif-exporter");
  exporter.clear();
}
```

::::

::::{tab-item} Rust
:sync: rust

```rust
use nemo_relay::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_relay::observability::atif::{AtifAgentInfo, AtifExporter};

let exporter = AtifExporter::new(
    "session-1".to_string(),
    AtifAgentInfo {
        name: "agent".to_string(),
        version: "1.0.0".to_string(),
        model_name: Some("demo-model".to_string()),
        tool_definitions: None,
        extra: None,
    },
);
register_subscriber("atif-exporter", exporter.subscriber())?;

// Run instrumented application work here.

let trajectory = exporter.export();
let trajectory_json = serde_json::to_string_pretty(&trajectory)?;
println!("{trajectory_json}");

let _ = deregister_subscriber("atif-exporter")?;
exporter.clear();
```

::::

:::::

## Common Validation Failures

- `filename_template` does not contain `{session_id}`.
- The output directory is not writable at runtime.
- Tool definitions or `extra` metadata are not JSON-compatible.
- The application never opens a top-level `agent` scope, so no trajectory file
  is created.
