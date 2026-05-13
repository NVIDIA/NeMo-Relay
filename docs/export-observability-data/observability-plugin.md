<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Basic Guide: Configure the Observability Plugin

Use the built-in Observability plugin when an application should install
standard exporters from one plugin configuration document instead of manually
registering each subscriber.

The plugin kind is `observability`. It is registered by the core runtime, so
applications do not need to register a plugin implementation before validation
or initialization.

## What It Installs

The component accepts four optional sections:

| Section | Runtime behavior |
|---|---|
| `atof` | Registers a global ATOF JSONL exporter for raw lifecycle events. |
| `atif` | Registers one ATIF dispatcher that writes one trajectory file for each top-level agent scope. |
| `opentelemetry` | Registers a global OpenTelemetry OTLP subscriber. |
| `openinference` | Registers a global OpenInference OTLP subscriber. |

Every section defaults to disabled. A section is active only when it includes
`enabled: true`.

## Top-Level Shape

The generic plugin config wraps the observability component:

```json
{
  "version": 1,
  "components": [
    {
      "kind": "observability",
      "enabled": true,
      "config": {
        "version": 1,
        "atof": { "enabled": true, "filename": "events.jsonl" }
      }
    }
  ]
}
```

`subscriber_name` is not part of this config. The runtime infers subscriber
names from the plugin namespace:

- ATOF: `atof`
- ATIF dispatcher: `atif`
- Per-agent ATIF scope subscriber: `atif-{agent_scope_uuid}`
- OpenTelemetry: `opentelemetry`
- OpenInference: `openinference`

The active runtime names include the component namespace prefix used by the
plugin system.

## CLI Gateway `plugin.toml`

The `nemo-flow` CLI gateway can activate one process-level plugin config at
startup. Define it with one of these sources:

- `--plugin-config` JSON on the command line.
- `[plugins].config` in `config.toml`.
- `plugin.toml` next to the resolved `config.toml`, or in the same discovered
  system, project, and user scopes as `config.toml`.

When multiple discovered `plugin.toml` files are present, the gateway loads
them from lowest to highest precedence:

1. System: `/etc/nemo-flow/plugin.toml`
2. Project: `.nemo-flow/plugin.toml`
3. User: `$XDG_CONFIG_HOME/nemo-flow/plugin.toml`, or
   `~/.config/nemo-flow/plugin.toml`

Later files override earlier files. TOML tables merge recursively, so a
higher-precedence file can override one nested key while preserving sibling
keys from lower-precedence files.

`[[components]]` is different because TOML represents it as an array. Arrays
are replaced rather than appended. If a project `plugin.toml` defines an
observability component and a user `plugin.toml` defines an adaptive component,
the effective config contains only the user adaptive component. Put the full
component list in the highest-precedence `plugin.toml` that should control the
run.

Use only one source for plugin config. The gateway reports an error when
`plugin.toml`, `[plugins].config`, or `--plugin-config` are used together.

`plugin.toml` uses the generic plugin config shape at the file root:

`version = 1` is recommended for clarity but not required. The root plugin
config version and observability component config version both default to `1`
when omitted; unsupported non-`1` versions fail validation by default.

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "logs"
filename = "events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
output_directory = "logs"
filename_template = "trajectory-{session_id}.json"
```

The file format is generic. Other plugin kinds can use the same `components`
array when their plugin implementation is registered in the gateway process.

## ATOF Section

Use ATOF when you want the raw ATOF `0.1` event stream as JSONL.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to write events. |
| `output_directory` | Current working directory | Directory containing the JSONL file. |
| `filename` | Timestamped `nemo-flow-events-*.jsonl` | Explicit output filename. |
| `mode` | `append` | `append` or `overwrite`. |

## ATIF Section

Use ATIF when you want one trajectory artifact per top-level agent run.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to write trajectories. |
| `agent_name` | `NeMo Flow` | Agent metadata written into the trajectory. |
| `agent_version` | NeMo Flow crate version | Agent version metadata. |
| `model_name` | `unknown` | Default model metadata when no call-level model is present. |
| `tool_definitions` | Omitted | Optional ATIF tool metadata. |
| `extra` | Omitted | Optional ATIF agent metadata. |
| `output_directory` | Current working directory | Directory containing trajectory files. |
| `filename_template` | `nemo-flow-atif-{session_id}.json` | Must contain `{session_id}`. |

A top-level agent is a scope start event with category `agent` whose parent is
the implicit root scope. The ATIF plugin creates a separate exporter for each
direct child agent scope, records that start event, attaches a scope-local
subscriber to the agent scope, and writes the file when the agent scope ends.
If the plugin is cleared while an agent is still open, teardown flushes the
partial trajectory.

Nested agent scopes under a top-level agent remain in the parent trajectory.
Direct child scopes that are not `agent` scopes do not create ATIF files.

## OpenTelemetry and OpenInference Sections

OpenTelemetry and OpenInference use the same section shape:

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Must be `true` to construct and register the subscriber. |
| `transport` | `http_binary` | `http_binary` or `grpc`. |
| `endpoint` | Exporter default | OTLP endpoint. |
| `headers` | `{}` | String-to-string exporter headers. |
| `resource_attributes` | `{}` | String-to-string OTLP resource attributes. |
| `service_name` | `nemo-flow` | `service.name` resource attribute. |
| `service_namespace` | Omitted | Optional `service.namespace`. |
| `service_version` | Omitted | Optional `service.version`. |
| `instrumentation_scope` | Omitted | Optional instrumentation scope name. |
| `timeout_millis` | `3000` | Export timeout. |

Disabled OTLP sections do not construct exporters and do not contact endpoints.

## Configure

:::::{tab-set}
:sync-group: language

::::{tab-item} Python
:sync: python

```python
from nemo_flow import plugin, scope, ScopeType
from nemo_flow.observability import (
    AtifConfig,
    AtofConfig,
    ComponentSpec,
    ObservabilityConfig,
)

config = plugin.PluginConfig(
    components=[
        ComponentSpec(
            ObservabilityConfig(
                atof=AtofConfig(
                    enabled=True,
                    output_directory="logs",
                    filename="events.jsonl",
                    mode="overwrite",
                ),
                atif=AtifConfig(
                    enabled=True,
                    output_directory="logs",
                    filename_template="trajectory-{session_id}.json",
                ),
            )
        )
    ]
)

report = plugin.validate(config)
if report["diagnostics"]:
    raise RuntimeError(report["diagnostics"])

await plugin.initialize(config)
try:
    with scope.scope("agent", ScopeType.Agent):
        pass
finally:
    plugin.clear()
```

::::

::::{tab-item} Node.js
:sync: node

```js
const plugin = require('nemo-flow-node/plugin');
const observability = require('nemo-flow-node/observability');

await plugin.initialize({
  version: 1,
  components: [
    observability.ComponentSpec({
      version: 1,
      atof: observability.atofConfig({
        enabled: true,
        output_directory: 'logs',
        filename: 'events.jsonl',
        mode: 'overwrite',
      }),
      atif: observability.atifConfig({
        enabled: true,
        output_directory: 'logs',
        filename_template: 'trajectory-{session_id}.json',
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
use nemo_flow::observability::plugin_component::{
    AtifSectionConfig, AtofSectionConfig, ComponentSpec, ObservabilityConfig,
};
use nemo_flow::plugin::{PluginConfig, initialize_plugins};

let component = ComponentSpec::new(ObservabilityConfig {
    atof: Some(AtofSectionConfig {
        enabled: true,
        output_directory: Some("logs".into()),
        filename: Some("events.jsonl".into()),
        mode: "overwrite".into(),
    }),
    atif: Some(AtifSectionConfig {
        enabled: true,
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

let report = initialize_plugins(config).await?;
assert!(!report.has_errors());
```

::::

:::::

## Validation and Teardown

Validate plugin configuration before activating it. The plugin reports
unsupported transports, unsupported ATOF modes, unsafe ATIF filename templates,
unknown fields according to policy, and enabled exporters that are unavailable
in the current build or target.

Call `plugin.clear()` or `clear_plugin_configuration()` during teardown. Clearing
the plugin config deregisters inferred subscribers, flushes file exporters, and
shuts down owned OTLP subscribers.
