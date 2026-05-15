<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Observability Configuration

Use this page when an application should install standard observability
exporters from one plugin configuration document instead of manually registering
each subscriber.

The plugin kind is `observability`. It is registered by the core runtime, so
applications do not need to register a plugin implementation before validation
or initialization.

For plugin file discovery, precedence, merge behavior, editor controls, and
gateway conflict rules, see
[Plugin Configuration Files](../../build-plugins/plugin-configuration-files.md).

:::{note}
Observability plugin configuration uses the generic NeMo Flow plugin document
shape, so field names are `snake_case` in every binding. This differs from
Node.js runtime classes such as `OpenTelemetrySubscriber`, which use
Node-native `camelCase` option names outside the plugin system.
:::

## What It Installs

Every exporter section is optional and defaults to disabled. A section is active
only when it includes `enabled: true`.

| Section | Runtime behavior |
|---|---|
| `atof` | Registers a global ATOF JSONL exporter for raw lifecycle events. |
| `atif` | Registers one ATIF dispatcher that writes one trajectory file for each top-level agent scope. |
| `opentelemetry` | Registers a global OpenTelemetry OTLP subscriber. |
| `openinference` | Registers a global OpenInference OTLP subscriber. |

`subscriber_name` is not part of this config. The runtime infers subscriber
names from the plugin namespace:

- ATOF: `atof`
- ATIF dispatcher: `atif`
- Per-agent ATIF scope subscriber: `atif-{agent_scope_uuid}`
- OpenTelemetry: `opentelemetry`
- OpenInference: `openinference`

## `plugins.toml` Example

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

[components.config.opentelemetry]
enabled = true
transport = "http_binary"
endpoint = "http://localhost:4318/v1/traces"
service_name = "nemo-flow"
service_namespace = "agent"
service_version = "0.2.0"
instrumentation_scope = "nemo-flow-observability"
timeout_millis = 3000

[components.config.opentelemetry.headers]
authorization = "Bearer <token>"

[components.config.opentelemetry.resource_attributes]
"deployment.environment" = "dev"
"service.instance.id" = "local"

[components.config.openinference]
enabled = true
transport = "http_binary"
endpoint = "http://localhost:6006/v1/traces"
service_name = "nemo-flow"
service_namespace = "agent"
service_version = "0.2.0"
instrumentation_scope = "nemo-flow-openinference"
timeout_millis = 3000

[components.config.openinference.headers]
authorization = "Bearer <token>"

[components.config.openinference.resource_attributes]
"deployment.environment" = "dev"
"service.instance.id" = "local"

[components.config.policy]
unknown_component = "warn"
unknown_field = "warn"
unsupported_value = "error"
```

Include only the sections you want to configure. In layered `plugins.toml`
files, omission inherits lower-precedence values; write `enabled = false` to
disable an inherited section.

## Activate From Code

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
if any(diagnostic["level"] == "error" for diagnostic in report["diagnostics"]):
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
const plugin = require("nemo-flow-node/plugin");
const observability = require("nemo-flow-node/observability");

await plugin.initialize({
  version: 1,
  components: [
    observability.ComponentSpec({
      version: 1,
      atof: observability.atofConfig({
        enabled: true,
        output_directory: "logs",
        filename: "events.jsonl",
        mode: "overwrite",
      }),
      atif: observability.atifConfig({
        enabled: true,
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
use nemo_flow::observability::plugin_component::{
    AtifSectionConfig, AtofSectionConfig, ComponentSpec, ObservabilityConfig,
};
use nemo_flow::plugin::{initialize_plugins, validate_plugin_config, PluginConfig};

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

let report = validate_plugin_config(&config);
assert!(!report.has_errors());

let active = initialize_plugins(config).await?;
```

::::

:::::

## Validation And Teardown

Validate plugin configuration before activating it. The plugin reports
unsupported transports, unsupported ATOF modes, unsafe ATIF filename templates,
unknown fields according to policy, and enabled exporters that are unavailable
in the current build or target.

Call `plugin.clear()` or `clear_plugin_configuration()` during teardown.
Clearing the plugin config deregisters inferred subscribers, flushes file
exporters, and shuts down owned OTLP subscribers.

Use manual subscriber/exporter APIs instead of the plugin when you need custom
subscriber names, explicit per-run exporter objects, or direct control over the
collection window.
