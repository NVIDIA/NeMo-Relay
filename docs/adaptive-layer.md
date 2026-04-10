<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Layer

Adaptive is a top-level plugin component configured through the core plugin
host. There is no public adaptive runtime object, and adaptive config does not
own a nested `components` list.

The model is intentionally simple:

- `PluginConfig.components` holds all active top-level components.
- `adaptive` is one of those top-level components.
- adaptive built-ins are flat config sections: `telemetry`, `adaptive_hints`,
  and `tool_parallelism`.
- custom plugins are separate top-level plugin components, not children of the
  adaptive config.

## Canonical Shape

```json
{
  "version": 1,
  "components": [
    {
      "kind": "adaptive",
      "enabled": true,
      "config": {
        "version": 1,
        "state": {
          "backend": {
            "kind": "in_memory",
            "config": {}
          }
        },
        "telemetry": {
          "learners": ["latency_sensitivity"]
        },
        "adaptive_hints": {},
        "tool_parallelism": {},
        "policy": {
          "unknown_component": "warn",
          "unknown_field": "warn",
          "unsupported_value": "error"
        }
      }
    },
    {
      "kind": "example.header_plugin",
      "enabled": true,
      "config": {
        "priority": 25
      }
    }
  ]
}
```

## Adaptive Sections

- `state`
  Shared backend used by adaptive features that persist or learn from runs.
- `telemetry`
  Registers the adaptive telemetry subscriber and learner pipeline.
- `adaptive_hints`
  Registers the request intercept that injects learned hints.
- `tool_parallelism`
  Registers the tool execution intercept for built-in scheduling behavior.
- `policy`
  Controls diagnostics behavior for unknown fields and unsupported values.

The adaptive component is flat by design. If you want to activate another
plugin, add another top-level `PluginComponentSpec`.

## Plugins

Custom plugins are managed by the shared core plugin host:

- register a plugin kind with the host
- activate it with another top-level component in `PluginConfig.components`
- configure adaptive separately with its own `adaptive` component

See [Plugins](hosted-plugins.md) for the focused guide on configuring plugins,
writing plugin handlers, registration context behavior, and rollback semantics.

## Examples

### Python

```python
from nemo_flow import adaptive, plugin

config = plugin.PluginConfig(
    components=[
        adaptive.ComponentSpec(
            adaptive.AdaptiveConfig(
                state=adaptive.StateConfig(
                    backend=adaptive.BackendSpec.in_memory()
                ),
                telemetry=adaptive.TelemetryConfig(
                    learners=["latency_sensitivity"]
                ),
                adaptive_hints=adaptive.AdaptiveHintsConfig(),
                tool_parallelism=adaptive.ToolParallelismConfig(),
            )
        )
    ]
)

report = plugin.validate(config)
configured = await plugin.initialize(config)
```

### Node.js

```javascript
const adaptive = require("./adaptive.js");
const plugin = require("./plugin.js");

const config = plugin.defaultConfig();
config.components = [
  adaptive.ComponentSpec({
    version: 1,
    state: { backend: adaptive.inMemoryBackend() },
    telemetry: adaptive.telemetryConfig({
      learners: ["latency_sensitivity"],
    }),
    adaptive_hints: adaptive.adaptiveHintsConfig(),
    tool_parallelism: adaptive.toolParallelismConfig(),
  }),
];

const report = plugin.validate(config);
await plugin.initialize(config);
```

### Go

```go
import (
    nemo_flow "github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

config := nemo_flow.NewPluginConfig()
adaptiveConfig := nemo_flow.NewAdaptiveConfig()
adaptiveConfig.State = &nemo_flow.AdaptiveStateConfig{
    Backend: nemo_flow.NewInMemoryAdaptiveBackend(),
}
telemetry := nemo_flow.NewTelemetryConfig()
telemetry.Learners = []string{"latency_sensitivity"}
adaptiveConfig.Telemetry = &telemetry
adaptiveConfig.AdaptiveHints = &nemo_flow.AdaptiveHintsConfig{
    Priority:       100,
    InjectHeader:   true,
    InjectBodyPath: "nvext.agent_hints",
}

config.Components = append(
    config.Components,
    nemo_flow.NewAdaptiveComponentSpec(adaptiveConfig).PluginComponent(),
)

report, err := nemo_flow.ValidatePluginConfig(config)
if err != nil {
    panic(err)
}
_ = report

_, err = nemo_flow.InitializePlugins(config)
if err != nil {
    panic(err)
}
```

### Rust

```rust
use nemo_flow_adaptive::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, BackendSpec, ComponentSpec, StateConfig,
    TelemetryComponentConfig, ToolParallelismComponentConfig,
};
use nemo_flow::{initialize_plugins, PluginConfig};

let config = PluginConfig {
    components: vec![ComponentSpec::new(AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        telemetry: Some(TelemetryComponentConfig {
            subscriber_name: None,
            learners: vec!["latency_sensitivity".into()],
        }),
        adaptive_hints: Some(AdaptiveHintsComponentConfig::default()),
        tool_parallelism: Some(ToolParallelismComponentConfig::default()),
        ..AdaptiveConfig::default()
    })
    .into()],
    ..PluginConfig::default()
};

let report = initialize_plugins(config).await?;
```

## Validation

Validate through the core plugin host, not through an adaptive-specific runtime:

- Rust: `validate_plugin_config(&config)`
- Python: `nemo_flow.plugin.validate(config)`
- Node.js: `plugin.validate(config)`
- Go: `nemo_flow.ValidatePluginConfig(config)`
- WASM: `plugin.validate(config)`

## Related Docs

- [Adaptive API Reference](adaptive-api-reference.md)
- [Plugins](hosted-plugins.md)
- [Online Learning Engine](online-learning-engine.md)
- [Language Bindings](language-bindings.md)
