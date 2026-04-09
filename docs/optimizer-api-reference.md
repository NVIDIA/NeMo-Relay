<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optimizer API Reference

The optimizer surface is a config document plus a runtime handle. Every
supported language maps back to the same dynamic config model, even when that
language also exposes typed helper constructors.

## Canonical Config Shape

All bindings mirror the same logical document:

```json
{
  "version": 1,
  "state": {
    "backend": {
      "kind": "in_memory",
      "config": {}
    }
  },
  "components": [
    { "kind": "telemetry", "enabled": true, "config": { "learners": ["latency_sensitivity"] } },
    { "kind": "dynamo_hints", "enabled": true, "config": {} },
    { "kind": "tool_parallelism", "enabled": true, "config": {} },
    {
      "kind": "external_component",
      "enabled": true,
      "config": {
        "plugin_kind": "example.header_plugin",
        "instance_id": "plugin-1",
        "plugin_config": { "priority": 17 }
      }
    }
  ],
  "policy": {
    "unknown_component": "warn",
    "unknown_field": "warn",
    "unsupported_value": "error"
  }
}
```

## Shared Hosted Plugin Context Surface

Hosted plugin contexts are intentionally narrow. Across all bindings they allow:

- subscriber registration
- LLM request intercept registration
- LLM execution intercept registration
- LLM stream execution intercept registration
- tool request intercept registration
- tool execution intercept registration

All hosted plugins are activated through `external_component`.

## Python

Primary exports from `nemo_flow.optimizer`:

- `OptimizerConfig`
- `StateConfig`
- `BackendSpec`
- `ComponentSpec`
- `ConfigPolicy`
- `TelemetryComponent`
- `DynamoHintsComponent`
- `ToolParallelismComponent`
- `ExternalComponent`
- `OptimizerRuntime`
- `validate_optimizer_config`
- `register_optimizer_plugin`
- `deregister_optimizer_plugin`
- `set_latency_sensitivity`

Runtime lifecycle:

```python
from nemo_flow.optimizer import (
    BackendSpec,
    OptimizerConfig,
    OptimizerRuntime,
    StateConfig,
    TelemetryComponent,
)

runtime = OptimizerRuntime(
    OptimizerConfig(
        state=StateConfig(backend=BackendSpec.in_memory()),
        components=[TelemetryComponent(learners=["latency_sensitivity"])],
    )
)

report = runtime.report()
await runtime.register()
runtime.deregister()
await runtime.shutdown()
```

Hosted plugin entry points:

- `register_optimizer_plugin(plugin_kind, handler)`
- `deregister_optimizer_plugin(plugin_kind)`
- optional `handler.validate(instance_id, plugin_config)`
- required `handler.register(instance_id, plugin_config, context)`

## Node.js

Primary optimizer exports are available from `optimizer.js`:

- `Runtime`
- `validateConfig`
- `defaultConfig`
- `inMemoryBackend`
- `redisBackend`
- `telemetryComponent`
- `dynamoHintsComponent`
- `toolParallelismComponent`
- `externalComponent`
- `registerPlugin`
- `deregisterPlugin`

Runtime lifecycle:

```javascript
import {
  Runtime,
  defaultConfig,
  inMemoryBackend,
  telemetryComponent,
  validateConfig,
} from "./optimizer.js";

const config = defaultConfig();
config.state = { backend: inMemoryBackend() };
config.components = [telemetryComponent({ learners: ["latency_sensitivity"] })];

const validation = validateConfig(config);
const runtime = new Runtime(config);
const report = await runtime.report();
await runtime.register();
await runtime.deregister();
await runtime.shutdown();
```

Hosted plugin entry points:

- `registerPlugin(pluginKind, handler)`
- `deregisterPlugin(pluginKind)`
- optional `handler.validate(instanceId, pluginConfig)`
- required `handler.register(instanceId, pluginConfig, context)`

## Go

Primary exports from `go/nemo_flow/optimizer`:

- `Config`
- `StateConfig`
- `BackendSpec`
- `ComponentSpec`
- `ConfigPolicy`
- `ConfigReport`
- `ConfigDiagnostic`
- `Runtime`
- `NewConfig`
- `NewInMemoryBackend`
- `NewRedisBackend`
- `TelemetryComponent`
- `DynamoHintsComponent`
- `ToolParallelismComponent`
- `ExternalComponent`
- `ValidateConfig`
- `RegisterPlugin`
- `DeregisterPlugin`

Runtime lifecycle:

```go
import optimizer "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/optimizer"

config := optimizer.NewConfig()
config.State = &optimizer.StateConfig{
    Backend: optimizer.NewInMemoryBackend(),
}
config.Components = []optimizer.ComponentSpec{
    optimizer.TelemetryComponent(optimizer.TelemetryComponentConfig{
        Learners: []string{"latency_sensitivity"},
    }),
}

report, err := optimizer.ValidateConfig(config)
if err != nil {
    panic(err)
}
_ = report

runtime, err := optimizer.NewRuntime(config)
if err != nil {
    panic(err)
}
defer runtime.Close()

if err := runtime.Register(); err != nil {
    panic(err)
}
if err := runtime.Deregister(); err != nil {
    panic(err)
}
if err := runtime.Shutdown(); err != nil {
    panic(err)
}
```

Hosted plugin entry points:

- `RegisterPlugin(pluginKind, handler)`
- `DeregisterPlugin(pluginKind)`
- optional `handler.Validate(instanceID, pluginConfig)`
- required `handler.Register(instanceID, pluginConfig, ctx)`

## WebAssembly

Primary optimizer exports are available from `optimizer.js`:

- `Runtime`
- `validateConfig`
- `defaultConfig`
- `inMemoryBackend`
- `redisBackend`
- `telemetryComponent`
- `dynamoHintsComponent`
- `toolParallelismComponent`
- `externalComponent`
- `registerPlugin`
- `deregisterPlugin`

The generated WASM package still provides the low-level runtime and registration
hooks consumed by `optimizer.js`.

Runtime lifecycle:

```javascript
import init from "./pkg/nemo_flow_wasm.js";
import {
  Runtime,
  validateConfig,
} from "./optimizer.js";

await init();

const config = {
  version: 1,
  state: {
    backend: { kind: "in_memory", config: {} },
  },
  components: [
    {
      kind: "telemetry",
      enabled: true,
      config: { learners: ["latency_sensitivity"] },
    },
  ],
};

const validation = validateConfig(config);
const runtime = new Runtime(config);
runtime.report();
await runtime.register();
runtime.deregister();
await runtime.shutdown();
```

Hosted plugin entry points:

- `registerPlugin(pluginKind, handler)`
- `deregisterPlugin(pluginKind)`
- optional `handler.validate(instanceId, pluginConfig)`
- required `handler.register(instanceId, pluginConfig, context)`

## Rust

Primary exports from `nemo-flow-optimizer`:

- `OptimizerConfig`
- `StateConfig`
- `BackendSpec`
- `ComponentSpec`
- `ConfigPolicy`
- `ConfigReport`
- `ConfigDiagnostic`
- `OptimizerRuntime`
- `TelemetryComponentConfig`
- `DynamoHintsComponentConfig`
- `ToolParallelismComponentConfig`
- `ExternalComponentConfig`
- `OptimizerComponentFactory`
- `OptimizerComponent`
- `HostedPluginHandler`
- `HostedRegistrationContext`
- `register_component_factory`
- `deregister_component_factory`
- `register_hosted_plugin_handler`
- `deregister_hosted_plugin_handler`

Runtime lifecycle:

```rust
use nemo_flow_optimizer::{
    BackendSpec, OptimizerConfig, OptimizerRuntime, StateConfig, TelemetryComponentConfig,
};

let report = OptimizerRuntime::validate_config(&OptimizerConfig::default());

let mut runtime = OptimizerRuntime::new(OptimizerConfig {
    state: Some(StateConfig {
        backend: BackendSpec::in_memory(),
    }),
    components: vec![
        TelemetryComponentConfig {
            subscriber_name: None,
            learners: vec!["latency_sensitivity".into()],
        }
        .into(),
    ],
    ..OptimizerConfig::default()
})
.await?;

runtime.register().await?;
runtime.deregister()?;
runtime.shutdown().await?;
```

Hosted plugin entry points:

- `register_hosted_plugin_handler(handler)`
- `deregister_hosted_plugin_handler(plugin_kind)`
- optional `HostedPluginHandler::validate(...)`
- required `HostedPluginHandler::register(...)`

## Compatibility Boundary

The dynamic config document is the compatibility boundary. New built-in
features should extend the `components` list rather than adding new top-level
optimizer lifecycle functions.
