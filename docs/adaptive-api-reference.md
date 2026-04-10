<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive API Reference

Adaptive exposes config helpers for one top-level plugin component. Activation,
validation, and hosted plugin registration all happen through the core plugin
host.

## Top-Level Model

- adaptive config is represented by `AdaptiveConfig`
- adaptive is activated with a top-level plugin component whose `kind` is
  `adaptive`
- hosted plugins use separate top-level plugin components with their own `kind`
- there is no public adaptive runtime object
- there is no adaptive-specific hosted-plugin registration API

## Adaptive Config Fields

- `version: u32`
  Schema version for the adaptive config document.
- `agent_id: string | null`
  Optional explicit agent identifier used by adaptive state.
- `state`
  Shared state backend configuration.
- `telemetry`
  Telemetry subscriber and learner settings.
- `adaptive_hints`
  LLM request hint injection settings.
- `tool_parallelism`
  Built-in tool scheduling settings.
- `policy`
  Unknown-field and unsupported-value handling.

## State

`state.backend` is required for adaptive features that need persistence.

Supported backend kinds:

- `in_memory`
- `redis`

## Canonical Plugin Config

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
        "tool_parallelism": {}
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

## Shared Adaptive Config Types

- `BackendSpec`
  Description: selects the adaptive state backend.
  Arguments: `kind` is the backend kind string and `config` is the backend-specific JSON object.
  Returns: a backend document embedded under `state.backend`.
  Behavior: use the helpers below instead of constructing raw backend maps when possible.

- `BackendSpec.in_memory()` / `inMemoryBackend()` / `NewInMemoryBackend()`
  Description: creates an in-memory backend spec.
  Arguments: none.
  Returns: a backend spec with `kind = "in_memory"` and an empty config object.
  Behavior: state lives in-process only and is lost when the process exits.

- `BackendSpec.redis(url, key_prefix)` / `redisBackend(url, keyPrefix)` / `NewRedisBackend(url, keyPrefix)`
  Description: creates a Redis-backed state spec.
  Arguments: `url` is the Redis connection URL and `key_prefix` / `keyPrefix` scopes adaptive keys.
  Returns: a backend spec with `kind = "redis"` and the corresponding config object.
  Behavior: use this when adaptive state must survive process restarts or be shared across workers.

- `StateConfig`
  Description: wraps the required adaptive state backend selection.
  Arguments: `backend`.
  Returns: a `state` section for `AdaptiveConfig`.
  Behavior: sections that depend on learned state are effectively disabled when `state` is omitted.

- `TelemetryConfig`
  Description: configures the built-in adaptive telemetry subscriber and learners.
  Arguments: `subscriber_name` is an optional override for the subscriber registration name and `learners` is the enabled learner set.
  Returns: a `telemetry` section for `AdaptiveConfig`.
  Behavior: telemetry only emits adaptive learning signals; hosted plugins are still configured separately as top-level plugin components.

- `AdaptiveHintsConfig`
  Description: configures built-in LLM request hint injection.
  Arguments: `priority`, `break_chain`, `inject_header`, and `inject_body_path`.
  Returns: an `adaptive_hints` section for `AdaptiveConfig`.
  Behavior: defaults are `priority = 100`, `break_chain = false`, `inject_header = true`, and `inject_body_path = "nvext.agent_hints"`.

- `ToolParallelismConfig`
  Description: configures built-in tool scheduling behavior.
  Arguments: `priority` and `mode`.
  Returns: a `tool_parallelism` section for `AdaptiveConfig`.
  Behavior: defaults are `priority = 100` and `mode = "observe_only"`. Other modes opt into progressively stronger adaptive scheduling behavior.

- `AdaptiveConfig`
  Description: the canonical config document for the top-level adaptive component.
  Arguments: `version`, `agent_id`, `state`, `telemetry`, `adaptive_hints`, `tool_parallelism`, and `policy`.
  Returns: a serializable adaptive config object.
  Behavior: this document configures only the adaptive component. It does not contain nested hosted-plugin components.

- `ComponentSpec`
  Description: wraps one `AdaptiveConfig` as a top-level plugin component.
  Arguments: `config` and `enabled`.
  Returns: a component whose `kind` is always `adaptive`.
  Behavior: this is the value placed into `PluginConfig.components` / `components`. Adaptive is always a top-level component alongside any hosted plugins.

- `set_latency_sensitivity(level)`
  Description: writes a request-local latency-sensitivity hint into the current runtime context.
  Arguments: `level` is the desired sensitivity value.
  Returns: no value.
  Behavior: this helper affects the current request/scope context only. It is an execution-time hint, not persistent configuration.

## Core Plugin Integration

Adaptive is activated and managed through the shared plugin host documented in
[API Reference](api-reference.md#plugin-host).

Relevant shared operations:

- `validate_plugin_config(...)` / `plugin.validate(...)` / `validate(...)` / `ValidatePluginConfig(...)`
  Description: validates the full plugin host configuration, including the adaptive component.
  Returns: a `ConfigReport`.

- `initialize_plugins(...)` / `plugin.initialize(...)` / `initialize(...)` / `InitializePlugins(...)`
  Description: validates and activates the full plugin host configuration.
  Returns: the successful `ConfigReport`.
  Behavior: activation is replace-with-rollback. Partial adaptive registration is undone on failure.

- `clear_plugin_configuration()` / `plugin.clear()` / `clear()` / `ClearPluginConfiguration()`
  Description: removes the active adaptive component registration together with any other active plugin components.
  Returns: no value.

## Language Entry Points

### Rust

- `nemo_flow_adaptive::AdaptiveConfig`
  Description: typed adaptive config object.
- `nemo_flow_adaptive::ComponentSpec::new(config)`
  Description: wraps an adaptive config as a top-level plugin component.
- `nemo_flow_adaptive::register_adaptive_component()`
  Description: registers the adaptive kind with the core plugin registry.
  Behavior: safe to call during startup before validating or initializing plugin configs.
- `nemo_flow_adaptive::deregister_adaptive_component()`
  Description: removes the adaptive kind from the core plugin registry.
- `nemo_flow_core::PluginConfig`
  Description: shared plugin host config that contains both adaptive and hosted plugin components.

### Python

- `nemo_flow.adaptive.AdaptiveConfig`
  Description: typed adaptive config object.
- `nemo_flow.adaptive.ComponentSpec(config, enabled=True)`
  Description: wraps adaptive config as a top-level plugin component.
- `nemo_flow.plugin.PluginConfig(components=[...])`
  Description: shared host config that mixes `adaptive.ComponentSpec(...)` with `plugin.ComponentSpec(...)`.
- `nemo_flow.plugin.validate(config)`
  Description: validates the full config and returns a `ConfigReport`.
- `await nemo_flow.plugin.initialize(config)`
  Description: activates the full config and returns the applied `ConfigReport`.

### Node.js and WASM

- `adaptive.defaultConfig()`
  Description: creates a default adaptive config object with `version = 1`.
- `adaptive.ComponentSpec(config, { enabled })`
  Description: wraps adaptive config as a top-level component whose kind is fixed to `adaptive`.
- `plugin.defaultConfig()`
  Description: creates a default plugin host config with `version = 1` and no components.
- `plugin.validate(config)`
  Description: validates the full config and returns a `ConfigReport`.
- `await plugin.initialize(config)`
  Description: activates the full config and resolves to the applied `ConfigReport`.

### Go

- `adaptive.NewConfig()`
  Description: creates a default adaptive config value.
- `adaptive.NewComponentSpec(config)`
  Description: creates an adaptive-owned component wrapper.
- `(adaptive.ComponentSpec).PluginComponent()`
  Description: converts the adaptive component wrapper into the shared `PluginComponentSpec` type used by `nemo_flow.PluginConfig`.
- `nemo_flow.ValidatePluginConfig(config)`
  Description: validates the full config and returns `(ConfigReport, error)`.
- `nemo_flow.InitializePlugins(config)`
  Description: activates the full config and returns `(ConfigReport, error)`.

## Hosted Plugins

Hosted plugins are always top-level plugin components.

Registration pattern:

1. Register a handler through the core plugin API.
2. Add a top-level plugin component with that plugin kind.
3. Configure adaptive separately with its own top-level `adaptive` component.

Registration context surface:

- `registerSubscriber(...)`
- `registerLlmRequestIntercept(...)`
- `registerLlmExecutionIntercept(...)`
- `registerLlmStreamExecutionIntercept(...)`
- `registerToolRequestIntercept(...)`
- `registerToolExecutionIntercept(...)`

Registration names are local to each component. The runtime namespaces them
internally, so users do not need to provide component instance ids.

## Example

```python
from nemo_flow import adaptive, plugin

class HeaderPlugin:
    def validate(self, plugin_config):
        return []

    def register(self, plugin_config, context):
        context.register_tool_request_intercept(
            "tool",
            25,
            False,
            lambda name, args: {**args, "x_plugin": "enabled", "tool": name},
        )

plugin.register("example.header_plugin", HeaderPlugin())

config = plugin.PluginConfig(
    components=[
        adaptive.ComponentSpec(
            adaptive.AdaptiveConfig(
                state=adaptive.StateConfig(
                    backend=adaptive.BackendSpec.in_memory()
                ),
                adaptive_hints=adaptive.AdaptiveHintsConfig(),
            )
        ),
        plugin.ComponentSpec(
            kind="example.header_plugin",
            config={"priority": 25},
        ),
    ]
)

await plugin.initialize(config)
```
