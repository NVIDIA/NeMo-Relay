<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Configuration

Use this page when you want to configure the built-in Adaptive plugin component
as a whole. The component kind is `adaptive`.

Adaptive plugin configuration uses the generic NeMo Flow plugin document shape.
Field names stay `snake_case` in every binding and in `plugins.toml`, even when
language helper functions use language-native naming conventions.

For plugin file discovery, precedence, merge behavior, editor controls, and
gateway conflict rules, see
[Plugin Configuration Files](../../build-plugins/plugin-configuration-files.md).

## Component Shape

The top-level adaptive object contains:

| Field | Purpose |
|---|---|
| `version` | Adaptive config schema version. Defaults to `1`. |
| `agent_id` | Stable logical agent or workflow identifier for learned state. |
| `state` | Adaptive state backend. |
| `telemetry` | Adaptive subscriber and learner settings. |
| `adaptive_hints` | Request hint-injection behavior. |
| `tool_parallelism` | Tool scheduling observation or scheduling behavior. |
| `acg` | Adaptive Cache Governor prompt-cache planning. |
| `policy` | Adaptive-local handling for unknown fields and unsupported values. |

The requested area pages cover [Adaptive Cache Governor (ACG)](acg.md) and
[Adaptive Hints](adaptive-hints.md). State, telemetry, tool parallelism, and
policy remain whole-plugin settings:

- Use `state.backend.kind = "in_memory"` for local experiments.
- Use Redis state when learned state must survive restarts or be shared across
  workers.
- Enable `telemetry` when adaptive learners should consume runtime events.
- Keep `tool_parallelism.mode = "observe_only"` until scheduling behavior has
  been validated.
- Keep `policy.unsupported_value = "error"` for rollout safety.

## `plugins.toml` Example

```toml
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config]
version = 1
agent_id = "planner"

[components.config.state.backend]
kind = "in_memory"

[components.config.telemetry]
subscriber_name = "adaptive.telemetry"
learners = ["tool_parallelism"]

[components.config.tool_parallelism]
mode = "observe_only"
priority = 100

[components.config.adaptive_hints]
priority = 100
break_chain = false
inject_header = true
inject_body_path = "nvext.agent_hints"

[components.config.acg]
provider = "passthrough"
observation_window = 100
priority = 50

[components.config.acg.stability_thresholds]
stable_threshold = 0.95
semi_stable_threshold = 0.50
min_observations_for_full_confidence = 20

[components.config.policy]
unknown_component = "warn"
unknown_field = "warn"
unsupported_value = "error"
```

This configuration activates adaptive telemetry, keeps tool parallelism
observational, injects adaptive hints, and leaves ACG in `passthrough` mode so
requests can be observed without provider-specific cache translation.

## Per-Language Plugin Configuration

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow

adaptive_config = nemo_flow.adaptive.AdaptiveConfig(
    agent_id="planner",
    state=nemo_flow.adaptive.StateConfig(
        backend=nemo_flow.adaptive.BackendSpec.in_memory(),
    ),
    telemetry=nemo_flow.adaptive.TelemetryConfig(
        subscriber_name="adaptive.telemetry",
        learners=["tool_parallelism"],
    ),
    tool_parallelism=nemo_flow.adaptive.ToolParallelismConfig(mode="observe_only"),
    adaptive_hints=nemo_flow.adaptive.AdaptiveHintsConfig(
        inject_body_path="nvext.agent_hints",
    ),
    acg=nemo_flow.adaptive.AcgConfig(provider="passthrough"),
)

plugin_config = nemo_flow.plugin.PluginConfig(
    components=[nemo_flow.adaptive.ComponentSpec(adaptive_config)]
)

report = nemo_flow.plugin.validate(plugin_config)
if any(diagnostic["level"] == "error" for diagnostic in report["diagnostics"]):
    raise RuntimeError(report["diagnostics"])

active = await nemo_flow.plugin.initialize(plugin_config)
```
:::

:::{tab-item} Node.js
:sync: node

```js
const adaptive = require("nemo-flow-node/adaptive");
const plugin = require("nemo-flow-node/plugin");

const adaptiveConfig = adaptive.defaultConfig();
adaptiveConfig.agent_id = "planner";
adaptiveConfig.state = { backend: adaptive.inMemoryBackend() };
adaptiveConfig.telemetry = adaptive.telemetryConfig({
  subscriber_name: "adaptive.telemetry",
  learners: ["tool_parallelism"],
});
adaptiveConfig.tool_parallelism = adaptive.toolParallelismConfig({ mode: "observe_only" });
adaptiveConfig.adaptive_hints = adaptive.adaptiveHintsConfig({
  inject_body_path: "nvext.agent_hints",
});
adaptiveConfig.acg = adaptive.acgConfig({ provider: "passthrough" });

const pluginConfig = plugin.defaultConfig();
pluginConfig.components = [adaptive.ComponentSpec(adaptiveConfig)];

const report = plugin.validate(pluginConfig);
if (report.diagnostics.some((diagnostic) => diagnostic.level === "error")) {
  throw new Error(JSON.stringify(report.diagnostics));
}

const active = await plugin.initialize(pluginConfig);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::plugin::{initialize_plugins, validate_plugin_config, PluginConfig};
use nemo_flow_adaptive::{
    AdaptiveConfig,
    BackendSpec,
    ComponentSpec,
    StateConfig,
    TelemetryComponentConfig,
    ToolParallelismComponentConfig,
    AdaptiveHintsComponentConfig,
    AcgComponentConfig,
};

let mut adaptive = AdaptiveConfig::default();
adaptive.agent_id = Some("planner".into());
adaptive.state = Some(StateConfig {
    backend: BackendSpec::in_memory(),
});
adaptive.telemetry = Some(TelemetryComponentConfig {
    subscriber_name: Some("adaptive.telemetry".into()),
    learners: vec!["tool_parallelism".into()],
});
adaptive.tool_parallelism = Some(ToolParallelismComponentConfig::default());
adaptive.adaptive_hints = Some(AdaptiveHintsComponentConfig {
    inject_body_path: "nvext.agent_hints".into(),
    ..AdaptiveHintsComponentConfig::default()
});
adaptive.acg = Some(AcgComponentConfig {
    provider: "passthrough".into(),
    ..AcgComponentConfig::default()
});

let mut plugin_config = PluginConfig::default();
plugin_config.components.push(ComponentSpec::new(adaptive).into());

let report = validate_plugin_config(&plugin_config);
assert!(!report.has_errors());

let active = initialize_plugins(plugin_config).await?;
```
:::

::::

## Manual API

Use the manual runtime API when an integration needs to own adaptive lifecycle
directly instead of activating the top-level plugin component.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_flow

adaptive_config = nemo_flow.adaptive.AdaptiveConfig(
    agent_id="planner",
    state=nemo_flow.adaptive.StateConfig(
        backend=nemo_flow.adaptive.BackendSpec.in_memory(),
    ),
    telemetry=nemo_flow.adaptive.TelemetryConfig(
        subscriber_name="adaptive.telemetry",
        learners=["tool_parallelism"],
    ),
    tool_parallelism=nemo_flow.adaptive.ToolParallelismConfig(mode="observe_only"),
    adaptive_hints=nemo_flow.adaptive.AdaptiveHintsConfig(
        inject_body_path="nvext.agent_hints",
    ),
    acg=nemo_flow.adaptive.AcgConfig(provider="passthrough"),
)

runtime = nemo_flow.adaptive.AdaptiveRuntime(adaptive_config.to_dict())
await runtime.register()
try:
    # Run instrumented application work here.
    runtime.wait_for_idle()
finally:
    await runtime.shutdown()
```
:::

:::{tab-item} Node.js
:sync: node

The Node.js binding exposes the built-in adaptive runtime through the adaptive
plugin component helpers. Use the Plugin Configuration example above when
activating adaptive behavior from Node.js.
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow_adaptive::{
    AcgComponentConfig, AdaptiveConfig, AdaptiveHintsComponentConfig, AdaptiveRuntime,
    BackendSpec, StateConfig, TelemetryComponentConfig, ToolParallelismComponentConfig,
};

let mut adaptive = AdaptiveConfig::default();
adaptive.agent_id = Some("planner".into());
adaptive.state = Some(StateConfig {
    backend: BackendSpec::in_memory(),
});
adaptive.telemetry = Some(TelemetryComponentConfig {
    subscriber_name: Some("adaptive.telemetry".into()),
    learners: vec!["tool_parallelism".into()],
});
adaptive.tool_parallelism = Some(ToolParallelismComponentConfig::default());
adaptive.adaptive_hints = Some(AdaptiveHintsComponentConfig {
    inject_body_path: "nvext.agent_hints".into(),
    ..AdaptiveHintsComponentConfig::default()
});
adaptive.acg = Some(AcgComponentConfig {
    provider: "passthrough".into(),
    ..AcgComponentConfig::default()
});

let mut runtime = AdaptiveRuntime::new(adaptive).await?;
runtime.register().await?;

// Run instrumented application work here.

runtime.wait_for_idle();
runtime.shutdown().await?;
```
:::

::::

## Validation And Teardown

Validate plugin configuration before initialization. Disabled adaptive
components are still validated, which lets operators prepare a rollout before
setting `enabled = true`.

Common validation failures include:

- Unknown adaptive fields when policy treats unknown fields as errors.
- Unsupported backend kinds, tool-parallelism modes, or ACG providers.
- Unsupported schema versions.
- Backend-specific fields that do not match the selected backend.

Clear plugin configuration during shutdown or test cleanup. Clearing the plugin
configuration deregisters adaptive subscribers and intercepts owned by the
plugin runtime.

## Rollout Guidance

Start by enabling state and telemetry in a development environment. Run
representative instrumented workflows, inspect emitted events and adaptive
reports, and then enable active behavior one area at a time. Keep rollback as a
configuration change.
