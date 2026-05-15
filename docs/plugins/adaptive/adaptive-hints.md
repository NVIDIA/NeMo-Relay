<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Hints

Use Adaptive Hints when downstream model calls or provider adapters can safely
receive guidance metadata from the adaptive runtime.

Adaptive hints register as LLM request intercepts. Lower numeric priority values
run earlier in the intercept chain. The default priority is chosen relative to
other middleware rather than as a standalone importance score.

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

[components.config.adaptive_hints]
priority = 100
break_chain = false
inject_header = true
inject_body_path = "nvext.agent_hints"
```

This configuration injects adaptive guidance into outgoing model requests while
allowing later request intercepts to continue running.

## Plugin Configuration

Use plugin configuration when the application should let NeMo Flow own the
Adaptive Hints request-intercept lifecycle.

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
    telemetry=nemo_flow.adaptive.TelemetryConfig(learners=["tool_parallelism"]),
    adaptive_hints=nemo_flow.adaptive.AdaptiveHintsConfig(
        inject_body_path="nvext.agent_hints",
    ),
)

plugin_config = nemo_flow.plugin.PluginConfig(
    components=[nemo_flow.adaptive.ComponentSpec(adaptive_config)]
)

report = nemo_flow.plugin.validate(plugin_config)
if any(diagnostic["level"] == "error" for diagnostic in report["diagnostics"]):
    raise RuntimeError(report["diagnostics"])

await nemo_flow.plugin.initialize(plugin_config)
try:
    # Run instrumented application work here.
    pass
finally:
    nemo_flow.plugin.clear()
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
adaptiveConfig.telemetry = adaptive.telemetryConfig({ learners: ["tool_parallelism"] });
adaptiveConfig.adaptive_hints = adaptive.adaptiveHintsConfig({
  inject_body_path: "nvext.agent_hints",
});

const pluginConfig = plugin.defaultConfig();
pluginConfig.components = [adaptive.ComponentSpec(adaptiveConfig)];

const report = plugin.validate(pluginConfig);
if (report.diagnostics.some((diagnostic) => diagnostic.level === "error")) {
  throw new Error(JSON.stringify(report.diagnostics));
}

await plugin.initialize(pluginConfig);
try {
  // Run instrumented application work here.
} finally {
  plugin.clear();
}
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::plugin::{initialize_plugins, validate_plugin_config, PluginConfig};
use nemo_flow_adaptive::plugin_component::ComponentSpec;
use nemo_flow_adaptive::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, BackendSpec, StateConfig, TelemetryComponentConfig,
};

let mut adaptive = AdaptiveConfig::default();
adaptive.agent_id = Some("planner".into());
adaptive.state = Some(StateConfig {
    backend: BackendSpec::in_memory(),
});
adaptive.telemetry = Some(TelemetryComponentConfig {
    learners: vec!["tool_parallelism".into()],
    ..TelemetryComponentConfig::default()
});
adaptive.adaptive_hints = Some(AdaptiveHintsComponentConfig {
    inject_body_path: "nvext.agent_hints".into(),
    ..AdaptiveHintsComponentConfig::default()
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
    telemetry=nemo_flow.adaptive.TelemetryConfig(learners=["tool_parallelism"]),
    adaptive_hints=nemo_flow.adaptive.AdaptiveHintsConfig(
        inject_body_path="nvext.agent_hints",
    ),
)

runtime = nemo_flow.adaptive.AdaptiveRuntime(adaptive_config.to_dict())
await runtime.register()
try:
    # Run instrumented application work here.
    nemo_flow.adaptive.set_latency_sensitivity(8)
finally:
    await runtime.shutdown()
```
:::

:::{tab-item} Node.js
:sync: node

The Node.js binding exposes Adaptive Hints through the adaptive plugin component
helpers. Use the Plugin Configuration example above when activating Adaptive
Hints from Node.js.
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow_adaptive::{
    set_latency_sensitivity, AdaptiveConfig, AdaptiveHintsComponentConfig, AdaptiveRuntime,
    BackendSpec, StateConfig, TelemetryComponentConfig,
};

let mut adaptive = AdaptiveConfig::default();
adaptive.agent_id = Some("planner".into());
adaptive.state = Some(StateConfig {
    backend: BackendSpec::in_memory(),
});
adaptive.telemetry = Some(TelemetryComponentConfig {
    learners: vec!["tool_parallelism".into()],
    ..TelemetryComponentConfig::default()
});
adaptive.adaptive_hints = Some(AdaptiveHintsComponentConfig {
    inject_body_path: "nvext.agent_hints".into(),
    ..AdaptiveHintsComponentConfig::default()
});

let mut runtime = AdaptiveRuntime::new(adaptive).await?;
runtime.register().await?;

// Run instrumented application work here.
set_latency_sensitivity(8).ok();

runtime.shutdown().await?;
```
:::

::::

## Fields

| Field | Default | Notes |
|---|---|---|
| `priority` | `100` | Request intercept priority. Lower values run earlier. |
| `break_chain` | `false` | Whether this intercept stops later request intercepts. |
| `inject_header` | `true` | Whether to add adaptive hints as request header metadata. |
| `inject_body_path` | `nvext.agent_hints` | JSON body path for request-body hint injection. |

Disable `break_chain` unless the adaptive hint should be the final request
transform. Adjust `priority` only when adaptive hints need to run before or
after known application middleware.

## Expected Output

Outgoing managed LLM requests receive adaptive hint metadata in the configured
header and body location. The hints do not replace the application callback or
change the returned value by themselves. Downstream code must explicitly
interpret the metadata before behavior changes.

## Common Validation Failures

- Unknown adaptive hint fields when unknown fields are treated as errors.
- `inject_body_path` does not match the request shape expected by downstream
  provider adapters.
- Hint injection is enabled before downstream model paths can consume or ignore
  the metadata safely.
