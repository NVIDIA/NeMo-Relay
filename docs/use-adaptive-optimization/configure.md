<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Basic Guide: Configure Adaptive Optimization

Use this guide when you want to enable NeMo Flow adaptive behavior through the adaptive plugin component.

## What You Build

You will create an adaptive component configuration, validate it, initialize it through the plugin system, and run instrumented work while adaptive telemetry observes lifecycle events.

Start with the smallest configuration that gives you observability before you let adaptive behavior influence execution. A conservative first configuration uses:

- In-memory state.
- Telemetry enabled.
- Tool parallelism in `observe_only` mode.
- Optional adaptive hints disabled or enabled only for a controlled model path.
- Cache governor disabled until you have representative LLM traffic.

## Before You Start

Complete these steps:

1. Instrument tool or LLM calls with NeMo Flow managed helpers.
2. Register observability during development so you can inspect lifecycle events.
3. Choose a stable `agent_id` for the logical agent or workflow.
4. Decide whether adaptive state can be process-local or must be shared across workers.

Use [Advanced Guide: Tune Adaptive Behavior](advanced-guide.md) after the basic configuration is running.

## Configuration Areas

The table below summarizes the adaptive configuration areas and when to use them.

| Area | Start With | Use When |
|---|---|---|
| State | In-memory backend | You want local development or process-local experiments |
| Telemetry | Enabled | You want adaptive components to observe runtime events |
| Adaptive hints | Disabled or low-priority | Downstream model calls can safely receive hint metadata |
| Tool parallelism | `observe_only` | You want measurements before adaptive scheduling |
| Cache governor | Disabled | Enable later for stable prompt-cache planning |
| Policy | Conservative defaults | You need explicit rollout or reporting controls |

The top-level adaptive object contains `version`, `agent_id`, `state`, `telemetry`, `adaptive_hints`, `tool_parallelism`, `acg`, and `policy`. You wrap that object with `ComponentSpec(...)`, insert it into a plugin configuration, validate, and initialize through the plugin system.

:::{note}
Adaptive plugin configuration uses the generic NeMo Flow plugin document shape,
so configuration field names stay `snake_case` in every binding. Node.js helper
function names such as `toolParallelismConfig(...)` are `camelCase`, but the
config object they create uses canonical `snake_case` keys.
:::

## Create a Minimal Adaptive Component

The examples below create a minimal adaptive component configuration for each binding.

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
)

plugin_config = nemo_flow.plugin.PluginConfig(
    components=[nemo_flow.adaptive.ComponentSpec(adaptive_config)]
)

report = nemo_flow.plugin.validate(plugin_config)
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

const pluginConfig = plugin.defaultConfig();
pluginConfig.components = [adaptive.ComponentSpec(adaptiveConfig)];

const report = plugin.validate(pluginConfig);
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

let mut plugin_config = PluginConfig::default();
plugin_config.components.push(ComponentSpec::new(adaptive).into());

let report = validate_plugin_config(&plugin_config);
let active = initialize_plugins(plugin_config).await?;
```
:::

::::

## Validate the Configuration

Before you run production traffic, confirm these points:

- Validation diagnostics do not include errors.
- Initialization returns an active plugin handle or active configuration object.
- Instrumented tool or LLM calls still return the same application result.
- The adaptive telemetry subscriber appears in emitted events or logs.
- Tool parallelism stays observational when mode is `observe_only`.

If validation reports warnings, decide whether the configuration is acceptable for the environment before initialization. If validation reports errors, fix the component config before initialization.

## Roll Out Safely

Use this sequence for first deployment:

1. Enable in-memory state and telemetry in a development environment.
2. Run representative agent workflows.
3. Review emitted events and adaptive reports.
4. Enable persistent state only after the observed signals are useful.
5. Enable adaptive hints or scheduling only after downstream components can interpret them safely.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **No adaptive activity appears**: Confirm that application work emits NeMo Flow lifecycle events.
- **State resets unexpectedly**: In-memory state is process-local. Use Redis-backed state when state must survive restarts or be shared across workers.
- **Tool execution changes too early**: Keep tool parallelism in `observe_only` until you are ready for adaptive scheduling.
- **Model requests receive unexpected metadata**: Disable adaptive hints or lower their priority while debugging.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Tune advanced behavior with [Advanced Guide: Tune Adaptive Behavior](advanced-guide.md).
- Review the plugin-facing shape in [Advanced Guide: Configure Adaptive Components](adaptive-components.md).
- Use [Code Examples](code-examples.md) for binding-level helper names, ACG threshold overrides, defaults, and runtime-adjacent variables.
