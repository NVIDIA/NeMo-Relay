<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Guide: Configure Adaptive Components

Use this guide when you are configuring the built-in adaptive component that enables NeMo Flow adaptive behavior across frameworks and bindings.

## What You Build

You will configure the built-in adaptive component, validate it through the plugin system, initialize it, and understand which runtime surfaces adaptive behavior can register.

## Before You Start

You need:

- Instrumented tool or LLM calls that emit lifecycle events.
- A plugin configuration path for the target binding.
- A stable logical agent ID.
- A decision about whether adaptive state is process-local or shared.

## What Adaptive Covers

The adaptive component is a plugin that enables built-in optimization and learning behaviors across frameworks and bindings.

Its top-level config is a single component with kind `adaptive`.

## What It Registers

Adaptive is not one middleware hook. It is a bundle of runtime behavior that can register:

- Subscribers for adaptive telemetry
- LLM request intercepts for adaptive hint injection
- Tool-related behaviors for parallelism guidance or scheduling
- LLM execution intercepts that plan prompt-cache breakpoints via the adaptive cache governor (ACG)
- State backends used by those features

## Main Config Areas

These areas organize the adaptive configuration fields by runtime responsibility.

### State

Select where adaptive state lives:

- In-memory backend
- Redis backend

### Telemetry

Configure the built-in adaptive subscriber and the enabled learners.

### Adaptive Hints

Configure the request-intercept behavior that injects adaptive hints:

- `priority`
- `break_chain`
- `inject_header`
- `inject_body_path`

Adaptive hints register as request intercepts. Lower numeric priority values run
earlier in the intercept chain, so the default priority is chosen relative to
other middleware rather than as a standalone importance score.

### Tool Parallelism

Configure the tool scheduling behavior:

- `priority`
- `mode`

Tool-parallelism priority follows the same middleware ordering rule: lower
numbers run earlier, higher numbers leave earlier intercepts and guardrails in
front of adaptive scheduling behavior.

Supported modes include:

- `observe_only`
- `inject_hints`
- `schedule`

### Cache Governor (ACG)

Configure the adaptive cache governor, the internal subsystem that decomposes
LLM requests into an addressable Prompt IR, scores block stability across
observed runs, and plans provider-specific prompt-cache breakpoints:

- `provider`
- `observation_window`
- `priority`
- `stability_thresholds`

Supported providers:

- `passthrough`
- `anthropic`
- `openai`

ACG is an optional section on the adaptive config; omit it to keep cache
planning disabled.

## End-To-End Example

The examples below show a complete adaptive component configuration in each binding
style.

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
    adaptive_hints=nemo_flow.adaptive.AdaptiveHintsConfig(),
    tool_parallelism=nemo_flow.adaptive.ToolParallelismConfig(mode="observe_only"),
    acg=nemo_flow.adaptive.AcgConfig(provider="anthropic"),
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

```ts
import * as adaptive from 'nemo-flow-node/adaptive';
import * as plugin from 'nemo-flow-node/plugin';

const adaptiveConfig = adaptive.defaultConfig();
adaptiveConfig.agent_id = 'planner';
adaptiveConfig.state = { backend: adaptive.inMemoryBackend() };
adaptiveConfig.telemetry = adaptive.telemetryConfig({
  subscriber_name: 'adaptive.telemetry',
  learners: ['tool_parallelism'],
});
adaptiveConfig.adaptive_hints = adaptive.adaptiveHintsConfig();
adaptiveConfig.tool_parallelism = adaptive.toolParallelismConfig({ mode: 'observe_only' });
adaptiveConfig.acg = adaptive.acgConfig({ provider: 'anthropic' });

const pluginConfig = plugin.defaultConfig();
pluginConfig.components = [adaptive.ComponentSpec(adaptiveConfig)];

const report = plugin.validate(pluginConfig);
const active = await plugin.initialize(pluginConfig);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow::plugin::{PluginConfig, initialize_plugins, validate_plugin_config};
use nemo_flow_adaptive::{
    AcgComponentConfig,
    AdaptiveConfig,
    AdaptiveHintsComponentConfig,
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
adaptive.adaptive_hints = Some(AdaptiveHintsComponentConfig::default());
adaptive.tool_parallelism = Some(ToolParallelismComponentConfig::default());
adaptive.acg = Some(AcgComponentConfig {
    provider: "anthropic".into(),
    ..AcgComponentConfig::default()
});

let mut plugin_config = PluginConfig::default();
plugin_config.components.push(ComponentSpec::new(adaptive).into());

let report = validate_plugin_config(&plugin_config);
let active = initialize_plugins(plugin_config).await?;
```
:::
::::

## Field-Level Guidance

These notes describe how to set individual adaptive configuration fields.

### `agent_id`

Use `agent_id` when one adaptive configuration should consistently identify a logical agent across requests.

### `state`

Choose `in_memory` for local development and process-local experiments. Choose `redis` when adaptive state must survive restarts or be shared across workers.

### `telemetry`

Set `subscriber_name` when you want the adaptive subscriber to appear under a predictable runtime name. Use `learners` to control which internal learners consume the observed event stream.

### `adaptive_hints`

Use this when you want the runtime to add hint metadata into outgoing model
requests. The most important controls are priority, whether the request-intercept
chain should stop, and where the injected data should live. The default
`adaptive_hints.priority`, `100`, means the hint injector runs after intercepts
with lower numeric values and before intercepts with higher numeric values;
adjust it only when you need adaptive hints to run before or after known
application middleware.

### `tool_parallelism`

Start with `observe_only`. Move to `inject_hints` when downstream code can interpret guidance safely. Use `schedule` only when you explicitly want adaptive behavior to influence execution strategy.

### `acg`

Enable the cache governor when you want adaptive to plan prompt-cache
breakpoints for outgoing LLM requests. Set `provider` to match the backend
API surface the agent actually calls (`anthropic`, `openai`, or
`passthrough` to disable the planner while keeping observations). Tune
`observation_window` to control how many recent PromptIR samples feed
stability analysis, and use `stability_thresholds` to adjust when a block
is classified as stable enough to cache.

## Why It Matters for Plugin Authors

Adaptive shows how to build a cross-language component that:

- Validates config
- Registers multiple runtime surfaces
- Uses shared state
- Stays visible through the generic plugin system

## Validate the Component

Before enabling adaptive behavior beyond observation:

1. Validate the plugin config and inspect diagnostics.
2. Initialize the component in a development environment.
3. Run representative instrumented workflows.
4. Confirm telemetry receives scope, tool, and LLM events.
5. Keep tool parallelism in `observe_only` until scheduling behavior is verified.
6. Enable ACG only after repeated LLM requests show stable prompt sections.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **Adaptive appears inactive**: Confirm the app emits NeMo Flow lifecycle events.
- **State disappears after restart**: Use Redis-backed state instead of in-memory state.
- **Provider cache hints are wrong**: Set `provider` to match the actual model API surface or use `passthrough` while observing.
- **Scheduling changes behavior**: Return tool parallelism to `observe_only` and validate tool idempotency.
