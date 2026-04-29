<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Examples

Use these examples when you need the binding-level adaptive helper names or a concrete configuration shape.

- [Advanced Guide: Configure Adaptive Components](adaptive-components.md)

## Complete Adaptive Component

Adaptive is configured as one plugin component with fixed kind `adaptive`.

At a high level:

1. Identify the component version and optional `agent_id`.
2. Choose a state backend.
3. Optionally enable telemetry.
4. Optionally configure adaptive hints.
5. Optionally configure tool parallelism.
6. Optionally configure the adaptive cache governor.
7. Apply config validation policy.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nemo_flow import adaptive

config = adaptive.AdaptiveConfig()
config.agent_id = "planner"
config.state = adaptive.StateConfig(
    backend=adaptive.BackendSpec.in_memory(),
)
config.telemetry = adaptive.TelemetryConfig(
    subscriber_name="adaptive.telemetry",
    learners=["tool_parallelism"],
)
config.adaptive_hints = adaptive.AdaptiveHintsConfig()
config.tool_parallelism = adaptive.ToolParallelismConfig(mode="observe_only")
config.acg = adaptive.AcgConfig(provider="openai")

component = adaptive.ComponentSpec(config)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import * as adaptive from 'nemo-flow-node/adaptive';

const config = adaptive.defaultConfig();
config.agent_id = 'planner';
config.state = {
  backend: adaptive.inMemoryBackend(),
};
config.telemetry = adaptive.telemetryConfig({
  subscriber_name: 'adaptive.telemetry',
  learners: ['tool_parallelism'],
});
config.adaptive_hints = adaptive.adaptiveHintsConfig();
config.tool_parallelism = adaptive.toolParallelismConfig({ mode: 'observe_only' });
config.acg = adaptive.acgConfig({ provider: 'openai' });

const component = adaptive.ComponentSpec(config);
```
:::

:::{tab-item} Rust
:sync: rust

```rust
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

let mut config = AdaptiveConfig::default();
config.agent_id = Some("planner".into());
config.state = Some(StateConfig {
    backend: BackendSpec::in_memory(),
});
config.telemetry = Some(TelemetryComponentConfig {
    subscriber_name: Some("adaptive.telemetry".into()),
    learners: vec!["tool_parallelism".into()],
});
config.adaptive_hints = Some(AdaptiveHintsComponentConfig::default());
config.tool_parallelism = Some(ToolParallelismComponentConfig::default());
config.acg = Some(AcgComponentConfig {
    provider: "openai".into(),
    ..AcgComponentConfig::default()
});

let component = ComponentSpec::new(config);
```
:::

::::

## Adaptive Cache Governor Thresholds

Override stability thresholds when cache breakpoint planning is too conservative or too aggressive for representative prompts.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nemo_flow import adaptive

config = adaptive.AdaptiveConfig()
config.acg = adaptive.AcgConfig(
    provider="openai",
    stability_thresholds=adaptive.AcgStabilityThresholds(
        stable_threshold=0.99,
        semi_stable_threshold=0.75,
        min_observations_for_full_confidence=12,
    ),
)
```
:::

:::{tab-item} Node.js
:sync: node

```ts
import * as adaptive from 'nemo-flow-node/adaptive';

const config = adaptive.defaultConfig();
config.acg = adaptive.acgConfig({
  provider: 'openai',
  stability_thresholds: {
    stable_threshold: 0.99,
    semi_stable_threshold: 0.75,
    min_observations_for_full_confidence: 12,
  },
});
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow_adaptive::acg::stability::StabilityThresholds;
use nemo_flow_adaptive::{AcgComponentConfig, AdaptiveConfig};

let mut config = AdaptiveConfig::default();
config.acg = Some(AcgComponentConfig {
    provider: "openai".into(),
    stability_thresholds: StabilityThresholds {
        stable_threshold: 0.99,
        semi_stable_threshold: 0.75,
        min_observations_for_full_confidence: 12,
    },
    ..AcgComponentConfig::default()
});
```
:::

::::

## Configuration Fields

The top-level `AdaptiveConfig` contains:

- `version`
- `agent_id`
- `state`
- `telemetry`
- `adaptive_hints`
- `tool_parallelism`
- `acg`
- `policy`

Important nested fields:

| Section | Fields |
|---|---|
| `state` | `backend.kind`, `backend.config` |
| `telemetry` | `subscriber_name`, `learners` |
| `adaptive_hints` | `priority`, `break_chain`, `inject_header`, `inject_body_path` |
| `tool_parallelism` | `priority`, `mode` |
| `acg` | `provider`, `observation_window`, `priority`, `stability_thresholds` |
| `policy` | `unknown_component`, `unknown_field`, `unsupported_value` |

Supported state backends are `in_memory` and `redis`. Supported tool-parallelism modes are `observe_only`, `inject_hints`, and `schedule`. Supported ACG providers are `passthrough`, `anthropic`, and `openai`.

## Defaults To Know

These defaults are important when reading or overriding adaptive configuration examples.

Middleware priorities are ordered by ascending number. For example,
`adaptive_hints.priority = 100` is not a percentage or severity level; it places
adaptive hint injection after middleware with lower numeric priorities and before
middleware with higher numeric priorities.

- `version = 1`
- `agent_id = None`
- `state = None`
- `telemetry = None`
- `adaptive_hints.priority = 100`
- `adaptive_hints.break_chain = false`
- `adaptive_hints.inject_header = true`
- `adaptive_hints.inject_body_path = "nvext.agent_hints"`
- `tool_parallelism.mode = "observe_only"`
- `acg.provider = "passthrough"`
- `acg.observation_window = 100`
- `acg.priority = 50`
- `acg.stability_thresholds.stable_threshold = 0.95`
- `acg.stability_thresholds.semi_stable_threshold = 0.50`
- `acg.stability_thresholds.min_observations_for_full_confidence = 20`

## Runtime-Adjacent Variables

NeMo Flow does not require application-level environment variables for normal adaptive runtime use. These variables are available for adjacent workflows:

| Variable | Scope | Purpose |
|---|---|---|
| `NEMO_FLOW_ACG_DEBUG` | Adaptive cache-governor diagnostics | Enables cache-governor debug diagnostics in adaptive internals. |
| `NEMO_FLOW_RUN_REDIS_TESTS` | Test workflows | Enables Redis-backed adaptive tests. |

Internal variables such as `NEMO_FLOW_BINDING_KIND` and `NEMO_FLOW_RUNTIME_OWNER` are binding and test controls. Do not set them in application code unless a maintainer asks you to debug runtime ownership behavior.
