<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Cache Governor (ACG)

Use the Adaptive Cache Governor (ACG) when repeated LLM requests contain stable
prompt sections that can benefit from provider prompt caching.

ACG decomposes LLM requests into Prompt IR, scores block stability across
observed runs, and plans provider-specific prompt-cache breakpoints. The `acg`
section is optional. Omit it to keep cache planning disabled.

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
learners = ["acg"]

[components.config.acg]
provider = "anthropic"
observation_window = 100
priority = 50

[components.config.acg.stability_thresholds]
stable_threshold = 0.95
semi_stable_threshold = 0.50
min_observations_for_full_confidence = 20
```

This configuration enables adaptive telemetry and configures ACG to plan cache
breakpoints for Anthropic-style request surfaces after it has enough observed
prompt samples.

## Helper Example

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
adaptive_config = nemo_flow.adaptive.AdaptiveConfig(
    agent_id="planner",
    state=nemo_flow.adaptive.StateConfig(
        backend=nemo_flow.adaptive.BackendSpec.in_memory(),
    ),
    telemetry=nemo_flow.adaptive.TelemetryConfig(learners=["acg"]),
    acg=nemo_flow.adaptive.AcgConfig(provider="anthropic"),
)
```
:::

:::{tab-item} Node.js
:sync: node

```js
const adaptiveConfig = adaptive.defaultConfig();
adaptiveConfig.agent_id = "planner";
adaptiveConfig.state = { backend: adaptive.inMemoryBackend() };
adaptiveConfig.telemetry = adaptive.telemetryConfig({ learners: ["acg"] });
adaptiveConfig.acg = adaptive.acgConfig({ provider: "anthropic" });
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow_adaptive::{AcgComponentConfig, AdaptiveConfig};

let mut adaptive = AdaptiveConfig::default();
adaptive.agent_id = Some("planner".into());
adaptive.acg = Some(AcgComponentConfig {
    provider: "anthropic".into(),
    ..AcgComponentConfig::default()
});
```
:::

::::

## Fields

| Field | Default | Notes |
|---|---|---|
| `provider` | `passthrough` | `passthrough`, `anthropic`, or `openai`. |
| `observation_window` | `100` | Rolling Prompt IR sample window for stability analysis. |
| `priority` | `50` | LLM execution intercept priority. Lower values run earlier. |
| `stability_thresholds.stable_threshold` | `0.95` | Minimum effective score classified as stable. |
| `stability_thresholds.semi_stable_threshold` | `0.50` | Minimum effective score classified as semi-stable. |
| `stability_thresholds.min_observations_for_full_confidence` | `20` | Observation count required for full confidence. |

Use `passthrough` when you want ACG observations without provider-specific hint
translation. Set `provider` to the backend API surface the agent actually calls
when you are ready to apply cache planning.

## Expected Output

When ACG is active, instrumented LLM calls still return the same application
result. ACG records observations and, when enough stable prompt structure is
available, emits adaptive diagnostics and cache-planning decisions through the
adaptive runtime.

Provider-specific cache hints are useful only when the request surface supports
them. Validate against representative LLM traffic before enabling ACG in
production.

## Common Validation Failures

- `provider` is not one of `passthrough`, `anthropic`, or `openai`.
- Stability thresholds are outside the supported numeric range.
- ACG is enabled before the application emits managed LLM events.
- The configured provider does not match the real model API surface.
