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
    telemetry=nemo_flow.adaptive.TelemetryConfig(learners=["tool_parallelism"]),
    adaptive_hints=nemo_flow.adaptive.AdaptiveHintsConfig(
        inject_body_path="nvext.agent_hints",
    ),
)
```
:::

:::{tab-item} Node.js
:sync: node

```js
const adaptiveConfig = adaptive.defaultConfig();
adaptiveConfig.agent_id = "planner";
adaptiveConfig.state = { backend: adaptive.inMemoryBackend() };
adaptiveConfig.telemetry = adaptive.telemetryConfig({ learners: ["tool_parallelism"] });
adaptiveConfig.adaptive_hints = adaptive.adaptiveHintsConfig({
  inject_body_path: "nvext.agent_hints",
});
```
:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_flow_adaptive::{AdaptiveConfig, AdaptiveHintsComponentConfig};

let mut adaptive = AdaptiveConfig::default();
adaptive.agent_id = Some("planner".into());
adaptive.adaptive_hints = Some(AdaptiveHintsComponentConfig::default());
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
