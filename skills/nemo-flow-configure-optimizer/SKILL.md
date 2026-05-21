---
name: nemo-flow-configure-optimizer
description: Deprecated compatibility alias for nemo-flow-tune-adaptive-config; retained through v0.3 for users who still ask to configure the optimizer
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Deprecated Compatibility Alias For `nemo-flow-tune-adaptive-config`

This legacy skill name is retained through v0.3 for compatibility. NeMo Flow
now describes this area as adaptive tuning through the shared plugin system, not
a separate optimizer configuration surface.

If this alias is selected, configure the current adaptive plugin component
directly.

## Compatibility Workflow

1. Model adaptive as a top-level plugin component with kind `adaptive`.
2. Use the shared plugin config `components` list.
3. Start with `state.backend = in_memory`.
4. Enable telemetry before active behavior.
5. Add only one active section at a time: `adaptive_hints`,
   `tool_parallelism`, or `acg`.
6. Validate the plugin config before initialization.
7. Keep active behavior behind a measured rollout and rollback path.

## Current Surfaces

- Python: `nemo_flow.adaptive.AdaptiveConfig(...)`,
  `nemo_flow.adaptive.ComponentSpec(...)`, and
  `nemo_flow.plugin.PluginConfig(...)`.
- Node.js: `require("nemo-flow-node/adaptive")` plus
  `nemo-flow-node/plugin`.
- Rust: `nemo_flow_adaptive::{AdaptiveConfig, ComponentSpec, ...}` and
  `nemo_flow::plugin::{validate_plugin_config, initialize_plugins}`.

If the user says "optimizer config," translate that to adaptive plugin
configuration and continue with `nemo-flow-tune-adaptive-config`.
