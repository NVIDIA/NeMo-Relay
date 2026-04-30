<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when you want NeMo Flow to collect runtime signals and activate adaptive behavior through the plugin system.

Adaptive optimization uses the same runtime model as the rest of NeMo Flow:
instrumented scopes and calls emit events, subscribers and learners observe
those events, request intercepts can add hints, and plugin configuration
controls activation. The adaptive component coordinates state, telemetry,
adaptive hints, tool parallelism, cache-governor behavior, and rollout policy.

Agent workflows often repeat similar work, call tools with different dependency
patterns, and send prompts with stable and variable sections. Adaptive
optimization gives the runtime a place to observe those patterns and expose
controlled behavior changes without hard-coding optimization logic into every
application.

## Start Here When

Use these signals to decide whether this documentation path matches your current task.

- Collect runtime signals before changing behavior
- Evaluate tool parallelism opportunities
- Add model-request hints in a controlled way
- Plan prompt-cache breakpoints for supported providers
- Share adaptive state across workers when needed
- Roll out optimization through config instead of code changes

If instrumentation is not in place yet, start with
[Instrument Applications](../instrument-applications/about.md) or
[Integrate into Frameworks](../integrate-frameworks/about.md).

## Guides

Use these guide links to move from the overview into task-specific instructions.

- [Basic Guide: Configure Adaptive Optimization](configure.md) shows the conservative first configuration and validation workflow.
- [Advanced Guide: Configure Adaptive Components](adaptive-components.md) explains the adaptive plugin component and its config fields in more detail.
- [Advanced Guide: Tune Adaptive Behavior](advanced-guide.md) explains state tuning, telemetry tuning, adaptive hints, tool parallelism, cache-governor tuning, and diagnostics.
- [Code Examples](code-examples.md) provides binding-level adaptive helper names, defaults, ACG threshold overrides, and runtime-adjacent variables.

Validate the basic workflow before tuning lower-level adaptive settings. First
confirm that instrumented work emits lifecycle events with the configuration in
[Basic Guide: Configure Adaptive Optimization](configure.md). After that
baseline is visible, keep tool parallelism in `observe_only`, leave cache
planning disabled until you have stable prompt samples, and enable active
behavior one area at a time.

Treat every adaptive change as a measured rollout. Record a baseline, change one
setting, compare events and reports, and keep rollback as a configuration
change.
