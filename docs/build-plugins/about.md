<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when you want to package reusable NeMo Relay behavior as a plugin that can be activated from configuration.

Plugins are the configuration-driven packaging layer for shared runtime
behavior. A plugin can validate component-local config, register middleware and
subscribers through a component-scoped context, and rely on the plugin system to
report diagnostics and roll back partial setup when activation fails.

Plugins prevent repeated registration code for policies, request transforms,
exporters, and related runtime components. They give shared behavior a stable
kind name, a structured config document, and a clear activation lifecycle.

## Start Here When

Use these signals to decide whether this documentation path matches your current task.

- Ship policy bundles across applications
- Install observability exporters consistently
- Package framework-agnostic request transforms
- Validate operator-supplied config before runtime behavior changes

If the behavior applies to only one request or tenant, consider scope-local middleware before turning it into a process-level plugin.

## Guides

Use these guide links to move from the overview into task-specific instructions.

- [Define a Plugin](basic-guide.md) explains plugin kinds, shape, runtime ownership, and the activation lifecycle.
- [Validate Plugin Configuration](validate-configuration.md) covers JSON-compatible config, validation rules, and structured diagnostics.
- [Plugin Configuration Files](plugin-configuration-files.md) documents `plugins.toml` file discovery, precedence, merge behavior, and editor controls for the CLI gateway.
- [Register Plugin Behavior](register-behavior.md) shows how to initialize config and install subscribers or middleware through `PluginContext`.
- [Design Plugin Configuration](advanced-configuration.md) covers validation rules, advanced configuration patterns, rollout controls, and `PluginContext` usage.
- [NeMo Guardrails Example Plugin](nemoguardrails.md) shows an external Python plugin that applies NeMo Guardrails checks around NeMo Relay LLM and tool calls.
- [Code Examples](code-examples.md) provides patterns for dynamic header injection, subscriber-oriented export, multi-surface bundles, and framework-facing plugins.

Start by deciding which runtime surfaces the plugin owns: middleware,
subscribers, or a combination of related runtime behavior. Define the smallest
JSON-compatible config that can drive that behavior, validate it before
registration, and keep external objects or callables out of the config document.

Use plugins for reusable process-level behavior. Keep request-specific behavior scope-local so it is cleaned up with the owning scope.
