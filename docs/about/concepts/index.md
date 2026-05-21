<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Concepts

Use these pages to understand the NeMo Relay runtime model before applying it in a use-case workflow.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.2em` Scopes
:link: scopes
:link-type: doc

Ownership boundaries for agent runs, requests, workflows, tools, LLM calls, and nested runtime work.
:::

:::{grid-item-card} {octicon}`workflow;1.2em` Middleware
:link: middleware
:link-type: doc

Guardrails and intercepts that sanitize, transform, block, or wrap tool and LLM execution.
:::

:::{grid-item-card} {octicon}`package;1.2em` Plugins
:link: plugins
:link-type: doc

Configuration-driven bundles that install reusable middleware, subscribers, and adaptive behavior.
:::

:::{grid-item-card} {octicon}`pulse;1.2em` Events
:link: events
:link-type: doc

Canonical lifecycle records for scopes, tool calls, LLM calls, marks, subscribers, and exporters.
:::

:::{grid-item-card} {octicon}`broadcast;1.2em` Subscribers
:link: subscribers
:link-type: doc

Consumers for lifecycle events, including logs, traces, trajectories, analytics, and diagnostics.
:::

:::{grid-item-card} {octicon}`plug;1.2em` Framework Integrations
:link: framework-integrations
:link-type: doc

Integration patterns for frameworks that own invocation boundaries, scheduling, retries, or provider payloads.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

scopes
middleware
plugins
events
subscribers
framework-integrations
```
