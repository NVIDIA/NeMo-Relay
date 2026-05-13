<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python API

These pages are generated from the `python/nemo_flow` package source.

## Binding At A Glance

This summary lists the package identity and support status for the binding.

- Package name: `nemo-flow`
- Local development path: repository root `pyproject.toml` with `uv sync`
- Generated package root: `nemo_flow`

The Python binding exposes the runtime through a public package layer in
`python/nemo_flow` and a compiled native extension exposed as `nemo_flow._native`.
Most users should work from the public package modules rather than the native
layer directly.

## Main Binding Surfaces

These entry points are the primary APIs to use from this binding.

- `nemo_flow.scope`: create scopes, emit mark events, and manage scope handles
- `nemo_flow.tools` and `nemo_flow.llm`: run tool and LLM lifecycles from Python
- `nemo_flow.guardrails` and `nemo_flow.intercepts`: register global middleware
- `nemo_flow.scope_local`: register middleware against a specific scope hierarchy
- `nemo_flow.subscribers`: observe emitted runtime lifecycle events
- `nemo_flow.plugin`, `nemo_flow.adaptive`, and `nemo_flow.observability`: configure plugin-backed, adaptive, and exporter behavior
- `nemo_flow.typed` and `nemo_flow.codecs`: use typed wrappers and request/response codecs

## How To Read The Generated Pages

The generated `nemo_flow` package page is the package root. Under that page you
will find submodule pages for the public binding surface, including:

- `llm`
- `tools`
- `scope`
- `scope_local`
- `guardrails`
- `intercepts`
- `subscribers`
- `plugin`
- `adaptive`
- `observability`
- `typed`
- `codecs`

Use the {doc}`generated Python package index <_generated/nemo_flow/index>`
when you want the docstring-level details for a specific symbol or module.

```{toctree}
:maxdepth: 1

nemo_flow <_generated/nemo_flow/index>
```

## Related Guides

Use these links to continue from the API reference into task-focused guides.

- [Quick Start](../../../getting-started/quick-start.md)
- [Python Quick Start](../../../getting-started/python/index.md)
- [Scopes](../../../about/concepts/scopes.md)
- [Middleware](../../../about/concepts/middleware.md)
- [Subscribers](../../../about/concepts/subscribers.md)
- [Plugins](../../../about/concepts/plugins.md)
- [Adaptive Optimization](../../../use-adaptive-optimization/about.md)
- [Configure the Observability Plugin](../../../export-observability-data/observability-plugin.md)
- [Typed Wrappers and Codecs](../../../integrate-frameworks/using-codecs.md)
- [Framework Integration Surfaces](../../../integrate-frameworks/about.md)
