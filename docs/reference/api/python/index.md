<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python API

These pages are generated from the `python/nemo_relay` package source.

## Binding At A Glance

This summary lists the package identity and support status for the binding.

- Package name: `nemo-relay`
- Local development path: repository root `pyproject.toml` with `uv sync`
- Generated package root: `nemo_relay`

The Python binding exposes the runtime through a public package layer in
`python/nemo_relay` and a compiled native extension exposed as `nemo_relay._native`.
Most users should work from the public package modules rather than the native
layer directly.

## Main Binding Surfaces

These entry points are the primary APIs to use from this binding.

- `nemo_relay.scope`: create scopes, emit mark events, and manage scope handles
- `nemo_relay.tools` and `nemo_relay.llm`: run tool and LLM lifecycles from Python
- `nemo_relay.guardrails` and `nemo_relay.intercepts`: register global middleware
- `nemo_relay.scope_local`: register middleware against a specific scope hierarchy
- `nemo_relay.subscribers`: observe emitted runtime lifecycle events
- `nemo_relay.plugin`, `nemo_relay.adaptive`, and `nemo_relay.observability`: configure plugin-backed, adaptive, and exporter behavior
- `nemo_relay.typed` and `nemo_relay.codecs`: use typed wrappers and request/response codecs

## How To Read The Generated Pages

The generated `nemo_relay` package page is the package root. Under that page you
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

Use the {doc}`generated Python package index <_generated/nemo_relay/index>`
when you want the docstring-level details for a specific symbol or module.

```{toctree}
:maxdepth: 1

nemo_relay <_generated/nemo_relay/index>
```

## Related Guides

Use these links to continue from the API reference into task-focused guides.

- [Quick Start](../../../getting-started/quick-start.md)
- [Python Quick Start](../../../getting-started/python/index.md)
- [Scopes](../../../about/concepts/scopes.md)
- [Middleware](../../../about/concepts/middleware.md)
- [Subscribers](../../../about/concepts/subscribers.md)
- [Plugins](../../../about/concepts/plugins.md)
- [Adaptive Tuning](../../../plugins/adaptive/about.md)
- [Observability Configuration](../../../plugins/observability/configuration.md)
- [Typed Wrappers and Codecs](../../../integrate-frameworks/using-codecs.md)
- [Framework Integration Surfaces](../../../integrate-frameworks/about.md)
