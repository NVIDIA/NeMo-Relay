<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Node.js API

These pages are generated from the exported TypeScript declaration surfaces in `crates/node`, including the generated root `index.d.ts`.

## Binding At A Glance

This summary lists the package identity and support status for the binding.

- Package name: `nemo-relay-node`
- Runtime requirement: Node.js `>=24`
- Local development path: `crates/node`

The Node.js binding is built with `napi-rs`. The package root exports the core
runtime lifecycle APIs, and the package also publishes focused subpath exports
for typed helpers, plugin helpers, adaptive helpers, and observability helpers.

## Main Binding Surfaces

These entry points are the primary APIs to use from this binding.

- Package root: scope stack, event, tool, LLM, middleware, and subscriber APIs
- `nemo-relay-node/typed`: typed wrappers and codec-aware execution helpers
- `nemo-relay-node/plugin`: plugin-facing helpers and configuration types
- `nemo-relay-node/adaptive`: adaptive helpers layered on top of the runtime
- `nemo-relay-node/observability`: built-in observability plugin helpers

## How To Read The Generated Pages

The generated pages are organized around the package export map:

- `Runtime`: the package root declarations from `index.d.ts`
- `Typed Helpers`: the `./typed` export
- `Plugins`: the `./plugin` export
- `Adaptive`: the `./adaptive` export
- `Observability`: the `./observability` export

Use the generated Node.js pages for symbol-level documentation:

- {doc}`Runtime <_generated/runtime>`
- {doc}`Typed Helpers <_generated/typed>`
- {doc}`Plugins <_generated/plugin>`
- {doc}`Adaptive <_generated/adaptive>`
- {doc}`Observability <_generated/observability>`

```{toctree}
:maxdepth: 1

runtime <_generated/runtime>
typed <_generated/typed>
plugin <_generated/plugin>
adaptive <_generated/adaptive>
observability <_generated/observability>
```

## Related Guides

Use these links to continue from the API reference into task-focused guides.

- [Quick Start](../../../getting-started/quick-start.md)
- [Node.js Quick Start](../../../getting-started/nodejs.md)
- [Scopes](../../../about/concepts/scopes.md)
- [Middleware](../../../about/concepts/middleware.md)
- [Subscribers](../../../about/concepts/subscribers.md)
- [Plugins](../../../about/concepts/plugins.md)
- [Adaptive Tuning](../../../plugins/adaptive/about.md)
- [Observability Configuration](../../../plugins/observability/configuration.md)
- [Instrument a Tool Call](../../../instrument-applications/instrument-tool-call.md)
- [Typed Wrappers and Codecs](../../../integrate-frameworks/using-codecs.md)
- [Framework Integration Surfaces](../../../integrate-frameworks/about.md)
