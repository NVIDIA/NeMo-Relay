<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust API

These pages are generated from the public Rust crates that back the core runtime and adaptive components.

## Binding At A Glance

This summary lists the package identity and support status for the binding.

- Published crates: `nemo-flow`, `nemo-flow-adaptive`, `nemo-flow-ffi`, and
  `nemo-flow-cli`
- Local development paths: `crates/core`, `crates/adaptive`, `crates/ffi`,
  and `crates/cli`
- Primary audience: Rust consumers who want the native runtime surface directly

The Rust docs are organized by crate because the Rust binding is the source
implementation of the runtime. The generated pages mirror each crate's public
module tree.

## Main Binding Surfaces

These entry points are the primary APIs to use from this binding.

- `nemo-flow`: core runtime APIs for scopes, tools, LLMs, registries, subscribers, codecs, streams, observability exporters, and the built-in observability plugin
- `nemo-flow-adaptive`: adaptive runtime helpers, learner implementations, storage backends, and adaptive configuration
- `nemo-flow-cli`: binary gateway for coding-agent hooks and passthrough LLM observability
- `nemo-flow-ffi`: raw C ABI used by downstream native bindings

Within `nemo-flow`, most integrations start in `api`, especially the `scope`,
`tool`, `llm`, `registry`, and `subscriber` modules. Other important public
modules include `codec`, `observability`, `stream`, `error`, and `json`. The
`observability::plugin_component` module contains the built-in `observability`
plugin config types.

Within `nemo-flow-adaptive`, the main surfaces include adaptive configuration,
plugin components, storage abstractions, learners, trie-backed data
structures, and optional Redis-backed helpers when the feature is enabled.
`nemo-flow-cli` is a binary crate, so its end-user surface is documented in
the NeMo Flow CLI guides rather than generated Rust API pages.

## How To Read The Generated Pages

Use the crate pages first, then expand into the public modules under each crate:

- `nemo-flow` for core runtime behavior
- `nemo-flow-adaptive` for adaptive and learning-oriented behavior
- `nemo-flow-cli` for coding-agent observability through hooks and the
  passthrough LLM gateway

That structure matches how Rust consumers import items from the crates.

Use the generated crate entry points when you need symbol-level detail:

- {doc}`nemo_flow <_generated/nemo-flow/src>`
- {doc}`nemo_flow_adaptive <_generated/nemo-flow-adaptive/src>`

```{toctree}
:maxdepth: 1

nemo-flow <_generated/nemo-flow/src>
nemo-flow-adaptive <_generated/nemo-flow-adaptive/src>
```

## Related Guides

Use these links to continue from the API reference into task-focused guides.

- [Quick Start](../../../getting-started/quick-start.md)
- [Rust Quick Start](../../../getting-started/rust.md)
- [Scopes](../../../about/concepts/scopes.md)
- [Middleware](../../../about/concepts/middleware.md)
- [Subscribers](../../../about/concepts/subscribers.md)
- [Plugins](../../../about/concepts/plugins.md)
- [Adaptive Optimization](../../../use-adaptive-optimization/about.md)
- [Configure the Observability Plugin](../../../export-observability-data/observability-plugin.md)
- [Typed Wrappers and Codecs](../../../integrate-frameworks/using-codecs.md)
- [Framework Integration Surfaces](../../../integrate-frameworks/about.md)
- [NeMo Flow CLI Basic Usage](../../../nemo-flow-cli/basic-usage.md)
