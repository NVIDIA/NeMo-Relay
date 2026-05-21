<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust API

These pages are generated from the public Rust crates that back the core runtime and adaptive components.

## Binding At A Glance

This summary lists the package identity and support status for the binding.

- Published crates: `nemo-relay`, `nemo-relay-adaptive`, `nemo-relay-ffi`, and
  `nemo-relay-cli`
- Local development paths: `crates/core`, `crates/adaptive`, `crates/ffi`,
  and `crates/cli`
- Primary audience: Rust consumers who want the native runtime surface directly

The Rust docs are organized by crate because the Rust binding is the source
implementation of the runtime. The generated pages mirror each crate's public
module tree.

## Main Binding Surfaces

These entry points are the primary APIs to use from this binding.

- `nemo-relay`: core runtime APIs for scopes, tools, LLMs, registries, subscribers, codecs, streams, observability exporters, and the built-in observability plugin
- `nemo-relay-adaptive`: adaptive runtime helpers, learner implementations, storage backends, and adaptive configuration
- `nemo-relay-cli`: binary gateway for coding-agent hooks and passthrough LLM observability
- `nemo-relay-ffi`: raw C ABI used by downstream native bindings

Within `nemo-relay`, most integrations start in `api`, especially the `scope`,
`tool`, `llm`, `registry`, and `subscriber` modules. Other important public
modules include `codec`, `observability`, `stream`, `error`, and `json`. The
`observability::plugin_component` module contains the built-in `observability`
plugin config types.

Within `nemo-relay-adaptive`, the main surfaces include adaptive configuration,
plugin components, storage abstractions, learners, trie-backed data
structures, and optional Redis-backed helpers when the feature is enabled.
`nemo-relay-cli` is a binary crate, so its end-user surface is documented in
the NeMo Relay CLI guides rather than generated Rust API pages.

## How To Read The Generated Pages

Use the crate pages first, then expand into the public modules under each crate:

- `nemo-relay` for core runtime behavior
- `nemo-relay-adaptive` for adaptive and learning-oriented behavior
- `nemo-relay-cli` for coding-agent observability through hooks and the
  passthrough LLM gateway

That structure matches how Rust consumers import items from the crates.

Use the generated crate entry points when you need symbol-level detail:

- {doc}`nemo_relay <_generated/nemo-relay/src>`
- {doc}`nemo_relay_adaptive <_generated/nemo-relay-adaptive/src>`

```{toctree}
:maxdepth: 1

nemo-relay <_generated/nemo-relay/src>
nemo-relay-adaptive <_generated/nemo-relay-adaptive/src>
```

## Related Guides

Use these links to continue from the API reference into task-focused guides.

- [Quick Start](../../../getting-started/quick-start.md)
- [Rust Quick Start](../../../getting-started/rust.md)
- [Scopes](../../../about/concepts/scopes.md)
- [Middleware](../../../about/concepts/middleware.md)
- [Subscribers](../../../about/concepts/subscribers.md)
- [Plugins](../../../about/concepts/plugins.md)
- [Adaptive Tuning](../../../plugins/adaptive/about.md)
- [Observability Configuration](../../../plugins/observability/configuration.md)
- [Typed Wrappers and Codecs](../../../integrate-frameworks/using-codecs.md)
- [Framework Integration Surfaces](../../../integrate-frameworks/about.md)
- [NeMo Relay CLI Basic Usage](../../../nemo-relay-cli/basic-usage.md)
