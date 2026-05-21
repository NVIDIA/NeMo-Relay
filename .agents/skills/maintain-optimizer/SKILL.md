---
name: maintain-optimizer
description: Maintain or extend the NeMo Relay adaptive surface across config, plugins, docs, and bindings; use this when users still say optimizer
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain Adaptive Surfaces

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when changing adaptive config schema, built-in sections, shared
plugin lifecycle, plugin registration, or binding-native helper
APIs.

## Public Boundary

The stable adaptive boundary is the config document plus the shared plugin
lifecycle:

- Config types and policies
- Built-in adaptive section helpers
- Plugin registration and composition
- Plugin lifecycle
- Reports and diagnostics

There is no separate public adaptive runtime handle.

See `docs/plugins/adaptive/configuration.md` and
`docs/about/concepts/plugins.md`.

## Keep In Sync

- `crates/adaptive`
- Shared plugin behavior in core and bindings
- Python adaptive/plugin wrappers in `python/nemo_relay/adaptive.py` and
  `python/nemo_relay/plugin.py`
- Go adaptive helpers under `go/nemo_relay/adaptive` plus shared plugin
  helpers in `go/nemo_relay`
- Node/WebAssembly adaptive helpers and plugin wrappers
- Docs and examples that show canonical config shapes

## Checklist

- [ ] Dynamic config shape still matches the documented canonical model
- [ ] Typed helper constructors still map cleanly to the same config document
- [ ] Plugin lifecycle is consistent across languages
- [ ] Plugin context surfaces remain aligned
- [ ] Validation/report behavior remains documented and tested
- [ ] Any new component kind has docs, examples, and binding coverage

## Validation

- Run adaptive-focused Rust tests
- Run binding tests for every changed adaptive or plugin surface
- Update adaptive docs and any examples in the same branch

## References

- `docs/plugins/adaptive/configuration.md`
- `docs/plugins/adaptive/about.md`
- `docs/plugins/adaptive/acg.md`
- `docs/plugins/adaptive/adaptive-hints.md`
- `docs/build-plugins/basic-guide.md`
- `docs/build-plugins/validate-configuration.md`
- `docs/about/concepts/plugins.md`
- `validate-change`
