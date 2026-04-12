<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Maintain or extend the NeMo Flow adaptive surface across config, plugin host, docs, and bindings; use this when users still say optimizer
---

# Maintain Adaptive Surfaces

Use this skill when changing adaptive config schema, built-in sections, shared
plugin-host lifecycle, hosted plugin registration, or binding-native helper
APIs.

## Public Boundary

The stable adaptive boundary is the config document plus the shared plugin-host
lifecycle:

- config types and policies
- built-in adaptive section helpers
- hosted plugin registration and composition
- plugin-host lifecycle
- reports and diagnostics

There is no separate public adaptive runtime handle.

See `docs/adaptive-layer.md` and `docs/adaptive-api-reference.md`.

## Keep In Sync

- `crates/adaptive`
- shared plugin-host behavior in core and bindings
- Python adaptive/plugin wrappers in `python/nemo_flow/adaptive.py` and
  `python/nemo_flow/plugin.py`
- Go adaptive helpers under `go/nemo_flow/adaptive` plus shared plugin-host
  helpers in `go/nemo_flow`
- Node/WASM adaptive helpers and plugin wrappers
- docs and examples that show canonical config shapes

## Checklist

- [ ] Dynamic config shape still matches the documented canonical model
- [ ] Typed helper constructors still map cleanly to the same config document
- [ ] Plugin-host lifecycle is consistent across languages
- [ ] Hosted plugin context surfaces remain aligned
- [ ] Validation/report behavior remains documented and tested
- [ ] Any new component kind has docs, examples, and binding coverage

## Validation

- Run adaptive-focused Rust tests
- Run binding tests for every changed adaptive or plugin-host surface
- Update adaptive docs and any examples in the same branch

## References

- `docs/adaptive-layer.md`
- `docs/adaptive-api-reference.md`
- `docs/hosted-plugins.md`
- `docs/recipes.md`
- `validate-change`
