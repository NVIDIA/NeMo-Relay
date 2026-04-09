<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Maintain or extend the NeMo Flow optimizer surface across config, runtime, plugins, docs, and bindings
---

# Maintain Optimizer Surfaces

Use this skill when changing optimizer config schema, built-in components,
runtime lifecycle, hosted plugin registration, or binding-native helper APIs.

## Public Boundary

The stable optimizer boundary is the config document plus runtime handle:

- config types and policies
- built-in component helpers
- hosted plugin registration
- runtime lifecycle
- reports and diagnostics

See `docs/optimizer-layer.md` and `docs/optimizer-api-reference.md`.

## Keep In Sync

- `crates/optimizer`
- Python optimizer wrappers in `python/nemo_flow/optimizer.py`
- Go optimizer package under `go/nemo_flow/optimizer`
- Node/WASM optimizer helpers and runtime wrappers
- docs and examples that show canonical config shapes

## Checklist

- [ ] Dynamic config shape still matches the documented canonical model
- [ ] Typed helper constructors still map cleanly to the same config document
- [ ] Runtime lifecycle is consistent across languages
- [ ] Hosted plugin context surfaces remain aligned
- [ ] Validation/report behavior remains documented and tested
- [ ] Any new component kind has docs, examples, and binding coverage

## Validation

- Run optimizer-focused Rust tests
- Run binding tests for every changed optimizer surface
- Update optimizer docs and any examples in the same branch

## References

- `docs/optimizer-layer.md`
- `docs/optimizer-api-reference.md`
- `docs/recipes.md`
- `validate-change`
