<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-wasm

`nemo-flow-wasm` provides the WebAssembly binding crate for NeMo Flow.

It uses wasm-bindgen to expose the shared runtime contract to JavaScript
environments that consume the generated WebAssembly package.

Most JavaScript users should consume the generated npm package rather than
depend on this crate directly.

For project-level documentation, start with:

- the repo root `README.md`
- `docs/getting-started/installation.md`
- `crates/wasm/package.json`
