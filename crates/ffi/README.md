<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Flow)](https://github.com/NVIDIA/NeMo-Flow/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Flow/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Flow?color=green)](https://github.com/NVIDIA/NeMo-Flow/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Flow/branch/main/graph/badge.svg)](https://codecov.io/gh/NVIDIA/NeMo-Flow)
[![PyPI](https://img.shields.io/pypi/v/nemo-flow?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-flow/)
[![npm node](https://img.shields.io/npm/v/nemo-flow-node?label=nemo-flow-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-flow-wasm?label=nemo-flow-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow?label=nemo-flow&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-adaptive?label=nemo-flow-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-adaptive)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Flow)

# nemo-flow-ffi

`nemo-flow-ffi` provides the C-compatible ABI for NeMo Flow. Use it when a
native integration or downstream language binding needs direct access to the
shared Rust runtime contract.

This surface is experimental and source-first. The repository-maintained Go
binding consumes it through CGo.

## Why Use It?

- 🔌 **Expose NeMo Flow to native consumers**: Call the shared Rust runtime from
  C-compatible hosts and downstream language bindings.
- 🧱 **Build on one ABI**: Keep native integrations aligned with the same scope,
  middleware, lifecycle event, and observability contract.
- 📦 **Consume a generated C header**: Use the committed `nemo_flow.h` surface
  produced by the crate build.
- 🚧 **Work source-first**: Use this experimental surface when Rust, Python, and
  Node.js packages are not the right integration layer.

## What You Get

- ✅ **Exported `nemo_flow_*` symbols**: APIs for scopes, tool calls, LLM calls,
  middleware, subscribers, plugins, observability exporters, and scope stack
  isolation.
- ✅ **Generated header**: A committed `nemo_flow.h` file for C-compatible
  consumers.
- ✅ **Native library outputs**: Shared and static libraries for platform
  linking.
- ✅ **JSON payload contract**: Cross-language request, response, metadata, and
  event data carried as JSON.
- ✅ **Go binding foundation**: The repository-maintained Go binding consumes
  this ABI through CGo.

## Installation

Build the FFI library from a repository checkout:

```bash
cargo build --release -p nemo-flow-ffi
```

The generated header is available at:

```text
crates/ffi/nemo_flow.h
```

Cargo writes the shared and static libraries under `target/release/`.

## Getting Started

Include the generated header and link against the release library for your
platform:

```c
#include "nemo_flow.h"
```

Use the FFI surface only when you need a native ABI. Rust, Python, and Node.js
applications should prefer the supported packages for those languages.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
