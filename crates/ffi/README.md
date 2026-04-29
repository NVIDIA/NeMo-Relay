<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

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
