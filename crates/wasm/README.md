<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Flow)](https://github.com/NVIDIA/NeMo-Flow/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Flow/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Flow?color=green)](https://github.com/NVIDIA/NeMo-Flow/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Flow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Flow)
[![PyPI](https://img.shields.io/pypi/v/nemo-flow?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-flow/)
[![npm node](https://img.shields.io/npm/v/nemo-flow-node?label=nemo-flow-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-flow-wasm?label=nemo-flow-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-flow-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow?label=nemo-flow&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow)
[![Crates.io](https://img.shields.io/crates/v/nemo-flow-adaptive?label=nemo-flow-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-flow-adaptive)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Flow)

# nemo-flow-wasm

`nemo-flow-wasm` is the NeMo Flow WebAssembly package for JavaScript
environments that load the runtime through WebAssembly. It exposes the same execution
scope, middleware, plugin, lifecycle event, and observability concepts as the
Rust runtime.

The Rust crate in this directory is build machinery for the generated npm
package. JavaScript users should install the npm package rather than depend on
the Rust crate directly.

This surface is experimental and source-first. Use the repository source tree
and WebAssembly tests when validating behavior, and prefer Rust, Python, or
Node.js for primary documented application integrations.

## Why Use It?

- 🌐 **Bring NeMo Flow to WebAssembly**: Use the shared runtime model from
  JavaScript environments that load the package through WebAssembly.
- 🧭 **Keep execution context visible**: Group scope, tool, LLM, middleware, and
  subscriber behavior into the same runtime event tree.
- 🛡️ **Register JavaScript policy callbacks**: Apply guardrails and intercepts
  around managed tool and LLM execution.
- 📦 **Consume it as npm**: Install the generated package instead of depending
  on the Rust crate directly.

## What You Get

- ✅ **WebAssembly runtime bindings**: Access to NeMo Flow scope, tool, LLM,
  middleware, subscriber, plugin, typed, and adaptive APIs.
- ✅ **Managed tool and LLM execution**: Helpers that emit lifecycle events for
  JavaScript-managed callbacks.
- ✅ **Middleware registration**: Guardrail and intercept APIs for JavaScript
  callbacks.
- ✅ **Additional entry points**: `nemo-flow-wasm/typed`,
  `nemo-flow-wasm/plugin`, and `nemo-flow-wasm/adaptive`.
- ✅ **Generated npm package**: A `wasm-pack` build prepared for JavaScript
  package consumption.

## Installation

Install the npm package in a JavaScript project:

```bash
npm install nemo-flow-wasm
```

For local source development from the repository root:

```bash
npm run build:pkg --workspace=nemo-flow-wasm
npm run test:pkg --workspace=nemo-flow-wasm
```

## Getting Started

Register a subscriber and emit a mark inside a scope:

```js
const {
  ScopeType,
  deregisterSubscriber,
  event,
  registerSubscriber,
  withScope,
} = require("nemo-flow-wasm");

async function main() {
  registerSubscriber("printer", (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
  });

  await withScope("demo-agent", ScopeType.Agent, async (handle) => {
    event("initialized", handle, { binding: "wasm" }, null);
  });

  deregisterSubscriber("printer");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

The main runtime API is exported from `nemo-flow-wasm`. Additional entry points
are available at `nemo-flow-wasm/typed`, `nemo-flow-wasm/plugin`, and
`nemo-flow-wasm/adaptive`.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
