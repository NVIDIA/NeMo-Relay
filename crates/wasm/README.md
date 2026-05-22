<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-relay-wasm?label=nemo-relay-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay

`nemo-relay-wasm` is the NeMo Relay WebAssembly package for JavaScript
environments that load the runtime through WebAssembly. It exposes the same execution
scope, middleware, plugin, lifecycle event, and observability concepts as the
Rust runtime.

The Rust crate in this directory is build machinery for the generated npm
package. JavaScript users should install the npm package rather than depend on
the Rust crate directly.

This surface is experimental and source-first. Use the repository source tree
and WebAssembly tests when validating behavior, and prefer Rust, Python, or
Node.js for primary documented application integrations.

Observability support is also experimental. The WebAssembly target does not
support `grpc` OTLP transport, and file-backed observability plugin sinks
require a host runtime with filesystem access.

## Why Use It?

- 🌐 **Bring NeMo Relay to WebAssembly**: Use the shared runtime model from
  JavaScript environments that load the package through WebAssembly.
- 🧭 **Keep execution context visible**: Group scope, tool, LLM, middleware, and
  subscriber behavior into the same runtime event tree.
- 🛡️ **Register JavaScript policy callbacks**: Apply guardrails and intercepts
  around managed tool and LLM execution.
- 📦 **Consume it as npm**: Install the generated package instead of depending
  on the Rust crate directly.

## What You Get

- ✅ **WebAssembly runtime bindings**: Access to NeMo Relay scope, tool, LLM,
  middleware, subscriber, plugin, typed, and adaptive APIs.
- ✅ **Managed tool and LLM execution**: Helpers that emit lifecycle events for
  JavaScript-managed callbacks.
- ✅ **Middleware registration**: Guardrail and intercept APIs for JavaScript
  callbacks.
- ✅ **Additional entry points**: `nemo-relay-wasm/typed`,
  `nemo-relay-wasm/plugin`, `nemo-relay-wasm/adaptive`, and
  `nemo-relay-wasm/observability`.
- ✅ **Generated npm package**: A `wasm-pack` build prepared for JavaScript
  package consumption.

## Installation

Install the npm package in a JavaScript project:

```bash
npm install nemo-relay-wasm
```

For local source validation from the repository root:

```bash
npm run build:pkg --workspace=nemo-relay-wasm
npm run test:pkg --workspace=nemo-relay-wasm
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
} = require("nemo-relay-wasm");

async function main() {
  registerSubscriber("printer", (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
    console.log(JSON.stringify(runtimeEvent));
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

The main runtime API is exported from `nemo-relay-wasm`. Additional entry points
are available at `nemo-relay-wasm/typed`, `nemo-relay-wasm/plugin`,
`nemo-relay-wasm/adaptive`, and `nemo-relay-wasm/observability`.

## Documentation

NeMo Relay Documentation: https://nvidia.github.io/NeMo-Relay
