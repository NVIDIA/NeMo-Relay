<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-wasm

`nemo-flow-wasm` is the NeMo Flow WebAssembly package for JavaScript
environments that load the runtime through WASM. It exposes the same execution
scope, middleware, plugin, lifecycle event, and observability concepts as the
Rust runtime.

The Rust crate in this directory is build machinery for the generated npm
package. JavaScript users should install the npm package rather than depend on
the Rust crate directly.

## Why Use It?

- 🌐 **Bring NeMo Flow to WebAssembly**: Use the shared runtime model from
  JavaScript environments that load the package through WASM.
- 🧭 **Keep execution context visible**: Group scope, tool, LLM, middleware, and
  subscriber behavior into the same runtime event tree.
- 🛡️ **Register JavaScript policy callbacks**: Apply guardrails and intercepts
  around managed tool and LLM execution.
- 📦 **Consume it as npm**: Install the generated package instead of depending
  on the Rust crate directly.

## What You Get

- ✅ **WASM runtime bindings**: Access to NeMo Flow scope, tool, LLM,
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

For local source development from this repository:

```bash
cd crates/wasm
npm run build:pkg
npm run test:pkg
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
