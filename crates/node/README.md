<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-flow-node

`nemo-flow-node` is the NeMo Flow package for Node.js applications. It gives
JavaScript and TypeScript code access to the same execution scopes, middleware,
plugins, lifecycle events, and observability model used by the Rust runtime.

The package is implemented as a napi-rs native extension, but Node.js users
should install it from npm rather than depend on the Rust crate directly.

## Why Use It?

- 🧭 **Own execution context in Node.js**: Group agent, tool, and LLM work into
  one scope tree from JavaScript or TypeScript.
- 🛡️ **Put policy around callbacks**: Register guardrails and intercepts for
  request rewriting, blocking, sanitization, and execution wrapping.
- 📡 **Emit one lifecycle stream**: Send runtime events to in-process
  subscribers, ATIF, OpenTelemetry, or OpenInference workflows.
- 🧩 **Use package entry points by need**: Import the main runtime surface plus
  typed, plugin, and adaptive helpers from npm.

## What You Get

- ✅ **npm package for Node.js**: A Node.js 20 or newer package backed by a
  napi-rs native extension.
- ✅ **Managed tool and LLM execution**: Helpers that emit lifecycle events and
  run middleware in a consistent order.
- ✅ **Middleware APIs**: Guardrails and intercepts for tool and LLM boundaries.
- ✅ **Observability exporters**: Subscriber and exporter support for common
  runtime telemetry flows.
- ✅ **Additional entry points**: `nemo-flow-node/typed`,
  `nemo-flow-node/plugin`, and `nemo-flow-node/adaptive`.

## Installation

Install the npm package in a Node.js 20 or newer project:

```bash
npm install nemo-flow-node
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
} = require("nemo-flow-node");

async function main() {
  registerSubscriber("printer", (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
  });

  await withScope("demo-agent", ScopeType.Agent, async (handle) => {
    event("initialized", handle, { binding: "node" }, null);
  });

  deregisterSubscriber("printer");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

The main runtime API is exported from `nemo-flow-node`. Additional entry points
are available at `nemo-flow-node/typed`, `nemo-flow-node/plugin`, and
`nemo-flow-node/adaptive`.

## Documentation

NeMo Flow Documentation: https://nvidia.github.io/NeMo-Flow
