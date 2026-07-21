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
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay

`nemo-relay-node` is the NeMo Relay package for Node.js applications. It gives
JavaScript and TypeScript code access to the same execution scopes, middleware,
plugins, lifecycle events, and observability model used by the Rust runtime.

The package is implemented as a napi-rs native extension, but Node.js users
should install it from npm rather than depend on the Rust crate directly.

## Why Use It?

Use the Node.js binding for the following tasks:

- **Own execution context in Node.js**: Group agent, tool, and LLM work into
  one scope tree from JavaScript or TypeScript.
- **Put policy around callbacks**: Register guardrails and intercepts for
  request rewriting, blocking, sanitization, and execution wrapping.
- **Emit one lifecycle stream**: Send runtime events to in-process
  subscribers, Agent Trajectory Interchange Format (ATIF), OpenTelemetry, or
  OpenInference workflows.
- **Use package entry points by need**: Import the main runtime surface plus
  typed, plugin, adaptive, and observability helpers from npm.

## What You Get

The Node.js package provides the following capabilities:

- **npm package for Node.js**: A Node.js 24 or newer package backed by a
  napi-rs native extension.
- **Managed tool and LLM execution**: Helpers that emit lifecycle events and
  run middleware in a consistent order.
- **Middleware APIs**: Guardrails and intercepts for tool and LLM boundaries,
  plus mark and scope event sanitizers for `data`, `categoryProfile`, and
  `metadata`.
- **Observability exporters**: Subscriber and exporter support for common
  runtime telemetry flows.
- **Additional entry points**: `nemo-relay-node/typed`,
  `nemo-relay-node/plugin`, `nemo-relay-node/adaptive`, and
  `nemo-relay-node/observability`.

## Installation

Install the npm package in a Node.js 24 or newer project:

```bash
npm install nemo-relay-node
```

## Getting Started

Register a subscriber and emit a mark inside a scope:

```js
const {
  ScopeType,
  deregisterSubscriber,
  event,
  flushSubscribers,
  registerSubscriber,
  withScope,
} = require("nemo-relay-node");

async function main() {
  registerSubscriber("printer", (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
    console.log(JSON.stringify(runtimeEvent));
  });

  await withScope("demo-agent", ScopeType.Agent, async (handle) => {
    event("initialized", handle, { binding: "node" }, null);
  });

  flushSubscribers();
  await new Promise((resolve) => setImmediate(resolve));
  deregisterSubscriber("printer");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

Native subscriber delivery is asynchronous. `flushSubscribers()` drains the
native dispatcher. The extra event-loop turn lets queued JavaScript callback
side effects complete before deregistration or exit.

The main runtime API is exported from `nemo-relay-node`. Additional entry points
are available at `nemo-relay-node/typed`, `nemo-relay-node/plugin`,
`nemo-relay-node/adaptive`, and `nemo-relay-node/observability`.

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
