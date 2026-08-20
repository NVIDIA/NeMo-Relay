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
  subscribers, Agent Trajectory Interchange Format (ATIF), or typed
  OpenTelemetry workflows.
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
- **Observability exporters**: `OpenTelemetrySubscriber` exports traces;
  `OpenTelemetryLogSubscriber` and `OpenTelemetryMetricSubscriber` export
  severity-tagged marks and typed metric measurements. Bare OTLP/HTTP origins
  resolve to `/v1/traces`, `/v1/logs`, or `/v1/metrics` for the selected signal.
  The `nemo-relay-node/observability` helper configures plugin-owned endpoint
  fan-out.
- **Additional entry points**: `nemo-relay-node/typed`,
  `nemo-relay-node/plugin`, `nemo-relay-node/adaptive`, and
  `nemo-relay-node/observability`.

## Installation

Install the npm package in a Node.js 24 or newer project:

```bash
npm install nemo-relay-node@0.8.0
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
} = require('nemo-relay-node');

async function main() {
  registerSubscriber('printer', (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
    console.log(JSON.stringify(runtimeEvent));
  });

  await withScope('demo-agent', ScopeType.Agent, async (handle) => {
    event('initialized', handle, { binding: 'node' }, null);
  });

  await flushSubscribers();
  deregisterSubscriber('printer');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

Tool producers return the canonical `{ result, annotation? }` object. Typed
helpers apply result codecs only to the application-owned `result`, while Relay
preserves the optional opaque `annotation` as adjacent metadata:

```js
const { toolCallExecuteAsync } = require('nemo-relay-node');

const execution = await toolCallExecuteAsync('lookup', { query: 'relay' }, async (args) => ({
  result: { answer: args.query.toUpperCase() },
  annotation: { provider: 'example' },
}));

console.log(execution.result.answer);
```

The core mark contract uses positional optional arguments. Pass `null` for
`dataSchema` before supplying the final `severity` argument:

```js
const { LogSeverity } = require('nemo-relay-node');

event('initialized', handle, { binding: 'node' }, null, null, null, LogSeverity.Info);
```

Call `metric()` with `MetricMeasurement` objects for metrics; Relay validates
the complete measurement group before publishing it.

## OTLP Logs and Metrics

For plugin-managed export, the `nemo-relay-node/observability` helpers create a
version-4 component. Enabling logs and metrics without signal endpoints derives
`/v1/logs` and `/v1/metrics` from the trace endpoint:

```js
const observability = require('nemo-relay-node/observability');

const component = observability.ComponentSpec({
  version: 4,
  opentelemetry: observability.openTelemetryConfig({
    enabled: true,
    endpoints: [observability.openTelemetryEndpoint({
      type: 'gen_ai', endpoint: 'http://localhost:4318/v1/traces',
    })],
    logs: observability.openTelemetryLogConfig({ enabled: true }),
    metrics: observability.openTelemetryMetricConfig({ enabled: true }),
  }),
});
```

Use the final `dataSchema` and `severity` arguments for a typed log mark, and
use `metric()` for an atomically validated metric group:

```js
const {
  event, metric, LogSeverity, MetricKind, MetricValueType,
} = require('nemo-relay-node');

event(
  'cache-nearly-full', null, { entries: 900 }, null, null,
  { name: 'example.cache', version: '1' }, LogSeverity.Warn,
);
metric('cache-entries', [{
  name: 'example.cache.entries', kind: MetricKind.Gauge,
  valueType: MetricValueType.U64, value: 900,
}]);
```

Direct log and metric subscribers are independently managed. Register each
before emitting marks, then deregister, force-flush, and shut it down during
graceful teardown. `runtimeDiagnostics()` returns bounded `code`, `message`,
and `count` entries:

```js
const {
  OpenTelemetryLogSubscriber, OpenTelemetryMetricSubscriber,
} = require('nemo-relay-node');

// Equivalent explicit OTLP/HTTP paths are /v1/logs and /v1/metrics, respectively.
const logs = new OpenTelemetryLogSubscriber({ endpoint: 'http://localhost:4318' });
const metrics = new OpenTelemetryMetricSubscriber({ endpoint: 'http://localhost:4318' });
logs.register('otlp-logs');
metrics.register('otlp-metrics');
try {
  for (const diagnostic of logs.runtimeDiagnostics()) {
    console.error(diagnostic.code, diagnostic.message);
  }
} finally {
  logs.deregister('otlp-logs');
  logs.forceFlush();
  logs.shutdown();
  metrics.deregister('otlp-metrics');
  metrics.forceFlush();
  metrics.shutdown();
}
```

Native subscriber delivery is asynchronous. Awaiting `flushSubscribers()` drains
the native dispatcher and waits for managed terminal publications registered
before the call and the JavaScript subscriber callbacks they queue, without
blocking the Node.js event loop. Native events emitted by a JavaScript subscriber
are separate publications; flush again if those events must also be observed.
Subscribers can return `Promise` objects. A synchronous throw or a rejected `Promise` from a subscriber is isolated:
it does not terminate the host or reject `flushSubscribers()`, and Relay reports the
failure to `stderr` and through `getLastCallbackError()`.

The main runtime API is exported from `nemo-relay-node`. Additional entry points
are available at `nemo-relay-node/typed`, `nemo-relay-node/plugin`,
`nemo-relay-node/adaptive`, and `nemo-relay-node/observability`.

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
