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

`nemo-relay-ffi` provides the C-compatible ABI for NeMo Relay. Use it when a
native integration or downstream language binding needs direct access to the
shared Rust runtime contract.

This surface is experimental and source-first. The repository-maintained Go
binding consumes it through CGo.

> **DO NOT TREAT AS PRODUCTION-READY:** the experimental
> `nemo_relay_initialize_with_dynamic_plugins` lifecycle needs a real consumer
> to validate shutdown, ownership, and error handling before it can be promoted
> to a stable contract.

## Why Use It?

- **Expose NeMo Relay to native consumers**: Call the shared Rust runtime from
  C-compatible hosts and downstream language bindings.
- **Build on one ABI**: Keep native integrations aligned with the same scope,
  middleware, lifecycle event, and observability contract.
- **Consume a generated C header**: Use the committed `nemo_relay.h` surface
  produced by the crate build.
- **Work source-first**: Use this experimental surface when Rust, Python, and
  Node.js packages are not the right integration layer.

## What You Get

- **Exported `nemo_relay_*` symbols**: APIs for scopes, tool calls, LLM calls,
  middleware, subscribers, plugins, observability exporters, and scope stack
  isolation.
- **Typed OpenTelemetry export**:
  `nemo_relay_otel_subscriber_create` constructs one `full`, `gen_ai`, or
  `openinference` trace subscriber. Independently managed log and metric
  subscribers use `nemo_relay_otel_log_subscriber_create` and
  `nemo_relay_otel_metric_subscriber_create`.
- **Structured marks and metrics**: `nemo_relay_event_v2` adds optional data
  schema JSON and `NemoRelayLogSeverity` to the compatible mark API.
  `nemo_relay_metric_json` and `nemo_relay_metric` emit atomically validated
  Relay metric measurements.
- **Generated header**: A committed `nemo_relay.h` file for C-compatible
  consumers.
- **Native library outputs**: Shared and static libraries for platform
  linking.
- **JSON payload contract**: Cross-language request, response, metadata, and
  event data carried as JSON.
- **Go binding foundation**: The repository-maintained Go binding consumes
  this ABI through CGo.

Middleware callbacks in the raw C ABI are synchronous. Relay invokes a
callback on a native thread and waits for it to return. Blocking I/O and other
long-running callback work therefore occupy that thread and can reduce
middleware throughput. The FFI does not expose completion-based middleware
registration.

## OTLP Logs and Metrics

Use `nemo_relay_event_v2` when a mark needs a data schema or OTLP log severity.
Pass `data_schema_json` as `{"name":"example.schema","version":"1"}` and a
`NemoRelayLogSeverity` pointer when applicable. The legacy `nemo_relay_event`
function remains valid for untyped marks.

Use `nemo_relay_metric_json` for a nonempty JSON array of canonical metric
measurements, or `nemo_relay_metric` for `NemoRelayMetricMeasurement` entries.
Relay validates a complete measurement group before it emits any recording
operation. Do not construct the reserved metric mark schema manually.

Create each direct OTLP log or metric subscriber independently. Register it
with `nemo_relay_otel_log_subscriber_register` or
`nemo_relay_otel_metric_subscriber_register`. During graceful shutdown,
deregister the name, force-flush the subscriber, shut it down, and free its
handle with the matching `nemo_relay_otel_*_subscriber_free` function. The log
and metric APIs also expose bounded JSON runtime-diagnostics snapshots through
their `nemo_relay_otel_*_subscriber_runtime_diagnostics_json` functions.

## Installation

Build the FFI library from a repository checkout:

```bash
cargo build --release -p nemo-relay-ffi
```

The generated header is available at:

```text
crates/ffi/nemo_relay.h
```

Cargo writes the shared and static libraries under `target/release/`.

## Getting Started

Include the generated header and link against the release library for your
platform:

```c
#include "nemo_relay.h"
```

Use the FFI surface only when you need a native ABI. Rust, Python, and Node.js
applications should prefer the supported packages for those languages.

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
