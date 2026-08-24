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

# NeMo Relay Go Binding

The Go binding exposes NeMo Relay runtime APIs through CGo and the raw
`nemo-relay-ffi` library. Use it when a Go application or integration needs the
same scope, middleware, lifecycle event, and observability model used by the
Rust runtime.

This binding is experimental and source-first. Rust, Python, and Node.js are the
primary supported surfaces.

> **DO NOT TREAT AS PRODUCTION-READY:** the experimental
> `InitializeWithDynamicPlugins` lifecycle needs a real consumer to validate
> shutdown, ownership, and error handling before it can be promoted to a stable
> contract.

## Why Use It?

Use the Go binding for the following tasks:

- **Use NeMo Relay from Go**: Group agent, tool, and LLM work into the same
  scope and lifecycle model as the Rust runtime.
- **Bridge through CGo and FFI**: Consume the shared runtime through the
  repository-maintained `nemo-relay-ffi` layer.
- **Observe runtime behavior**: Register subscribers for scope, tool, LLM,
  and mark events emitted by the runtime.
- **Evaluate an experimental binding**: Use the source-first Go surface when
  a Go integration needs NeMo Relay semantics.

## What You Get

The Go package provides the following capabilities:

- **Scope, tool, and LLM helpers**: Managed lifecycle APIs backed by the
  shared Rust runtime.
- **Middleware APIs**: Guardrails and intercepts for request rewriting,
  blocking, sanitization, and execution wrapping, including mark and scope
  event sanitizers at global, scope-local, and plugin-context levels.
- **Event metadata injection**: Global, scope-local, and plugin-context
  callbacks can inspect immutable events and propose flat metadata additions.
- **Event subscribers**: Runtime lifecycle callbacks for observability and
  diagnostics.
- **Typed OpenTelemetry export**: `NewOpenTelemetryConfig` returns configuration
  for a `full`, `gen_ai`, or `openinference` trace subscriber.
  `NewOpenTelemetrySubscriber` constructs the independently managed trace
  subscriber. `NewOpenTelemetryLogSubscriber` and
  `NewOpenTelemetryMetricSubscriber` construct independent log and metric
  subscribers.
- **Structured marks and metrics**: `EmitEvent` accepts data-schema and
  severity options, and `EmitMetric` validates a complete metric-measurement
  group before emission.
- **Observability configuration version 4**: `NewObservabilityConfig` and its
  OTLP helpers configure trace, log, and metric plugin-managed exporters.
- **Convenience subpackages**: Short imports for scopes, tools, LLM calls,
  guardrails, intercepts, subscribers, plugins, and adaptive helpers.
- **Local source-first workflow**: Build the FFI library locally, then test or
  consume the Go module from the checkout.

Go middleware callbacks are synchronous. Relay waits for each callback on a
native thread, so blocking I/O and other long-running callback work occupy that
thread and can reduce middleware throughput. The Go binding does not provide
completion-based middleware registration.

## Event Metadata Injection

Use `RegisterEventMetadataInjector` for application-wide metadata. The callback
receives an owned event snapshot and returns proposed additions. Relay validates
the map, preserves existing metadata, and applies injectors in priority order.
Returning an error rejects only that callback's additions; the event is still
delivered.

The following Go example registers application-wide metadata injection:

```go
err := nemo.RegisterEventMetadataInjector(
	"application-metadata",
	10,
	func(event nemo.Event) (nemo.EventMetadata, error) {
		return nemo.EventMetadata{
			"application.event_kind": event.Kind(),
		}, nil
	},
)
if err != nil {
	log.Fatal(err)
}
defer nemo.DeregisterEventMetadataInjector("application-metadata")
```

Use `ScopeRegisterEventMetadataInjector` for an active scope. Plugins use
`PluginContext.RegisterEventMetadataInjector` so component name qualification
and rollback cleanup apply to the registration.

## OTLP Logs and Metrics

Use `EmitEvent` with `WithEventDataSchema` and `WithEventSeverity` for a typed
mark. Use `EmitMetric` for an atomically validated metric group. The examples
below assume `nemo` is imported as
`nemo "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"`:

```go
if err := nemo.EmitEvent(
	"cache-nearly-full",
	nemo.WithEventData(json.RawMessage(`{"entries": 900}`)),
	nemo.WithEventDataSchema(nemo.DataSchema{Name: "example.cache", Version: "1"}),
	nemo.WithEventSeverity(nemo.LogSeverityWarn),
); err != nil {
	log.Fatal(err)
}

if err := nemo.EmitMetric("cache-entries", []nemo.MetricMeasurement{{
	Name: "example.cache.entries", Kind: nemo.MetricKindGauge,
	ValueType: nemo.MetricValueTypeU64, Value: uint64(900),
}}); err != nil {
	log.Fatal(err)
}
```

Create direct OTLP log and metric subscribers separately. A bare OTLP/HTTP
authority derives `/v1/logs` or `/v1/metrics` for its selected signal. Register
the subscriber before emitting marks. During graceful shutdown, deregister it,
force-flush it, shut it down, and call `Close` to free its FFI handle:

```go
// Equivalent explicit OTLP/HTTP paths are /v1/logs and /v1/metrics, respectively.
logs, err := nemo.NewOpenTelemetryLogSubscriber(
	nemo.NewOpenTelemetryLogConfig("http://localhost:4318"),
)
if err != nil {
	log.Fatal(err)
}
if err := logs.Register("otlp-logs"); err != nil {
	logs.Close()
	log.Fatal(err)
}
metrics, err := nemo.NewOpenTelemetryMetricSubscriber(
	nemo.NewOpenTelemetryMetricConfig("http://localhost:4318"),
)
if err != nil {
	_ = logs.Deregister("otlp-logs")
	logs.Close()
	log.Fatal(err)
}
if err := metrics.Register("otlp-metrics"); err != nil {
	metrics.Close()
	_ = logs.Deregister("otlp-logs")
	logs.Close()
	log.Fatal(err)
}
defer func() {
	_ = logs.Deregister("otlp-logs")
	_ = logs.ForceFlush()
	_ = logs.Shutdown()
	logs.Close()
	_ = metrics.Deregister("otlp-metrics")
	_ = metrics.ForceFlush()
	_ = metrics.Shutdown()
	metrics.Close()
}()

diagnostics, err := logs.RuntimeDiagnostics()
if err != nil {
	log.Fatal(err)
}
for _, diagnostic := range diagnostics {
	log.Printf("%s: %s", diagnostic.Code, diagnostic.Message)
}
```

Direct trace, log, and metric subscribers expose `RuntimeDiagnostics` for
bounded exporter and event-processing failures.

For plugin-managed export, `NewObservabilityConfig` creates version 4
configuration. Enable `Logs` and `Metrics` while leaving their `Endpoints`
pointers nil to derive signal destinations from each trace endpoint:

```go
config := nemo.NewObservabilityConfig()
otel := nemo.NewObservabilityOpenTelemetryConfig()
otel.Enabled = true
otel.Endpoints = []nemo.ObservabilityOpenTelemetryEndpointConfig{
	nemo.NewObservabilityOpenTelemetryEndpointConfig(
		nemo.OpenTelemetryTypeGenAI,
		"http://localhost:4318/v1/traces",
	),
}
logsConfig := nemo.NewObservabilityOpenTelemetryLogConfig()
logsConfig.Enabled = true
metricsConfig := nemo.NewObservabilityOpenTelemetryMetricConfig()
metricsConfig.Enabled = true
otel.Logs = &logsConfig
otel.Metrics = &metricsConfig
config.OpenTelemetry = &otel
```

In this configuration, Relay derives `/v1/logs` and `/v1/metrics` from the
trace endpoint. Assign an explicit signal endpoint list with
`ObservabilityOpenTelemetrySignalEndpoints` when a signal needs a different
destination. Wrap `config` with `NewObservabilityComponentSpec` before plugin
initialization.

## Installation

Build the FFI library from a repository checkout before using the Go binding:

```bash
git clone https://github.com/NVIDIA/NeMo-Relay.git
cd NeMo-Relay
cargo build --release -p nemo-relay-ffi
```

For a Go application that consumes a local checkout, point the module at the
checked-out binding:

```bash
go mod edit -replace github.com/NVIDIA/NeMo-Relay/go/nemo_relay=../NeMo-Relay/go/nemo_relay
go get github.com/NVIDIA/NeMo-Relay/go/nemo_relay
```

## Getting Started

Run the binding tests from the repository checkout to verify the CGo link path
and the FFI library:

```bash
cd go/nemo_relay
go test ./...
```

Then import the package from application code:

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	nemo "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/scope"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/tools"
)

func main() {
	defer func() {
		if err := nemo.ShutdownLogging(); err != nil {
			log.Printf("shut down NeMo Relay logging: %v", err)
		}
	}()

	if err := nemo.RegisterSubscriber("printer", func(event nemo.Event) {
		fmt.Printf("%s %s\n", event.Kind(), event.Name())
		fmt.Println(string(event.JSON()))
	}); err != nil {
		log.Fatal(err)
	}
	defer nemo.DeregisterSubscriber("printer")
	defer nemo.FlushSubscribers()

	handle, err := scope.Push("demo-agent", nemo.ScopeTypeAgent)
	if err != nil {
		log.Fatal(err)
	}
	defer scope.Pop(handle)

	if err := scope.Event("initialized"); err != nil {
		log.Fatal(err)
	}

	result, err := tools.Execute("search", json.RawMessage(`{"query":"hello"}`), func(args json.RawMessage) (nemo.ToolExecutionResult, error) {
		return nemo.ToolExecutionResult{Result: args}, nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(result.Result))
}
```
