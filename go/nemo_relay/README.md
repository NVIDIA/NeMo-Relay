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
> `Initialize` lifecycle needs a real consumer to validate
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
- `InspectLLMContent` selects a Go-owned snapshot from a normalized codec
  annotation. Read its typed parts and locations or call `Text()` for display.
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

## Application Execution Context

Use `WithApplicationExecutionContext` on a managed tool call to give execution
middleware application-owned information separately from the tool's arguments:

```go
result, err := nemo.ToolCallExecute(
    "write_file",
    json.RawMessage(`{"path":"notes.txt","content":"hello"}`),
    executeWrite,
    nemo.WithApplicationExecutionContext(&nemo.ApplicationExecutionContext{
        Data: map[string]json.RawMessage{
            "host_approval": json.RawMessage(`{"approval_id":"review-123","source":"application"}`),
        },
    }),
)
```

Here `executeWrite` is the application's tool callback. Middleware reads the
separate object from `ctx.ExecutionContext`; the business callback receives
only the tool arguments. Context is not automatically added to lifecycle
events or inherited by nested calls. Application assertions are not proof of
approval: middleware must decide which application sources and claims it trusts.

`ApplicationExecutionContext` reserves outer fields for the Relay API. Application
fields belong in its `Data` map; middleware owns their schema, interpretation and
trust checks. Relay does not interpret approval claims or other application data.
`WithToolExecutionContext(json.RawMessage(...))` remains available
for raw JSON producers and follows the same runtime schema validation.

The option snapshots its input and can be reused. A nil value omits context;
`{}` supplies an empty context and `{"data":{}}` explicitly supplies empty application data. Unknown outer fields and non-object `data` are rejected. Relay rejects invalid or non-object JSON, objects
over 16 KiB of serialized JSON, and nesting beyond 64 container levels before
middleware or execution. Manual `ToolCall` and `ToolCallEnd` reject this option.
Build the matching C FFI library when updating this source-first binding.

## LLM Codecs and Child Scope Stacks

Use `WithLLMCodec` with an owned built-in codec handle or a custom `CodecFunc`.
Built-ins include `NewOpenAIChatCodec`, `NewOpenAIResponsesCodec`,
`NewAnthropicMessagesCodec`, `NewGeminiGenerateContentCodec`, and
`NewOCIGenAIChatCodec`. Handles expose `Decode`, `Encode`, and `DecodeResponse`
for standalone JSON transformations. `Encode` returns an `LLMRequest` with
headers and content. Built-in handles preserve their codec identity in sanitizer
and execution-intercept contexts.

Pass the normalized JSON from `Decode` or `DecodeResponse` to
`InspectLLMContent` to read its content. The returned snapshot owns its parts
and source locations. It stays valid after the codec closes or the callback
returns. `Text()` joins text and refusal parts. Tool and media payloads remain
available as typed parts and do not contribute display text.

`NewCustomCodec(request, response)` creates an owned request codec, response
codec, or both. Request callbacks provide `Decode` and `Encode` together.
Response decoders receive provider JSON and return annotated response JSON.
Callbacks must support concurrent calls. `NewCustomResponseCodec` is a
response-only helper. Use `WithLLMCodec` and `WithLLMResponseCodec` to select
owned handles. Call `codec.Close()` to release a handle. Active calls keep their
own references. You can call `Close` more than once. Other operations on a
closed handle return an error.

Manual `LlmCall` accepts a request codec. `LlmCallEnd` accepts a response codec.
Use `WithLLMAnnotatedRequest` on Start or `WithLLMAnnotatedResponse` on End when
the integration already has an annotation. Managed execute functions reject
these options. The following example uses codecs with a manual call:

```go
codec := nemo_relay.NewOpenAIChatCodec()
defer codec.Close()
handle, err := nemo_relay.LlmCall("chat", request, nemo_relay.WithLLMCodec(codec))
if err != nil {
    return err
}
// Call the provider to get response JSON.
return nemo_relay.LlmCallEnd(handle, response, nemo_relay.WithLLMResponseCodec(codec))
```

Inside an execution intercept, call
`context.RecordOptimizationContribution(contribution)` to attach optimization
evidence to the managed call. It returns acceptance and an error. A false
acceptance value leaves provider execution unaffected. The context expires when
the callback returns. Later use returns `ErrLLMCodecExpired`. Call
`context.RetainOptimizationRecorder()` before returning to obtain an owned
recorder for decisions made during later stream reads. Store it with the source,
record from `Next`, and release it with `Close`. Recording after the managed
call ends returns false.

Call `ForkScopeStack()` in a bound parent goroutine before starting child work.
It saves the current parent and creates a separate child stack. Use
`child.Run(...)` in the child goroutine and call `child.Close()` after the child
finishes. Scope-local registrations are not copied.

## Streaming and Context Cancellation

`LlmStreamCallExecute` accepts `LLMStreamExecutionFunc`, which opens an owned
`LLMStreamSource`. A source implements this interface:

```go
type LLMStreamSource interface {
    Next(context.Context) (json.RawMessage, error)
    Close() error
}
```

`Next` returns one chunk or `io.EOF` and must honor context cancellation. Relay
serializes reads and closes the source once, after reads finish. Close errors
reach the managed stream's `Close`.

Owned callbacks must follow their argument, result, and ownership rules. They
can use the capabilities passed to them, such as a continuation, codec,
recorder, upstream source, or `PluginContext`. They can also emit observability
events. Other Relay API calls are invalid within these callbacks. This includes:

- Starting managed calls or controlling managed streams.
- Changing scopes or adding registrations outside a supplied `PluginContext`.
- Flushing or shutting down subscribers.

Close managed streams and flush subscribers from outside owned callbacks.

Register `LLMStreamExecutionInterceptFunc` for all calls, an active scope, or a
plugin to wrap a complete source. The returned source can wait until a read to
call `next(request)`. This continuation stays valid until the returned source
closes. Interceptors own sources returned by `next` and must close any they
discard. Existing single-value execution callbacks remain usable through
`LLMExecutionFunc(fn).AsStream()` and
`LLMExecutionInterceptFunc(fn).AsStream()`, preserving the remaining upstream
chunks.

Use `LlmCallExecuteContext`, `ToolCallExecuteContext`, and
`LlmStreamCallExecuteContext` for cancellation and deadlines. Context-aware
unary providers receive the same context and must cancel their provider work
when the context ends. Relay cannot force callback code to stop. The streaming
context covers setup and subsequent reads. `stream.NextContext(ctx)` adds a
deadline to one wait. Cancelling it closes the entire stream. The context APIs
preserve `context.Canceled` and `context.DeadlineExceeded`. The `llm` and
`tools` shorthand packages expose matching context methods.

## Event Metadata Injection

Use `RegisterEventMetadataInjector` for application-wide metadata. The callback
receives an owned event snapshot and returns proposed additions. Relay validates
the map, preserves existing metadata, and applies injectors in priority order.
Returning an error rejects only that callback's additions. Relay still delivers
the event.

The following Go example registers a metadata injector for all events:

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
	ValueType: nemo.MetricValueTypeU64, Value: json.Number("900"),
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


## Errors

Native calls that return a status code include a `*RelayError` in the error
chain. It includes the native code and message. Use these Go helpers to inspect it:

- Check a status with `errors.Is(err, nemo.ErrAlreadyExists)`.
- Read `Code` and `Message` with `errors.As(err, &nativeError)`.
- Check native cancellation with `errors.Is(err, context.Canceled)`.

Pointer-returning C calls and stream iteration failures provide a message
without a status code. These errors use `StatusUnknown` and match
`ErrUnknownStatus`. Go validation can return other error types. Context-aware
provider errors keep their Go error causes. Stream read, collector, finalizer,
and cleanup errors also keep their Go causes. If `Close` combines failures,
the returned error includes every selected cause. Use `errors.Is` or `errors.As` to inspect them. Retries keep
the cause of the selected failure, even when several errors share a message.

## Requests and Handle Fields

`LLMRequest` is the Go request type for lifecycle calls, codecs, and middleware.
Its `Headers` and `Content` fields hold JSON. To create a request directly,
supply a headers object and content in the provider's format:

```go
request := nemo_relay.LLMRequest{
    Headers: json.RawMessage(`{}`),
    Content: json.RawMessage(`{"model":"example-model","messages":[]}`),
}
```

You can also call `NewLLMRequest(headers, content)` with two `json.RawMessage`
values. It validates the JSON and returns `(LLMRequest, error)`. Check the error
before using the request. The constructor uses these defaults:

- Nil headers become an empty JSON object.
- Nil content becomes JSON null.

Request sanitizer results and callback-scoped request codecs use the same
defaults. To omit observability data from a request sanitizer, return `omit =
true`.

JSON serialization errors keep their Go error cause. Native validation errors
return `*RelayError` with `StatusUnknown`. The constructor releases its internal
C request handle before returning.

Scope, tool, and LLM handles provide these fields:

- `StartedAt()` returns an RFC 3339 string, as `Event.Timestamp()` does.
- `Data()` and `Metadata()` return JSON, or nil if the field is absent.
- `ToolHandle.ToolCallID()` returns the tool call ID.
- `LLMHandle.ModelName()` returns the model name.

The string getters return an empty string if the field is absent. These getters
read stored handle fields. Event sanitizers do not change these values.

## Event and Export Options

`EmitEvent` accepts these category options:

- `WithEventCategory` accepts an `EventCategory` constant or a custom string.
- `WithEventCategoryProfile(json.RawMessage(...))` sets a profile. Relay
  validates known fields and keeps unknown keys. Nil omits the profile.

`NewAtifExporter` accepts these options:

- `WithAtifTrajectoryID` sets the trajectory ID. It defaults to the session ID.
- `WithAtifToolDefinitions` sets the agent's tool definitions. Nil omits the
  field. A non-nil empty slice exports an empty array.
- `WithAtifAgentExtra` sets extra agent JSON. Nil omits the field.

`MetricMeasurement.Value` uses `json.Number` to keep integer precision. Set
`ValueType` to the matching native numeric type. Relay checks that the value
fits the metric's allowed range. Metric attributes use
`map[string]json.RawMessage`. Plugin configuration, event metadata, and
structured logging fields accept maps with values of different types.

## Stream Results and Cleanup

Streaming collectors return `error`. Finalizers return `(json.RawMessage,
error)`. Use these return values to report failures:

- A collector error stops collection. `Next` returns the error, or `Close`
  returns it if `Next` has not delivered it.
- A finalizer error reaches `Close`. Relay marks the end event as `ERROR` and
  omits the aggregate. It discards any partial result returned with the error.
- A successful nil finalizer result means JSON null.

Provider cleanup restores the scope and trace context from source creation. This
also applies during cancellation. Events emitted by the source's `Close` keep
their LLM parent and pass through the scope's event sanitizers.

An `.AsStream()` interceptor closes its previous upstream source before calling
`next` again. If cleanup fails, `next` returns that error without opening
another source. If the interceptor also fails, the returned error includes both
failures.

Call the managed stream's `Close` outside owned callbacks to ensure provider
cleanup runs. If a stream becomes unreachable, garbage collection cancels its
provider context and releases its native handle. A live parent context does not
keep the stream alive. This also applies if a provider source or factory holds a
reference to the stream. Garbage collection can skip provider callbacks whose Go
state is no longer live, so use explicit `Close` to ensure they run.

## Cache Telemetry

Cache telemetry has optional `*CacheMissReason` and `*CacheMissDiagnosis`
fields. `CacheMissReason.Reason` uses the native reason constants. Set
`Description` when the reason is `CacheMissReasonOther`.

Use a type switch on `diagnosis.Evidence` to read one of these evidence types:

- `*CacheMissPrefixMismatchEvidence`
- `*CacheMissBelowMinimumThresholdEvidence`
- `*CacheMissRetentionExpiredEvidence`
- `*CacheMissUnknownEvidence`

These types keep the native `reason` and `kind` JSON tags, including zero
values. The `adaptive` shorthand package also exports these types.

## Optimization Records for Manual Calls

Manual LLM calls can record optimization evidence:

1. After `LlmCall`, call `handle.RetainOptimizationRecorder()`.
2. Call the recorder's `RecordOptimizationContribution` method.
3. Call the recorder's `Close` method when finished.

Closing a recorder releases that reference. `LlmCallEnd` ends recording, emits
the recorded marks, and completes the summary. Later records return false
without an error. Retaining another recorder does not reopen recording.
