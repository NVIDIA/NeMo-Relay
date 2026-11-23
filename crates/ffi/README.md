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
> `nemo_relay_plugin_initialize` lifecycle needs a real consumer
> to validate shutdown, ownership, and error handling before it can be promoted
> to a stable contract.

## ABI and Constructors

The C ABI has one signature for each operation. Exported names have no version
suffix. Rebuild native consumers with the generated header when you update this
experimental ABI.

Constructors include the following functions:

- `nemo_relay_scope_stack_create` accepts optional propagation-context JSON.
  Null creates a new root scope.
- `nemo_relay_atof_exporter_create` accepts the complete `AtofExporterConfig`
  JSON object, including file and stream destinations.
- `nemo_relay_otel_subscriber_create` accepts all event mapping, header, parent trace,
  and TLS settings. A non-null output directory selects OTLP file export.
  Null selects network export. Log and metric subscribers each have their own
  constructor with complete settings and optional TLS.
- `nemo_relay_codec_new(provider, callbacks)` selects one of five built-in
  request/response codecs, including OCI GenAI, with a null callback pointer.
  `NEMO_RELAY_CODEC_CUSTOM` accepts `NemoRelayCodecCallbacks` for custom request
  codecs, response decoders, or both. Release the result with
  `nemo_relay_codec_free`.
  Relay owns the callback state when the call starts, even if validation fails.
  Its destructor runs once after all native codec references are released.

Managed tool, LLM, and streaming LLM execute functions accept `skip_stages`
directly. Pass zero to run all middleware stages.

## LLM Codecs and Optimization Evidence

`nemo_relay_codec_decode`, `nemo_relay_codec_encode`, and
`nemo_relay_codec_decode_response` operate on owned codec handles. Encode
returns complete request JSON containing both `headers` and `content`. Free
returned strings with `nemo_relay_string_free` and check null results with
`nemo_relay_last_error`. Custom request encode callbacks continue to return only
the content JSON. Relay keeps the original headers.

Managed LLM calls accept a borrowed request-codec handle or request callbacks.
Passing both returns `InvalidArg`. Request decode and encode callbacks must be
supplied together. A response-codec handle can refer to a built-in codec or a
custom response decoder. A handle must support the direction requested. Native
calls retain the codec references they need after returning.

Manual `nemo_relay_llm_call` accepts a request codec. `nemo_relay_llm_call_end`
accepts a response codec. Both accept optional annotation JSON computed by the
caller. These options affect observability and the codec context passed to
sanitizers. They do not change the provider's request or response.

An execution intercept receives a borrowed `optimization_recorder` in its
`NemoRelayLlmExecutionContext`. Call `nemo_relay_llm_optimization_record` with
contribution JSON in the runtime format and a boolean output to record evidence
on that managed call. A false output means Relay did not accept the record. This
does not fail the call. The borrowed recorder and other context pointers expire
when the callback returns. Retain a recorder with
`nemo_relay_llm_optimization_recorder_retain` before returning to record
evidence later during stream reads. Release the recorder with
`nemo_relay_llm_optimization_recorder_free`. Recording ends with the managed
call. Later records return false.

## LLM Content Inspection

`nemo_relay_inspect_llm_content_json` reads normalized JSON from a codec. It can
select content from a request, response, message, or standalone content value.
Use the `NEMO_RELAY_CONTENT_SOURCE_*` and `NEMO_RELAY_CONTENT_SELECT_*` constants
to choose the source and selection.

The result contains ordered parts, source locations, and text parts. An absent
selection returns success with a null output. A selected item with no text still
returns a snapshot. Free each non-null result with `nemo_relay_string_free`.

## Streaming Sources and Cancellation

`nemo_relay_stream_source_create` constructs an owned, pull-based source from
read, cancel, close, and free callbacks. The callbacks follow these rules:

- Read returns 1 and an owned JSON string for a chunk, 0 for EOF, or -1 after
  setting the last error. Reads run one at a time on blocking workers. They keep
  the active Relay scope and trace context.
- Cancel returns promptly and unblocks a pending read.
- Close runs once after reads stop. Its events use the scope captured when the
  source was created, even if the caller later changes scopes.
- Free runs once after close and after the last native reference is released.

Owned callbacks must follow their argument, result, and ownership rules. They
can use the capabilities passed to them, such as a continuation, codec,
recorder, upstream source, or plugin context. They can also emit observability
events. Other Relay API calls are invalid within these callbacks. This includes:

- Starting managed calls or controlling managed streams.
- Changing scopes or adding registrations outside a supplied plugin context.
- Flushing or shutting down subscribers.

Cancel and close managed streams from outside owned callbacks.

Pass a `NemoRelayOptionalLlmStreamExecCb` factory to the existing
`nemo_relay_llm_stream_call_execute` operation, with a null single-value
callback. The factory returns its owned source and transfers ownership to Relay.
The single-value provider route remains available for callers that produce one
JSON value.

Global, scope-local, and plugin-context stream intercept registrations accept a
`NemoRelayOptionalLlmStreamExecInterceptCb`. It receives a continuation that returns a
complete source. Return that source or a replacement to Relay. Close and free
any owned source that is discarded. To open the upstream source after the
interceptor returns, retain its next-context with
`nemo_relay_stream_continuation_retain`, invoke it with
`nemo_relay_stream_continuation_call`, and release it with
`nemo_relay_stream_continuation_free`. Retained continuations keep their
original scope and trace context. Callback-scoped codec capabilities still
expire. Language bindings can use `nemo_relay_stream_continuation_callback_data`
during an interceptor callback to borrow user data for the managed stream's
collector and finalizer. Retained continuations keep that state alive, even if
both optional callbacks are null.

Create a cancellation signal with `nemo_relay_cancellation_create` and pass it
to managed tool or LLM execution, or streaming setup. Another thread can call
`nemo_relay_cancellation_cancel`. Callbacks can poll
`nemo_relay_cancellation_is_cancelled`. Callbacks must cooperate with
cancellation. Relay waits for running foreign callbacks to return before
releasing their state. A cancelled managed operation returns
`NEMO_RELAY_STATUS_CANCELLED`. Release the signal with
`nemo_relay_cancellation_free` after foreign users have stopped. After streaming
setup completes, use `nemo_relay_stream_cancel` or close the returned stream to
stop consumption and await producer cleanup.

## Why Use It?

Use the FFI for these tasks:

- **Expose NeMo Relay to native consumers**: Call the shared Rust runtime from
  C-compatible hosts and downstream language bindings.
- **Build on one ABI**: Keep native integrations aligned with the same scope,
  middleware, lifecycle event, and observability contract.
- **Consume a generated C header**: Use the committed `nemo_relay.h` surface
  produced by the crate build.
- **Work source-first**: Use this experimental surface when Rust, Python, and
  Node.js packages are not the right integration layer.

## What You Get

The FFI provides these features:

- **Exported `nemo_relay_*` symbols**: APIs for scopes, tool calls, LLM calls,
  middleware, subscribers, plugins, observability exporters, and scope stack
  isolation.
- **Event metadata injection**: Register global, scope-local, or plugin-owned
  callbacks that inspect an immutable event and propose flat metadata additions.
- **Typed OpenTelemetry export**:
  `nemo_relay_otel_subscriber_create` constructs one `full`, `gen_ai`, or
  `openinference` trace subscriber. Independently managed log and metric
  subscribers use `nemo_relay_otel_log_subscriber_create` and
  `nemo_relay_otel_metric_subscriber_create`.
- **Structured marks and metrics**: `nemo_relay_event` accepts optional data
  schema JSON and `NemoRelayLogSeverity`.
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

## Event Metadata Injection

An event metadata injector receives a borrowed `FfiEvent` and returns a
heap-allocated JSON object containing proposed metadata additions. Relay frees
the returned string, validates the object, and inserts only keys that are not
already present. Return null after calling
`nemo_relay_set_last_error_message` to reject that callback's additions while
allowing the event to continue through sanitization and delivery.

The following C example registers application-wide metadata injection:

```c
#include <stdio.h>
#include <stdlib.h>

static char *inject_metadata(void *user_data, const FfiEvent *event) {
    (void)user_data;
    (void)event;
    const char payload[] = "{\"application.region\":\"us-central\"}";
    return nemo_relay_string_clone(payload);
}

NemoRelayStatus status = nemo_relay_register_event_metadata_injector(
    "application-metadata", 10, inject_metadata, NULL, NULL
);
if (status != NEMO_RELAY_STATUS_OK) {
    fprintf(stderr, "registration failed: %s\n", nemo_relay_last_error());
    return EXIT_FAILURE;
}
/* Emit scopes and marks while the callback is registered. */
status = nemo_relay_deregister_event_metadata_injector("application-metadata");
if (status != NEMO_RELAY_STATUS_OK) {
    fprintf(stderr, "cleanup failed: %s\n", nemo_relay_last_error());
    return EXIT_FAILURE;
}
```

Use `nemo_relay_scope_register_event_metadata_injector` for an active scope or
`nemo_relay_plugin_context_register_event_metadata_injector` during plugin
registration. The matching deregistration functions prevent future calls. Scope
and plugin cleanup also remove registrations that they own.

## OTLP Logs and Metrics

The raw C ABI remains experimental and source-first. Configure plugin-managed
export with the same version-4 JSON document used by every binding. Leaving
the log and metric endpoint lists absent derives their destinations from the
trace endpoint:

```c
const char *config_json =
    "{\"version\":1,\"components\":[{\"kind\":\"observability\",\"config\":{"
    "\"version\":4,\"opentelemetry\":{\"enabled\":true,\"endpoints\":[{"
    "\"type\":\"gen_ai\",\"endpoint\":\"http://localhost:4318/v1/traces\"}],"
    "\"logs\":{\"enabled\":true},\"metrics\":{\"enabled\":true}}}}]}";
char *report_json = NULL;
FfiPluginHostActivation *activation = NULL;
if (nemo_relay_plugin_initialize(config_json, NULL, &activation, &report_json) != NEMO_RELAY_STATUS_OK) {
    /* inspect nemo_relay_last_error() */
} else {
    nemo_relay_string_free(report_json);
    /* Close, then free, the activation during process teardown. */
    if (nemo_relay_plugin_host_activation_close(activation) != NEMO_RELAY_STATUS_OK) {
        /* inspect nemo_relay_last_error(); do not free an activation that did not close */
    } else {
        nemo_relay_plugin_host_activation_free(&activation);
    }
}
```

Use `nemo_relay_event` for a schema-tagged log mark and `nemo_relay_metric_json`
for an atomically validated metric group. Pass null schema and severity pointers
to emit an untyped mark:

```c
NemoRelayLogSeverity severity = NEMO_RELAY_LOG_SEVERITY_WARN;
nemo_relay_event(
    "cache-nearly-full", NULL, "{\"entries\":900}",
    "{\"name\":\"example.cache\",\"version\":\"1\"}", NULL, &severity, NULL, NULL, NULL
);
nemo_relay_metric_json(
    "cache-entries", NULL,
    "[{\"name\":\"example.cache.entries\",\"kind\":\"gauge\","
    "\"value_type\":\"u64\",\"value\":900}]",
    NULL, NULL
);
```

Create direct log and metric subscribers independently. Register each before
emitting marks, then deregister, force-flush, shut down, and free it during
graceful teardown. Runtime diagnostics are a caller-owned, bounded JSON array
of `{"code":"...","message":"...","count":N}` entries; release the result
with `nemo_relay_string_free` after parsing it. Check every returned status
before continuing. On failure, read `nemo_relay_last_error()` immediately,
before another FFI call clears the thread-local message:

```c
#include <stdbool.h>
#include <stdio.h>

static bool relay_ok(NemoRelayStatus status) {
    if (status == NEMO_RELAY_STATUS_OK) {
        return true;
    }
    const char *message = nemo_relay_last_error();
    fprintf(stderr, "nemo-relay: %s\n", message != NULL ? message : "unknown error");
    return false;
}

struct FfiOpenTelemetryLogSubscriber *logs = NULL;
struct FfiOpenTelemetryMetricSubscriber *metrics = NULL;
bool logs_registered = false;
bool metrics_registered = false;

/* Equivalent explicit OTLP/HTTP paths are /v1/logs and /v1/metrics, respectively. */
if (!relay_ok(nemo_relay_otel_log_subscriber_create(
    "http_binary", "http://localhost:4318", NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, 0, "info", 0, 0, 0, 60000, NULL, &logs
))) goto cleanup;
if (!relay_ok(nemo_relay_otel_metric_subscriber_create(
    "http_binary", "http://localhost:4318", NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, 0, 0, "cumulative", 0, 0, NULL, &metrics
))) goto cleanup;
if (!relay_ok(nemo_relay_otel_log_subscriber_register(logs, "otlp-logs"))) goto cleanup;
logs_registered = true;
if (!relay_ok(nemo_relay_otel_metric_subscriber_register(metrics, "otlp-metrics"))) goto cleanup;
metrics_registered = true;

char *diagnostics_json = NULL;
if (relay_ok(nemo_relay_otel_log_subscriber_runtime_diagnostics_json(
    logs, &diagnostics_json
))) {
    /* Parse diagnostics_json, then release it. */
    nemo_relay_string_free(diagnostics_json);
}

cleanup:
if (metrics_registered) {
    (void)relay_ok(nemo_relay_otel_metric_subscriber_deregister("otlp-metrics"));
}
if (metrics != NULL) {
    (void)relay_ok(nemo_relay_otel_metric_subscriber_force_flush(metrics));
    (void)relay_ok(nemo_relay_otel_metric_subscriber_shutdown(metrics));
    nemo_relay_otel_metric_subscriber_free(metrics);
}
if (logs_registered) {
    (void)relay_ok(nemo_relay_otel_log_subscriber_deregister("otlp-logs"));
}
if (logs != NULL) {
    (void)relay_ok(nemo_relay_otel_log_subscriber_force_flush(logs));
    (void)relay_ok(nemo_relay_otel_log_subscriber_shutdown(logs));
    nemo_relay_otel_log_subscriber_free(logs);
}
```

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

The release profile disables stripping for the FFI package to avoid a
[Mach-O string-table alignment bug](https://github.com/rust-lang/rust/issues/157750)
in the pinned Rust toolchain on macOS. Size optimization and thin LTO still apply.

## Getting Started

Include the generated header and link against the release library for your
platform:

```c
#include "nemo_relay.h"
```

Use the FFI surface only when you need a native ABI. Rust, Python, and Node.js
applications should prefer the supported packages for those languages.

## Event and Export Options

`nemo_relay_event` accepts an optional category string and category-profile JSON
before its timestamp argument. Categories can use custom strings. Relay
validates known profile fields and keeps unknown keys. Pass null to omit a
field.

`nemo_relay_atif_exporter_create` accepts these optional fields before its
output pointer:

- `trajectory_id` sets the trajectory ID. Null uses the session ID.
- `tool_definitions_json` sets a JSON array of agent tool definitions.
- `agent_extra_json` sets extra agent JSON.

Null JSON pointers omit the matching agent fields.

## Callback Error Causes

Language bindings can attach an opaque ID to a callback failure:

- Set the message with `nemo_relay_set_last_error_message`.
- Set a nonzero ID with `nemo_relay_set_last_error_cause_id` on the same thread.
- Read `nemo_relay_last_error_cause_id` with the returned error message on the
  calling thread. Zero means the error has no callback ID.

The binding owns the data behind each ID. Relay carries the ID with the selected
failure across worker threads. Setting or clearing the error message clears the
ID. These helpers do not change the error text.

## Requests and Handle Fields

`nemo_relay_llm_request_new` validates request JSON. Headers must be a JSON
object or JSON null. Omitted or JSON-null headers become an empty object.
Omitted content becomes JSON null. On failure, the function returns null and
sets `nemo_relay_last_error()`. Read that message at once on the same thread.

Handle getters provide these fields:

- `nemo_relay_{scope,tool,llm}_handle_started_at` returns an RFC 3339 start time.
- Tool and LLM handles provide `_data` and `_metadata` JSON getters.
- `nemo_relay_tool_handle_tool_call_id` returns the tool call ID.
- `nemo_relay_llm_handle_model_name` returns the model name.

Free each non-null result with `nemo_relay_string_free`. A null handle or absent
field returns null. An explicit JSON null returns the string `"null"`. These
getters read stored handle fields. Event sanitizers do not change these values.

## Optimization Records for Manual Calls

Use these steps to record evidence for a manual LLM call:

1. Call `nemo_relay_llm_handle_retain_optimization_recorder(handle, &recorder)`.
   Check the status before using the output. Failure clears a non-null output
   slot.
2. Record evidence with `nemo_relay_llm_optimization_record`.
3. Release each owned recorder with
   `nemo_relay_llm_optimization_recorder_free`.

The recorder stays valid after its original handle is freed. Ending the LLM call
ends recording. Later records return success with `accepted = false`.

## Callback Ownership and Cleanup

Allocate callback result strings with `nemo_relay_string_clone`. Relay frees
these strings after use. Callbacks can use this helper and the error helpers
required by their contracts.

Stream-source cancel and close callbacks restore the scope and trace context
from source creation. This also applies on blocking workers and before the first
read. Cleanup events keep their original parent and pass through the scope's
event middleware.

Optional callbacks use `NemoRelayOptional*` function-pointer types and accept
`NULL`. Streaming execution takes ownership of all callback state when the call
starts. This includes codec state if provider validation fails. Owned callbacks
follow the rules in “Streaming Sources and Cancellation.” Beyond those rules,
they can only call Relay APIs that emit observability events.

## Documentation

Refer to the [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay).
