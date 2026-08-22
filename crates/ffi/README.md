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
- **Event metadata injection**: Register global, scope-local, or plugin-owned
  callbacks that inspect an immutable event and propose flat metadata additions.
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
#include <string.h>

static char *inject_metadata(void *user_data, const FfiEvent *event) {
    (void)user_data;
    (void)event;
    const char payload[] = "{\"application.region\":\"us-central\"}";
    char *result = malloc(sizeof(payload));
    if (result == NULL) {
        nemo_relay_set_last_error_message("failed to allocate injector result");
        return NULL;
    }
    memcpy(result, payload, sizeof(payload));
    return result;
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
registration. The corresponding deregistration functions remove future
invocations; scope and plugin cleanup also remove their owned registrations.

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
if (nemo_relay_initialize_plugins(config_json, &report_json) != NEMO_RELAY_STATUS_OK) {
    /* inspect nemo_relay_last_error() */
}
nemo_relay_string_free(report_json);
/* Call nemo_relay_clear_plugin_configuration() during process teardown. */
```

Use `nemo_relay_event_v2` for a schema-tagged log mark and
`nemo_relay_metric_json` for an atomically validated metric group. The legacy
`nemo_relay_event` function remains valid for untyped marks:

```c
NemoRelayLogSeverity severity = NEMO_RELAY_LOG_SEVERITY_WARN;
nemo_relay_event_v2(
    "cache-nearly-full", NULL, "{\"entries\":900}",
    "{\"name\":\"example.cache\",\"version\":\"1\"}", NULL, &severity, NULL
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
    0, "info", 0, 0, 0, &logs
))) goto cleanup;
if (!relay_ok(nemo_relay_otel_metric_subscriber_create(
    "http_binary", "http://localhost:4318", NULL, NULL, NULL, NULL, NULL, NULL,
    0, 0, "cumulative", 0, 0, &metrics
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
