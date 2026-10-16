<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

The following can serve as the implementation-details section of the proposal. The core package is currently named **`nemo-relay`**, located in `crates/core`; the Rust examples below illustrate proposed APIs rather than existing symbols.

**1. Shared contracts and core implementation**

Implement serializable diagnostic contracts in `nemo-relay-types` and the executable foundation in `crates/core::diagnostics`. Generalize the existing independently owned [OpenTelemetry diagnostic recorder](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/core/src/observability/otel_signal.rs#L108), while making recording available throughout startup, operation, and teardown.

The foundation should provide separate write and read interfaces. Components receive scoped recording handles; consumers receive read-only query handles. Operational logging becomes a projection of recorded observations. Diagnostic state updates occur independently of logging configuration and verbosity.

- **Shared contracts:** Define source identity, component identity, diagnostic codes, events, conditions, counters, snapshots, and coverage metadata. Keep synchronization, logger handles, HTTP concepts, and runtime ownership out of `types`.
- **Identity:** Assign an incarnation identity before logging initialization. Give each component a logical identity and a registration generation, so recreated exporters or sinks are distinguishable. Keep request/session correlation outside aggregation keys to bound cardinality.
- **Registration and lifetime:** Register components explicitly and return scoped handles. Closing a component records its termination and retains bounded evidence. Closing a component must not imply that its unresolved conditions recovered.
- **State management:** Separate recent events, active conditions, cumulative counters, and current component facts. Distinguish unavailable, unsupported, never observed, and available values.
- **Resource limits:** Bound registered components, diagnostic keys, retained events, message sizes, and snapshot size. Expose truncation, eviction, rejected observations, and counter saturation through reserved metadata.
- **Recording semantics:** Count an occurrence once. Activating or querying an existing condition must not increment its occurrence count. Recovery requires an observed successful operation or another explicit recovery signal.
- **Compatibility:** Preserve existing subscriber diagnostics and plugin reports as projections. Replace mirrored counting paths with a single recording boundary.
- **Logging integration:** Publish log projections after releasing diagnostic-state locks. Sink failures update state through a path that avoids recursively writing into the failed sink.

Illustrative contracts and recording:

```rust
// nemo-relay-types: serializable contracts.
pub struct DiagnosticSnapshot {
    pub schema_version: u32,
    pub source: SourceIdentity,
    pub captured_at: ObservationTime,
    pub sequence: u64,
    pub components: Vec<ComponentSnapshot>,
    pub coverage: DiagnosticCoverage,
}

// nemo-relay: recorder, ownership, storage, and queries.
let diagnostics = Diagnostics::new(source_identity, limits);
let exporter = diagnostics.register_component(
    ComponentIdentity::exporter("otlp.logs", "collector-a"),
)?;

// Records one occurrence and updates its active condition.
let _receipt = exporter.observe_failure(
    "otel.logs_export_failed",
    SafeMessage::new("The export attempt failed"),
    correlation,
);

// A later observed success resolves the condition; history remains.
exporter.observe_recovery("otel.logs_export_failed");

let snapshot = diagnostics.reader().snapshot(query)?;
```

The common model should describe observed evidence and its coverage. Components and deployment owners retain responsibility for admission, retry, restart, and health policy.

**2. CLI deployment mode integration**

Initialize the diagnostic service at the CLI process boundary, before configuration resolution and logging initialization. Pass its handles through wrapped execution, persistent gateway startup, and supporting commands. The current CLI configures logging before command-start recording ([dispatch implementation](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/cli/src/commands/mod.rs#L149)); this phase establishes the earlier recording boundary.

Provide one reusable HTTP diagnostics module in the CLI crate. Wrapped and persistent gateways mount the same handlers over the core reader, using the daemon’s existing `/_nemo_relay/v1/` namespace. Short-lived commands use the same recording foundation and appropriate command/error output without requiring a listener.

- **Bootstrap:** Start with conservative default storage limits. Apply validated configuration afterward, reporting any evidence lost while operating under bootstrap limits.
- **Logging attachment:** Supply the diagnostic identity to logging so records correlate with the process snapshot. Define bounded startup-event replay and preserve event identities during replay.
- **Optional sinks:** Implement #1297 through explicit required/optional sink policy in core, with CLI/TOML exposure. Optional file-open failures permit startup and remain inspectable; required-sink failures and invalid configuration retain strict behavior.
- **Shared HTTP interface:** Implement common status, component, and event-query behavior with consistent serialization, limits, errors, versioning, and caching rules. `GET /_nemo_relay/v1/status` is the proposed common process-status route.
- **Access policy:** Apply host-appropriate authorization and redaction around the shared handlers. A richer snapshot must not automatically inherit the existing public health probe’s exposure.
- **Helpers:** Record MCP acquisition and hook-forwarding failures locally, reusing existing safe correlation identifiers. Keep diagnostic output separate from protocol stdout.
- **Shutdown:** Record teardown and logging-flush outcomes before releasing diagnostic state. Preserve existing sink-flushing behavior; the diagnostic foundation does not itself establish a whole-process shutdown deadline.

Illustrative process composition:

```rust
let diagnostics = Diagnostics::bootstrap(ProcessRole::Gateway);

let config = resolve_configuration(diagnostics.recorder())?;

let logging = LoggingRuntime::configure_with_diagnostics(
    config.logging,
    diagnostics.component("logging")?,
)?;

let log_projection = diagnostics.attach_log_projection(&logging);

let routes = diagnostics_http::router(
    diagnostics.reader(),
    config.diagnostics.access_policy(),
);

let result = run_gateway(config, routes, diagnostics.clone()).await;

diagnostics.record_shutdown_outcome(&result);
shutdown_logging_and_record_outcome(logging, &diagnostics);
drop(log_projection);
```

Fatal startup failures can be recorded before logger construction, but external evidence still requires a functioning terminal/error output path. No live HTTP API should be assumed after startup aborts.

**3. Daemon broker and worker integration**

Mount the same CLI diagnostics interface in both Relay broker and Relay worker processes. Each process owns its local diagnostic state. The broker additionally maintains a bounded collection of worker snapshots obtained through the generalized worker diagnostics API.

Preserve the distinction between authoritative broker observations and worker-authored evidence. Existing worker status endpoints read [broker registry state](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/cli/src/daemon/broker/registry.rs#L1002); collected worker diagnostics supplement that state.

- **Local instrumentation:** Register broker routing/control components and worker runtime/plugin/exporter components through the common recorder.
- **API consistency:** Share process/component snapshot schemas and handlers. Preserve existing `/healthz`, worker listing, aggregate status, and individual worker status contracts.
- **Assignment versus process state:** Keep broker assignment fields separate from the worker’s own lifecycle and component conditions.
- **Collection:** Query the worker’s shared diagnostics interface through its authenticated listener. Run collection independently of request forwarding and lifecycle control.
- **Freshness and identity:** Retain worker generation, process incarnation, source sequence, source observation time, and broker receipt time. Use broker-local elapsed time to determine freshness.
- **Cache behavior:** Replace cumulative snapshots rather than adding their counters. Reject obsolete generations, bound cache size, and expose missing, stale, unsupported, or evicted coverage.
- **Failures:** A disconnected worker becomes unreachable or stale. Record confirmed exit only where an observer obtains termination evidence.
- **Read-only inspection:** Do not use the existing readiness probe for diagnostic reads: that probe can activate request admission ([worker readiness behavior](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/cli/src/daemon/worker/runtime.rs#L584)).

Illustrative collection:

```rust
let worker_reader = WorkerDiagnosticsClient::new(
    assigned_worker.endpoint(),
    assigned_worker.credentials(),
);

match worker_reader.snapshot(query).await {
    Ok(snapshot) => worker_snapshots.replace_if_current(
        assigned_worker.generation(),
        snapshot,
        broker_clock.now(),
    )?,
    Err(error) => worker_snapshots.record_collection_failure(
        assigned_worker.generation(),
        classify_collection_error(error),
    ),
}

// Combines local evidence and collected views without recounting events.
let response = broker_view.snapshot(
    diagnostics.reader(),
    &registry,
    &worker_snapshots,
)?;
```

The broker can expose worker snapshots through its existing collection API family. Workers expose the same underlying process schema directly, subject to their access policy.

**4. Language bindings integration**

Expose the core inspection model through thin binding adapters. Bindings should reuse core registration, retention, counting, and lifecycle semantics rather than implement additional diagnostic stores.

Provide general process/runtime inspection separately from existing per-subscriber diagnostic methods. Applications should be able to inspect logging and initialization evidence without holding an exporter handle.

- **Rust:** Re-export shared types through the public diagnostics module and expose reader/recorder handles through established runtime ownership.
- **Python:** Add snapshot/query wrappers and public type declarations in the PyO3 bridge and Python package.
- **Node.js:** Add equivalent methods and declarations. Preserve evidence across environment registration and avoid releasing shared state while another Node environment remains active.
- **C FFI and Go:** Expose the common query model through established JSON/string ownership conventions and Go wrappers, retaining their existing experimental support status.
- **Initialization:** Establish diagnostic ownership before binding logging initialization. Repeated initialization must reuse the appropriate service rather than reset history.
- **Data representation:** Specify lossless counter and sequence representation across JSON and JavaScript. Preserve unavailable values explicitly rather than converting them to zero.
- **Compatibility:** Keep existing `runtime_diagnostics()` and `runtimeDiagnostics()` result shapes as compatibility projections.
- **Errors and teardown:** Translate query errors consistently and retain evidence through cleanup. Import or initialization failures still need the binding’s normal exception/error path.

Illustrative bridge implementation:

```rust
// Common Rust query implementation used by binding adapters.
fn read_runtime_diagnostics(
    query: DiagnosticQuery,
) -> FlowResult<DiagnosticSnapshot> {
    let reader = shared_runtime::diagnostics_reader()?;
    reader.snapshot(query)
}

// Illustrative Python adapter.
fn python_snapshot(query: PyDiagnosticQuery)
    -> PyResult<PyDiagnosticSnapshot>
{
    let snapshot =
        read_runtime_diagnostics(query.try_into()?)
            .map_err(to_python_error)?;

    Ok(PyDiagnosticSnapshot::from(snapshot))
}
```

Python currently initializes logging during module import, while Node tracks environment lifetimes ([Python initialization](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/python/src/lib.rs#L46), [Node lifecycle](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/node/src/api/mod.rs#L115)). The diagnostic service must fit those ownership rules.

**5. `worker` and `worker-proto` integration for dynamic plugins**

Extend the existing authenticated plugin-to-host bridge so out-of-process dynamic plugins can contribute structured observations to the host’s diagnostic foundation. Keep `crates/worker` as a thin SDK depending on shared types and protocol definitions.

The protocol already separates host-to-plugin `PluginWorker` calls from plugin-to-host `RelayHostRuntime` calls. Existing `GetRuntimeDiagnostics` belongs to the latter: it lets a plugin inspect the host ([service definitions](https://github.com/NVIDIA/NeMo-Relay/blob/fcd81c8c/crates/worker-proto/proto/nemo/relay/worker/v1/plugin_worker.proto#L8)).

- **Protocol additions:** Add typed recording requests and richer snapshot queries to `RelayHostRuntime`. Preserve the existing `{code, message, count}` diagnostic response as a compatibility projection.
- **Transport structures:** Use explicit protobuf fields for closed diagnostic structures. Keep genuinely open attributes lossless through existing `JsonValue`/`JsonEnvelope` conventions.
- **Host attribution:** Authenticate every request and bind observations to host-assigned plugin identity, activation generation, and permitted component names. Plugin payloads cannot select another component’s ownership.
- **SDK APIs:** Provide equivalent typed asynchronous methods in `crates/worker` and `python/plugin`. Generate protocol bindings through the owning workflows.
- **Host-observed failures:** Instrument launch, handshake, registration, callback transport, cancellation, and teardown independently of plugin-authored records.
- **Acceptance semantics:** A recording receipt means acceptance into host memory. Bound requests and any retry/deduplication state; retries of an accepted operation must not add another occurrence.
- **Compatibility:** Implement additive `grpc-v1` extensions. Older hosts returning `UNIMPLEMENTED` produce unsupported coverage, not an empty successful result.
- **Lifecycle:** Retire activation-owned sources on teardown while retaining bounded evidence. Conditions do not become recovered merely because the plugin exits.

Illustrative SDK usage:

```rust
let recorder = runtime
    .diagnostics()
    .component("remote_lookup");

let operation_id = DiagnosticOperationId::new();

let receipt = recorder.record(
    operation_id,
    DiagnosticUpdate::failure(
        "plugin.backend_unavailable",
        SafeMessage::new("The configured backend is unavailable"),
    ),
).await?;

// After an observed successful backend operation:
recorder.record(
    DiagnosticOperationId::new(),
    DiagnosticUpdate::recovery("plugin.backend_unavailable"),
).await?;
```

Illustrative host bridge:

```rust
async fn record_diagnostic(
    &self,
    request: RecordDiagnosticRequest,
) -> Result<RecordDiagnosticResponse, Status> {
    let activation = self.authorize(
        &request.activation_id,
        &request.auth_token,
    )?;

    let recorder =
        activation.resolve_component(&request.local_component)?;

    let update = decode_bounded_update(request.update)?;

    let receipt =
        recorder.accept_once(request.operation_id, update);

    Ok(encode_receipt(receipt))
}
```

Existing ordinary logging and configuration-validation diagnostics remain distinct operations. Typed observations enter the shared recorder once and use its logging projection; free-form log messages should not be parsed into conditions.

This phase does not require a second diagnostic store or HTTP server inside each dynamic-plugin process. Independently queryable plugin-local state would require a separate host-to-plugin snapshot capability.

**6. Runtime API consistency and documentation**

Use this phase to complete conformance, compatibility review, and documentation across all earlier phases. The foundational schema and ownership decisions must be established in phase 1 and exercised throughout implementation.

Define one contract for recording, querying, projecting, and collecting diagnostics. Keep adapters responsible for transport and language representation, with observable semantics supplied by core.

- **Schema and versioning:** Document required fields, identities, timestamp meanings, code namespaces, unknown-field handling, lossless integers, availability, freshness, retention gaps, and compatibility rules.
- **API consistency:** Publish the shared CLI endpoint contract and its relationship to existing health and worker-status endpoints. Distinguish request success, lifecycle state, active diagnostic conditions, and deployment health policy.
- **Producer coverage:** Catalogue which components report each condition and outcome. Explicitly identify observations the implementation cannot obtain.
- **Delivery semantics:** Define precisely what “recorded,” “exported,” “acknowledged,” and “recovered” mean. Collector partial rejection requires actual exporter instrumentation; a common API cannot infer it.
- **Compatibility:** Review existing subscriber reports, plugin reports, binding APIs, and dynamic-plugin consumers. Native-plugin extensions, if required, must use the stable ABI boundary rather than pass Rust runtime handles across libraries.
- **Documentation:** Update operational logging, CLI/gateway, daemon operations, embedded SDK, plugin authoring, troubleshooting, and release notes. Explain optional-sink behavior, bootstrap limitations, freshness, and loss.
- **Conformance validation:** Exercise common scenarios through each applicable adapter: startup failure, optional-sink failure, repeated failure and recovery, component replacement, stale worker reports, unsupported peers, and bounded-history loss.
- **Generated surfaces:** Update schemas, declarations, protocol clients, and API references through their owning generation workflows.

Illustrative conformance pseudocode:

```rust
let expected = fixture.record_failure_then_recovery();

let rust_view = rust_reader.snapshot(query.clone())?;
let http_view = decode_http_snapshot(http.get_status().await?)?;
let plugin_view = worker_client.snapshot(query.clone()).await?;

assert_same_diagnostic_facts(
    &expected,
    &[rust_view, http_view, plugin_view],
);

// Verify separately:
// - transport-specific access and errors;
// - legacy response compatibility;
// - reads do not change counters or trigger recovery;
// - source identity and freshness survive collection.
```

The conformance checks should compare the same facts and meanings while allowing appropriate language and transport representations—for example, a typed Rust snapshot, a Python object, or a versioned JSON response.
