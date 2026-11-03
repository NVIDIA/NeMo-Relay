<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Worker contract

The replacement is `nvidia.nemo.relay.worker.v1`, protobuf package `nvidia.nemo.relay.worker.v1`, contract
version `1.0.0`. These identify a new contract, not a crate or wheel release.
This layer introduces the schema, generated messages, converters, validation,
and current-contract tests. The host and maintained SDKs still use `grpc-v1`;
the legacy schema, `v1` Rust module, and JSON helpers remain available until their
coordinated adoption. Runtime and SDK requirements below describe that adoption
target; they do not claim that live traffic has switched in this layer.
Diagnostic read grants remain disabled by default.

## Source layout

| File under `proto/nvidia/nemo/relay/worker/v1/` | Ownership |
| --- | --- |
| `common.proto` | Credentials, protocol identity, lossless JSON, schema identity, errors, registration surfaces, scope types, severity |
| `llm.proto` | Request/response DTOs, multimodal variants, open codec-owned provider data, optimization contributions and summaries |
| `event.proto` | ATOF envelopes, scope/mark variants, category profiles, sanitizer fields, pending marks |
| `runtime.proto` | Lifecycle, per-surface invocation, continuations, codecs, scopes, registration, and logging messages |
| `diagnostics.proto` | Delivery/session contracts and the canonical diagnostic snapshot graph |
| `diagnostic_read.proto` | Complete shared query, discovery, producer support, domain/family selectors, and read reasons |
| `diagnostic_memory.proto` | Canonical memory page, backing footprints, row attribution, and selection totals |
| `diagnostic_resource.proto` | Typed resource memory samples, units, numeric kinds, availability, and fixed nullable limit slots |
| `plugin_worker_service.proto` | Worker service and its 8 dedicated request/response pairs |
| `relay_host_runtime_service.proto` | Host runtime service and its 17 dedicated request/response pairs |
| `relay_diagnostics_service.proto` | Diagnostic service and its 6 dedicated request/response pairs |

The runtime DTOs in `nemo-relay-types` remain the source of truth for application
semantics. The protobuf messages define the transport representation. Runtime
converters must explicitly preserve Relay-owned fields. Codec-owned `api_specific`
data uses lossless JSON with the same Serde representation as the native SDK. The
protocol does not enumerate providers or codec implementations.

## Version and evolution rules

Historical compatibility applies once this contract is released. Initial unreleased
development is checked against the current schema, without a previous-release baseline.

- Both sides must check the activation credentials, plugin identity, protocol
  name, and supported contract version before registration. Health may precede
  handshake, as in the existing host lifecycle.
- The worker protocol contract is `1.0.0` in the stable major-version namespace
  `nvidia.nemo.relay.worker.v1`. Backwards-compatible evolution stays in `v1`;
  breaking changes require a new major namespace and negotiated protocol identity.
  Diagnostic snapshot-transport and provenance contracts start at `1.0.0` and
  remain independently versioned. Package releases remain on the unified Relay release
  version. Core diagnostic schema/catalogue revisions, ATOF versions, optimization
  calculation/report versions, and application `DataSchema` versions retain their
  own meanings; do not rewrite them to the worker version.
- Incompatible changes require a distinct negotiated protocol identity. Do not
  reuse a v1 field number with a different type or silently accept a v1 payload.
- Add fields with new numbers. Preserve existing field numbers, names, types,
  presence, oneof membership, enum values, service names, and method signatures.
  Reserve deleted names and numbers. Removing fields still requires explicit
  compatibility review and an appropriate version boundary.
- Readers must reject unknown registration surfaces, invocation alternatives,
  mandatory enum values, and missing mandatory alternatives. They must not
  interpret them as success, allow, or an empty payload. Tolerating unknown wire
  fields is not proof that a runtime conversion preserves those fields.
- Provider extensions belong in explicit `extra` maps or provider-native variants.
  Unknown fields in those maps must round-trip. Encode and decode reject keys
  that shadow modeled native JSON fields, including `role` and `type`
  discriminators. Native names determine collisions even when protobuf uses a
  different field name: an image part reserves `image`, while `value` remains a
  valid extension. The literal key `extra` remains valid because the map is
  flattened into the native object. Open event categories, optimization kinds,
  diagnostic names, and diagnostic execution/phase names remain strings.

## Presence and validation

Protobuf decoders establish field types and alternative exclusivity. They do not
establish semantic validity. SDK and host validation must occur before callback
dispatch or state mutation. The schemas carry executable Protovalidate rules for
presence, enum membership, finite numbers, UUIDs, protocol identities, diagnostic
bounds, and cross-field relationships. Python contract tests execute these rules
with Protovalidate 2.1.0 and compile the entire message graph. Both SDKs and the host call validation at dispatch boundaries and check open JSON syntax
and well-known timestamp/duration validity.

JSON syntax, authentication, capability ownership, configured frame sizes, and
stateful sequencing still require SDK/host checks; annotations alone do not
establish those properties.

Required boundary values:

- Every RPC request requires valid `credentials`.
- Handshake requires `plugin_id`, `relay_version`, `protocol`, and `host_endpoint`.
  Worker identity in handshake and health responses is required. Unspecified
  health status is invalid.
- Validate/register require a plugin ID and config value. No config is encoded
  as JSON `null`, not absent bytes. Validate/register responses require one result;
  an empty diagnostics list or registration plan is a present successful result.
- Invoke requires a nonempty unique active invocation ID, registration name, and
  exactly one invocation alternative matching that registration. Streaming uses
  its own request type and the streaming registration surface.
- Every domain response, stream item, guardrail decision, event kind, LLM message
  role, content kind, tool definition/choice,
  and finish reason must select an alternative. Oneof absence is malformed.
- A required `JsonValue` must contain one valid JSON value. `LlmRequest.headers`
  and `content`, tool invocation arguments/values, tool execution `result`, codec
  inputs/outputs, and stream chunks are required even when their JSON value is null.
  Every `JsonObject`/`extra` map value must be valid; an empty map value is invalid.
- Tool execution outcomes require their result wrapper. LLM intercept outcomes
  require a request and, when a codec is active, an authoritative annotation.
  Contributions containing a payload require its schema identity.
- Events require their base envelope, UUID, timestamp, and kind-specific values.
  Scope events require a recognized lifecycle phase and a category. UUID strings
  use canonical hyphenated form. Diagnostic session UUID bytes have length 16.
- Codec invocations require a current invocation-bound capability with the correct
  direction. Non-streaming execution requires request and response codec contexts;
  streaming execution carries only the request context. Capability IDs and scope
  bindings remain host-issued opaque identities and must not reach user callbacks.

DTO presence follows the runtime's option types. In normalized requests, messages
and extra maps default to empty; all other top-level fields are optional. In
normalized responses, all modeled top-level fields are optional. Required nested
runtime fields remain required, including named/tool message content and function
call definitions. Assistant content and tool-call lists are optional. Function
message content absence maps to the runtime's explicit null content.

Optional collections use presence-bearing list messages: absent is `None`, a
present empty list is `Some([])`. Optional scalar fields preserve explicit false
and zero. A present `MessageContent.text = ""` is different from no content variant.

Optional JSON fields preserve explicit null where the runtime does. Tool result
annotations and category-profile tool annotations normalize null to absence.
Event sanitizer output replaces the three sanitizer fields; absent means clear,
not leave unchanged. LLM sanitizers use the explicit `unchanged` result instead.

Python adapters reject nonstring JSON object keys recursively before serialization.
Modeled integer fields require integer values and reject booleans, fractional
numbers, and strings. Decimal-string conversion is limited to diagnostic fields
whose native JSON representation requires it.
Modeled objects require dictionaries, arrays require lists or tuples, and strings
require string values. Iteration, `dict` construction, or protobuf setters must
not silently reinterpret an incorrect native value as a different valid payload.

All timestamps use `google.protobuf.Timestamp`, UTC seconds plus nanoseconds,
within its documented year 1–9999 range, with nanos in `0..999999999`. Absent scope
timestamps use host time; a present epoch-zero timestamp is explicit. Converters
must reject out-of-range times rather than overflow, and preserve the runtime's
available precision. Double fields must be finite.

## Invocation result matrix

`InvokeResponse.error` is available for every unary callback failure. Successful
alternatives are restricted by the registered surface:

| Invocation surface | Successful result |
| --- | --- |
| Subscriber | `completed` |
| Event metadata injector | `metadata_additions` |
| Mark/scope sanitizer | `event_sanitize` |
| Tool request/response sanitizer or request interceptor | `tool_value` |
| Tool/LLM conditional execution or conditional middleware guardrail | `guardrail` |
| Tool execution interceptor | `tool_execution_outcome` |
| LLM request sanitizer | `llm_request` or `unchanged` |
| LLM response sanitizer | `llm_response` or `unchanged` |
| LLM request interceptor | `llm_request_outcome` |
| LLM execution interceptor | `llm_response` |

Guardrail allow is explicit; block requires a nonempty reason. An absent response
result must never be accepted as subscriber success. The oneof on the request
eliminates a contradictory surface enum, but the worker must still compare that
alternative with the actual registration and the host must validate its response.

## Errors and lifecycle

| Failure | Wire representation |
| --- | --- |
| Missing/malformed values or invalid alternatives | gRPC `INVALID_ARGUMENT` |
| Missing or invalid authentication credentials | gRPC `UNAUTHENTICATED` |
| Authenticated caller lacks access to an activation or resource | gRPC `PERMISSION_DENIED` |
| Incompatible protocol/plugin identity or changed registration config | gRPC `FAILED_PRECONDITION` |
| Missing/expired continuation, scope binding, or codec capability | Appropriate `NOT_FOUND`, `PERMISSION_DENIED`, or `CANCELLED` status, matching ownership and lifetime |
| Callback/domain failure | The operation's `WorkerError` alternative |
| Stream-opening failure before an item exists | gRPC status with downstream provenance metadata when applicable |
| Failure after stream opening | One terminal `WorkerError` item, then close |
| Transport cancellation/deadline/capacity failure | gRPC `CANCELLED`, `DEADLINE_EXCEEDED`, or `RESOURCE_EXHAUSTED` |

Simple host operations return method-specific success responses and use gRPC
status for failure. Continuation/codec operations retain a domain error alternative
so downstream failure identity can be propagated. `http_status`, when supplied,
is an upstream HTTP error in `400..=599`; it is not a gRPC status. Converters must
preserve compatibility with the runtime's existing HTTP error helpers.

Preserve `downstream_continuation_id` only on the original downstream failure,
never a replacement callback error. Stream-opening provenance uses the existing
`nemo-relay-downstream-continuation` metadata. `retryable` does not authorize
replaying callbacks or continuations.

Registration is serialized and idempotent for semantically equal configuration
within one activation. Repeats return the original plan. Different config fails
before running registration again. Cancellation is idempotent and waits for
callback/continuation cleanup as required by the SDK lifecycle. Unknown/completed
invocations return `accepted = false`. Shutdown drains within existing bounds.
Dropping or terminalizing streams must release capabilities and scope ownership.

Successful scope and conditional-guardrail creation returns a nonempty host-owned
handle, which SDKs validate before exposing it to callers. Invalid registration
replies roll back local callback state. Explicit parent scope selection resolves
within the selected owned stack and rejects unknown or expired parent identities
before executing. A failed scope pop preserves its handle for a corrected retry;
ownership is consumed only after native closure succeeds.
Stack revocation fences in-flight scope creation: a push cannot publish an owned
handle after its stack has been dropped. Activation shutdown revokes new scope
admission and drains owned scopes and isolated stacks before reporting successful
cleanup.

## Diagnostics and limits

Diagnostic admission remains separate from execution and logs. Capabilities and
session-open settings must agree. Queue capacity excludes one in-flight request.
Request/response limits count the complete new protobuf messages, including the
new response wrappers; migrate size calculations rather than copying v1 sizes.

Diagnostic receipts specify evidence retention only for `recorded` observations.
Registration, closure, rejection, and disabled receipts leave evidence unspecified
because no observation was recorded; a recorded receipt must select retained,
truncated, or dropped evidence.

Registered receipts require a positive component handle and code count. Both SDKs
validate the complete receipt and check its status against the submitted operation, including
the code count for successful registration, before advancing the submission
sequence. Malformed receipts fence further delivery as uncertain; they never
acknowledge successful recording or create an inert registered component.

The sequence starts at one. Identical retries reuse the same operation, session,
and sequence. An admitted rejection consumes its sequence. Admission rejection
uses existing `nemo-relay-diagnostic-not-consumed: 1` metadata; missing replies may
mean uncertain acceptance. Never replay uncertain submissions into a new session.

Snapshot counters use uint64 on the protobuf wire and canonical decimal strings
in existing JSON projections. Snapshot, discovery, and memory reads require their respective separate grants
and remain denied to ordinary workers. Full DiagnosticQuery selectors and typed
cursors preserve the canonical query semantics. Cursor validation is bound to
the host/activation and operation kind; a cursor is not an authority grant.

When activating namespaced v1, update the host/SDK service-name and RPC-path checks used by
message-limit interceptors and diagnostic admission. Preserve per-callback limits,
pre-decode admission, deadlines, retry bounds, and cleanup behavior.

## Shared native and worker authoring contract

Rust domain types in `nemo-relay-types` are the authoring source of truth. SDK
shared authoring signatures must not expose generated protobuf messages, gRPC
statuses, activation credentials, or capability IDs. Backend-specific bootstrap
configuration and raw ABI APIs remain separate. Native and worker adapters
convert at their boundaries. In particular, migrate the legacy worker diagnostic API's
protobuf re-exports to shared diagnostic declarations, queries, and results.
`ToolExecutionContext` will become one nominal type re-exported by both Rust SDKs during adoption;
its existing native fields and worker accessor methods remain available.

| Authoring capability | Shared domain contract | New worker wire representation | Migration acceptance |
| --- | --- | --- | --- |
| Plugin identity and registration | Component configuration, diagnostics, and explicit multiple-component policy | Handshake, Validate, Register, RegistrationPlan | Resolve `plugin_kind`/`plugin_id`, mutable registration, `Send`/`Sync`, and differing defaults in the shared facade; never silently change legacy defaults. |
| Events and metadata | Event, EventSanitizeFields, PendingMarkSpec, typed metadata maps | Per-surface invocation and result alternatives | Same transformations, event parentage, and null/presence behavior through both adapters. |
| Tool invocation | ToolExecutionContext, ToolExecutionResult, ToolExecutionInterceptOutcome | ToolExecutionInvocation and canonical tool result/outcome | Same callback input type and lossless result/annotation/mark behavior. |
| LLM invocation and codecs | LlmRequest, AnnotatedLlmRequest/Response, LlmCodecIdentity, optimization DTOs | Typed invocation and normalized results, open codec reference metadata and capability | Async common facade supports RPC latency; native synchronous helpers may remain backend-specific. |
| Continuations | Shared async authoring interfaces over backend-owned handles | ToolNext, LlmNext, LlmStreamNext and invocation-bound continuation IDs | Repeat/concurrency, expiry, downstream failure identity, and cleanup agree with the rules below. |
| Scopes and runtime control | ScopeType, ScopeStackSnapshot, runtime registration identities and opaque local handles | Scope, stack, logging, registration, and gate RPCs | Preserve ownership, parent selection, idempotent removal, and activation cleanup. |
| Diagnostic recording | Shared component declarations, observations, outcomes, and delivery state | SubmitDiagnostic typed operation and receipt | Synchronous enqueue acceptance remains distinct from host admission/recording; delivery failures do not become callback failures. |
| Diagnostic discovery | DiagnosticDiscovery and DiagnosticProducerSupport | GetDiagnosticDiscovery | Explicit discovery grant; remains side-effect-free when collection or queries are disabled. |
| Diagnostic snapshots | Complete DiagnosticQuery, DiagnosticSnapshot, DiagnosticReadReason | ReadDiagnosticPage with typed query and result/read-reason oneof | Preserve every selector and cursor; reject unauthorized reads independently of query scope. |
| Diagnostic memory | DiagnosticMemoryPage, footprints, selection totals, ResourceMemoryView | InspectDiagnosticMemory and complete typed memory/resource graph | Explicit memory grant; fixed null slots and numeric kinds round-trip; never start or retarget a resource collector. |
| Errors | Shared callback/downstream error semantics, HTTP helper compatibility, portable diagnostic read reasons | WorkerError, DiagnosticReadReason, and standard gRPC status | Domain adapters preserve original failures; transport errors remain distinguishable without leaking tonic types. |

Capability parity does not mean automatic authority. `can_read_context`,
`can_read_discovery`, and `can_inspect_memory` are independent host grants,
false by default. Unsupported or denied features must be explicit; a generic
"connected" state must not imply read access. Credentials authenticate an
activation but do not authorize context-wide evidence. Read RPCs do not require
an admitted delivery session, so authorized discovery still works when diagnostic
collection or snapshot workspaces are disabled.

Shared continuation and resource-lifetime requirements:

- Continuation calls may repeat or run concurrently while the interceptor owns
  them. Each receives an isolated snapshot of the current scope stack. Callback
  completion/cancellation revokes the associated handles and settles outstanding
  work before reporting cleanup completion.
- Cloning a context or continuation never extends host authority or invocation
  lifetime. Expired, foreign, wrong-direction, and cross-activation capabilities
  fail before execution. Original downstream errors retain provenance only when
  returned unchanged; replacement callback errors must clear it.
- Stream opening, first item, exhaustion, terminal error, consumer drop, explicit
  cancellation, and deadline are distinct conformance cases. Delivery is ordered
  and bounded, applies backpressure, and produces at most one terminal result.
  Successful completion and error completion both release stream ownership.
  A successful public stream close also waits for host-owned downstream cleanup,
  with bounded teardown and cleanup failures reported to the caller.
- Cancellation acknowledgement proves the SDK's callback/continuation cleanup,
  not termination of arbitrary blocking tasks created by plugin code. Do not
  replay execution after ambiguous transport failure.
- Diagnostic query selectors intersect host authority. Cursor validation binds
  source, revision, selection, activation, and operation kind; memory cursors cannot
  resume snapshots and vice versa. Typed cursors are opaque authoring handles,
  not transferable read grants. Authorization failures use gRPC status; portable
  read failures use DiagnosticReadReason, never an empty successful page.
- Diagnostic `usize` memory quantities cross the wire as uint64; adapters must
  check target-width conversions. Exact integers never pass through double.
  Resource non-finite measurements use an explicit alternative. Eight nullable
  limit-event slots preserve positions instead of compacting away empty slots.

The host/SDK integration suites exercise these requirements through the active
namespaced services, including installed Python wheels. Schema-only tests remain
separate from runtime lifecycle and ownership tests. Rust authoring APIs retain
shared domain types; no Rust trait object crosses the native C ABI.

## Rust domain conversions

`nemo_relay_types` owns the shared authoring types used by native plugins and Rust
workers. Protobuf owns the transport structures. `conversions::WireValue` connects
them, and both the host and Rust worker reuse its implementations. Generated
messages therefore stay out of callback APIs. Closed DTOs are converted by field;
only the contract's open application and codec-owned data uses JSON serialization.

The conversion implementation has three responsibilities:

| File | Responsibility |
| --- | --- |
| `src/conversions/mod.rs` | Shared checked conversions and the private mapping macros. |
| `src/conversions/dto.rs` | Declarative struct fields, enum mappings, and collection wrappers. |
| `src/conversions/variants.rs` | Explicit mappings for Rust enum variants, protobuf oneofs, and structures with different layouts. |

A DTO declaration uses `field: mode` when names match and
`source => target: mode` when they differ. For example:

```text
dto!(nemo_relay_types::api::event::BaseEvent => BaseEvent {
    uuid: scalar,
    timestamp => time: required,
    // Remaining fields must also be declared.
});
```

The macro emits complete struct initializers in both directions, without a
`..Default::default()` fallback. Adding a field to either struct fails compilation
until its mapping is updated. Enum variants remain explicit; unknown or unsupported
values return conversion errors. `list_codec!` shares the implementation for list
wrapper messages, including explicitly empty lists.

| Field mode | Conversion rule |
| --- | --- |
| `scalar` | Convert a value directly, including enums, UUID strings, and checked integer widths. |
| `required` | Wrap a required domain value in protobuf message presence; reject absence when decoding. |
| `optional` | Convert a present value and preserve absence. A present JSON null remains a value. |
| `repeated` | Convert each list element, preserving order. |
| `map` | Convert each map value, preserving keys. |
| `codec_data` | Preserve codec-owned native JSON; decode absent data or JSON null as `None`, matching native Serde behavior. |
| `annotation` | Normalize absent or JSON-null tool annotations to absence. |
| `fixed_slots` | Preserve the eight positional diagnostic limit-event slots, including empty slots; reject the wrong length. |

Explicit adapters remain necessary for domain enums versus protobuf oneofs,
flattened event/outcome fields versus nested transport messages, timestamps,
resource numeric variants, and native ownership/name wrappers. Timestamp conversion
checks protobuf bounds and nanoseconds; diagnostic `usize` values use checked
`uint64` conversions. Losslessness applies to supported domain values and declared
open payloads; unknown protobuf fields are not retained through domain conversion.

When changing a shared type or its transport representation:

1. Update the schema and mapping together. Use the existing DTO/list macros for
   direct mappings and explicit conversions for differences in structure or semantics.
2. Preserve the documented absence, null, empty-collection, integer, and time rules.
3. Update the relevant round-trip cases in `tests/conversion_tests.rs` or existing
   host/worker integration tests. Exercise actual protobuf encoding and decoding;
   schema field coverage alone does not prove value preservation.
4. Run the focused conversion tests with
   `cargo test -p nemo-relay-worker-proto --test conversion_tests`, then the affected
   protocol checks and repository validation. Schema changes also require
   `just check-worker-proto` and compatibility review.

## NVIDIA API standards review

This contract follows NVIDIA gRPC API Standards and Guidance, including its
channel-based versioning principles. The table below records the applicable
requirements and their enforcement. AEP references in that standard are optional
design references; this contract does not claim blanket AEP conformance.

| Guideline area | Contract decision and enforcement |
| --- | --- |
| Package naming and versioning | `nvidia.nemo.relay.worker.v1`, matching directories, stable major suffix; all Buf STANDARD rules apply without an exception. |
| Retired namespace | `nemo.relay.worker.v1` remains available until the separate consumer cutover. The Rust module is `nvidia_worker_v1`; its negotiation identity is the fully qualified current package name. |
| Compatible evolution | Additive updates remain in the same major namespace. After the first release, compatibility checks should compare wire, source, JSON, and validation constraints against that released schema. New breaking changes require a new namespace. |
| Source layout | Open-source SPDX license, file overview, proto3 syntax, package, alphabetized imports, then declarations; Buf formats files with two-space indentation. No public or weak imports. |
| Service organization | One service per matching `_service.proto`, service first, dedicated request/response pairs in RPC order in that file. Supporting DTOs are in separate files. |
| Documentation and names | Complete-sentence declaration comments, PascalCase types/services/RPCs, lower_snake_case fields/oneofs, plural repeated fields. Buf checks standard names and comment presence; code review covers additional editorial requirements. |
| Fields and nesting | No explicit `json_name`, no `descriptor` field names, and no user-defined nested messages or enums. Code review checks these conventions; compiler-generated map entries are permitted. |
| Enums | PascalCase names, prefixed UPPER_SNAKE_CASE values, zero `_UNSPECIFIED` sentinel, no aliases; boundary validation rejects invalid submitted values. |
| Well-known types | `google.protobuf.Timestamp` for instants, `google.protobuf.Duration` for the bounded retry deadline, and `google.protobuf.Empty` for empty oneof alternatives. RPCs retain dedicated request/response messages. |
| Open application data | Lossless JSON bytes preserve `u64` integers and distinguish null/presence; `google.protobuf.Value` would lose integer precision. Only explicitly allowlisted fields use open payloads. |
| Time naming and compatibility | Timestamp fields use `time` or `_time`. Diagnostic elapsed spans retain explicit millisecond units to preserve saturating runtime `u64` values beyond the WKT Duration range. `time_to_first_chunk_seconds` retains the runtime's floating-point precision; conversion to integral nanos could round it. |
| Duration conversion | Convert SDK `retry_timeout_millis` to `retry_timeout_duration` exactly; bounds remain one millisecond through ten seconds. Do not serialize the runtime struct directly. |
| Field behavior annotations | This is a process-to-process gRPC API, not a resource API or HTTP-transcoded API. Requiredness is documented and enforced with Protovalidate. No HTTP gateway annotations are added. |
| Streaming | Only progressive LLM output uses server streaming. Clients and proxies must support HTTP/2 streaming, propagate cancellation/deadlines, and avoid response buffering. Diagnostics use bounded unary pages. |
| Errors | Authentication, admission, malformed input, and RPC failures use standard gRPC statuses. Callback result data remains typed, including the original downstream failure identity. Kaizen is optional under the standard. |

Code review covers service-file ownership, complete sentences, overview comments,
prohibited field names/options, plural repeated fields, top-level type declarations,
and timestamp/duration suffixes. Generated
stubs are compiled independently in Rust and Python.

Both SDKs register the
standard `grpc.health.v1.Health` service (`Check` and `Watch`) alongside the
worker's authenticated lifecycle `Health` RPC; propagate deadlines, cancellation,
and W3C `traceparent`/`tracestate` metadata when participating in tracing; and
apply the host's health/metrics policies. The custom lifecycle RPC is not a
replacement for standard gRPC probes. Standard health bindings can come from
`grpcio-health-checking` / `tonic-health`; they do not require a competing Relay
health schema. HTTP metrics/probes belong to the owning runtime deployment,
not to these worker DTO definitions.

At runtime, map invalid credentials to `UNAUTHENTICATED`, denied diagnostic reads
to `PERMISSION_DENIED`, malformed requests to `INVALID_ARGUMENT`, and incompatible
protocol/lifecycle state to `FAILED_PRECONDITION`. Preserve `CANCELLED`,
`DEADLINE_EXCEEDED`, and other applicable standard statuses. Both SDKs
validate messages before callback dispatch or state mutation.

The adoption layer will move both SDKs and the Rust host to this namespace and
remove the legacy sources. Existing workers must then rebuild and update their
manifests; no adapter is provided. Namespace collision review is limited to
this repository; no external namespace registration or governance approval is claimed.

## Current-contract checks

This contract is unreleased. Tests validate the current schema and its Rust/Python
implementations. They do not compare against development snapshots, committed
descriptor fixtures, or historical wire bytes. After the first release, introduce
compatibility checks against that released schema when evolving the contract.

- `cargo test -p nemo-relay-worker-proto` builds descriptors from the current
  schema and tests service structure, message round trips, conversions, and validation.
- `crates/worker-proto/buf.yaml` is the only Buf workspace configuration. Its
  adjacent `buf.lock` pins dependencies and ships with the published Rust crate.
- `just check-worker-proto` runs Buf STANDARD and COMMENTS lint, formatting,
  independent Python generation, and current-contract tests. Install Buf 1.75.0
  or newer; `BUF` can select its executable. All STANDARD rules apply, including
  `PACKAGE_VERSION_SUFFIX`. Documentation is required for messages, fields,
  enums and values, oneofs, services, and RPCs. Client streaming and inline
  lint-ignore directives are forbidden. Stable imports are checked transitively.
- Pinned Buf pre-commit hooks run lint and formatting when the workspace config
  or schemas change. Python contract tests enforce stable major-version namespaces,
  the exact server-streaming allowlist, JSON/binary field allowlists, and required
  real oneofs. Additional NVIDIA editorial conventions remain code-review guidance.
- `just check-python-worker-proto` checks generation and runs the Python contract
  tests in an isolated environment. It does not replace packaged SDK bindings.

`buf.lock` pins Protovalidate v1.2.2 to an immutable BSR commit. Rust and Python
source builds require Buf on PATH (or selected with `BUF`) and export that exact
commit into an ignored build directory. Buf manages its download cache; an initial
build needs BSR access. `just resolve-worker-proto-deps` resolves the include path
for standalone generation and policy tests. No upstream schema is committed.
After SDK adoption, Python source distributions carry the canonical schema and lockfile; building a
wheel from them also requires Buf. Prebuilt wheels include private generated
bindings and the upstream license, and require neither Buf nor protoc to install.

Current-message round trips cover null, large integers, provider extensions,
presence of empty collections/default scalars, open variants, and typed pending
marks. Test inputs live in test source; serialized messages and descriptors are
generated during the build or test run. Python checks also compare runtime struct
fields with protobuf DTOs. SDK validation and Rust/Python host interoperability
are covered by the worker SDK and installed-wheel host integration suites.
