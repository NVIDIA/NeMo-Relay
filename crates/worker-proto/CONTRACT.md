<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Worker Contract

The new protobuf package is `nvidia.nemo.relay.worker.v1`. Workers use the same
name to identify the protocol during handshake. Its contract version is `1.0.0`.
Crates and wheels have separate release versions.
This change adds the schema, generated messages, converters, validation, and tests.
The host and SDKs still use `grpc-v1`. The old schema, Rust `v1` module, and JSON
helpers remain available until the host and SDKs switch together. The requirements
below describe that planned change. They do not describe the live protocol yet.
Diagnostic read access is disabled by default.

A domain type is an application data type from the shared runtime. A wire message
is its protobuf form. A capability ID is an access handle issued by the host.

## Source Layout

The schema files have the following roles:

| File Under `proto/nvidia/nemo/relay/worker/v1/` | Ownership |
| --- | --- |
| `common.proto` | Credentials, protocol identity, lossless JSON, schema identity, errors, registration surfaces, scope types, severity |
| `llm.proto` | Request/response DTOs, multimodal variants, open codec-owned provider data, optimization contributions and summaries |
| `event.proto` | ATOF envelopes, scope/mark variants, category profiles, sanitizer fields, pending marks |
| `runtime.proto` | Lifecycle, per-surface invocation, continuations, codecs, scopes, registration, and logging messages |
| `diagnostics.proto` | Delivery/session contracts and the canonical diagnostic snapshot graph |
| `diagnostic_read.proto` | Complete shared query, discovery, producer support, domain/family selectors, and read reasons |
| `diagnostic_memory.proto` | Canonical memory page, backing footprints, row attribution, and selection totals |
| `diagnostic_resource.proto` | Typed resource memory samples, units, numeric kinds, availability, and fixed nullable limit slots |
| `plugin_worker_service.proto` | Worker service and its eight request and response pairs |
| `relay_host_runtime_service.proto` | Host runtime service and its 17 request and response pairs |
| `relay_diagnostics_service.proto` | Diagnostic service and its six request and response pairs |

The shared data types in `nemo-relay-types` define application behavior. Protobuf
messages define how those values cross the process boundary. Converters must
preserve every Relay-owned field. Codec-owned `api_specific` data uses the native
SDK's JSON representation. Its number limits are described under
[Rust Domain Conversions](#rust-domain-conversions). The protocol does not list
all providers or codecs.

## Version and Evolution Rules

After the first release, compare changes against that released schema. Until
then, check the current schema without keeping earlier development snapshots.
Apply these version and compatibility rules:

- Both sides must check the activation credentials, plugin identity, protocol
  name, and supported contract version before registration. The host can check
  health before handshake.
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

## Presence and Validation

Protobuf checks field types and allows only one value in each `oneof`. It does
not check every rule that makes a message valid. The host and SDKs must validate
messages before calling a callback or changing state.

Protovalidate rules check required fields, enums, finite numbers, UUIDs, protocol
identities, diagnostic limits, and relationships between fields. Python contract
tests run these rules with Protovalidate 2.1.0 and compile every message type.
The host and SDKs also check JSON syntax, timestamps, and durations.

Schema annotations do not replace runtime checks. The host and SDKs still check
credentials, access rights, message sizes, and operation order.

Apply these rules at each RPC boundary:

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

Field presence follows the runtime's option types. In normalized requests, messages
and extra maps default to empty; all other top-level fields are optional. In
normalized responses, all modeled top-level fields are optional. Required nested
runtime fields remain required, including named/tool message content and function
call definitions. Assistant content and tool-call lists are optional. Function
message content absence maps to the runtime's explicit null content.

Optional lists use wrapper messages to preserve absence. An absent wrapper maps
to `None`. A wrapper with an empty list maps to `Some([])`. Optional scalar fields
preserve explicit false and zero. A present `MessageContent.text = ""` differs
from a missing content variant.

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

Use `google.protobuf.Timestamp` for UTC seconds and nanoseconds. Valid years run
from 1 through 9999, and nanos must be in `0..999999999`. An absent scope timestamp
uses host time. A present epoch-zero timestamp keeps that explicit value.
Converters reject times outside these bounds and preserve the runtime's time
precision. Double fields must be finite.

## Invocation Result Matrix

`InvokeResponse.error` is available for every unary callback failure. Successful
alternatives are restricted by the registered surface:

| Invocation Surface | Successful Result |
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

A guardrail must return an explicit allow result or a block result with a reason.
A missing result is invalid, including for subscribers. The worker must check
that the request alternative matches the registered callback type. The host must
validate the response.

## Errors and Lifecycle

Use these error forms for each failure:

| Failure | Wire Representation |
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

Simple host operations return their own success messages and use gRPC status
codes for failures. Continuation and codec responses also have an application
error alternative. It preserves the identity of a downstream failure.

When present, `http_status` is an upstream HTTP error in `400..=599`, not a gRPC
status. Converters must work with the runtime's existing HTTP error helpers.

Preserve `downstream_continuation_id` only on the original downstream failure,
never a replacement callback error. Stream-opening provenance uses the existing
`nemo-relay-downstream-continuation` metadata. `retryable` does not authorize
replaying callbacks or continuations.

Run registration calls one at a time. Repeating a call with equivalent config
returns the original plan. Different config fails before registration runs again.
Repeating cancellation is safe. Cancellation waits for callback and continuation
cleanup under the SDK's lifecycle rules. Unknown or completed invocations return
`accepted = false`. Shutdown uses the existing cleanup deadlines. Finishing or
dropping a stream must release its access handles and owned scopes.

Successful scope and conditional-guardrail creation returns a nonempty handle.
The host owns the handle, and the SDK validates it before returning it to callers.
Invalid registration replies roll back local callback state.

Resolve explicit parent scopes within the selected stack. Reject unknown or
expired parents before executing. If a scope pop fails, keep its handle so the
caller can correct the input and retry. Release ownership only after the native
close succeeds.

Dropping a stack also blocks pending scope creation. A push cannot return an owned
handle after its stack has been dropped. Shutdown blocks new scopes and cleans up
owned scopes and isolated stacks before reporting success.

## Diagnostics and Limits

Diagnostic admission is separate from callback execution and logging. The
session must confirm the advertised settings. Queue capacity excludes the one
request being sent. Size limits cover each complete protobuf request or response,
including its wrapper. Recalculate sizes when moving from the old protocol.

Diagnostic receipts specify evidence retention only for `recorded` observations.
Registration, closure, rejection, and disabled receipts leave evidence unspecified
because no observation was recorded; a recorded receipt must select retained,
truncated, or dropped evidence.

Registered receipts require a positive component handle and code count. Both SDKs
validate the whole receipt before advancing the sequence. Its status must match
the submitted operation. Successful registration must return the right code count.
An invalid receipt stops delivery and reports uncertain acceptance. It cannot
confirm recording or create a component that silently drops observations.

The sequence starts at one. Identical retries reuse the same operation, session,
and sequence. An admitted rejection consumes its sequence. Admission rejection
uses `nemo-relay-diagnostic-not-consumed: 1` metadata. Without a reply, acceptance
can be uncertain. Never replay an uncertain submission into a new session.

Snapshot counters use `uint64` on the wire and decimal strings in JSON output.
Snapshot, discovery, and memory reads each need a separate access grant. Ordinary
workers do not receive these grants. Preserve every `DiagnosticQuery` selector
and cursor field. Check the host, activation, and operation type when validating
a cursor. A cursor alone does not grant access.

When activating namespaced v1, update the host/SDK service-name and RPC-path checks used by
message-limit interceptors and diagnostic admission. Preserve per-callback limits,
pre-decode admission, deadlines, retry bounds, and cleanup behavior.

## Shared Native and Worker Authoring Contract

Use the Rust types in `nemo-relay-types` for shared plugin APIs. Keep generated
protobuf messages, gRPC statuses, credentials, and capability IDs out of callback
signatures. Each backend keeps its own startup settings and raw ABI APIs. Native
and worker adapters convert values at their boundaries.

During adoption, replace the worker diagnostic API's protobuf re-exports with
shared declarations, queries, and results. Both Rust SDKs will re-export the same
`ToolExecutionContext` type. Keep its native fields and worker accessor methods.
The following table lists the shared contracts and their adoption requirements:

| Authoring Capability | Shared Domain Contract | New Worker Wire Format | Adoption Requirements |
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

Matching features do not grant matching access rights. `can_read_context`,
`can_read_discovery`, and `can_inspect_memory` are independent host grants,
false by default. Report unsupported or denied access explicitly. A connected
worker does not automatically have read access. Credentials identify an
activation; they do not grant access to all evidence in a context.

Read RPCs do not need an admitted delivery session. An authorized caller can
therefore read discovery data even when collection or snapshot workspaces are
disabled.

Apply these rules to shared continuations and resource lifetimes:

- Continuation calls can repeat or run concurrently while the interceptor owns
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
  resume snapshots. Snapshot cursors cannot resume memory reads. Typed cursors are opaque authoring handles,
  not transferable read grants. Authorization failures use gRPC status; portable
  read failures use DiagnosticReadReason, never an empty successful page.
- Diagnostic `usize` memory quantities cross the wire as uint64; adapters must
  check target-width conversions. Exact integers never pass through double.
  Resource non-finite measurements use an explicit alternative. Eight nullable
  limit-event slots preserve positions instead of compacting away empty slots.

Use host and SDK integration tests to check these rules. Include the namespaced
services and installed Python wheels. Schema tests alone do not check lifecycle
or ownership. Rust authoring APIs use shared domain types. Rust trait objects
must not cross the native C ABI.

## Rust Domain Conversions

`nemo_relay_types` owns the shared authoring types used by native plugins and Rust
workers. Protobuf owns the transport structures. `conversions::WireValue` connects
them, and both the host and Rust worker reuse its implementations. Generated
messages therefore stay out of callback APIs. Closed DTOs are converted by field;
only open application data and codec-owned data use JSON serialization.

The conversion code is split across three files:

| File | Responsibility |
| --- | --- |
| `src/conversions/mod.rs` | Shared checked conversions and the private mapping macros. |
| `src/conversions/dto.rs` | Declarative struct fields, enum mappings, and collection wrappers. |
| `src/conversions/variants.rs` | Explicit mappings for Rust enum variants, protobuf oneofs, and structures with different layouts. |

A `dto!` declaration uses `field: mode` when names match and
`source => target: mode` when they differ. This example shows the declaration syntax:

```text
dto!(nemo_relay_types::api::event::BaseEvent => BaseEvent {
    uuid: scalar,
    timestamp => time: required,
    // Remaining fields must also be declared.
});
```

The macro fills every struct field in both directions. It does not use a
`..Default::default()` fallback. Adding a field to either struct fails to compile
until its mapping is updated. Enum mappings list each variant. Unknown or
unsupported values return conversion errors. `list_codec!` handles list wrappers, including
explicitly empty lists. Use these modes for each field:

| Field Mode | Conversion Rule |
| --- | --- |
| `scalar` | Convert a value directly, including enums, UUID strings, and checked integer widths. |
| `required` | Wrap a required domain value in protobuf message presence; reject absence when decoding. |
| `optional` | Convert a present value and preserve absence. A present JSON null remains a value. |
| `repeated` | Convert each list element, preserving order. |
| `map` | Convert each map value, preserving keys. |
| `codec_data` | Preserve codec-owned native JSON; decode absent data or JSON null as `None`, matching native Serde behavior. |
| `annotation` | Normalize absent or JSON-null tool annotations to absence. |
| `fixed_slots` | Preserve the eight positional diagnostic limit-event slots, including empty slots; reject the wrong length. |

Some values need explicit adapters. These include enum variants, nested transport
messages, timestamps, resource numbers, and native name or ownership wrappers.
Timestamp conversion checks the protobuf bounds and nanoseconds. Diagnostic
`usize` values use checked `uint64` conversions.

Conversions preserve supported domain values and the declared open payloads.
They do not retain unknown protobuf fields.

JSON numbers use the native SDK's number types: signed and unsigned 64-bit
integers, plus finite 64-bit floating-point values. Conversion does not preserve
decimal spelling or precision beyond those types. Native and worker callbacks
use the same number types for codec-owned JSON.

When you change a shared type or its wire format, follow these steps:

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

## NVIDIA API Standards Review

This contract uses NVIDIA gRPC API Standards and Guidance, including its
channel-based version rules. The following table lists the rules and checks. API Enhancement Proposals (AEPs) cited by that standard are optional references.
This contract does not claim to follow every AEP.

| Guideline Area | Contract Decision and Checks |
| --- | --- |
| Package naming and versioning | `nvidia.nemo.relay.worker.v1`, matching directories, stable major suffix; all Buf STANDARD rules apply without an exception. |
| Retired namespace | `nemo.relay.worker.v1` remains available until the separate consumer cutover. The Rust module is `nvidia_worker_v1`; its negotiation identity is the fully qualified current package name. |
| Compatible evolution | Additive updates remain in the same major namespace. After the first release, compatibility checks should compare wire, source, JSON, and validation constraints against that released schema. New breaking changes require a new namespace. |
| Source layout | Open-source SPDX license, file overview, `proto3` syntax, package, alphabetized imports, then declarations; Buf formats files with two-space indentation. No public or weak imports. |
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
| Errors | Authentication, admission, malformed input, and RPC failures use standard gRPC statuses. Callback result data remains typed, including the original downstream failure identity.  |

Code review checks file ownership, complete sentences, and file overviews. It
also checks forbidden field names and options, plural repeated-field names,
top-level declarations, and time suffixes. Rust and Python compile their generated
bindings independently.

SDKs must register the standard `grpc.health.v1.Health` service with `Check` and
`Watch`. Keep the worker's authenticated lifecycle `Health` RPC as well. That RPC
does not replace standard gRPC probes.

SDKs must propagate deadlines and cancellation. When tracing is enabled, they
must also propagate W3C `traceparent` and `tracestate` metadata. Apply the host's
health and metrics policies. Use `grpcio-health-checking` or `tonic-health` for
standard health bindings. The runtime deployment owns HTTP metrics and probes;
they do not belong in the worker message definitions.

At runtime, map invalid credentials to `UNAUTHENTICATED`, denied diagnostic reads
to `PERMISSION_DENIED`, malformed requests to `INVALID_ARGUMENT`, and incompatible
protocol/lifecycle state to `FAILED_PRECONDITION`. Preserve `CANCELLED`,
`DEADLINE_EXCEEDED`, and other applicable standard statuses. Both SDKs
validate messages before callback dispatch or state mutation.

The adoption layer will move both SDKs and the Rust host to this namespace and
remove the legacy sources. Existing workers must then rebuild and update their
manifests; no adapter is provided. Namespace collision review is limited to
this repository; no external namespace registration or governance approval is claimed.

## Current-Contract Checks

This contract is unreleased. Tests check the current schema and its Rust and
Python implementations. They do not compare earlier development snapshots,
committed descriptors, or old wire bytes. After the first release, check future
changes against that released schema.

Use the following checks:

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
