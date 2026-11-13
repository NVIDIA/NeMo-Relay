# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check contract structure, validation, and independently generated worker bindings."""

from __future__ import annotations

import importlib
import re
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

protoc = pytest.importorskip("grpc_tools.protoc")
grpc_tools = pytest.importorskip("grpc_tools")
from google.protobuf import descriptor, message_factory  # noqa: E402

# Protobuf exposes generated descriptor classes and C-extension pools dynamically.
descriptor_pb2: Any = importlib.import_module("google.protobuf.descriptor_pb2")
descriptor_pool: Any = importlib.import_module("google.protobuf.descriptor_pool")

ROOT = Path(__file__).resolve().parents[3]
PROTO_ROOT = ROOT / "crates/worker-proto/proto"
PACKAGE = "nvidia.nemo.relay.worker.v1"


@pytest.fixture(scope="module")
def contract(tmp_path_factory):
    output = tmp_path_factory.mktemp("nvidia-worker-v1")
    dependency = runpy.run_path(str(ROOT / "python/plugin/build_backend.py"))["resolve_proto_dependencies"]()
    sources = sorted((PROTO_ROOT / "nvidia/nemo/relay/worker/v1").glob("*.proto"))
    assert (
        protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO_ROOT}",
                f"-I{dependency}",
                f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
                f"--python_out={output}",
                f"--grpc_python_out={output}",
                str(dependency / "buf/validate/validate.proto"),
                *map(str, sources),
            ]
        )
        == 0
    )
    sys.path.insert(0, str(output))
    try:
        services = importlib.import_module(f"{PACKAGE}.plugin_worker_service_pb2")
        for service in ("plugin_worker", "relay_host_runtime", "relay_diagnostics"):
            importlib.import_module(f"{PACKAGE}.{service}_service_pb2")
            importlib.import_module(f"{PACKAGE}.{service}_service_pb2_grpc")
        yield services.DESCRIPTOR.pool
    finally:
        sys.path.remove(str(output))


def test_all_rpcs_have_dedicated_request_response_types(contract):
    counts = {"PluginWorkerService": 8, "RelayHostRuntimeService": 18, "RelayDiagnosticsService": 6}
    types = set()
    for name, count in counts.items():
        service = contract.FindServiceByName(f"{PACKAGE}.{name}")
        assert len(service.methods) == count
        for method in service.methods:
            assert method.input_type.name == f"{method.name}Request"
            assert method.output_type.name == f"{method.name}Response"
            for message in (method.input_type, method.output_type):
                assert message.full_name not in types
                types.add(message.full_name)
            assert not method.client_streaming
            assert method.server_streaming == (
                method.full_name
                in {f"{PACKAGE}.PluginWorkerService.InvokeStream", f"{PACKAGE}.RelayHostRuntimeService.LlmStreamNext"}
            )


@pytest.mark.parametrize(
    "name,fields",
    [
        pytest.param("JsonValue", {"json": b"null"}, id="null"),
        pytest.param("JsonValue", {"json": b"9007199254740993"}, id="large_integer"),
        pytest.param(
            "JsonValue", {"json": b'{"vendor":{"enabled":false,"n":18446744073709551615}}'}, id="provider_extension"
        ),
        pytest.param("AnnotatedLlmRequest", {"tools": {}}, id="present_empty_tools"),
        pytest.param("AnnotatedLlmRequest", {"store": False}, id="present_false"),
        pytest.param("AnnotatedLlmRequest", {"max_output_tokens": 0}, id="present_zero"),
        pytest.param("GenerationParams", {"stop": {}}, id="present_empty_stop"),
        pytest.param("MessageContent", {"text": ""}, id="empty_text_is_a_selected_variant"),
        pytest.param("MessageContent", {"parts": {}}, id="empty_parts_is_a_selected_variant"),
        pytest.param("FinishReason", {"unknown": "vendor_reason"}, id="open_finish_reason"),
        pytest.param("GuardrailDecision", {"allow": {}}, id="explicit_guardrail_allow"),
        pytest.param("ToolExecutionResult", {"result": {"json": b"null"}}, id="null_tool_result"),
        pytest.param("DiagnosticCount", {"value": 18446744073709551615}, id="maximum_diagnostic_counter"),
        pytest.param(
            "ToolExecutionInterceptOutcome",
            {
                "value": {"result": {"json": b"null"}},
                "pending_marks": [{"name": "mark", "category": "vendor.custom", "data": {"json": b"9007199254740993"}}],
            },
            id="typed_pending_mark_with_open_payload",
        ),
        pytest.param(
            "InvokeResponse",
            {
                "error": {
                    "code": "upstream.error",
                    "message": "unavailable",
                    "downstream_continuation_id": "next",
                    "http_status": 503,
                }
            },
            id="downstream_error_provenance_and_http_status",
        ),
        pytest.param(
            "DiagnosticQuery", {"after_sequence": 0, "max_events": 0}, id="diagnostic_query_present_zero_limits"
        ),
        pytest.param("ResourceMemoryNumber", {"integer": 18446744073709551615}, id="resource_integer_max"),
        pytest.param("ResourceMemoryNumber", {"non_finite": {}}, id="resource_non_finite_marker"),
        pytest.param(
            "ReadDiagnosticPageResponse", {"read_reason": "DIAGNOSTIC_READ_REASON_BUSY"}, id="diagnostic_read_busy"
        ),
        pytest.param(
            "ResourceMemoryMetrics",
            {
                "limit_events": [
                    {"absent": {}},
                    {"absent": {}},
                    {"absent": {}},
                    {"absent": {}},
                    {"absent": {}},
                    {"absent": {}},
                    {"absent": {}},
                    {"absent": {}},
                ]
            },
            id="memory_fixed_empty_slots",
        ),
    ],
)
def test_generated_messages_preserve_payloads_presence_and_variants(contract, name, fields):
    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.{name}"))
    expected = message(**fields)
    wire = expected.SerializeToString()
    assert message.FromString(wire) == expected


def test_runtime_struct_fields_are_all_modeled(contract):
    # Guard against losing runtime-owned fields during conversion, or adding a DTO
    # field later without extending the transport. Variants are tested separately.
    groups = {
        "codec/request.rs": ["AnnotatedLlmRequest", "GenerationParams", "FunctionDefinition", "ToolCall"],
        "codec/response.rs": [
            "AnnotatedLlmResponse",
            "Usage",
            "CacheWriteTokenCounts",
            "CostEstimate",
            "ResponseToolCall",
        ],
        "codec/optimization.rs": [
            "LlmOptimizationModel",
            "LlmOptimizationModelTransition",
            "LlmOptimizationTokens",
            "LlmOptimizationTokenImpact",
            "LlmOptimizationContribution",
            "LlmOptimizationSummary",
        ],
        "api/event.rs": [
            "BaseEvent",
            "ScopeEvent",
            "MarkEvent",
            "CategoryProfile",
            "EventSanitizeFields",
            "PendingMarkSpec",
            "DataSchema",
        ],
        "api/llm.rs": ["LlmRequest", "LlmRequestInterceptOutcome"],
        "api/tool.rs": ["ToolExecutionResult"],
        "plugin.rs": ["ConfigDiagnostic"],
        "diagnostics/snapshot.rs": ["DiagnosticSnapshot"],
        "diagnostics/accounting.rs": [
            "DiagnosticCount",
            "DiagnosticTotals",
            "DiagnosticLifetimeSummary",
            "DiagnosticRefreshSummary",
            "DiagnosticCounter",
            "DiagnosticWindowBucket",
            "DiagnosticWindowCoverage",
            "DiagnosticWindow",
        ],
        "diagnostics/coverage.rs": [
            "DiagnosticRetention",
            "DiagnosticLoss",
            "DiagnosticCapability",
            "DiagnosticCoverage",
        ],
        "diagnostics/query.rs": ["DiagnosticCursor", "DiagnosticQuery"],
        "diagnostics.rs": ["DiagnosticDiscovery", "DiagnosticProducerSupport"],
        "diagnostics/memory.rs": [
            "DiagnosticMemoryUsage",
            "EntryMemoryFootprint",
            "ComponentMemoryFootprint",
            "EntryMemoryUsage",
            "ComponentMemoryUsage",
            "MemorySelectionTotals",
            "DiagnosticMemoryPage",
        ],
        "diagnostics/resource.rs": [
            "ResourceMemoryMeasurement",
            "MemoryLimitEvent",
            "ResourceMemoryMetrics",
            "ResourceProcessIdentity",
            "ResourceMemorySample",
            "ResourceMemoryView",
        ],
        "diagnostics/event.rs": ["DiagnosticEvent", "RemoteDiagnosticOrigin", "DiagnosticDetail"],
    }
    for source, names in groups.items():
        rust = (ROOT / "crates/types/src" / source).read_text()
        for name in names:
            body = re.search(rf"pub struct {name}\s*\{{(.*?)\n\}}", rust, re.DOTALL)
            assert body is not None, name
            fields = set(re.findall(r"pub (\w+):", body[1]))
            transport_names = {
                "timestamp": "time",
                "observed_at": "observe_time",
                "captured_at": "capture_time",
                "completed_at": "complete_time",
                "producer_support": "producer_support_entries",
                "received_at": "receive_time",
                "last_failure_at": "last_failure_time",
                "started_at": "start_time",
                "last_observed_at": "last_observe_time",
                "registered_at": "register_time",
                "closed_at": "close_time",
                "time_to_first_chunk": "time_to_first_chunk_seconds",
            }
            fields = {transport_names.get(field, field) for field in fields}
            proto = contract.FindMessageTypeByName(f"{PACKAGE}.{name}")
            transport_fields = set(proto.fields_by_name)
            if name == "Usage":
                fields.remove("cache_write_tokens")
                fields.add("cache_write_token_counts")
                for legacy in ("cache_write_tokens", "cache_write_1h_tokens"):
                    assert proto.fields_by_name[legacy].GetOptions().deprecated
                    transport_fields.remove(legacy)
            assert fields == transport_fields, name


# Open application data is intentional; new entries require contract review.
OPEN_JSON_FIELDS = frozenset(
    """
    AnnotatedLlmRequest.api_specific
    AnnotatedLlmResponse.api_specific
    AnnotatedLlmRequest.extra
    AnnotatedLlmRequest.include
    AnnotatedLlmRequest.metadata
    AnnotatedLlmRequest.reasoning
    AnnotatedLlmRequest.truncation
    AnnotatedLlmResponse.extra
    BaseEvent.data
    BaseEvent.metadata
    CategoryProfile.extra
    CategoryProfile.tool_result_annotation
    DecodeLlmCodecResponseRequest.response
    EventSanitizeFields.data
    EventSanitizeFields.metadata
    FunctionDefinition.extra
    FunctionDefinition.parameters
    FunctionToolDefinition.extra
    ImageUrlContentPart.extra
    InvokeResponse.llm_response
    InvokeResponse.metadata_additions
    InvokeResponse.tool_value
    InvokeStreamResponse.value
    JsonArray.values
    JsonContentPart.extra
    JsonContentPart.value
    JsonObject.fields
    LlmNextResponse.value
    LlmOptimizationContribution.extra
    LlmOptimizationContribution.payload
    LlmRequest.content
    LlmRequest.headers
    LlmContentEntry.image
    LlmContentEntry.audio
    LlmContentEntry.file
    LlmContentPayload.json
    LlmSanitizeResponseInvocation.response
    LlmStreamNextResponse.value
    LogRequest.fields
    PendingMarkSpec.data
    PendingMarkSpec.metadata
    PopScopeRequest.metadata
    PopScopeRequest.output
    ProviderNativeComponent.value
    PushScopeRequest.data
    PushScopeRequest.input
    PushScopeRequest.metadata
    RegisterRequest.config
    ResponseToolCall.arguments
    TextContentPart.extra
    ToolCallItem.arguments
    ToolCallItem.extra
    ToolExecutionInvocation.args
    ToolExecutionResult.annotation
    ToolExecutionResult.result
    ToolNextRequest.value
    ToolResultContentPart.content
    ToolResultContentPart.extra
    ToolResultItem.extra
    ToolResultItem.output
    ToolUseContentPart.extra
    ToolUseContentPart.input
    ToolValueInvocation.value
    ValidateRequest.config
    """.split()
)
BINARY_FIELDS = {"JsonValue.json", "OpenDiagnosticSessionResponse.session_id", "SubmitDiagnosticRequest.session_id"}


def test_closed_structures_do_not_hide_json_or_reference_legacy_v1(contract):
    seen_json = set()
    seen_binary = set()
    for path in sorted((PROTO_ROOT / "nvidia/nemo/relay/worker/v1").glob("*.proto")):
        file = contract.FindFileByName(str(path.relative_to(PROTO_ROOT)))
        assert re.fullmatch(r"nvidia\.nemo\.relay\.worker\.v[1-9][0-9]*", file.package)
        assert all(not item.name.startswith("nemo/relay/worker/v1/") for item in file.dependencies)
        pending = list(file.message_types_by_name.values())
        while pending:
            message = pending.pop()
            pending.extend(message.nested_types)
            if message.GetOptions().map_entry:
                continue
            assert message.name not in {"JsonEnvelope", "JsonResult", "HostAck", "WorkerAck"}
            for field in message.fields:
                target = field.message_type
                if target is not None and target.GetOptions().map_entry:
                    value = target.fields_by_name["value"]
                else:
                    value = field
                name = field.full_name.removeprefix(PACKAGE + ".")
                if value.type == descriptor.FieldDescriptor.TYPE_BYTES:
                    seen_binary.add(name)
                target = value.message_type
                if target is not None:
                    assert target.full_name not in {
                        "google.protobuf.Value",
                        "google.protobuf.Struct",
                        "google.protobuf.ListValue",
                    }, name
                    if target.full_name in {f"{PACKAGE}.{name}" for name in ("JsonValue", "JsonObject", "JsonArray")}:
                        seen_json.add(name)
                if field.name == "error":
                    assert field.containing_oneof is not None
    assert seen_json == OPEN_JSON_FIELDS
    assert seen_binary == BINARY_FIELDS


def test_real_oneofs_require_an_alternative(contract):
    required = contract.FindExtensionByName("buf.validate.oneof")
    for path in sorted((PROTO_ROOT / "nvidia/nemo/relay/worker/v1").glob("*.proto")):
        file = descriptor_pb2.FileDescriptorProto()
        contract.FindFileByName(str(path.relative_to(PROTO_ROOT))).CopyToProto(file)
        pending = list(file.message_type)
        while pending:
            message = pending.pop()
            pending.extend(message.nested_type)
            optional = {field.oneof_index for field in message.field if field.proto3_optional}
            for index, oneof in enumerate(message.oneof_decl):
                if index not in optional:
                    assert oneof.options.Extensions[required].required, f"{message.name}.{oneof.name}"


def test_contract_imports_only_stable_packages(contract):
    for path in sorted((PROTO_ROOT / "nvidia/nemo/relay/worker/v1").glob("*.proto")):
        _assert_stable_imports(contract.FindFileByName(str(path.relative_to(PROTO_ROOT))))


def _assert_stable_imports(root):
    # Buf's stable-package rule recognizes conventional version suffixes, so enforce the
    # same policy explicitly for our stable worker namespace, including transitive imports.
    unstable = re.compile(r"v\d+(?:(?:p|_)\d+)*_?(?:alpha|beta|rc|test)[A-Za-z0-9_]*")
    pending = [root]
    visited = set()
    while pending:
        file = pending.pop()
        if file.name in visited:
            continue
        visited.add(file.name)
        assert not any(unstable.fullmatch(part) for part in file.package.split(".")), file.package
        pending.extend(file.dependencies)


@pytest.mark.parametrize("package", ["example.v1alpha1", "example.v1p2beta3", "example.v2_rc1"])
def test_stable_namespace_rejects_transitive_prerelease_imports(package):
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor_pb2.FileDescriptorProto(name="experimental.proto", package=package, syntax="proto3"))
    pool.Add(
        descriptor_pb2.FileDescriptorProto(
            name="bridge.proto", package="example.v1", syntax="proto3", dependency=["experimental.proto"]
        )
    )
    root = pool.Add(
        descriptor_pb2.FileDescriptorProto(
            name="contract.proto", package=PACKAGE, syntax="proto3", dependency=["bridge.proto"]
        )
    )
    with pytest.raises(AssertionError, match=re.escape(package)):
        _assert_stable_imports(root)


def test_optional_collections_and_explicit_default_values_keep_presence(contract):
    request = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.AnnotatedLlmRequest"))
    value = request()
    assert not value.HasField("tools")
    value.tools.SetInParent()
    value.store = False
    value.max_output_tokens = 0
    decoded = request.FromString(value.SerializeToString())
    assert decoded.HasField("tools") and not decoded.tools.values
    assert decoded.HasField("store") and decoded.store is False
    assert decoded.HasField("max_output_tokens") and decoded.max_output_tokens == 0


def test_llm_request_headers_are_an_object_in_generated_clients(contract):
    request = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.LlmRequest"))
    # This malformed value was accepted by the v1 Python JSON envelope decoder.
    with pytest.raises(TypeError):
        request(headers=42)
    value = request(headers={"fields": {"x-provider": {"json": b'"custom"'}}}, content={"json": b"null"})
    decoded = request.FromString(value.SerializeToString())
    assert decoded.headers.fields["x-provider"].json == b'"custom"'
    assert decoded.content.json == b"null"


def test_invoke_alternatives_cover_every_nonstreaming_registration(contract):
    surfaces = contract.FindEnumTypeByName(f"{PACKAGE}.RegistrationSurface")
    invocation = contract.FindMessageTypeByName(f"{PACKAGE}.InvokeRequest")
    expected = {v.name.removeprefix("REGISTRATION_SURFACE_").lower() for v in surfaces.values}
    expected -= {"unspecified", "llm_stream_execution_intercept"}
    assert {field.name for field in invocation.oneofs_by_name["invocation"].fields} == expected
    assert "surface" not in invocation.fields_by_name
    stream = contract.FindMessageTypeByName(f"{PACKAGE}.InvokeStreamRequest")
    assert "response_codec" not in stream.fields_by_name


def test_current_descriptors_include_the_complete_import_graph(contract):
    pending = [
        contract.FindFileByName(str(path.relative_to(PROTO_ROOT)))
        for path in sorted((PROTO_ROOT / "nvidia/nemo/relay/worker/v1").glob("*.proto"))
    ]
    names = set()
    while pending:
        file = pending.pop()
        if file.name not in names:
            names.add(file.name)
            pending.extend(file.dependencies)
    assert "nemo/relay/worker/v1/plugin_worker.proto" not in names
    assert "nvidia/nemo/relay/worker/v1/plugin_worker_service.proto" in names
    assert "google/protobuf/timestamp.proto" in names
    assert "buf/validate/validate.proto" in names


@pytest.mark.parametrize(
    ("name", "valid", "invalid"),
    [
        (
            "ProtocolIdentity",
            {"name": "nvidia.nemo.relay.worker.v1", "version": "1.0.0"},
            {"name": "grpc-v1", "version": "1.0.0"},
        ),
        ("Credentials", {"activation_id": "activation", "auth_token": "secret"}, {"activation_id": "activation"}),
        ("ToolExecutionResult", {"result": {"json": b"null"}}, {}),
        (
            "DiagnosticObservation",
            {"component": 1, "code": 1, "severity": 3, "count": 1},
            {"component": 0, "code": 1, "severity": 3, "count": 1},
        ),
        (
            "DiagnosticObservation",
            {"component": 1, "code": 1, "severity": 3, "count": 1},
            {"component": 1, "code": 0, "severity": 3, "count": 1},
        ),
        (
            "DiagnosticObservation",
            {"component": 1, "code": 1, "severity": 3, "count": 1},
            {"component": 1, "code": 1, "severity": 3, "count": 0},
        ),
        (
            "DiagnosticObservation",
            {"component": 1, "code": 1, "severity": 3, "count": 1},
            {"component": 1, "code": 1, "severity": 99, "count": 1},
        ),
        ("GenerationParams", {"temperature": 0.0}, {"temperature": float("inf")}),
        ("LlmMessage", {"assistant": {}}, {}),
        ("LlmCodecContext", {"codec": {"kind": "none"}}, {"codec": {"kind": ""}}),
        ("JsonValue", {"json": b"null"}, {"json": b""}),
        ("WorkerError", {"code": "upstream", "http_status": 400}, {"code": "upstream", "http_status": 399}),
        ("WorkerError", {"code": "upstream", "http_status": 599}, {"code": "upstream", "http_status": 600}),
        (
            "LlmOptimizationContribution",
            {"producer": "plugin", "kind": "custom", "applied": False},
            {"producer": "plugin", "kind": "custom", "id": "invalid-uuid"},
        ),
        (
            "LlmOptimizationContribution",
            {
                "producer": "plugin",
                "kind": "custom",
                "payload": {"json": b"null"},
                "payload_schema": {"name": "custom", "version": "1"},
            },
            {"producer": "plugin", "kind": "custom", "payload": {"json": b"null"}},
        ),
    ],
)
def test_semantic_validation_accepts_valid_and_rejects_invalid_messages(contract, name, valid, invalid):
    protovalidate: Any = importlib.import_module("protovalidate")

    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.{name}"))
    protovalidate.validate(message(**valid))
    with pytest.raises(protovalidate.ValidationError):
        protovalidate.validate(message(**invalid))


@pytest.mark.parametrize("status", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("evidence", [0, 1, 2, 3, 999])
def test_diagnostic_receipt_evidence_matches_recording_status(contract, status, evidence):
    protovalidate: Any = importlib.import_module("protovalidate")
    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.DiagnosticReceipt"))
    receipt = message(
        submission_sequence=1,
        status=status,
        evidence=evidence,
        component=1 if status == 4 else 0,
        code_count=1 if status == 4 else 0,
    )
    if evidence in {0, 1, 2, 3} and ((status == 1) == (evidence != 0)):
        protovalidate.validate(receipt)
    else:
        with pytest.raises(protovalidate.ValidationError):
            protovalidate.validate(receipt)


@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("code_count", [0, 1])
def test_registered_diagnostic_receipt_requires_usable_handles(contract, component, code_count):
    protovalidate: Any = importlib.import_module("protovalidate")
    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.DiagnosticReceipt"))
    receipt = message(submission_sequence=1, status=4, component=component, code_count=code_count)
    if component and code_count:
        protovalidate.validate(receipt)
    else:
        with pytest.raises(protovalidate.ValidationError):
            protovalidate.validate(receipt)


@pytest.mark.parametrize(
    "name,field",
    [
        ("PushScopeResponse", "scope_handle_id"),
        ("CreateScopeStackResponse", "scope_stack_id"),
        ("RegisterConditionalMiddlewareGuardrailResponse", "handle"),
    ],
)
def test_created_runtime_handles_must_be_nonempty(contract, name, field):
    protovalidate: Any = importlib.import_module("protovalidate")
    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.{name}"))
    protovalidate.validate(message(**{field: "host-issued-handle"}))
    with pytest.raises(protovalidate.ValidationError):
        protovalidate.validate(message())


def test_every_message_has_compilable_validation_rules(contract):
    protovalidate: Any = importlib.import_module("protovalidate")

    # Compile CEL/rules for the entire graph. Empty messages may violate rules,
    # but every message's rules must compile.
    for path in sorted((PROTO_ROOT / "nvidia/nemo/relay/worker/v1").glob("*.proto")):
        file = contract.FindFileByName(str(path.relative_to(PROTO_ROOT)))
        for message_descriptor in file.message_types_by_name.values():
            message = message_factory.GetMessageClass(message_descriptor)
            try:
                protovalidate.validate(message())
            except protovalidate.ValidationError:
                pass


@pytest.mark.parametrize("timeout", [{"nanos": 999999}, {"seconds": 10, "nanos": 1}])
def test_retry_timeout_duration_enforces_advertised_bounds(contract, timeout):
    protovalidate: Any = importlib.import_module("protovalidate")
    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.DiagnosticCapabilities"))
    capabilities = message(
        protocol={"name": PACKAGE, "version": "1.0.0"},
        snapshot_contract_version="1.0.0",
        provenance_contract_version="1.0.0",
        max_components=1,
        max_request_bytes=258,
        max_response_bytes=256,
        max_pending_submissions=1,
        retry_attempts=1,
        retry_timeout_duration={"nanos": 1000000},
    )
    protovalidate.validate(capabilities)
    capabilities.retry_timeout_duration.seconds = timeout.get("seconds", 0)
    capabilities.retry_timeout_duration.nanos = timeout.get("nanos", 0)
    with pytest.raises(protovalidate.ValidationError):
        protovalidate.validate(capabilities)


def test_full_diagnostic_query_preserves_selectors_and_presence(contract):
    protovalidate: Any = importlib.import_module("protovalidate")
    message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.DiagnosticQuery"))
    query = message(
        schema_version=1,
        component_key="plugin.lookup",
        component_instance_id="00000000-0000-4000-8000-000000000001",
        origin_generation=2**64 - 1,
        code="plugin.operation.failed",
        domain=3,
        plugin_family=2,
        failure_class="timeout",
        ownership={"configuration_id": "config-a"},
        after_sequence=0,
        max_components=0,
        max_events=0,
        max_response_bytes=0,
        continuation={"source_instance_id": "00000000-0000-4000-8000-000000000002", "scope": 2**64 - 1},
    )
    protovalidate.validate(query)
    decoded = message.FromString(query.SerializeToString())
    assert decoded == query
    assert decoded.HasField("max_events") and decoded.max_events == 0
    assert decoded.HasField("after_sequence") and decoded.after_sequence == 0
    assert decoded.continuation.scope == 2**64 - 1
    assert not message().HasField("max_events")


def test_diagnostic_read_failures_are_typed_and_never_empty_success(contract):
    protovalidate: Any = importlib.import_module("protovalidate")
    for method in ("ReadDiagnosticPage", "GetDiagnosticDiscovery", "InspectDiagnosticMemory"):
        message = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.{method}Response"))
        reason = (
            contract.FindEnumTypeByName(f"{PACKAGE}.DiagnosticReadReason")
            .values_by_name["DIAGNOSTIC_READ_REASON_BUSY"]
            .number
        )
        response = message(read_reason=reason)
        protovalidate.validate(response)
        assert message.FromString(response.SerializeToString()).WhichOneof("result") == "read_reason"
        with pytest.raises(protovalidate.ValidationError):
            protovalidate.validate(message())


def test_resource_memory_values_preserve_numeric_kinds_and_fixed_slots(contract):
    protovalidate: Any = importlib.import_module("protovalidate")
    number = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.ResourceMemoryNumber"))
    for value in (number(integer=2**64 - 1), number(decimal=0.125), number(non_finite={})):
        protovalidate.validate(value)
        assert number.FromString(value.SerializeToString()) == value
    with pytest.raises(protovalidate.ValidationError):
        protovalidate.validate(number(decimal=float("nan")))
    metrics = message_factory.GetMessageClass(contract.FindMessageTypeByName(f"{PACKAGE}.ResourceMemoryMetrics"))
    valid = metrics(limit_events=[{"absent": {}} for _ in range(8)])
    protovalidate.validate(valid)
    assert len(metrics.FromString(valid.SerializeToString()).limit_events) == 8
    for invalid in (metrics(), metrics(limit_events=[{} for _ in range(8)])):
        with pytest.raises(protovalidate.ValidationError):
            protovalidate.validate(invalid)


def test_diagnostic_enums_cover_shared_runtime_variants(contract):
    groups = {
        "diagnostics/catalogue.rs": ("DiagnosticDomain", "PluginDiagnosticFamily"),
        "diagnostics/reason.rs": ("DiagnosticReadReason",),
        "diagnostics/resource.rs": (
            "ResourceMemoryAvailability",
            "ResourceSampleStream",
            "ResourceAssociation",
            "MemoryFieldScope",
        ),
        "api/resource_metrics.rs": (
            "ResourceMeasurementScope",
            "ResourceOperatingSystem",
            "CapacityUnit",
            "DurationUnit",
            "CountUnit",
            "ResourceLimitResource",
            "ResourceLimitEventKind",
        ),
    }
    for filename, enums in groups.items():
        source = (ROOT / "crates/types/src" / filename).read_text()
        for name in enums:
            body = re.search(rf"pub enum {name}\s*\{{(.*?)\n\}}", source, re.DOTALL)
            assert body is not None
            variants = set(re.findall(r"^\s+(\w+),", body[1], re.MULTILINE))
            prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper() + "_"
            expected = {prefix + re.sub(r"(?<!^)(?=[A-Z])", "_", variant).upper() for variant in variants}
            actual = contract.FindEnumTypeByName(f"{PACKAGE}.{name}")
            assert {value.name for value in actual.values if value.number} == expected


def test_diagnostic_reads_are_independent_of_delivery_admission(contract):
    for name in ("GetDiagnosticDiscovery", "ReadDiagnosticPage", "InspectDiagnosticMemory"):
        request = contract.FindMessageTypeByName(f"{PACKAGE}.{name}Request")
        assert "credentials" in request.fields_by_name
        assert "session_id" not in request.fields_by_name
    capabilities = message_factory.GetMessageClass(
        contract.FindMessageTypeByName(f"{PACKAGE}.DiagnosticCapabilities")
    )()
    assert not capabilities.can_read_context
    assert not capabilities.can_read_discovery
    assert not capabilities.can_inspect_memory
