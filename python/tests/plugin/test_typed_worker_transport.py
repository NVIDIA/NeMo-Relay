# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the active typed transport through a real gRPC server."""

import asyncio

import grpc
import protovalidate
import pytest

from nemo_relay_plugin import PluginContext, PluginRuntime, WorkerPlugin
from nemo_relay_plugin._api import _WorkerService
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    common_pb2 as common_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    diagnostic_memory_pb2 as diagnostic_memory_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import diagnostic_resource_pb2 as resource_pb
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    diagnostics_pb2 as diagnostics_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    event_pb2 as event_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    llm_pb2 as llm_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    plugin_worker_service_pb2 as worker_pb,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    plugin_worker_service_pb2_grpc as worker_grpc,
)
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import (
    runtime_pb2 as runtime_pb,
)
from nemo_relay_plugin._protocol import PACKAGE, VERSION, pack, unpack, validate, validate_json


class Plugin(WorkerPlugin):
    plugin_id = "test.typed"

    def __init__(self):
        self.registrations = 0

    def register(self, ctx: PluginContext, config):
        self.registrations += 1
        ctx.register_tool_request_intercept("echo", lambda name, args: args)
        ctx.register_llm_sanitize_request_guardrail("unchanged", lambda request, context: None)
        ctx.register_llm_sanitize_response_guardrail("response", lambda response, context: response)


@pytest.mark.parametrize(
    ("message_type", "value", "expected_error"),
    [
        (llm_pb.LlmMessage, {}, "LlmMessage.role must be a string"),
        (llm_pb.ContentPart, {"type": None}, "ContentPart.type must be a string"),
        (
            diagnostics_pb.DiagnosticObservation,
            {"severity": "invalid"},
            "DiagnosticObservation.severity has unknown value 'invalid'",
        ),
    ],
)
def test_protocol_packing_rejects_missing_tags_and_unknown_enum_values(message_type, value, expected_error):
    with pytest.raises(ValueError) as error:
        pack(message_type, value)
    assert str(error.value) == expected_error


@pytest.mark.asyncio
async def test_typed_lifecycle_and_invocation():
    plugin = Plugin()
    runtime = PluginRuntime(activation_id="act", auth_token="token", host_stub=None)
    service = _WorkerService(plugin, runtime, asyncio.Event())
    server = grpc.aio.server()
    worker_grpc.add_PluginWorkerServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    credentials = common_pb.Credentials(activation_id="act", auth_token="token")
    codec = runtime_pb.LlmCodecContext(codec=runtime_pb.LlmCodecReference(kind="none"))
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = worker_grpc.PluginWorkerServiceStub(channel)
            handshake = await stub.Handshake(
                worker_pb.HandshakeRequest(
                    credentials=credentials,
                    plugin_id=plugin.plugin_id,
                    relay_version="0.11.0",
                    host_endpoint="http://127.0.0.1:1",
                    protocol=common_pb.ProtocolIdentity(name=PACKAGE, version=VERSION),
                )
            )
            validate(handshake)
            assert handshake.identity.protocol.name == PACKAGE
            request = worker_pb.RegisterRequest(
                credentials=credentials, plugin_id=plugin.plugin_id, config=pack(common_pb.JsonValue, None)
            )
            first, second = await asyncio.gather(stub.Register(request), stub.Register(request))
            validate(first)
            assert first == second
            assert plugin.registrations == 1
            response = await stub.Invoke(
                worker_pb.InvokeRequest(
                    credentials=credentials,
                    invocation_id="echo",
                    registration_name="echo",
                    tool_request_intercept=runtime_pb.ToolValueInvocation(
                        tool_name="echo", value=pack(common_pb.JsonValue, {"value": None, "large": 2**64 - 1})
                    ),
                )
            )
            validate(response)
            assert unpack(response.tool_value) == {"value": None, "large": 2**64 - 1}
            response = await stub.Invoke(
                worker_pb.InvokeRequest(
                    credentials=credentials,
                    invocation_id="request",
                    registration_name="unchanged",
                    llm_sanitize_request_guardrail=runtime_pb.LlmSanitizeRequestInvocation(
                        request=pack(llm_pb.LlmRequest, {"headers": {}, "content": {}}), context=codec
                    ),
                )
            )
            assert response.WhichOneof("result") == "unchanged"
            response = await stub.Invoke(
                worker_pb.InvokeRequest(
                    credentials=credentials,
                    invocation_id="response",
                    registration_name="response",
                    llm_sanitize_response_guardrail=runtime_pb.LlmSanitizeResponseInvocation(
                        response=pack(common_pb.JsonValue, {"text": "ok"}), context=codec
                    ),
                )
            )
            assert response.WhichOneof("result") == "llm_response"
            assert unpack(response.llm_response) == {"text": "ok"}
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Invoke(
                    worker_pb.InvokeRequest(credentials=credentials, invocation_id="invalid", registration_name="echo")
                )
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            request.config.CopyFrom(pack(common_pb.JsonValue, {"changed": True}))
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Register(request)
            assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Health(
                    worker_pb.HealthRequest(credentials=common_pb.Credentials(activation_id="act", auth_token="wrong"))
                )
            assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
            await stub.Shutdown(worker_pb.ShutdownRequest(credentials=credentials))
            assert service._shutdown_event.is_set()
    finally:
        await server.stop(0)


@pytest.mark.parametrize(
    "name,fixture",
    [
        (diagnostics_pb.DiagnosticSnapshot, "diagnostics/diagnostic-snapshot-schema-1.json"),
        (diagnostic_memory_pb.DiagnosticMemoryPage, "diagnostics/diagnostic-memory-schema-1.json"),
        (llm_pb.LlmOptimizationContribution, "llm_optimization_contribution_v1.json"),
    ],
)
def test_domain_fixtures_roundtrip(name, fixture):
    import json
    from pathlib import Path

    value = json.loads((Path(__file__).resolve().parents[3] / "crates/types/tests/fixtures" / fixture).read_text())
    wire = pack(name, value)
    validate(wire)
    assert unpack(wire) == value


@pytest.mark.asyncio
async def test_shutdown_fences_registration_in_progress():
    started, release = asyncio.Event(), asyncio.Event()

    class DelayedPlugin(Plugin):
        async def register(self, ctx, config):
            started.set()
            await release.wait()
            super().register(ctx, config)

    plugin = DelayedPlugin()
    service = _WorkerService(
        plugin, PluginRuntime(activation_id="act", auth_token="token", host_stub=None), asyncio.Event()
    )
    server = grpc.aio.server()
    worker_grpc.add_PluginWorkerServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    credentials = common_pb.Credentials(activation_id="act", auth_token="token")
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = worker_grpc.PluginWorkerServiceStub(channel)
            registration = stub.Register(
                worker_pb.RegisterRequest(
                    credentials=credentials, plugin_id=plugin.plugin_id, config=pack(common_pb.JsonValue, None)
                )
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            await stub.Shutdown(worker_pb.ShutdownRequest(credentials=credentials))
            health = await stub.Health(worker_pb.HealthRequest(credentials=credentials))
            assert health.status == runtime_pb.WORKER_HEALTH_STATUS_UNHEALTHY
            release.set()
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await registration
            assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            assert not service._handlers.registrations
    finally:
        release.set()
        await server.stop(0)


@pytest.mark.parametrize("kind", ["builtin", "runtime", "dynamic.example"])
def test_open_codec_reference_and_provider_data(kind):
    reference = runtime_pb.LlmCodecContext(
        codec=runtime_pb.LlmCodecReference(kind=kind, id="future.codec/v7"), capability_id="invocation-capability"
    )
    validate(reference)
    value = {
        "messages": [],
        "api_specific": {
            "api": "custom",
            "api_name": "dynamic.example/v7",
            "data": {"future": [None, False, 2**64 - 1]},
        },
    }
    wire = pack(llm_pb.AnnotatedLlmRequest, value)
    validate(wire)
    assert unpack(wire)["api_specific"] == value["api_specific"]
    # The transport accepts codec-owned shapes without interpreting provider semantics.
    wire.api_specific.CopyFrom(pack(common_pb.JsonValue, {"arbitrary": [1, None, "value"]}))
    validate(wire)
    assert unpack(wire)["api_specific"] == {"arbitrary": [1, None, "value"]}


def test_optional_codec_data_null_matches_native_absence():
    wire = pack(llm_pb.AnnotatedLlmRequest, {"messages": [], "api_specific": None})
    assert not wire.HasField("api_specific")


@pytest.mark.parametrize(
    "message_type,value",
    [
        (llm_pb.MessageContent, ""),
        (llm_pb.MessageContent, []),
        (llm_pb.MessageContent, [{"type": "image", "image": {"source": "custom", "data": None}}]),
        (llm_pb.ContentPart, {"type": "refusal", "refusal": "declined", "custom": True}),
        (llm_pb.LlmMessage, {"role": "function", "name": "lookup", "content": None}),
        (llm_pb.ToolChoice, "auto"),
        (llm_pb.ToolChoice, {"type": "function", "function": {"name": "lookup"}}),
        (llm_pb.ToolChoice, {"provider": "custom", "kind": "future", "value": {"tools": []}}),
        (llm_pb.FinishReason, "complete"),
        (llm_pb.FinishReason, {"unknown": "provider.stop"}),
        (resource_pb.ResourceMemoryNumber, {"kind": "integer", "value": str(2**64 - 1)}),
        (resource_pb.ResourceMemoryNumber, {"kind": "decimal", "value": 0.25}),
        (resource_pb.ResourceMemoryNumber, {"kind": "non_finite"}),
        (resource_pb.ResourceMemoryUnit, {"dimension": "capacity", "name": "bytes"}),
    ],
)
def test_callback_representations_preserve_variants_and_presence(message_type, value):
    wire = pack(message_type, value)
    validate(wire)
    assert unpack(wire) == value


@pytest.mark.parametrize("message_type", [llm_pb.ContentPart, llm_pb.LlmMessage, llm_pb.MessageContent])
def test_callback_decode_rejects_unselected_variants(message_type):
    with pytest.raises(ValueError, match="missing .* alternative"):
        unpack(message_type())


def test_json_validation_checks_nested_payloads_and_well_known_types():
    from google.protobuf.duration_pb2 import Duration
    from google.protobuf.timestamp_pb2 import Timestamp

    request = worker_pb.InvokeRequest(
        tool_request_intercept=runtime_pb.ToolValueInvocation(value=common_pb.JsonValue(json=b"NaN"))
    )
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        validate_json(request)
    for invalid in (Timestamp(seconds=253402300800), Timestamp(nanos=-1), Duration(seconds=1, nanos=-1)):
        with pytest.raises(ValueError, match="invalid google.protobuf"):
            validate_json(invalid)
    validate_json(Duration(seconds=-1, nanos=-1))


def test_annotated_tool_calls_and_choices_use_native_type_keys():
    value = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }
    wire = pack(llm_pb.AnnotatedLlmRequest, value)
    validate(wire)
    assert wire.messages[0].assistant.tool_calls.values[0].call_type == "function"
    assert wire.tool_choice.specific.choice_type == "function"
    assert unpack(llm_pb.AnnotatedLlmRequest.FromString(wire.SerializeToString())) == value


@pytest.mark.parametrize(
    "part,payload",
    [
        ("image", {"source": "image"}),
        ("audio", {"source": "audio"}),
        ("file", {"source": "file"}),
        ("refusal", "declined"),
        ("text", "accepted"),
    ],
)
def test_content_part_aliases_preserve_other_variant_and_wire_names_as_extensions(part, payload):
    value = {
        "type": part,
        "image": {"extra": "image"},
        "audio": {"extra": "audio"},
        "file": {"extra": "file"},
        "refusal": "extra refusal",
        "text": "extra text",
        "value": {"extra": "value"},
    }
    value[part] = payload
    wire = pack(llm_pb.ContentPart, value)
    validate(wire)
    child = getattr(wire, part)
    assert set(child.extra) == set(value) - {"type", part}
    if part in {"image", "audio", "file"}:
        assert unpack(child.value) == payload
    else:
        assert child.text == payload
    assert unpack(llm_pb.ContentPart.FromString(wire.SerializeToString())) == value


@pytest.mark.parametrize("part", ["image", "audio", "file", "refusal", "text"])
@pytest.mark.parametrize("shadow", ["payload", "type"])
def test_content_part_decode_rejects_extensions_shadowing_native_fields(part, shadow):
    payload = "content" if part in {"text", "refusal"} else {"source": part}
    wire = pack(llm_pb.ContentPart, {"type": part, part: payload})
    getattr(wire, part).extra[part if shadow == "payload" else "type"].CopyFrom(pack(common_pb.JsonValue, "shadow"))
    with pytest.raises(ValueError, match="extension fields shadow modeled fields"):
        unpack(wire)


def test_decode_rejects_extra_shadowing_an_absent_optional_modeled_field():
    wire = llm_pb.AnnotatedLlmRequest(extra={"model": pack(common_pb.JsonValue, 42)})
    validate(wire)
    assert not wire.HasField("model")
    with pytest.raises(ValueError, match="extension fields shadow modeled fields"):
        unpack(wire)


def test_literal_extra_extension_is_preserved():
    value = {"messages": [], "extra": {"provider": "value"}}
    wire = pack(llm_pb.AnnotatedLlmRequest, value)
    validate(wire)
    assert set(wire.extra) == {"extra"}
    assert unpack(llm_pb.AnnotatedLlmRequest.FromString(wire.SerializeToString())) == value


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mark", "scope"])
@pytest.mark.parametrize("profile", [None, {}])
async def test_root_event_callbacks_preserve_native_nulls_and_omissions(kind, profile):
    events = []

    class SubscriberPlugin(WorkerPlugin):
        plugin_id = "test.event-shape"

        def register(self, ctx, config):
            ctx.register_subscriber("capture", events.append)

    plugin = SubscriberPlugin()
    service = _WorkerService(
        plugin, PluginRuntime(activation_id="act", auth_token="token", host_stub=None), asyncio.Event()
    )
    server = grpc.aio.server()
    worker_grpc.add_PluginWorkerServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    credentials = common_pb.Credentials(activation_id="act", auth_token="token")
    base = event_pb.BaseEvent(atof_version="1", uuid="00000000-0000-0000-0000-000000000001", name="root")
    base.time.FromJsonString("2026-01-01T00:00:00Z")
    event = event_pb.Event()
    selected = getattr(event, kind)
    selected.base.CopyFrom(base)
    if profile is not None:
        selected.category_profile.SetInParent()
    expected = {
        "kind": kind,
        "atof_version": "1",
        "parent_uuid": None,
        "uuid": base.uuid,
        "timestamp": "2026-01-01T00:00:00Z",
        "name": "root",
        "data": None,
        "data_schema": None,
        "metadata": None,
        "category": None,
        "category_profile": profile,
    }
    if kind == "scope":
        selected.scope_category = event_pb.SCOPE_CATEGORY_START
        selected.category = "custom"
        expected.update(scope_category="start", attributes=[], category="custom")
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = worker_grpc.PluginWorkerServiceStub(channel)
            registered = await stub.Register(
                worker_pb.RegisterRequest(
                    credentials=credentials,
                    plugin_id=plugin.plugin_id,
                    config=pack(common_pb.JsonValue, None),
                )
            )
            assert registered.WhichOneof("result") == "plan"
            response = await stub.Invoke(
                worker_pb.InvokeRequest(
                    credentials=credentials,
                    invocation_id="root-event",
                    registration_name="capture",
                    subscriber=event,
                )
            )
            assert response.WhichOneof("result") == "completed"
            assert events == [expected]
    finally:
        await server.stop(0)


@pytest.mark.parametrize("currency,expected", [(None, "USD"), ("EUR", "EUR"), ("", "")])
def test_sanitizer_cost_uses_native_default_only_for_omitted_currency(currency, expected):
    cost = {"total": 1.0, "source": "provider_reported"}
    if currency is not None:
        cost["currency"] = currency
    value = {"category_profile": {"annotated_response": {"usage": {"cost": cost}}}}
    wire = pack(event_pb.EventSanitizeFields, value)
    validate(wire)
    assert wire.category_profile.annotated_response.usage.cost.currency == expected


@pytest.mark.parametrize("source", [True, False, 0, 1, 2, 1.0])
def test_sanitizer_cost_rejects_non_string_provenance(source):
    value = {"category_profile": {"annotated_response": {"usage": {"cost": {"source": source}}}}}
    with pytest.raises(ValueError, match="CostEstimate.source must be a string"):
        pack(event_pb.EventSanitizeFields, value)


@pytest.mark.parametrize("source", ["model_pricing", "provider_reported"])
def test_sanitizer_cost_preserves_provenance(source):
    value = {"category_profile": {"annotated_response": {"usage": {"cost": {"source": source}}}}}
    wire = pack(event_pb.EventSanitizeFields, value)
    validate(wire)
    assert unpack(wire)["category_profile"]["annotated_response"]["usage"]["cost"]["source"] == source


@pytest.mark.parametrize("value", [3.9, True, "12"])
def test_modeled_token_counts_reject_integer_coercion(value):
    with pytest.raises(ValueError, match="GenerationParams.max_tokens must be an integer"):
        pack(llm_pb.AnnotatedLlmRequest, {"messages": [], "params": {"max_tokens": value}})


@pytest.mark.parametrize("field,value", [("stream", 1), ("store", 0)])
def test_modeled_booleans_reject_numeric_coercion(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        pack(llm_pb.AnnotatedLlmRequest, {"messages": [], field: value})


def test_modeled_float_rejects_boolean_coercion():
    with pytest.raises(ValueError, match="temperature must be a number"):
        pack(llm_pb.GenerationParams, {"temperature": True})


@pytest.mark.parametrize("value", [0, 2**64 - 1])
def test_modeled_integers_and_native_diagnostic_decimal_strings_preserve_precision(value):
    request = pack(llm_pb.AnnotatedLlmRequest, {"messages": [], "params": {"max_tokens": value}})
    validate(request)
    assert unpack(request)["params"]["max_tokens"] == value
    counter = pack(diagnostics_pb.DiagnosticCount, {"value": str(value), "saturated": False})
    validate(counter)
    assert unpack(counter)["value"] == str(value)
    resource = pack(resource_pb.ResourceMemoryNumber, {"kind": "integer", "value": str(value)})
    validate(resource)
    assert unpack(resource) == {"kind": "integer", "value": str(value)}


@pytest.mark.parametrize("value", [{1: "value"}, {"nested": [{1: "first", "1": "second"}]}])
def test_json_payloads_reject_recursive_key_coercion(value):
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        pack(common_pb.JsonValue, value)


@pytest.mark.parametrize(
    "value,message",
    [
        ({"messages": [], "params": {"stop": "END"}}, "StringList must be an array"),
        ({"messages": [], "params": {"stop": {"first": "END"}}}, "StringList must be an array"),
        ({"messages": [], "params": {"stop": [b"END"]}}, "elements must be strings"),
        ({"messages": [], "params": [["max_tokens", 12]]}, "GenerationParams must be an object"),
        ({"messages": [], "model": b"example"}, "model must be a string"),
        ({"messages": [], "tools": {}}, "ToolDefinitionList must be an array"),
        ({"messages": {}}, "messages must be an array"),
        ({"messages": ""}, "messages must be an array"),
        ({"messages": None}, "messages must be an array"),
    ],
)
def test_modeled_containers_and_strings_reject_native_shape_coercion(value, message):
    with pytest.raises(ValueError, match=message):
        pack(llm_pb.AnnotatedLlmRequest, value)


def test_tagged_objects_and_required_strings_reject_non_native_values():
    with pytest.raises(ValueError, match="LlmMessage must be an object"):
        pack(llm_pb.LlmMessage, [("role", "user"), ("content", "text")])
    with pytest.raises(ValueError, match="FunctionCall.name must be a string"):
        pack(llm_pb.FunctionCall, {"name": None, "arguments": "{}"})
    with pytest.raises(ValueError, match="JsonObject must be an object"):
        pack(common_pb.JsonObject, [("key", "value")])


def test_arrays_and_optional_nulls_keep_native_json_serializer_semantics():
    value = {
        "messages": (),
        "params": {"stop": ("END", "STOP")},
        "tools": (),
        "metadata": None,
    }
    wire = pack(llm_pb.AnnotatedLlmRequest, value)
    validate(wire)
    assert unpack(wire) == {"messages": [], "params": {"stop": ["END", "STOP"]}, "tools": [], "metadata": None}
    absent = pack(llm_pb.AnnotatedLlmRequest, {"messages": [], "params": {"stop": None}, "model": None})
    validate(absent)
    assert not absent.params.HasField("stop")
    assert not absent.HasField("model")
    assert absent.HasField("params")
    for raw in (None, "text", {"stop": "END"}, ("END", "STOP")):
        assert unpack(pack(common_pb.JsonValue, raw)) == (["END", "STOP"] if isinstance(raw, tuple) else raw)


def test_cache_write_usage_roundtrip_preserves_native_list():
    value = {
        "prompt_tokens": 0,
        "completion_tokens": 2,
        "total_tokens": 2,
        "cache_read_tokens": 0,
        "uncached_input_tokens": 0,
        "cache_write_tokens": [
            {"count": 0},
            {"count": 3, "duration": 3600},
            {"count": 4, "duration": 0},
            {"count": 5, "duration": 3600},
            {"count": 2**64 - 1, "duration": 2**64 - 1},
        ],
        "cost": {"total": 0, "currency": "USD", "source": "provider_reported"},
    }
    wire = pack(llm_pb.Usage, value)
    returned = llm_pb.Usage.FromString(wire.SerializeToString())
    validate(returned)
    assert unpack(returned) == value


@pytest.mark.parametrize(
    "value,expected",
    [
        ({}, {}),
        ({"cache_write_tokens": None}, {}),
        ({"cache_write_tokens": []}, {"cache_write_tokens": []}),
        ({"cache_write_tokens": [{"count": 0}]}, {"cache_write_tokens": [{"count": 0}]}),
        (
            {"cache_write_tokens": [{"count": 0, "duration": 0}]},
            {"cache_write_tokens": [{"count": 0, "duration": 0}]},
        ),
    ],
)
def test_cache_write_usage_optional_list_preserves_presence(value, expected):
    wire = llm_pb.Usage.FromString(pack(llm_pb.Usage, value).SerializeToString())
    validate(wire)
    assert wire.HasField("cache_write_token_counts") is ("cache_write_tokens" in expected)
    assert wire.HasField("cache_write_tokens") is ("cache_write_tokens" in expected)
    assert not wire.HasField("cache_write_1h_tokens")
    assert unpack(wire) == expected


@pytest.mark.parametrize(
    "writes,total,one_hour",
    [
        ([], 0, None),
        ([{"count": 0, "duration": 3600}], 0, 0),
        ([{"count": 3}, {"count": 4, "duration": 3600}, {"count": 5, "duration": 3600}], 12, 9),
        (
            [{"count": 2**64 - 1, "duration": 3600}, {"count": 1, "duration": 3600}],
            2**64 - 1,
            2**64 - 1,
        ),
    ],
)
def test_cache_write_usage_legacy_projection_saturates_without_changing_entries(writes, total, one_hour):
    wire = llm_pb.Usage.FromString(pack(llm_pb.Usage, {"cache_write_tokens": writes}).SerializeToString())
    validate(wire)
    assert wire.cache_write_tokens == total
    assert wire.HasField("cache_write_1h_tokens") is (one_hour is not None)
    if one_hour is not None:
        assert wire.cache_write_1h_tokens == one_hour
    assert unpack(wire) == {"cache_write_tokens": writes}


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ({}, {}),
        ({"cache_write_1h_tokens": 3}, {}),
        ({"cache_write_1h_tokens": 0}, {}),
        ({"cache_write_tokens": 0}, {"cache_write_tokens": [{"count": 0}]}),
        ({"cache_write_tokens": 8}, {"cache_write_tokens": [{"count": 8}]}),
        (
            {"cache_write_tokens": 8, "cache_write_1h_tokens": 3},
            {"cache_write_tokens": [{"count": 5}, {"count": 3, "duration": 3600}]},
        ),
        (
            {"cache_write_tokens": 8, "cache_write_1h_tokens": 0},
            {"cache_write_tokens": [{"count": 8}]},
        ),
        (
            {"cache_write_tokens": 8, "cache_write_1h_tokens": 9},
            {"cache_write_tokens": [{"count": 8, "duration": 3600}]},
        ),
        (
            {"cache_write_tokens": 0, "cache_write_1h_tokens": 3},
            {"cache_write_tokens": [{"count": 0}]},
        ),
    ],
)
def test_cache_write_usage_legacy_wire_uses_authoritative_total_normalization(legacy, expected):
    wire = llm_pb.Usage.FromString(llm_pb.Usage(**legacy).SerializeToString())
    validate(wire)
    assert unpack(wire) == expected


@pytest.mark.parametrize("writes", [[], [{"count": 0}], [{"count": 2, "duration": 7}]])
def test_cache_write_usage_canonical_list_overrides_contradictory_legacy_counts(writes):
    wire = pack(llm_pb.Usage, {"cache_write_tokens": writes})
    wire.cache_write_tokens = 1
    wire.cache_write_1h_tokens = 9
    wire = llm_pb.Usage.FromString(wire.SerializeToString())
    validate(wire)
    assert unpack(wire) == {"cache_write_tokens": writes}


@pytest.mark.parametrize("key", ["cache_write_1h_tokens", "cache_write_token_counts"])
def test_cache_write_usage_callback_dictionary_rejects_transport_only_keys(key):
    with pytest.raises(ValueError, match=key):
        pack(llm_pb.Usage, {key: 3})


@pytest.mark.parametrize("value", [3, {}, "3"])
def test_cache_write_usage_callback_dictionary_rejects_non_array_cache_writes(value):
    with pytest.raises(ValueError, match="must be an array"):
        pack(llm_pb.Usage, {"cache_write_tokens": value})


@pytest.mark.parametrize("field", ["count", "duration"])
@pytest.mark.parametrize("value", [True, "3", 1.5, -1, 2**64])
def test_cache_write_usage_entries_reject_invalid_integer_values(field, value):
    with pytest.raises(ValueError):
        pack(llm_pb.Usage, {"cache_write_tokens": [{"count": 1, field: value}]})


def test_cache_write_usage_wire_entry_without_count_fails_validation():
    wire = llm_pb.Usage()
    wire.cache_write_token_counts.values.add(duration=3600)
    wire = llm_pb.Usage.FromString(wire.SerializeToString())
    with pytest.raises(protovalidate.ValidationError, match="count"):
        validate(wire)


def test_cache_write_usage_callback_entry_without_count_is_rejected():
    with pytest.raises((ValueError, protovalidate.ValidationError), match="count"):
        validate(pack(llm_pb.Usage, {"cache_write_tokens": [{"duration": 3600}]}))


def test_cache_write_usage_nested_response_and_optimization_usage_roundtrip():
    usage = {"cache_write_tokens": [{"count": 0, "duration": 0}, {"count": 2, "duration": 7}]}
    value = {
        "usage": usage,
        "optimization_summary": {
            "schema_version": "1.0.0",
            "calculation_version": "1.0.0",
            "status": "complete",
            "tokens_saved": {},
            "effective_usage": usage,
            "baseline_usage": {"cache_write_tokens": []},
        },
    }
    wire = pack(llm_pb.AnnotatedLlmResponse, value)
    wire = llm_pb.AnnotatedLlmResponse.FromString(wire.SerializeToString())
    validate(wire)
    returned = unpack(wire)
    assert returned["usage"] == usage
    assert returned["optimization_summary"]["effective_usage"] == usage
    assert returned["optimization_summary"]["baseline_usage"] == {"cache_write_tokens": []}
