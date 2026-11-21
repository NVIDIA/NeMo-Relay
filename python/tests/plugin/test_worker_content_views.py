# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker content adapters consume typed host projections, never local extraction policy."""

from __future__ import annotations

import asyncio
import gc
import inspect
from dataclasses import FrozenInstanceError

import grpc
import pytest

from nemo_relay_plugin import PluginRuntime, WorkerSdkError, _protocol, codecs
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import content_pb2 as cp
from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import relay_host_runtime_service_pb2 as hp


def _location(field=cp.LLM_CONTENT_FIELD_CONTENT, index=None):
    return cp.LlmContentLocation(root=cp.LLM_CONTENT_ROOT_REQUEST_MESSAGE, message_index=7, field=field, index=index)


def _payload(kind, value=None):
    if kind == "null":
        return cp.LlmContentPayload(null={})
    if kind == "string":
        return cp.LlmContentPayload(string=value)
    from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import common_pb2, llm_pb2

    cls = common_pb2.JsonValue if kind == "json" else llm_pb2.MessageContent
    return cp.LlmContentPayload(**{kind: _protocol.pack(cls, value)})


def _snapshot():
    from nemo_relay_plugin._proto.nvidia.nemo.relay.worker.v1 import common_pb2, llm_pb2

    def json_value(value):
        return _protocol.pack(common_pb2.JsonValue, value)

    return cp.LlmContentSnapshot(
        parts=[
            cp.LlmContentEntry(location=_location(index=0), text="not the extraction result"),
            cp.LlmContentEntry(location=_location(index=1), refusal=""),
            cp.LlmContentEntry(
                location=_location(index=2),
                tool_call=cp.LlmContentToolCall(
                    id="item", call_id="correlation", name="lookup", arguments=_payload("string", "{invalid}")
                ),
            ),
            cp.LlmContentEntry(
                location=_location(index=3),
                tool_result=cp.LlmContentToolResult(
                    call_id="correlation", is_error=False, payload=_payload("json", {"q": [1]})
                ),
            ),
            cp.LlmContentEntry(location=_location(index=4), image_url=llm_pb2.ImageUrl(url="https://example.com/i")),
            cp.LlmContentEntry(location=_location(index=5), image=json_value({"text": "not eligible"})),
            cp.LlmContentEntry(location=_location(index=6), audio=json_value({"data": "sound"})),
            cp.LlmContentEntry(location=_location(index=7), file=json_value({"id": "file"})),
            cp.LlmContentEntry(
                location=_location(index=8),
                provider_native=llm_pb2.ProviderNativeComponent(
                    provider="custom", kind="extension", value=json_value({"text": "also not eligible"})
                ),
            ),
        ],
        text_parts=[" host selected ", "", '{"literal": true}'],
    )


class Host:
    def __init__(self, response=None, error=None):
        self.requests = []
        self.response = response if response is not None else hp.InspectLlmContentResponse(snapshot=_snapshot())
        self.error = error

    async def InspectLlmContent(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def _runtime(host):
    return PluginRuntime(activation_id="activation", auth_token="token", host_stub=host)


@pytest.mark.asyncio
async def test_one_rpc_returns_authoritative_text_and_all_typed_kinds() -> None:
    host = Host()
    view = await codecs.last_message(_runtime(host), {"messages": [{"role": "user", "content": "client input"}]})
    assert view is not None
    assert len(host.requests) == 1
    request = host.requests[0]
    assert request.credentials.activation_id == "activation"
    assert request.credentials.auth_token == "token"
    assert request.selection.WhichOneof("kind") == "last_message"
    assert request.WhichOneof("source") == "request"
    assert _protocol.unpack(request.request)["messages"] == [{"role": "user", "content": "client input"}]
    # The client does not extract from its input or even reclassify returned raw entries.
    assert view.text() == ' host selected \n\n{"literal": true}'
    assert list(view.text_parts()) == [" host selected ", "", '{"literal": true}']
    entries = list(view.parts())
    assert [entry.kind for entry in entries] == [
        "text",
        "refusal",
        "tool_call",
        "tool_result",
        "image_url",
        "image",
        "audio",
        "file",
        "provider_native",
    ]
    assert entries[0].text == "not the extraction result"
    assert entries[1].text == ""
    call = entries[2].tool_call
    assert call is not None
    assert (call.id, call.call_id, call.name, call.arguments.kind, call.arguments.value) == (
        "item",
        "correlation",
        "lookup",
        "string",
        "{invalid}",
    )
    result = entries[3].tool_result
    assert result is not None
    assert (result.id, result.call_id, result.name, result.is_error) == (None, "correlation", None, False)
    assert result.payload.kind == "json"
    assert result.payload.value == {"q": [1]}
    assert entries[4].value == {"url": "https://example.com/i"}
    assert [entry.value for entry in entries[5:8]] == [{"text": "not eligible"}, {"data": "sound"}, {"id": "file"}]
    assert (entries[8].provider, entries[8].native_kind, entries[8].value) == (
        "custom",
        "extension",
        {"text": "also not eligible"},
    )
    position = entries[3].location
    assert (position.root, position.message_index, position.field, position.index) == (
        "request_message",
        7,
        "content",
        3,
    )
    assert len(host.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper,selection",
    [
        (codecs.last_message, "last_message"),
        (codecs.instructions_view, "instructions"),
        (codecs.system_prompt, "system_prompt"),
        (codecs.last_user_message, "last_user_message"),
    ],
)
async def test_request_helpers_delegate_selection_instead_of_searching_locally(helper, selection) -> None:
    host = Host(hp.InspectLlmContentResponse(no_selection={}))
    annotation = {
        "messages": [{"role": "system", "content": "local fallback forbidden"}],
        "instructions": "also forbidden",
    }
    assert await helper(_runtime(host), annotation) is None
    assert len(host.requests) == 1
    assert host.requests[0].selection.WhichOneof("kind") == selection


@pytest.mark.asyncio
async def test_signed_indices_and_standalone_and_response_sources() -> None:
    host = Host()
    runtime = _runtime(host)
    view = await codecs.message(runtime, {"messages": []}, -1)
    assert view.text() == ' host selected \n\n{"literal": true}'
    assert host.requests[-1].selection.message_index == -1
    for factory, value, source in [
        (codecs.MessageView.from_message, {"role": "function", "name": "f", "content": None}, "message"),
        (codecs.MessageView.from_content, "", "content"),
    ]:
        await factory(runtime, value)
        assert host.requests[-1].WhichOneof("source") == source
        assert host.requests[-1].selection.WhichOneof("kind") == "all"
    await codecs.output(runtime, {"message": "response", "usage": object(), "extra": object()})
    assert host.requests[-1].WhichOneof("source") == "response"
    assert host.requests[-1].selection.WhichOneof("kind") == "output"
    assert not host.requests[-1].response.HasField("usage")
    assert await codecs.response_text(runtime, {}) == ' host selected \n\n{"literal": true}'
    with pytest.raises(IndexError):
        await codecs.message(runtime, {}, 2**100)
    with pytest.raises(TypeError):
        await codecs.message(runtime, {}, 1.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("text_parts,expected", [([], None), ([""], ""), (["  "], "  "), (["", "", "x"], "\n\nx")])
async def test_absent_empty_and_literal_text_come_from_host(text_parts, expected) -> None:
    host = Host(hp.InspectLlmContentResponse(snapshot=cp.LlmContentSnapshot(text_parts=text_parts)))
    assert (await codecs.output(_runtime(host), {})).text() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,value",
    [
        ("null", None),
        ("string", "hidden"),
        ("json", "{invalid}"),
        ("json", None),
        ("content", [{"type": "text", "text": "hidden"}]),
    ],
)
async def test_payload_representation_survives_transport(kind, value) -> None:
    snapshot = cp.LlmContentSnapshot(
        parts=[
            cp.LlmContentEntry(location=_location(), tool_result=cp.LlmContentToolResult(payload=_payload(kind, value)))
        ]
    )
    host = Host(hp.InspectLlmContentResponse(snapshot=snapshot))
    view = await codecs.output(_runtime(host), {})
    result = next(view.parts()).tool_result
    assert result is not None
    assert result.payload.kind == kind
    assert result.payload.value == value
    assert view.text() is None


@pytest.mark.asyncio
async def test_snapshot_and_payload_lifetimes_are_independent_of_runtime_and_response() -> None:
    host = Host()
    runtime = _runtime(host)
    annotation = {"message": "input"}
    view = await codecs.output(runtime, annotation)
    entries = view.parts()
    text = view.text_parts()
    annotation["message"] = "edited"
    host.response.snapshot.parts.clear()
    host.response.snapshot.text_parts.clear()
    del annotation, host, runtime, view
    gc.collect()
    next(entries)
    next(entries)
    call_entry = next(entries)
    result_entry = next(entries)
    result = result_entry.tool_result
    assert result is not None
    value = result.payload.value
    value["q"].append(2)
    assert result.payload.value == {"q": [1]}
    assert list(text) == [" host selected ", "", '{"literal": true}']
    with pytest.raises(FrozenInstanceError):
        setattr(call_entry, "kind", "text")
    with pytest.raises(FrozenInstanceError):
        setattr(result_entry.location, "index", 99)
    with pytest.raises(TypeError):
        result.payload._value["q"] = (7,)
    native = list(entries)[-1]
    value = native.value
    value["text"] = "changed"
    assert native.value == {"text": "also not eligible"}


def _rpc_error(code):
    return grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata(), "fixture error")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,exception,match",
    [
        (grpc.StatusCode.UNIMPLEMENTED, WorkerSdkError, "unsupported"),
        (grpc.StatusCode.OUT_OF_RANGE, IndexError, "out of range"),
        (grpc.StatusCode.UNAUTHENTICATED, WorkerSdkError, "InspectLlmContent failed"),
        (grpc.StatusCode.FAILED_PRECONDITION, WorkerSdkError, "InspectLlmContent failed"),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, WorkerSdkError, "InspectLlmContent failed"),
    ],
)
async def test_host_errors_are_explicit_and_have_no_local_fallback(code, exception, match) -> None:
    host = Host(error=_rpc_error(code))
    with pytest.raises(exception, match=match):
        await codecs.last_message(_runtime(host), {"messages": [{"role": "user", "content": "never return this"}]})
    assert len(host.requests) == 1


@pytest.mark.asyncio
async def test_cancellation_does_not_retry_or_fall_back() -> None:
    started = asyncio.Event()

    class SlowHost:
        async def InspectLlmContent(self, request):
            started.set()
            await asyncio.Future()

    task = asyncio.create_task(codecs.last_message(_runtime(SlowHost()), {"messages": []}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_invalid_projections_and_inputs_fail_closed() -> None:
    host = Host(hp.InspectLlmContentResponse())
    with pytest.raises(WorkerSdkError):
        await codecs.last_message(_runtime(host), {})
    host = Host(hp.InspectLlmContentResponse(snapshot=cp.LlmContentSnapshot(parts=[cp.LlmContentEntry(text="x")])))
    with pytest.raises(WorkerSdkError):
        await codecs.output(_runtime(host), {})
    host = Host(hp.InspectLlmContentResponse(no_selection={}))
    with pytest.raises(WorkerSdkError, match="required content"):
        await codecs.output(_runtime(host), {})
    with pytest.raises(WorkerSdkError):
        await codecs.last_message(_runtime(host), [])
    with pytest.raises(WorkerSdkError):
        await codecs.MessageView.from_content(_runtime(host), object())
    with pytest.raises(TypeError):
        codecs.MessageView()


def test_public_content_api_is_documented() -> None:
    for name in codecs.__all__:
        value = getattr(codecs, name)
        assert inspect.getdoc(value), name
        if inspect.isclass(value):
            for member_name, member in inspect.getmembers_static(value):
                if member_name.startswith("_"):
                    continue
                if isinstance(member, property):
                    member = member.fget
                if inspect.isfunction(member):
                    assert inspect.getdoc(member), f"{name}.{member_name}"
