# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool codec round trips and observable LangChain result semantics."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal, cast
from unittest.mock import MagicMock

import pytest

import nemo_relay


@pytest.mark.parametrize("command", [False, True])
@pytest.mark.parametrize("status", ["success", "error"])
def test_tool_codec_round_trip_and_projection(command, status):
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.types import Command, Send

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    message = ToolMessage(
        content=["answer", {"type": "text", "text": "42"}],
        tool_call_id="call-1",
        status=status,
        artifact={"data": [1, 2]},
        additional_kwargs={"kind": "command", "value": "user-owned"},
        response_metadata={"source": "test"},
        name="lookup",
        id="message-1",
    )
    result = (
        Command(
            graph=Command.PARENT,
            update={"messages": [AIMessage(content="thinking"), message], "state": (1, {"kind": "send"})},
            resume={"answer": True},
            goto=[Send("next", {"messages": [message]}), "other"],
        )
        if command
        else message
    )
    encoded = LangChainToolResultCodec("call-1").to_json(result)
    assert encoded["content"] == [{"type": "text", "text": "answer"}, {"type": "text", "text": "42"}]
    assert encoded["is_error"] == (status == "error")
    # A new codec can restore JSON-serializable results, including nested native messages.
    restored = LangChainToolResultCodec("call-1").from_json(json.loads(json.dumps(encoded)))
    assert restored == result
    if command:
        assert isinstance(restored, Command)
        assert isinstance(restored.update["messages"][1], ToolMessage)
        assert isinstance(restored.goto[0], Send)
        assert isinstance(restored.update["state"], tuple)


@pytest.mark.parametrize("command", [False, True])
@pytest.mark.parametrize("replayed", [False, True])
@pytest.mark.parametrize("edit", [None, "status", "content"])
def test_tool_codec_preserves_mixed_content_unless_content_is_edited(command, replayed, edit):
    """Rebinding IDs or editing status must preserve the native content representation."""
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    content = ["answer", {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]
    message = ToolMessage(content=content, tool_call_id="call-1", id="message-1")
    result = Command(update={"messages": [message]}) if command else message
    encoded = json.loads(json.dumps(LangChainToolResultCodec("call-1").to_json(result)))
    if edit == "status":
        encoded["is_error"] = True
    elif edit == "content":
        encoded["content"] = ["edited", content[1]]
    restored = LangChainToolResultCodec("call-2" if replayed else "call-1").from_json(encoded)
    if command:
        assert isinstance(restored, Command)
        restored_message = restored.update["messages"][0]
    else:
        restored_message = restored
    assert isinstance(restored_message, ToolMessage)
    assert restored_message.content == (["edited", content[1]] if edit == "content" else content)
    assert restored_message.status == ("error" if edit == "status" else "success")
    assert restored_message.tool_call_id == ("call-2" if replayed else "call-1")
    assert restored_message.id == (None if replayed else "message-1")
    assert message.content == content


@pytest.mark.parametrize("message_key", ["type", "role"])
def test_replayed_graph_destination_rebinds_tool_message_dictionaries(message_key):
    """Dictionary tool messages in graph destinations receive fresh call and message IDs."""
    from langchain_core.messages import ToolMessage
    from langgraph.graph.message import add_messages
    from langgraph.types import Command, Send

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    message = {message_key: "tool", "content": "answer", "tool_call_id": "call-1", "id": "message-1"}
    unrelated = {message_key: "tool", "content": "other", "tool_call_id": "other", "id": "message-other"}
    command = Command(update={"state": 42}, goto=Send("next", {"messages": [message, unrelated]}))
    encoded = json.loads(json.dumps(LangChainToolResultCodec("call-1").to_json(command)))
    restored = LangChainToolResultCodec("call-2").from_json(encoded)
    assert isinstance(restored, Command)
    assert isinstance(restored.goto, Send)
    replayed_message, restored_unrelated = restored.goto.arg["messages"]
    assert replayed_message == {**message, "tool_call_id": "call-2", "id": None}
    assert restored_unrelated == unrelated
    history = [ToolMessage(content="original", tool_call_id="call-1", id="message-1")]
    assert len(add_messages(history, [replayed_message])) == 2


def test_tool_codec_rejects_invalid_cached_tool_call_id():
    """Reject malformed invocation metadata from a persisted tool result."""
    from langchain_core.messages import ToolMessage

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    encoded = LangChainToolResultCodec("call-1").to_json(ToolMessage(content="answer", tool_call_id="call-1"))
    encoded["cache"] = {"replayable": True, "tool_call_id": 5}
    with pytest.raises(ValueError, match="cached tool call ID"):
        LangChainToolResultCodec("call-2").from_json(encoded)


def test_results_cached_by_release_candidates_replay_in_a_new_call() -> None:
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command, Send

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    def kind(name: str, value: nemo_relay.Json) -> nemo_relay.JsonObject:
        return {"kind": name, "value": value}

    def text(value: str | None) -> nemo_relay.JsonObject:
        return kind("json", value)

    data = kind("dict", {"content": text("answer"), "tool_call_id": text("call-1"), "id": text("message-1")})
    message = kind("message", kind("dict", {"type": text("tool"), "data": data}))
    send = kind("send", kind("dict", {"node": text("next"), "arg": kind("dict", {})}))
    fields = {"graph": text(None), "update": kind("dict", {"messages": kind("list", [message])}), "resume": text(None)}
    cached: nemo_relay.JsonObject = {
        "schema": "nemo.relay.tool_result@1",
        "content": "answer",
        "is_error": False,
        "cache": {"replayable": True, "tool_call_id": "call-1"},
        "result": kind("command", kind("dict", {**fields, "goto": send})),
    }

    restored = LangChainToolResultCodec("call-2").from_json(cached)

    assert isinstance(restored, Command)
    assert restored.update == {"messages": [ToolMessage(content="answer", tool_call_id="call-2")]}
    assert restored.goto == Send("next", {})


@pytest.mark.parametrize(
    ("result", "error", "match"),
    [
        pytest.param({"kind": 5, "value": {}}, TypeError, "needs a kind", id="kind-not-text"),
        pytest.param({"kind": "json", "value": [1]}, ValueError, "Invalid encoded json value", id="scalar-shape"),
        pytest.param({"kind": "list", "value": {}}, ValueError, "Invalid encoded list value", id="container-shape"),
        pytest.param(
            {"kind": "dict_items", "value": [[{"kind": "json", "value": 1}]]},
            ValueError,
            "needs a key and a value",
            id="entry-shape",
        ),
        pytest.param(
            {"kind": "dict_items", "value": [[{"kind": "json", "value": 1}, {"kind": "json", "value": 1}]] * 2},
            ValueError,
            "repeats a key",
            id="repeated-key",
        ),
        pytest.param(
            {"kind": "dataclass", "type": "langgraph.types.Overwrite", "value": {"kind": {"kind": "json", "value": 1}}},
            ValueError,
            r"needs the fields \['type', 'value'\]",
            id="dataclass-fields",
        ),
        pytest.param(
            {"kind": "model", "type": "langchain_core.messages.tool.ToolMessage", "fields_set": ["other"], "value": {}},
            ValueError,
            "needs a list of its set fields",
            id="model-fields-set",
        ),
        pytest.param({"kind": "typed", "type": 5, "value": 1}, TypeError, "type name must be text", id="type-name"),
    ],
)
def test_malformed_encodings_are_rejected(result: nemo_relay.JsonObject, error: type[Exception], match: str) -> None:
    from langchain_core.messages import ToolMessage

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    codec = LangChainToolResultCodec("call-1")
    encoded = codec.to_json(ToolMessage(content="answer", tool_call_id="call-1"))
    encoded["result"] = result
    with pytest.raises(error, match=match):
        codec.from_json(encoded)


def test_custom_objects_are_recorded_with_types_and_stay_with_their_call() -> None:
    from dataclasses import dataclass

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    @dataclass
    class Artifact:
        answer: int

    artifact = Artifact(42)
    state = {42: object()}
    message = ToolMessage(content="answer", tool_call_id="call-1", artifact=artifact)
    result = Command(update={"messages": [message], "state": state})
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    assert encoded == LangChainToolResultCodec("call-1").to_json(result)
    assert encoded["cache"]["replayable"] is False
    update = encoded["result"]["value"]["update"]["value"]
    assert update["messages"]["value"][0]["value"]["artifact"] == {
        "kind": "dataclass",
        "type": f"{__name__}.{Artifact.__qualname__}",
        "value": {"answer": {"kind": "json", "value": 42}},
    }
    assert update["state"] == {
        "kind": "dict_items",
        "value": [
            [
                {"kind": "json", "value": 42},
                {"kind": "local", "type": "builtins.object", "ref": 0, "value": "<builtins.object>"},
            ]
        ],
    }
    restored = codec.from_json(encoded)
    assert isinstance(restored, Command)
    assert restored is result
    with pytest.raises(ValueError, match="Only the call that returned"):
        LangChainToolResultCodec("call-1").from_json(encoded)


@pytest.mark.parametrize("integration", ["langchain", "langgraph"])
@pytest.mark.parametrize("result_kind", ["message", "command", "dict_command", "role_dict_command", "list"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("status", ["success", "error"])
def test_tool_cache_replays_current_call_identity_or_runs_local_results_live(
    integration: str, result_kind: str, local: bool, status: str, subscribed_events: list[nemo_relay.Event]
) -> None:
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.graph.message import add_messages
    from langgraph.types import Command, Send

    from nemo_relay.adaptive import (
        AdaptiveConfig,
        AdaptiveRuntime,
        ResponseCacheConfig,
        ToolCacheConfig,
        ToolOverride,
    )
    from nemo_relay.integrations.langchain import NemoRelayMiddleware
    from nemo_relay.integrations.langgraph.tool_node import awrap_tool_call

    artifact = object() if local else {"answer": 42}
    calls = 0
    unrelated = ToolMessage(content="other", tool_call_id="other", id="message-other")

    async def handler(request: ToolCallRequest) -> ToolMessage | Command[Any] | list[ToolMessage | Command[Any]]:
        """Count live invocations and return the selected message or graph command shape."""
        nonlocal calls
        calls += 1
        call_id = request.tool_call["id"]
        message = ToolMessage(
            content="answer", tool_call_id=call_id, id=f"message-{call_id}", artifact=artifact, status=status
        )
        if result_kind == "message":
            return message
        written: ToolMessage | dict[str, object] = message
        if result_kind in ("dict_command", "role_dict_command"):
            written = message.model_dump(mode="python")
            if result_kind == "role_dict_command":
                written["role"] = written.pop("type")
        command = Command(
            update={"messages": [unrelated, written], "state": (1, 2)},
            goto=Send("next", {"messages": [written]}),
        )
        return [unrelated, command] if result_kind == "list" else command

    def projected_message(result: object, call_id: str) -> object:
        if result_kind == "list":
            assert isinstance(result, list)
            assert result[0] == unrelated
            result = result[1]
        if result_kind != "message":
            assert isinstance(result, Command)
            assert isinstance(result.update, dict)
            assert isinstance(result.goto, Send)
            assert result.update["messages"][0] == unrelated
            assert result.update["state"] == (1, 2)
            sent = result.goto.arg["messages"][0]
            assert (sent["tool_call_id"] if isinstance(sent, dict) else sent.tool_call_id) == call_id
            result = result.update["messages"][1]
            if isinstance(result, dict):
                result = ToolMessage.model_validate(result)
        return result

    async def run() -> None:
        """Exercise repeated calls through the integration and verify results and graph history."""
        runtime = AdaptiveRuntime(
            AdaptiveConfig(
                response_cache=ResponseCacheConfig(
                    namespace="framework-tool-replay",
                    tools=ToolCacheConfig(
                        enabled=True, cache_errors=True, overrides={"lookup": ToolOverride(cacheable=True)}
                    ),
                )
            ).to_dict()
        )
        await runtime.register()
        try:
            with nemo_relay.scope.scope("cache-replay", nemo_relay.ScopeType.Agent):
                history = []
                for call_id in ["call-1", "call-2", "call-3"]:
                    request = ToolCallRequest(
                        tool_call={"name": "lookup", "args": {"query": "same"}, "id": call_id},
                        tool=MagicMock(),
                        state={},
                        runtime=MagicMock(),
                    )
                    result = await (
                        NemoRelayMiddleware().awrap_tool_call(request, handler)
                        if integration == "langchain"
                        else awrap_tool_call(request, handler)
                    )
                    result = projected_message(result, call_id)
                    assert isinstance(result, ToolMessage)
                    assert result.content == "answer"
                    assert result.status == status
                    assert result.tool_call_id == call_id
                    assert result.id == (f"message-{call_id}" if local or call_id == "call-1" else None)
                    if local:
                        assert result.artifact is artifact
                    else:
                        assert result.artifact == artifact
                    history = add_messages(history, [result])
                assert len(history) == 3
        finally:
            await runtime.shutdown()
            await nemo_relay.subscribers.flush_async()

    asyncio.run(run())
    assert calls == (3 if local else 1)
    hits = [
        event
        for event in subscribed_events
        if event.name == "response_cache" and isinstance(event.data, dict) and event.data.get("status") == "hit"
    ]
    assert len(hits) == (0 if local else 2)


def test_replayed_graph_destination_binds_messages_without_display_content():
    """Rebind destination messages even when the command has no display projection."""
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command, Send

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    message = ToolMessage(content="answer", tool_call_id="call-1", id="message-1")
    command = Command(update={"state": 42}, goto=Send("next", {"messages": [message]}))
    encoded = LangChainToolResultCodec("call-1").to_json(command)
    restored = LangChainToolResultCodec("call-2").from_json(encoded)
    assert isinstance(restored, Command)
    assert isinstance(restored.goto, Send)
    assert restored.update == {"state": 42}
    assert restored.goto.arg["messages"][0].tool_call_id == "call-2"
    assert restored.goto.arg["messages"][0].id is None


@pytest.mark.parametrize("command", [False, True])
@pytest.mark.parametrize("custom_type", [False, True])
@pytest.mark.parametrize("edited", [False, True])
def test_custom_tool_message_class_and_fields_are_preserved(command, custom_type, edited):
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class CustomToolMessage(ToolMessage):
        source: str = "custom"

    class CustomKindToolMessage(CustomToolMessage):
        type: Literal["custom_tool"] = "custom_tool"  # ty: ignore[invalid-attribute-override]

    cls = CustomKindToolMessage if custom_type else CustomToolMessage
    artifact = object()
    message = cls(content="answer", tool_call_id="call-1", artifact=artifact)
    result = Command(update={"messages": [message]}, goto="next") if command else message
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    assert encoded["content"] == "answer"
    assert encoded["is_error"] is False
    if edited:
        encoded["content"] = "edited failure"
        encoded["is_error"] = True
    restored = codec.from_json(encoded)
    if command:
        assert isinstance(restored, Command)
        assert restored.goto == "next"
        restored_message = restored.update["messages"][0]
    else:
        restored_message = restored
    assert isinstance(restored_message, CustomToolMessage)
    assert type(restored_message) is cls
    assert restored_message.source == "custom"
    assert restored_message.artifact is artifact
    assert restored_message.content == ("edited failure" if edited else "answer")
    assert restored_message.status == ("error" if edited else "success")
    if not edited:
        assert restored_message is message
    assert message.content == "answer"
    assert message.status == "success"
    with pytest.raises(ValueError, match="Only the call that returned"):
        LangChainToolResultCodec("call-1").from_json(encoded)


@pytest.mark.parametrize(
    ("message_name", "fields"),
    [
        ("AIMessage", {"content": "context"}),
        ("AIMessageChunk", {"content": "context"}),
        ("ChatMessage", {"content": "context", "role": "assistant"}),
        ("ChatMessageChunk", {"content": "context", "role": "assistant"}),
        ("FunctionMessage", {"content": "context", "name": "lookup"}),
        ("FunctionMessageChunk", {"content": "context", "name": "lookup"}),
        ("HumanMessage", {"content": "context"}),
        ("HumanMessageChunk", {"content": "context"}),
        ("RemoveMessage", {"id": "remove-1"}),
        ("SystemMessage", {"content": "context"}),
        ("SystemMessageChunk", {"content": "context"}),
        ("ToolMessage", {"content": "context", "tool_call_id": "other"}),
        ("ToolMessageChunk", {"content": "context", "tool_call_id": "other"}),
    ],
)
def test_standard_message_classes_can_be_restored_by_a_fresh_codec(message_name, fields):
    from langchain_core import messages
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    message = getattr(messages, message_name)(**fields)
    matching = messages.ToolMessage(content="answer", tool_call_id="call-1")
    result = Command(update={"messages": [message, matching]})
    encoded = LangChainToolResultCodec("call-1").to_json(result)
    restored = LangChainToolResultCodec("call-1").from_json(json.loads(json.dumps(encoded)))
    assert isinstance(restored, Command)
    assert type(restored.update["messages"][0]) is type(message)
    assert restored.update["messages"][0] == message


def test_custom_messages_in_graph_state_keep_their_class():
    from langchain_core.messages import BaseMessage, ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class CustomMessage(BaseMessage):
        type: str = "custom"
        source: str = "custom"

    custom = CustomMessage(content="context")
    message = ToolMessage(content="answer", tool_call_id="call-1")
    codec = LangChainToolResultCodec("call-1")
    result = Command(update={"messages": [custom, message]})
    restored = codec.from_json(json.loads(json.dumps(codec.to_json(result))))
    assert isinstance(restored, Command)
    assert restored.update["messages"][0] is custom
    assert restored.update["messages"][1] == message


def test_tool_codec_rejects_unrecognized_payloads():
    from langchain_core.messages import ToolMessage

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    codec = LangChainToolResultCodec("call-1")
    for data in ["text", {"schema": "nemo.relay.tool_result@2"}]:
        with pytest.raises(ValueError, match="versioned Relay tool result"):
            codec.from_json(data)
    encoded = codec.to_json(ToolMessage(content="answer", tool_call_id="call-1"))
    encoded["result"] = {"kind": "import.module.Class", "value": {}}
    with pytest.raises(ValueError, match="Unsupported LangChain tool result kind"):
        codec.from_json(encoded)


@pytest.mark.parametrize("update_kind", ["dict", "list", "dict_messages", "role_messages"])
def test_command_projection_matches_call_id_and_applies_edits(update_kind):
    """Apply content and failure edits only to the message matching the active tool call."""
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    unrelated = ToolMessage(content="other failure", status="error", tool_call_id="other")
    matching = ToolMessage(content="answer", tool_call_id="target", artifact={"keep": True})
    messages = [unrelated, matching]
    if update_kind in ("dict_messages", "role_messages"):
        messages = [message.model_dump() for message in messages]
        if update_kind == "role_messages":
            for message in messages:
                message["role"] = message.pop("type")
    update = messages if update_kind == "list" else {"messages": messages, "state": {"keep": 42}}
    result = Command(update=update, goto="next")
    codec = LangChainToolResultCodec("target")
    encoded = codec.to_json(result)
    assert encoded["content"] == "answer"
    assert encoded["is_error"] is False
    encoded["content"] = "edited failure"
    encoded["is_error"] = True
    restored = codec.from_json(encoded)
    assert isinstance(restored, Command)
    updated_messages = restored.update if update_kind == "list" else restored.update["messages"]
    assert updated_messages[0] == messages[0]
    assert updated_messages[1].content == "edited failure"
    assert updated_messages[1].status == "error"
    assert updated_messages[1].artifact == {"keep": True}
    assert restored.goto == "next"
    assert matching.content == "answer"


def test_command_without_matching_message_has_no_projection():
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    for result in [
        Command(update={"state": 42}, goto="next"),
        Command(update={"messages": [ToolMessage(content="other", status="error", tool_call_id="other")]}),
    ]:
        codec = LangChainToolResultCodec("target")
        encoded = codec.to_json(result)
        assert encoded["content"] is None
        assert encoded["is_error"] is False
        assert codec.from_json(encoded) == result
        encoded["content"] = "unrelated"
        with pytest.raises(ValueError, match="without a matching ToolMessage"):
            codec.from_json(encoded)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("command", [False, True])
@pytest.mark.parametrize("status", ["success", "error"])
def test_middleware_records_normalized_content_and_status(subscribed_events, use_async, command, status):
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware

    request = ToolCallRequest(
        tool_call={"name": "lookup", "args": {}, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(),
    )
    message = ToolMessage(content="tool answer", status=status, tool_call_id="call-1")
    result = Command(update={"messages": [message], "state": 42}, goto="next") if command else message

    def handler(_request):
        return result

    async def ahandler(_request):
        return result

    middleware = NemoRelayMiddleware()
    exporter = nemo_relay.AtifExporter("tool-results", "agent", "1")
    exporter.register("langchain-tool-codec")
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            returned = (
                asyncio.run(middleware.awrap_tool_call(request, ahandler))
                if use_async
                else middleware.wrap_tool_call(request, handler)
            )
        nemo_relay.subscribers.flush()
        trajectory = cast(dict[str, Any], exporter.export())
    finally:
        exporter.deregister("langchain-tool-codec")
    observations = [result for step in trajectory["steps"] for result in step.get("observation", {}).get("results", [])]
    assert len(observations) == 1
    assert observations[0]["content"] == "tool answer"
    assert observations[0]["extra"].get("is_error") == (True if status == "error" else None)
    assert returned == result
    nemo_relay.subscribers.flush()
    event = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "lookup" and event.scope_category == "end"
    )
    assert event.category_profile["tool_call_id"] == "call-1"
    assert event.data["content"] == "tool answer"
    assert event.data["is_error"] == (status == "error")
    assert event.metadata["otel.status_code"] == ("ERROR" if status == "error" else "OK")


@pytest.mark.parametrize("command", [False, True])
def test_execution_intercept_edits_normalized_tool_result(subscribed_events, command):
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware

    message = ToolMessage(content="original", tool_call_id="edited", artifact={"keep": 42})
    result = Command(update={"messages": [message], "state": 42}, goto="next") if command else message
    request = ToolCallRequest(
        tool_call={"name": "lookup", "args": {}, "id": "edited"},
        tool=None,
        state={},
        runtime=MagicMock(),
    )

    async def intercept(context, next_call):
        downstream = await next_call(context.args)
        return nemo_relay.ToolExecutionInterceptOutcome(
            {**downstream.result, "content": "intercepted failure", "is_error": True},
            annotation={"source": "intercept"},
        )

    nemo_relay.intercepts.register_tool_execution("edit-normalized-result", 1, intercept)
    try:
        with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
            restored = NemoRelayMiddleware().wrap_tool_call(request, lambda _request: result)
    finally:
        nemo_relay.intercepts.deregister_tool_execution("edit-normalized-result")
    updated = restored.update["messages"][0] if isinstance(restored, Command) else restored
    assert updated.content == "intercepted failure"
    assert updated.status == "error"
    assert updated.artifact == {"keep": 42}
    assert message.content == "original"
    if command:
        assert isinstance(restored, Command)
        assert restored.goto == "next"
        assert restored.update["state"] == 42
    nemo_relay.subscribers.flush()
    event = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "lookup" and event.scope_category == "end"
    )
    assert event.metadata["otel.status_code"] == "ERROR"
    assert event.category_profile["tool_result_annotation"] == {"source": "intercept"}


def test_list_edits_only_change_the_matching_message():
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    other = Command(update={"state": 42}, goto="other")
    message = ToolMessage(content="answer", tool_call_id="target")
    codec = LangChainToolResultCodec("target")
    encoded = codec.to_json([other, [Command(update={"messages": [message]}, goto="next")]])
    encoded["content"] = "edited failure"
    encoded["is_error"] = True
    restored = codec.from_json(encoded)
    assert isinstance(restored, list)
    assert restored[0] == other
    inner = restored[1]
    assert isinstance(inner, list)
    command = inner[0]
    assert isinstance(command, Command)
    assert command.goto == "next"
    updated = command.update["messages"][0]
    assert updated.content == "edited failure"
    assert updated.status == "error"
    assert message.content == "answer"


def test_values_without_plain_json_form_replay_exactly(subscribed_events: list[nemo_relay.Event]) -> None:
    import math

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command, Overwrite, Send

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec, ToolResultValue

    state = {
        "nan": math.nan,
        "inf": -math.inf,
        "big": 2**70,
        "text": "a\ud800",
        "\udc00": "key",
        "bytes": b"\xff",
        "buffer": bytearray(b"\x00"),
        "set": {"b", "a"},
        "frozen": frozenset({(1, 2)}),
        "keys": {1: "int", (1, 2): "tuple"},
        "tuple": (1, [2]),
    }
    message = ToolMessage(content="answer", tool_call_id="call-1")
    command = Command(
        update={"messages": Overwrite([message]), "state": state}, goto=[Send("next", {"x": 1}, timeout=5.0)]
    )

    def call(_args: nemo_relay.Json) -> nemo_relay.ToolExecutionResult[ToolResultValue]:
        return nemo_relay.ToolExecutionResult(command)

    async def run() -> ToolResultValue:
        outcome = await nemo_relay.typed.tool_execute(
            name="lookup",
            args={},
            func=call,
            args_codec=nemo_relay.typed.JsonPassthrough(),
            result_codec=LangChainToolResultCodec("call-1"),
            tool_call_id="call-1",
        )
        return outcome.result

    assert asyncio.run(run()) is command
    nemo_relay.subscribers.flush()
    event = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "lookup" and event.scope_category == "end"
    )
    assert event.data["cache"]["replayable"] is True
    restored = LangChainToolResultCodec("call-1").from_json(event.data)
    assert isinstance(restored, Command)
    assert restored.goto == command.goto
    assert restored.update["messages"] == Overwrite([message])
    restored_state = dict(restored.update["state"])
    assert math.isnan(restored_state.pop("nan"))
    assert restored_state == {key: item for key, item in state.items() if key != "nan"}
    assert [type(restored_state[key]) for key in ("buffer", "set", "frozen")] == [bytearray, set, frozenset]


@pytest.mark.parametrize("update_kind", ["model", "dataclass", "pairs", "overwrite", "single"])
def test_projection_edits_messages_in_each_state_update_shape(update_kind: str) -> None:
    from dataclasses import dataclass

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command, Overwrite
    from pydantic import BaseModel

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class ModelState(BaseModel):
        messages: list[ToolMessage]

    @dataclass
    class DataclassState:
        messages: list[ToolMessage]

    message = ToolMessage(content="answer", tool_call_id="call-1")
    updates: dict[str, object] = {
        "model": ModelState(messages=[message]),
        "dataclass": DataclassState(messages=[message]),
        "pairs": [("messages", [message])],
        "overwrite": {"messages": Overwrite([message])},
        "single": {"messages": message},
    }
    update = updates[update_kind]
    result = Command(update=update)
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    assert encoded["content"] == "answer"
    encoded["content"] = "edited"
    encoded["is_error"] = True
    restored = codec.from_json(encoded)
    assert isinstance(restored, Command)
    assert type(restored.update) is type(update)
    reencoded = LangChainToolResultCodec("call-1").to_json(restored)
    assert (reencoded["content"], reencoded["is_error"]) == ("edited", True)
    assert LangChainToolResultCodec("call-1").to_json(result)["content"] == "answer"
    assert message.content == "answer"


def test_result_edits_rebuild_changed_values_and_keep_other_objects() -> None:
    from dataclasses import dataclass
    from datetime import datetime
    from enum import StrEnum

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class Color(StrEnum):
        RED = "red"
        BLUE = "blue"

    @dataclass
    class Artifact:
        rows: list[int]
        color: Color

    handle = object()
    artifact = Artifact([1], Color.RED)
    message = ToolMessage(content="answer", tool_call_id="call-1", artifact=artifact)
    result = Command(update={"messages": [message], "handle": handle, "at": datetime(2026, 1, 1)})
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    update = encoded["result"]["value"]["update"]["value"]
    update["messages"]["value"][0]["value"]["artifact"]["value"]["color"]["value"] = "blue"
    update["at"]["value"] = "2027-01-01T00:00:00"
    restored = codec.from_json(encoded)
    assert isinstance(restored, Command)
    assert restored.update["handle"] is handle
    assert restored.update["at"] == datetime(2027, 1, 1)
    edited = restored.update["messages"][0].artifact
    assert edited == Artifact([1], Color.BLUE)
    assert type(edited.color) is Color
    assert edited.rows is artifact.rows
    update["handle"]["value"] = "edited"
    with pytest.raises(ValueError, match="has no JSON form"):
        codec.from_json(encoded)


def test_content_shows_str_and_int_subclasses_by_their_values() -> None:
    from enum import Enum, IntEnum

    from langchain_core.messages import ToolMessage

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class Color(str, Enum):
        RED = "red"

    class Level(IntEnum):
        LOW = 1

    message = ToolMessage(content=[{"type": "text", "text": Color.RED, "level": Level.LOW}], tool_call_id="call-1")
    codec = LangChainToolResultCodec("call-1")
    encoded = codec.to_json(message)
    assert encoded["content"] == [{"type": "text", "text": "red", "level": 1}]
    assert codec.from_json(json.loads(json.dumps(encoded))) is message


def test_iterators_and_reference_cycles_stay_with_their_call() -> None:
    from collections import OrderedDict
    from collections.abc import Iterator
    from dataclasses import dataclass

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    @dataclass
    class Holder:
        rows: Iterator[int]

    def numbers() -> Iterator[int]:
        yield 1

    iterator = numbers()
    holder = Holder(numbers())
    cycle: list[object] = []
    cycle.append(cycle)
    ordered: OrderedDict[str, object] = OrderedDict()
    ordered["self"] = ordered
    message = ToolMessage(content="answer", tool_call_id="call-1")
    result = Command(
        update={"messages": [message], "iterator": iterator, "holder": holder, "cycle": cycle, "ordered": ordered}
    )
    codec = LangChainToolResultCodec("call-1")
    encoded = codec.to_json(result)
    update = encoded["result"]["value"]["update"]["value"]
    assert update["iterator"] == {
        "kind": "local",
        "type": "builtins.generator",
        "ref": 0,
        "value": "<builtins.generator>",
    }
    assert update["holder"] == {
        "kind": "dataclass",
        "type": f"{__name__}.{Holder.__qualname__}",
        "value": {"rows": {"kind": "local", "type": "builtins.generator", "ref": 1, "value": "<builtins.generator>"}},
    }
    assert update["cycle"]["value"][0] == {
        "kind": "local",
        "type": "builtins.list",
        "ref": 2,
        "value": ["<builtins.list>"],
    }
    assert update["ordered"] == {
        "kind": "mapping",
        "type": "collections.OrderedDict",
        "value": [
            [
                {"kind": "json", "value": "self"},
                {
                    "kind": "local",
                    "type": "collections.OrderedDict",
                    "ref": 3,
                    "value": {"self": "<collections.OrderedDict>"},
                },
            ]
        ],
    }
    assert encoded["cache"]["replayable"] is False
    assert codec.from_json(json.loads(json.dumps(encoded))) is result
    assert list(iterator) == [1]
    assert list(holder.rows) == [1]


def test_mapping_subclasses_are_recorded_as_entries_with_their_type_and_rebuilt_by_their_call() -> None:
    from collections import OrderedDict, defaultdict

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    ordered = OrderedDict([(1, "a"), ("1", "b")])
    rows: defaultdict[int, list[str]] = defaultdict(list, {1: ["a"]})
    message = ToolMessage(content="answer", tool_call_id="call-1")
    result = Command(update={"messages": [message], "ordered": ordered, "rows": rows})
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    update = encoded["result"]["value"]["update"]["value"]
    assert update["ordered"] == {
        "kind": "mapping",
        "type": "collections.OrderedDict",
        "value": [
            [{"kind": "json", "value": 1}, {"kind": "json", "value": "a"}],
            [{"kind": "json", "value": "1"}, {"kind": "json", "value": "b"}],
        ],
    }
    assert encoded["cache"]["replayable"] is False
    assert codec.from_json(encoded) is result

    update["ordered"]["value"].reverse()
    update["ordered"]["value"][0][1] = {"kind": "json", "value": "c"}
    update["rows"]["value"].append(
        [{"kind": "json", "value": 2}, {"kind": "list", "value": [{"kind": "json", "value": "b"}]}]
    )
    restored = codec.from_json(encoded)

    assert isinstance(restored, Command)
    assert type(restored.update["ordered"]) is OrderedDict
    assert list(restored.update["ordered"].items()) == [("1", "c"), (1, "a")]
    assert type(restored.update["rows"]) is defaultdict
    assert restored.update["rows"].default_factory is list
    assert restored.update["rows"] == {1: ["a"], 2: ["b"]}
    assert restored.update["rows"][1] is rows[1]
    with pytest.raises(ValueError, match="Only the call that returned"):
        LangChainToolResultCodec("call-1").from_json(encoded)


def test_mappings_that_cannot_take_edited_entries_fail_the_edit() -> None:
    from collections import ChainMap
    from types import MappingProxyType

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class Lower(dict[str, int]):
        def __setitem__(self, key: str, value: int) -> None:
            super().__setitem__(key.lower(), value)

    message = ToolMessage(content="answer", tool_call_id="call-1")
    frozen = MappingProxyType({1: "a"})
    layered = ChainMap({1: "a"}, {2: "b"})
    lowered = Lower()
    lowered["A"] = 1
    result = Command(update={"messages": [message], "frozen": frozen, "layered": layered, "lowered": lowered})
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    update = encoded["result"]["value"]["update"]["value"]
    assert update["frozen"]["type"] == "builtins.mappingproxy"
    assert [entry[0]["value"] for entry in update["layered"]["value"]] == [2, 1]
    assert codec.from_json(encoded) is result

    for name in ("frozen", "layered"):
        edited = json.loads(json.dumps(encoded))
        edited["result"]["value"]["update"]["value"][name]["value"][0][1] = {"kind": "json", "value": "c"}
        with pytest.raises(ValueError, match="cannot take edited entries"):
            codec.from_json(edited)
    assert dict(layered) == {1: "a", 2: "b"}
    lowered_edit = json.loads(json.dumps(encoded))
    lowered_edit["result"]["value"]["update"]["value"]["lowered"]["value"].append(
        [{"kind": "json", "value": "B"}, {"kind": "json", "value": 2}]
    )
    with pytest.raises(ValueError, match="did not take the edited entries"):
        codec.from_json(lowered_edit)
    assert dict(lowered) == {"a": 1}


def test_typed_values_whose_json_cannot_rebuild_them_stay_with_their_call() -> None:
    from typing import NamedTuple

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
    from pydantic import RootModel

    from nemo_relay.integrations.langchain._tool_codec import LangChainToolResultCodec

    class Mixed(RootModel[dict[int | str, str]]):
        pass

    class Ints(RootModel[dict[int, str]]):
        pass

    class Row(NamedTuple):
        cells: dict[object, str]

    class Cell(NamedTuple):
        text: str

    class Grid(NamedTuple):
        cells: list[Cell]

    mixed = Mixed({1: "a", "1": "b"})
    row = Row({1: "a"})
    grid = Grid([Cell("a")])
    message = ToolMessage(content="answer", tool_call_id="call-1")
    result = Command(update={"messages": [message], "mixed": mixed, "ints": Ints({1: "a"}), "row": row, "grid": grid})
    codec = LangChainToolResultCodec("call-1")
    encoded = json.loads(json.dumps(codec.to_json(result)))
    update = encoded["result"]["value"]["update"]["value"]
    assert (update["mixed"]["kind"], update["mixed"]["type"]) == ("local", f"{__name__}.{Mixed.__qualname__}")
    assert (update["row"]["kind"], update["row"]["type"]) == ("local", f"{__name__}.{Row.__qualname__}")
    assert (update["grid"]["kind"], update["grid"]["type"]) == ("local", f"{__name__}.{Grid.__qualname__}")
    assert update["ints"] == {"kind": "typed", "type": f"{__name__}.{Ints.__qualname__}", "value": {"1": "a"}}
    assert codec.from_json(encoded) is result

    update["ints"]["value"] = {"2": "b"}
    restored = codec.from_json(encoded)

    assert isinstance(restored, Command)
    assert restored.update["ints"] == Ints({2: "b"})
    assert restored.update["mixed"] is mixed
    assert restored.update["row"] is row
    assert restored.update["grid"] is grid
