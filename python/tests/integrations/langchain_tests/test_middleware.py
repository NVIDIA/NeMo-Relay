# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Relay middleware."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_relay

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
    from langchain_core.messages import AIMessage, ToolMessage

    from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware

_DEFAULT_MOCK_RESPONSE_MSG = "nemo_relay unittest result"


@pytest.fixture(name="model_request_handler")
def model_request_handler_fixture() -> tuple[
    Callable[[ModelRequest[Any]], ModelResponse[Any]], dict[str, ModelRequest[Any]]
]:
    from langchain.agents.middleware import ModelResponse
    from langchain_core.messages import AIMessage

    seen_request: dict[str, ModelRequest[Any]] = {}

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen_request["request"] = request
        return ModelResponse(result=[AIMessage(content="done")])

    return handler, seen_request


@pytest.fixture(name="async_model_request_handler")
def async_model_request_handler_fixture(
    model_request_handler: tuple[Callable[[ModelRequest[Any]], ModelResponse[Any]], dict[str, ModelRequest[Any]]],
) -> tuple[Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]], dict[str, ModelRequest[Any]]]:
    (sync_handler, seen_request) = model_request_handler

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        return sync_handler(request)

    return handler, seen_request


@pytest.fixture(name="tool_request_handler")
def tool_request_handler_fixture() -> tuple[Callable[[ToolCallRequest], ToolMessage], dict[str, ToolCallRequest]]:
    from langchain_core.messages import ToolMessage

    seen_request: dict[str, ToolCallRequest] = {}

    def handler(request: ToolCallRequest) -> ToolMessage:
        seen_request["request"] = request
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    return handler, seen_request


@pytest.fixture(name="async_tool_request_handler")
def async_tool_request_handler_fixture(
    tool_request_handler: tuple[Callable[[ToolCallRequest], ToolMessage], dict[str, ToolCallRequest]],
) -> tuple[Callable[[ToolCallRequest], Awaitable[ToolMessage]], dict[str, ToolCallRequest]]:
    (sync_handler, seen_request) = tool_request_handler

    async def handler(request: ToolCallRequest) -> ToolMessage:
        return sync_handler(request)

    return handler, seen_request


@pytest.fixture(name="mock_tool_execute")
def mock_tool_execute_fixture() -> AsyncMock:
    async def execute_side_effect(*, func: Any, **kwargs: Any) -> nemo_relay.ToolExecutionResult[ToolMessage]:
        result = func({"query": "intercepted"})
        if inspect.isawaitable(result):
            return await result
        return result

    return AsyncMock(side_effect=execute_side_effect)


def _mk_mock_model(returned_message: str | list[AIMessage] = _DEFAULT_MOCK_RESPONSE_MSG) -> MagicMock:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind.return_value = mock_model
    mock_model.bind_tools.return_value = mock_model
    mock_model.model = "mock-model"

    if isinstance(returned_message, str):
        msg = AIMessage(content=returned_message)
        mock_model.invoke.return_value = msg
        mock_model.ainvoke = AsyncMock(return_value=msg)
    else:
        mock_model.invoke.side_effect = list(returned_message)
        mock_model.ainvoke = AsyncMock(side_effect=list(returned_message))

    return mock_model


@pytest.fixture(name="nemo_relay_middleware")
def nemo_relay_middleware_fixture() -> NemoRelayMiddleware:
    from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware

    return NemoRelayMiddleware()


class RecordingMiddleware(Protocol):
    calls: list[dict[str, Any]]
    wrap_model_call: Callable
    awrap_model_call: Callable


@pytest.fixture(name="recording_middleware")
def recording_middleware_fixture() -> RecordingMiddleware:
    from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware

    class _RecordingMiddleware(NemoRelayMiddleware, RecordingMiddleware):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        async def _llm_execute(
            self,
            model_name: str,
            request: nemo_relay.LLMRequest,
            codec: Any,
            response_codec: Any,
            func: Any,
        ) -> Any:
            self.calls.append(
                {
                    "model_name": model_name,
                    "request": request,
                    "codec": codec,
                    "response_codec": response_codec,
                }
            )
            intercepted = nemo_relay.LLMRequest(
                request.headers,
                {
                    **request.content,
                    "model_settings": {"temperature": 0.25},
                },
            )
            return await func(intercepted)

    return _RecordingMiddleware()


@pytest.fixture(name="model_request")
def model_request_fixture() -> ModelRequest[Any]:
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import HumanMessage

    mock_model = _mk_mock_model()

    return ModelRequest(
        model=mock_model,
        messages=[HumanMessage(content="hello")],
        model_settings={"temperature": 1.0},
    )


@pytest.fixture(name="tool_call_request")
def tool_call_request_fixture() -> ToolCallRequest:
    from langchain.agents.middleware import ToolCallRequest

    return ToolCallRequest(
        tool_call={"name": "lookup", "args": {"query": "original"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(),
    )


def test_wrap_model_call_routes_through_llm_execute(
    model_request: ModelRequest[Any],
    model_request_handler: tuple[Callable[[ModelRequest[Any]], ModelResponse[Any]], dict[str, ModelRequest[Any]]],
    recording_middleware: RecordingMiddleware,
):
    (handler, seen_request) = model_request_handler

    response = recording_middleware.wrap_model_call(model_request, handler)

    assert response.result[0].content == "done"
    assert seen_request["request"].model_settings == {"temperature": 0.25}
    assert recording_middleware.calls[0]["model_name"] == "mock-model"
    assert recording_middleware.calls[0]["request"].content["model"] == "mock-model"
    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    assert isinstance(recording_middleware.calls[0]["codec"], LangChainCodec)
    assert recording_middleware.calls[0]["response_codec"] is recording_middleware.calls[0]["codec"]


def test_awrap_model_call_routes_through_llm_execute(
    model_request: ModelRequest[Any],
    async_model_request_handler: tuple[
        Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]], dict[str, ModelRequest[Any]]
    ],
    recording_middleware: RecordingMiddleware,
):
    (handler, seen_request) = async_model_request_handler

    response = asyncio.run(recording_middleware.awrap_model_call(model_request, handler))

    assert response.result[0].content == "done"
    assert seen_request["request"].model_settings == {"temperature": 0.25}
    assert recording_middleware.calls[0]["model_name"] == "mock-model"
    assert recording_middleware.calls[0]["request"].content["model"] == "mock-model"
    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    assert isinstance(recording_middleware.calls[0]["codec"], LangChainCodec)
    assert recording_middleware.calls[0]["response_codec"] is recording_middleware.calls[0]["codec"]


def test_langchain_model_request_codec_round_trips_messages(model_request: ModelRequest[Any]):
    from nemo_relay.integrations.langchain._serialization import (
        LangChainCodec,
        model_request_to_payload,
        payload_to_model_request,
    )

    codec = LangChainCodec()
    request = nemo_relay.LLMRequest({}, model_request_to_payload("mock-model", model_request))

    annotated = codec.decode(request)
    assert annotated.messages == [{"role": "user", "content": "hello"}]

    annotated.messages = [{"role": "user", "content": "hello from intercept"}]
    encoded = codec.encode(annotated, request)
    round_tripped = payload_to_model_request(model_request, encoded)

    assert round_tripped.messages[0].content == "hello from intercept"


def test_langchain_request_codec_preserves_provider_tool_calls():
    from langchain_core.messages import AIMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    provider_tool_calls = [
        {
            "id": "call-weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]
    request = nemo_relay.LLMRequest(
        {},
        {
            "messages": messages_to_dict(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-weather",
                                "name": "get_weather",
                                "args": {"city": "SF"},
                                "type": "tool_call",
                            }
                        ],
                        additional_kwargs={"tool_calls": provider_tool_calls},
                    )
                ]
            )
        },
    )

    codec = LangChainCodec()
    encoded = codec.encode(codec.decode(request), request)
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], encoded.content["messages"]))[0]
    assert isinstance(rebuilt, AIMessage)

    assert rebuilt.tool_calls == [
        {
            "id": "call-weather",
            "name": "get_weather",
            "args": {"city": "SF"},
            "type": "tool_call",
        }
    ]
    assert rebuilt.additional_kwargs["tool_calls"] == provider_tool_calls


def test_langchain_request_codec_preserves_chat_nvidia_tool_call_payload_after_prepending_message():
    from langchain_core.messages import AIMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    convert_message_to_dict = pytest.importorskip("langchain_nvidia_ai_endpoints._utils").convert_message_to_dict
    provider_tool_calls = [
        {
            "id": "call-weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]
    request = nemo_relay.LLMRequest(
        {},
        {
            "messages": messages_to_dict(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-weather",
                                "name": "get_weather",
                                "args": {"city": "SF"},
                                "type": "tool_call",
                            }
                        ],
                        additional_kwargs={"tool_calls": provider_tool_calls},
                    )
                ]
            )
        },
    )

    codec = LangChainCodec()
    annotated = codec.decode(request)
    annotated.messages = [{"role": "user", "content": "Prepended by an interceptor"}, *annotated.messages]
    encoded = codec.encode(annotated, request)
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], encoded.content["messages"]))[1]

    assert convert_message_to_dict(rebuilt) == {
        "role": "assistant",
        "content": None,
        "tool_calls": provider_tool_calls,
    }


def test_langchain_request_codec_preserves_reordered_provider_tool_calls():
    from langchain_core.messages import AIMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    first_provider_tool_calls = [
        {
            "id": "call-one",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            "provider_field": "first",
        }
    ]
    second_provider_tool_calls = [
        {
            "id": "call-two",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "NY"}'},
            "provider_field": "second",
        }
    ]
    request = nemo_relay.LLMRequest(
        {},
        {
            "messages": messages_to_dict(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-one",
                                "name": "get_weather",
                                "args": {"city": "SF"},
                                "type": "tool_call",
                            }
                        ],
                        additional_kwargs={"tool_calls": first_provider_tool_calls},
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-two",
                                "name": "get_weather",
                                "args": {"city": "NY"},
                                "type": "tool_call",
                            }
                        ],
                        additional_kwargs={"tool_calls": second_provider_tool_calls},
                    ),
                ]
            )
        },
    )

    codec = LangChainCodec()
    annotated = codec.decode(request)
    annotated.messages = list(reversed(annotated.messages))
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))

    assert [message.additional_kwargs["tool_calls"] for message in rebuilt] == [
        second_provider_tool_calls,
        first_provider_tool_calls,
    ]


def test_langchain_request_codec_rebuilds_provider_tool_calls_after_content_edit():
    from langchain_core.messages import AIMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest(
        {},
        {
            "messages": messages_to_dict(
                [
                    AIMessage(
                        content="sensitive content",
                        tool_calls=[
                            {
                                "id": "call-weather",
                                "name": "get_weather",
                                "args": {"city": "SF"},
                                "type": "tool_call",
                            }
                        ],
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                                }
                            ]
                        },
                    )
                ]
            )
        },
    )

    codec = LangChainCodec()
    annotated = codec.decode(request)
    annotated.messages = [{**annotated.messages[0], "content": "[redacted]"}]
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))[0]

    assert isinstance(rebuilt, AIMessage)
    assert rebuilt.additional_kwargs["tool_calls"] == [
        {
            "id": "call-weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]


def test_langchain_request_codec_builds_provider_tool_calls_for_new_assistant():
    from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest({}, {"messages": messages_to_dict([HumanMessage(content="hello")])})

    codec = LangChainCodec()
    annotated = codec.decode(request)
    annotated.messages = [
        *annotated.messages,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                }
            ],
        },
    ]
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))[1]

    assert isinstance(rebuilt, AIMessage)
    assert rebuilt.additional_kwargs["tool_calls"] == [
        {
            "id": "call-weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]


def test_langchain_request_codec_preserves_multi_block_assistant_message():
    from langchain_core.messages import AIMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest(
        {},
        {
            "messages": messages_to_dict(
                [
                    AIMessage(
                        content=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
                        tool_calls=[
                            {
                                "id": "call-weather",
                                "name": "get_weather",
                                "args": {"city": "SF"},
                                "type": "tool_call",
                            }
                        ],
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                                }
                            ]
                        },
                    )
                ]
            )
        },
    )

    codec = LangChainCodec()
    rebuilt = messages_from_dict(
        cast(list[dict[str, Any]], codec.encode(codec.decode(request), request).content["messages"])
    )
    assert len(rebuilt) == 1
    rebuilt_assistant = rebuilt[0]
    assert isinstance(rebuilt_assistant, AIMessage)

    assert rebuilt_assistant.content == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert rebuilt_assistant.tool_calls == [
        {"id": "call-weather", "name": "get_weather", "args": {"city": "SF"}, "type": "tool_call"}
    ]
    assert rebuilt_assistant.additional_kwargs == {
        "tool_calls": [
            {
                "id": "call-weather",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }
        ]
    }


def test_langchain_request_codec_preserves_unchanged_bare_string_content_parts():
    from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    original_content = ["first", {"type": "text", "text": "second"}]
    request = nemo_relay.LLMRequest({}, {"messages": messages_to_dict([HumanMessage(content=original_content)])})

    codec = LangChainCodec()
    rebuilt = messages_from_dict(
        cast(list[dict[str, Any]], codec.encode(codec.decode(request), request).content["messages"])
    )

    assert rebuilt[0].content == original_content


def test_langchain_request_codec_uses_canonical_blocks_after_bare_string_content_edit():
    from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest(
        {},
        {"messages": messages_to_dict([HumanMessage(content=["first", {"type": "text", "text": "second"}])])},
    )

    codec = LangChainCodec()
    annotated = codec.decode(request)
    message = dict(annotated.messages[0])
    content = [dict(part) for part in cast(list[dict[str, Any]], message["content"])]
    content[0]["text"] = "first from intercept"
    message["content"] = content
    annotated.messages = [message]
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))

    assert rebuilt[0].content == [
        {"type": "text", "text": "first from intercept"},
        {"type": "text", "text": "second"},
    ]


@pytest.mark.parametrize("message_list_edit", ["prepend", "append", "delete"])
def test_langchain_request_codec_preserves_bare_strings_across_message_list_edits(message_list_edit: str):
    from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest(
        {},
        {"messages": messages_to_dict([HumanMessage(content=["first"]), AIMessage(content=["second"])])},
    )

    codec = LangChainCodec()
    annotated = codec.decode(request)
    if message_list_edit == "prepend":
        annotated.messages = [{"role": "system", "content": "new"}, *annotated.messages]
    elif message_list_edit == "append":
        annotated.messages = [*annotated.messages, {"role": "assistant", "content": "new", "tool_calls": []}]
    else:
        annotated.messages = annotated.messages[:1]
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))

    preserved_messages = [message for message in rebuilt if message.content != "new"]
    assert preserved_messages[0].content == ["first"]
    if message_list_edit != "delete":
        assert preserved_messages[1].content == ["second"]


def test_langchain_request_codec_preserves_bare_strings_after_name_edit():
    from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest({}, {"messages": messages_to_dict([HumanMessage(content=["first"])])})

    codec = LangChainCodec()
    annotated = codec.decode(request)
    message = dict(annotated.messages[0])
    message["name"] = "renamed"
    annotated.messages = [message]
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))

    assert rebuilt[0].name == "renamed"
    assert rebuilt[0].content == ["first"]


def test_langchain_request_codec_does_not_guess_bare_string_shape_for_ambiguous_messages():
    from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    request = nemo_relay.LLMRequest(
        {},
        {
            "messages": messages_to_dict(
                [
                    HumanMessage(content=["same"]),
                    HumanMessage(content=[{"type": "text", "text": "same"}]),
                ]
            )
        },
    )

    codec = LangChainCodec()
    annotated = codec.decode(request)
    annotated.messages = annotated.messages[1:]
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))

    assert rebuilt[0].content == [{"type": "text", "text": "same"}]


def test_langchain_request_codec_preserves_distinct_system_messages_and_native_blocks():
    from langchain_core.messages import HumanMessage, SystemMessage, messages_from_dict, messages_to_dict

    from nemo_relay.integrations.langchain._serialization import LangChainCodec

    original_messages = [
        SystemMessage(content="first system message"),
        SystemMessage(content="second system message"),
        HumanMessage(content=[{"type": "image", "url": "https://example.com/image.png"}]),
    ]
    request = nemo_relay.LLMRequest({}, {"messages": messages_to_dict(original_messages)})

    codec = LangChainCodec()
    annotated = codec.decode(request)
    rebuilt = messages_from_dict(cast(list[dict[str, Any]], codec.encode(annotated, request).content["messages"]))

    assert [message.type for message in rebuilt] == ["system", "system", "human"]
    assert [message.content for message in rebuilt] == [message.content for message in original_messages]


def test_model_call_applies_edit_to_one_system_content_block(
    nemo_relay_middleware: NemoRelayMiddleware,
):
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    content_blocks = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
        {"type": "text", "text": "third"},
        {"type": "text", "text": "fourth"},
    ]
    request = ModelRequest(
        model=_mk_mock_model(),
        system_message=SystemMessage(content_blocks=content_blocks),
        messages=[HumanMessage(content="hello")],
    )
    seen_request: dict[str, ModelRequest[Any]] = {}

    def edit_system_block(
        _name: str,
        llm_request: nemo_relay.LLMRequest,
        annotated: nemo_relay.AnnotatedLLMRequest | None,
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        assert annotated is not None
        messages = annotated.messages
        system_message = dict(messages[0])
        system_content = [dict(part) for part in cast(list[dict[str, Any]], system_message["content"])]
        system_content[1]["text"] = "second from intercept"
        system_message["content"] = system_content
        annotated.messages = [system_message, *messages[1:]]
        return nemo_relay.LLMRequestInterceptOutcome(llm_request, annotated)

    def handler(next_request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen_request["request"] = next_request
        return ModelResponse(result=[AIMessage(content="done")])

    nemo_relay.intercepts.register_llm_request("edit_system_content_block", 1, False, edit_system_block)
    try:
        response = nemo_relay_middleware.wrap_model_call(request, handler)
    finally:
        nemo_relay.intercepts.deregister_llm_request("edit_system_content_block")

    assert response.result[0].content == "done"
    assert seen_request["request"].system_message is not None
    assert seen_request["request"].system_message.content == [
        content_blocks[0],
        {"type": "text", "text": "second from intercept"},
        content_blocks[2],
        content_blocks[3],
    ]
    assert request.system_message is not None
    assert request.system_message.content == content_blocks


def test_model_call_intercept_rebuilds_provider_tool_calls(
    nemo_relay_middleware: NemoRelayMiddleware,
    model_request: ModelRequest[Any],
    model_request_handler: tuple[Callable[[ModelRequest[Any]], ModelResponse[Any]], dict[str, ModelRequest[Any]]],
):
    from langchain_core.messages import AIMessage

    provider_tool_calls = [
        {
            "id": "call-weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]
    original = model_request.override(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-weather",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={"tool_calls": provider_tool_calls},
            )
        ]
    )

    def change_tool_call(_: str, request: nemo_relay.LLMRequest, annotated: Any):
        assert annotated is not None
        annotated.messages = [
            {
                **message,
                "tool_calls": [
                    {
                        **message["tool_calls"][0],
                        "function": {
                            **message["tool_calls"][0]["function"],
                            "arguments": '{"city":"San Jose"}',
                        },
                    }
                ],
            }
            if message.get("role") == "assistant"
            else message
            for message in annotated.messages
        ]
        return nemo_relay.LLMRequestInterceptOutcome(request, annotated)

    nemo_relay.intercepts.register_llm_request("test_langchain_change_tool_call", 1, False, change_tool_call)
    try:
        (handler, seen_request) = model_request_handler
        nemo_relay_middleware.wrap_model_call(original, handler)
    finally:
        nemo_relay.intercepts.deregister_llm_request("test_langchain_change_tool_call")

    rebuilt = next(message for message in seen_request["request"].messages if message.type == "ai")
    assert isinstance(rebuilt, AIMessage)
    assert rebuilt.tool_calls[0]["args"] == {"city": "San Jose"}
    assert rebuilt.additional_kwargs["tool_calls"] == [
        {
            "id": "call-weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"San Jose"}'},
        }
    ]


def test_payload_to_model_request_moves_relay_headers_to_chat_nvidia_transport(model_request: ModelRequest[Any]):
    from nemo_relay.integrations.langchain._serialization import (
        model_request_to_payload,
        payload_to_model_request,
    )

    ChatNVIDIA = pytest.importorskip("langchain_nvidia_ai_endpoints").ChatNVIDIA
    original_model = ChatNVIDIA.model_construct(default_headers={"x-existing": "existing", "X-Shared": "provider"})
    original = model_request.override(model=original_model)
    relay_request = nemo_relay.LLMRequest(
        {
            "x-dynamo-session-id": "session-1",
            "x-shared": "relay",
            "x-json": {"enabled": True},
        },
        model_request_to_payload("mock-model", original),
    )

    converted = payload_to_model_request(original, relay_request)

    assert converted.model is not original_model
    assert cast(Any, converted.model).default_headers == {
        "x-existing": "existing",
        "x-shared": "relay",
        "x-dynamo-session-id": "session-1",
        "x-json": '{"enabled":true}',
    }
    assert original_model.default_headers == {"x-existing": "existing", "X-Shared": "provider"}
    assert converted.model_settings == {"temperature": 1.0}


def test_payload_to_model_request_leaves_chat_nvidia_model_unchanged_without_relay_headers(
    model_request: ModelRequest[Any],
):
    from nemo_relay.integrations.langchain._serialization import (
        model_request_to_payload,
        payload_to_model_request,
    )

    ChatNVIDIA = pytest.importorskip("langchain_nvidia_ai_endpoints").ChatNVIDIA
    original_model = ChatNVIDIA.model_construct(default_headers={"x-existing": "existing"})
    original = model_request.override(model=original_model)
    converted = payload_to_model_request(
        original,
        nemo_relay.LLMRequest({}, model_request_to_payload("mock-model", original)),
    )

    assert converted.model is original_model
    assert converted.model_settings == {"temperature": 1.0}


def test_payload_to_model_request_keeps_generic_model_headers_in_model_settings(model_request: ModelRequest[Any]):
    from nemo_relay.integrations.langchain._serialization import (
        model_request_to_payload,
        payload_to_model_request,
    )

    relay_request = nemo_relay.LLMRequest(
        {"x-relay-header": "value"},
        model_request_to_payload("mock-model", model_request),
    )

    converted = payload_to_model_request(model_request, relay_request)

    assert converted.model is model_request.model
    assert converted.model_settings == {
        "temperature": 1.0,
        "extra_headers": {"x-relay-header": "value"},
    }


def test_langchain_model_response_codec_decodes_text_and_tool_calls():
    from langchain.agents.middleware import ModelResponse
    from langchain_core.messages import AIMessage

    from nemo_relay import AnnotatedLLMResponse
    from nemo_relay.integrations.langchain._serialization import LangChainCodec, model_response_to_json

    codec = LangChainCodec()
    response = ModelResponse(
        result=[
            AIMessage(
                content="I will search docs.",
                tool_calls=[
                    {
                        "name": "search_docs",
                        "args": {"query": "Deep Agents"},
                        "id": "call-search-docs",
                    }
                ],
                response_metadata={"finish_reason": "tool_calls", "model_name": "mock-model"},
                usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            )
        ]
    )

    annotated = codec.decode_response(model_response_to_json(response, nemo_relay.typed.BestEffortAnyCodec()))

    assert isinstance(annotated, AnnotatedLLMResponse)
    assert annotated.model == "mock-model"
    assert annotated.response_text() == "I will search docs."
    assert annotated.finish_reason == "tool_use"
    assert annotated.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert annotated.tool_calls == [
        {
            "id": "call-search-docs",
            "name": "search_docs",
            "arguments": {"query": "Deep Agents"},
        }
    ]

    unknown_response = ModelResponse(
        result=[
            AIMessage(
                content="done",
                response_metadata={"finish_reason": "provider_custom_stop"},
            )
        ]
    )
    unknown_annotated = codec.decode_response(
        model_response_to_json(unknown_response, nemo_relay.typed.BestEffortAnyCodec())
    )
    assert unknown_annotated.finish_reason == "provider_custom_stop"


@pytest.mark.parametrize("use_async", [False, True])
def test_model_call_applies_annotated_llm_request_intercept(
    use_async: bool,
    nemo_relay_middleware: NemoRelayMiddleware,
    model_request: ModelRequest[Any],
    model_request_handler: tuple[Callable[[ModelRequest[Any]], ModelResponse[Any]], dict[str, ModelRequest[Any]]],
    async_model_request_handler: tuple[
        Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]], dict[str, ModelRequest[Any]]
    ],
):
    captured: dict[str, Any] = {}

    def change_request(name: str, request: nemo_relay.LLMRequest, annotated: Any):
        assert name == "mock-model"
        assert annotated is not None
        captured["before"] = annotated.messages
        annotated.messages = [
            {
                **message,
                "content": str(message["content"]).replace("hello", "hello from intercept"),
            }
            if message.get("role") == "user"
            else message
            for message in annotated.messages
        ]
        return nemo_relay.LLMRequestInterceptOutcome(request, annotated)

    nemo_relay.intercepts.register_llm_request("test_langchain_change_request", 1, False, change_request)
    try:
        if use_async:
            (handler, seen_request) = async_model_request_handler
            response = asyncio.run(nemo_relay_middleware.awrap_model_call(model_request, handler))
        else:
            (handler, seen_request) = model_request_handler
            response = nemo_relay_middleware.wrap_model_call(model_request, handler)
    finally:
        nemo_relay.intercepts.deregister_llm_request("test_langchain_change_request")

    assert response.result[0].content == "done"
    assert captured["before"] == [{"role": "user", "content": "hello"}]
    assert seen_request["request"].messages[0].content == "hello from intercept"


def test_wrap_tool_call_routes_through_tool_execute(
    monkeypatch: pytest.MonkeyPatch,
    nemo_relay_middleware: NemoRelayMiddleware,
    mock_tool_execute: AsyncMock,
    tool_call_request: ToolCallRequest,
    tool_request_handler: tuple[Callable[[ToolCallRequest], ToolMessage], dict[str, ToolCallRequest]],
):
    (handler, seen_request) = tool_request_handler
    parent_handle = MagicMock()

    monkeypatch.setattr(nemo_relay.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_relay.typed, "tool_execute", mock_tool_execute)

    response = nemo_relay_middleware.wrap_tool_call(tool_call_request, handler)

    assert response.content == "done"
    assert seen_request["request"].tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    assert mock_tool_execute.await_args is not None
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert kwargs["tool_call_id"] == "call-1"
    assert isinstance(kwargs["args_codec"], nemo_relay.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_relay.typed.BestEffortAnyCodec)


def test_awrap_tool_call_routes_through_tool_execute(
    monkeypatch: pytest.MonkeyPatch,
    nemo_relay_middleware: NemoRelayMiddleware,
    mock_tool_execute: AsyncMock,
    tool_call_request: ToolCallRequest,
    async_tool_request_handler: tuple[Callable[[ToolCallRequest], Awaitable[ToolMessage]], dict[str, ToolCallRequest]],
):
    parent_handle = MagicMock()
    (handler, seen_request) = async_tool_request_handler

    monkeypatch.setattr(nemo_relay.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_relay.typed, "tool_execute", mock_tool_execute)

    response = asyncio.run(nemo_relay_middleware.awrap_tool_call(tool_call_request, handler))

    assert response.content == "done"
    assert seen_request["request"].tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    assert mock_tool_execute.await_args is not None
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert kwargs["tool_call_id"] == "call-1"
    assert isinstance(kwargs["args_codec"], nemo_relay.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_relay.typed.BestEffortAnyCodec)


def test_complete_skill_read_emits_mark_through_langchain_middleware(
    subscribed_events: list[nemo_relay.Event],
    nemo_relay_middleware: NemoRelayMiddleware,
):
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage

    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {"path": "/skills/review/SKILL.md"},
            "id": "call-skill",
        },
        tool=None,
        state={},
        runtime=MagicMock(),
    )
    with nemo_relay.scope.scope("langchain-skill", nemo_relay.ScopeType.Agent):
        response = nemo_relay_middleware.wrap_tool_call(
            request,
            lambda next_request: ToolMessage(
                content="loaded",
                tool_call_id=next_request.tool_call["id"],
            ),
        )
    nemo_relay.subscribers.flush()

    assert response.content == "loaded"
    mark = next(
        event for event in subscribed_events if isinstance(event, nemo_relay.MarkEvent) and event.name == "skill.load"
    )
    tool_start = next(
        event
        for event in subscribed_events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "read_file" and event.scope_category == "start"
    )
    assert mark.parent_uuid == tool_start.uuid
    assert mark.data == {"skill_name": "review"}
    assert mark.metadata == {
        "skill_load_source": "structured_read",
        "tool_name": "read_file",
    }


@pytest.mark.parametrize("use_async", [False, True])
def test_agent_integration(use_async: bool, nemo_relay_middleware: NemoRelayMiddleware):
    """An integration test to verify that the middleware correctly wraps a model call end-to-end."""
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    model_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                    "id": "call-1",
                }
            ],
        ),
        AIMessage(content=_DEFAULT_MOCK_RESPONSE_MSG),
    ]

    mock_model = _mk_mock_model(model_responses)

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72 degrees."

    agent = create_agent(model=mock_model, tools=[get_weather], middleware=[nemo_relay_middleware])

    input_payload = {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in San Francisco?",
            }
        ]
    }

    events = []
    expected_events = [
        "scope.start.langchain-request",
        "scope.start.mock-model",
        "scope.end.mock-model",
        "scope.start.get_weather",
        "scope.end.get_weather",
        "scope.start.mock-model",
        "scope.end.mock-model",
        "scope.end.langchain-request",
    ]

    def event_recorder(event):
        events.append(f"{event.kind}.{event.scope_category}.{event.name}")

    nemo_relay.subscribers.register("event_recorder", event_recorder)

    try:
        with nemo_relay.scope.scope("langchain-request", nemo_relay.ScopeType.Agent):
            if use_async:
                result = asyncio.run(agent.ainvoke(input_payload))
            else:
                result = agent.invoke(input_payload)
    finally:
        nemo_relay.subscribers.flush()
        nemo_relay.subscribers.deregister("event_recorder")

    assert any(
        message.content == "The weather in San Francisco is sunny and 72 degrees." for message in result["messages"]
    )
    assert result["messages"][-1].content == _DEFAULT_MOCK_RESPONSE_MSG

    assert events == expected_events
