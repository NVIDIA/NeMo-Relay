# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Relay middleware."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol
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
    async def execute_side_effect(*, func: Any, **kwargs: Any) -> ToolMessage:
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
    assert isinstance(kwargs["args_codec"], nemo_relay.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_relay.typed.BestEffortAnyCodec)


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
