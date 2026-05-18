# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Flow middleware."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_flow
from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
    from langchain_core.messages import AIMessage, ToolMessage

    from nemo_flow.integrations.langchain.middleware import NemoFlowMiddleware

_DEFAULT_MOCK_RESPONSE_MSG = "nemo_flow unittest result"


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


@pytest.fixture(name="nemo_flow_middleware")
def nemo_flow_middleware_fixture() -> NemoFlowMiddleware:
    from nemo_flow.integrations.langchain.middleware import NemoFlowMiddleware

    return NemoFlowMiddleware()


@pytest.fixture(name="recording_middleware")
def recording_middleware_fixture() -> NemoFlowMiddleware:
    from nemo_flow.integrations.langchain.middleware import NemoFlowMiddleware

    class RecordingMiddleware(NemoFlowMiddleware):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        async def _llm_execute(
            self,
            model_name: str,
            request: nemo_flow.LLMRequest,
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
            intercepted = nemo_flow.LLMRequest(
                request.headers,
                {
                    **request.content,
                    "model_settings": {"temperature": 0.25},
                },
            )
            return await func(intercepted)

    return RecordingMiddleware()


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
    recording_middleware: NemoFlowMiddleware,
):
    (handler, seen_request) = model_request_handler

    response = recording_middleware.wrap_model_call(model_request, handler)

    assert response.result[0].content == "done"
    assert seen_request["request"].model_settings == {"temperature": 0.25}
    assert recording_middleware.calls[0]["model_name"] == "mock-model"
    assert recording_middleware.calls[0]["request"].content["model"] == "mock-model"
    assert recording_middleware.calls[0]["codec"] is None
    assert recording_middleware.calls[0]["response_codec"] is None


def test_awrap_model_call_routes_through_llm_execute(
    model_request: ModelRequest[Any],
    async_model_request_handler: tuple[
        Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]], dict[str, ModelRequest[Any]]
    ],
    recording_middleware: NemoFlowMiddleware,
):
    (handler, seen_request) = async_model_request_handler

    response = asyncio.run(recording_middleware.awrap_model_call(model_request, handler))

    assert response.result[0].content == "done"
    assert seen_request["request"].model_settings == {"temperature": 0.25}
    assert recording_middleware.calls[0]["model_name"] == "mock-model"
    assert recording_middleware.calls[0]["request"].content["model"] == "mock-model"
    assert recording_middleware.calls[0]["codec"] is None
    assert recording_middleware.calls[0]["response_codec"] is None


def test_wrap_tool_call_routes_through_tool_execute(
    monkeypatch: pytest.MonkeyPatch,
    nemo_flow_middleware: NemoFlowMiddleware,
    mock_tool_execute: AsyncMock,
    tool_call_request: ToolCallRequest,
    tool_request_handler: tuple[Callable[[ToolCallRequest], ToolMessage], dict[str, ToolCallRequest]],
):
    (handler, seen_request) = tool_request_handler
    parent_handle = MagicMock()

    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_tool_execute)

    response = nemo_flow_middleware.wrap_tool_call(tool_call_request, handler)

    assert response.content == "done"
    assert seen_request["request"].tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert isinstance(kwargs["args_codec"], nemo_flow.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_flow.typed.BestEffortAnyCodec)


def test_awrap_tool_call_routes_through_tool_execute(
    monkeypatch: pytest.MonkeyPatch,
    nemo_flow_middleware: NemoFlowMiddleware,
    mock_tool_execute: AsyncMock,
    tool_call_request: ToolCallRequest,
    async_tool_request_handler: tuple[Callable[[ToolCallRequest], Awaitable[ToolMessage]], dict[str, ToolCallRequest]],
):
    parent_handle = MagicMock()
    (handler, seen_request) = async_tool_request_handler

    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_tool_execute)

    response = asyncio.run(nemo_flow_middleware.awrap_tool_call(tool_call_request, handler))

    assert response.content == "done"
    assert seen_request["request"].tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert isinstance(kwargs["args_codec"], nemo_flow.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_flow.typed.BestEffortAnyCodec)


def test_infer_codec_from_supported_model_classes(monkeypatch: pytest.MonkeyPatch):
    from nemo_flow.integrations.langchain import _serialization

    MockChatAnthropic = MagicMock(spec=type("MockChatAnthropic", (), {}))
    MockChatOpenAI = MagicMock(spec=type("MockChatOpenAI", (), {}))
    MockChatOpenAIResponses = MagicMock(spec=MockChatOpenAI.__class__)
    MockChatOpenAIResponses.use_responses_api = True
    MockChatNVIDIA = MagicMock(spec=type("MockChatNVIDIA", (), {}))

    monkeypatch.setattr(_serialization, "ChatAnthropic", MockChatAnthropic.__class__, raising=False)
    monkeypatch.setattr(_serialization, "ChatOpenAI", MockChatOpenAI.__class__, raising=False)
    monkeypatch.setattr(_serialization, "ChatNVIDIA", MockChatNVIDIA.__class__, raising=False)
    monkeypatch.setattr(_serialization, "_HAS_ANTHROPIC", True)
    monkeypatch.setattr(_serialization, "_HAS_OPENAI", True)
    monkeypatch.setattr(_serialization, "_HAS_NVIDIA", True)

    assert isinstance(_serialization.infer_codec_from_model(MockChatAnthropic), AnthropicMessagesCodec)
    assert isinstance(_serialization.infer_codec_from_model(MockChatOpenAI), OpenAIChatCodec)
    assert isinstance(
        _serialization.infer_codec_from_model(MockChatOpenAIResponses),
        OpenAIResponsesCodec,
    )
    assert isinstance(_serialization.infer_codec_from_model(MockChatNVIDIA), OpenAIChatCodec)
    assert _serialization.infer_codec_from_model(object()) is None


@pytest.mark.parametrize("use_async", [False, True])
def test_agent_integration(use_async: bool, nemo_flow_middleware: NemoFlowMiddleware):
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

    agent = create_agent(model=mock_model, tools=[get_weather], middleware=[nemo_flow_middleware])

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

    nemo_flow.subscribers.register("event_recorder", event_recorder)

    try:
        with nemo_flow.scope.scope("langchain-request", nemo_flow.ScopeType.Agent):
            if use_async:
                result = asyncio.run(agent.ainvoke(input_payload))
            else:
                result = agent.invoke(input_payload)
    finally:
        nemo_flow.subscribers.deregister("event_recorder")

    assert any(
        message.content == "The weather in San Francisco is sunny and 72 degrees." for message in result["messages"]
    )
    assert result["messages"][-1].content == _DEFAULT_MOCK_RESPONSE_MSG
    assert events == expected_events
