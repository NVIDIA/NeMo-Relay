# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain NeMo Flow middleware."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import nemo_flow
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from nemo_flow.codecs import AnthropicMessagesCodec, OpenAIChatCodec, OpenAIResponsesCodec

from langchain_nemo_flow import _serialization
from langchain_nemo_flow.middleware import NemoFlowMiddleware

_DEFAULT_MOCK_RESPONSE_MSG = "nemo_flow unittest result"

def _mk_mock_model(returned_message: str = _DEFAULT_MOCK_RESPONSE_MSG) -> MagicMock:
    msg = AIMessage(content=returned_message)

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind.return_value = mock_model
    mock_model.model = "mock-model"
    mock_model.invoke.return_value = msg
    mock_model.ainvoke = AsyncMock(return_value=msg)

    return mock_model


class RecordingMiddleware(NemoFlowMiddleware):
    def __init__(self) -> None:
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


def _model_request() -> ModelRequest[Any]:
    mock_model = _mk_mock_model()

    return ModelRequest(
        model=mock_model,
        messages=[HumanMessage(content="hello")],
        model_settings={"temperature": 1.0},
    )


def _tool_call_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "lookup", "args": {"query": "original"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=MagicMock(),
    )


def test_wrap_model_call_routes_through_llm_execute() -> None:
    middleware = RecordingMiddleware()
    seen_request: ModelRequest[Any] | None = None

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal seen_request
        seen_request = request
        return ModelResponse(result=[AIMessage(content="done")])

    response = middleware.wrap_model_call(_model_request(), handler)

    assert response.result[0].content == "done"
    assert seen_request is not None
    assert seen_request.model_settings == {"temperature": 0.25}
    assert middleware.calls[0]["model_name"] == "mock-model"
    assert middleware.calls[0]["request"].content["model"] == "mock-model"
    assert middleware.calls[0]["codec"] is None
    assert middleware.calls[0]["response_codec"] is None


def test_awrap_model_call_routes_through_llm_execute() -> None:
    middleware = RecordingMiddleware()
    seen_request: ModelRequest[Any] | None = None

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal seen_request
        seen_request = request
        return ModelResponse(result=[AIMessage(content="done")])

    response = asyncio.run(middleware.awrap_model_call(_model_request(), handler))

    assert response.result[0].content == "done"
    assert seen_request is not None
    assert seen_request.model_settings == {"temperature": 0.25}
    assert middleware.calls[0]["model_name"] == "mock-model"
    assert middleware.calls[0]["request"].content["model"] == "mock-model"
    assert middleware.calls[0]["codec"] is None
    assert middleware.calls[0]["response_codec"] is None


def test_wrap_tool_call_routes_through_tool_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = NemoFlowMiddleware()
    parent_handle = MagicMock()
    seen_request: ToolCallRequest | None = None

    async def execute_side_effect(
        *,
        func: Any,
        **kwargs: Any
    ) -> ToolMessage:
        return func({"query": "intercepted"})

    mock_tool_execute = AsyncMock(side_effect=execute_side_effect)

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal seen_request
        seen_request = request
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_tool_execute)

    response = middleware.wrap_tool_call(_tool_call_request(), handler)

    assert response.content == "done"
    assert seen_request is not None
    assert seen_request.tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert isinstance(kwargs["args_codec"], nemo_flow.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_flow.typed.BestEffortAnyCodec)


def test_awrap_tool_call_routes_through_tool_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = NemoFlowMiddleware()
    parent_handle = MagicMock()
    seen_request: ToolCallRequest | None = None

    async def execute_side_effect(
        *,
        func: Any,
        **kwargs: Any
    ) -> ToolMessage:
        return await func({"query": "intercepted"})

    mock_tool_execute = AsyncMock(side_effect=execute_side_effect)

    async def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal seen_request
        seen_request = request
        return ToolMessage(content="done", tool_call_id=request.tool_call["id"])

    monkeypatch.setattr(nemo_flow.scope, "get_handle", lambda: parent_handle)
    monkeypatch.setattr(nemo_flow.typed, "tool_execute", mock_tool_execute)

    response = asyncio.run(middleware.awrap_tool_call(_tool_call_request(), handler))

    assert response.content == "done"
    assert seen_request is not None
    assert seen_request.tool_call["args"] == {"query": "intercepted"}
    mock_tool_execute.assert_awaited_once()
    kwargs = mock_tool_execute.await_args.kwargs
    assert kwargs["name"] == "lookup"
    assert kwargs["args"] == {"query": "original"}
    assert kwargs["handle"] is parent_handle
    assert isinstance(kwargs["args_codec"], nemo_flow.typed.BestEffortAnyCodec)
    assert isinstance(kwargs["result_codec"], nemo_flow.typed.BestEffortAnyCodec)


def test_infer_codec_from_supported_model_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeChatAnthropic:
        pass

    class FakeChatOpenAI:
        def __init__(self, *, use_responses_api: bool = False) -> None:
            self.use_responses_api = use_responses_api

    monkeypatch.setattr(_serialization, "ChatAnthropic", FakeChatAnthropic)
    monkeypatch.setattr(_serialization, "ChatOpenAI", FakeChatOpenAI)

    assert isinstance(_serialization.infer_codec_from_model(FakeChatAnthropic()), AnthropicMessagesCodec)
    assert isinstance(_serialization.infer_codec_from_model(FakeChatOpenAI()), OpenAIChatCodec)
    assert isinstance(
        _serialization.infer_codec_from_model(FakeChatOpenAI(use_responses_api=True)),
        OpenAIResponsesCodec,
    )
    assert _serialization.infer_codec_from_model(object()) is None


@pytest.mark.parametrize("use_async", [False, True])
def test_wrap_model_integration_test(use_async: bool) -> None:
    """An integration test to verify that the middleware correctly wraps a model call end-to-end."""
    mock_model = _mk_mock_model()

    agent = create_agent(
        model=mock_model,
        middleware=[NemoFlowMiddleware()]
    )

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
        "scope.end.langchain-request",
    ]

    def event_recorder(event) -> None:
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

    assert result['messages'][-1].content == _DEFAULT_MOCK_RESPONSE_MSG
    assert events == expected_events
