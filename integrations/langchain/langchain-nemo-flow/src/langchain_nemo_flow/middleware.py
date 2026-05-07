# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain AgentMiddleware implementation for NeMo Flow."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import nemo_flow
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from nemo_flow._native import LLMRequest

from langchain_nemo_flow._serialization import (
    ModelRequestHeaders,
    get_model_name,
    model_request_to_payload,
    model_response_from_json,
    model_response_to_json,
    payload_to_model_request,
)

if TYPE_CHECKING:
    from nemo_flow.codecs import LlmCodec, LlmResponseCodec


class NemoFlowMiddleware(AgentMiddleware):
    """Route LangChain agent model and tool calls through NeMo Flow.

    This uses LangChain's public ``AgentMiddleware`` hooks. It applies to agents
    built with ``langchain.agents.create_agent(..., middleware=[...])``.
    """

    def __init__(
        self,
        *,
        name: str = "NemoFlowMiddleware",
        model_request_headers: ModelRequestHeaders | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._model_request_headers = model_request_headers

    @property
    def name(self) -> str:
        """Middleware name used by LangChain graph nodes and traces."""
        return self._name

    async def llm_execute(
        self,
        model_name: str,
        request: "LLMRequest",
        codec: LlmCodec | None,
        response_codec: LlmResponseCodec | None,
        func: Callable[..., Any],
    ) -> Any:
        """Execute a non-streaming LLM call through the NeMo Flow pipeline."""
        return await nemo_flow.llm.execute(
            model_name,
            request,
            func,
            model_name=model_name,
            codec=codec,
            response_codec=response_codec,
        )

    async def llm_stream_execute(
        self,
        model_name: str,
        request: "LLMRequest",
        func: Callable[..., Any],
        collector: Callable[[Any], None],
        finalizer: Callable[[], Any],
        codec: LlmCodec | None,
        response_codec: LlmResponseCodec | None,
    ) -> Any:
        """Execute a streaming LLM call through the NeMo Flow pipeline."""
        return await nemo_flow.llm.stream_execute(
            model_name,
            request,
            func,
            collector,
            finalizer,
            model_name=model_name,
            codec=codec,
            response_codec=response_codec,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Wrap a sync LangChain agent model call in NeMo Flow LLM execution."""
        object_codec = nemo_flow.typed.BestEffortAnyCodec()
        llm_request = nemo_flow.LLMRequest(self._headers_for(request), model_request_to_payload(request))
        model_name = get_model_name(request.model)
        # model_codec = infer_codec_from_model(request.model)

        async def _call(req: Any) -> Any:
            response = handler(payload_to_model_request(request, req.content))
            return model_response_to_json(response, object_codec)

        result = nemo_flow.utils.run_sync(
            self.llm_execute(
                model_name=model_name,
                request=llm_request,
                func=_call,
                # TODO:  Whenever I set these I get an attribute error about the codec missing a type attribute.
                codec=None,
                response_codec=None,
            )
        )
        return model_response_from_json(result, object_codec)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Wrap an async LangChain agent model call in NeMo Flow LLM execution."""

        object_codec = nemo_flow.typed.BestEffortAnyCodec()
        llm_request = nemo_flow.LLMRequest(self._headers_for(request), model_request_to_payload(request))
        model_name = get_model_name(request.model)
        # model_codec = infer_codec_from_model(request.model)

        async def _call(req: Any) -> Any:
            response = await handler(payload_to_model_request(request, req.content))
            return model_response_to_json(response, object_codec)

        result = await self.llm_execute(
            model_name=model_name,
            request=llm_request,
            func=_call,
            codec=None,
            response_codec=None,
        )
        return model_response_from_json(result, object_codec)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap a sync LangChain agent tool call in NeMo Flow tool execution."""

        parent = nemo_flow.scope.get_handle()
        codec = nemo_flow.typed.BestEffortAnyCodec()
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args") or {}

        def _call(args: Any) -> ToolMessage | Command[Any]:
            return handler(request.override(tool_call={**request.tool_call, "args": args}))

        return nemo_flow.utils.run_sync(nemo_flow.typed.tool_execute(name=tool_name, args=tool_args, func=_call, args_codec=codec, result_codec=codec, handle=parent))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap an async LangChain agent tool call in NeMo Flow tool execution."""

        codec = nemo_flow.typed.BestEffortAnyCodec()
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args") or {}

        async def _call(args: Any) -> ToolMessage | Command[Any]:
            return await handler(request.override(tool_call={**request.tool_call, "args": args}))

        parent = nemo_flow.scope.get_handle()
        return await nemo_flow.typed.tool_execute(name=tool_name, args=tool_args, func=_call, args_codec=codec, result_codec=codec, handle=parent)

    def _headers_for(self, request: ModelRequest[Any]) -> dict[str, str]:
        return self._model_request_headers(request) if self._model_request_headers else {}
