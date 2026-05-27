# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain AgentMiddleware implementation for NeMo Relay."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware

import nemo_relay
from nemo_relay.integrations.langchain._serialization import (
    LangChainModelRequestCodec,
    get_model_name,
    infer_codec_from_model,
    model_request_to_payload,
    model_response_from_json,
    model_response_to_json,
    payload_to_model_request,
)
from nemo_relay.utils import run_sync

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from nemo_relay.codecs import LlmCodec, LlmResponseCodec


class NemoRelayMiddleware(AgentMiddleware):
    """Route LangChain agent model and tool calls through NeMo Relay.

    This uses LangChain's public ``AgentMiddleware`` hooks. It applies to agents
    built with ``langchain.agents.create_agent(..., middleware=[...])``.
    """

    def __init__(
        self,
        *,
        name: str = "NemoRelayMiddleware",
    ) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """Middleware name used by LangChain graph nodes and traces."""
        return self._name

    async def _llm_execute(
        self,
        model_name: str,
        request: nemo_relay.LLMRequest,
        codec: LlmCodec | None,
        response_codec: LlmResponseCodec | None,
        func: Callable[..., Any],
    ) -> Any:
        """Execute a non-streaming LLM call through the NeMo Relay pipeline."""
        return await nemo_relay.llm.execute(
            model_name,
            request,
            func,
            model_name=model_name,
            codec=codec,
            response_codec=response_codec,
        )

    def _prepare_model_call(self, request: ModelRequest[Any]) -> tuple:
        """Boilerplate code common to both wrap_model_call and awrap_model_call"""
        object_codec = nemo_relay.typed.BestEffortAnyCodec()
        model_name = get_model_name(request.model)
        llm_request = nemo_relay.LLMRequest({}, model_request_to_payload(model_name, request))
        model_codec = LangChainModelRequestCodec()
        return (object_codec, llm_request, model_name, model_codec)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Wrap a sync LangChain agent model call in NeMo Relay LLM execution."""
        (object_codec, llm_request, model_name, model_codec) = self._prepare_model_call(request)

        async def _call(req: Any) -> Any:
            response = handler(payload_to_model_request(request, req.content))
            return model_response_to_json(response, object_codec)

        result = run_sync(
            self._llm_execute(
                model_name=model_name,
                request=llm_request,
                func=_call,
                codec=model_codec,
                response_codec=model_codec,
            )
        )
        return model_response_from_json(result, object_codec)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Wrap an async LangChain agent model call in NeMo Relay LLM execution."""
        (object_codec, llm_request, model_name, model_codec) = self._prepare_model_call(request)

        async def _call(req: Any) -> Any:
            response = await handler(payload_to_model_request(request, req.content))
            return model_response_to_json(response, object_codec)

        result = await self._llm_execute(
            model_name=model_name,
            request=llm_request,
            func=_call,
            codec=model_codec,
            response_codec=model_codec,
        )
        return model_response_from_json(result, object_codec)

    def _prepare_tool_call(self, request: ToolCallRequest) -> tuple:
        """Boilerplate code common to both wrap_tool_call and awrap_tool_call"""
        parent = nemo_relay.scope.get_handle()
        codec = nemo_relay.typed.BestEffortAnyCodec()
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args") or {}
        return (parent, codec, tool_name, tool_args)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap a sync LangChain agent tool call in NeMo Relay tool execution."""

        (parent, codec, tool_name, tool_args) = self._prepare_tool_call(request)

        def _call(args: Any) -> ToolMessage | Command[Any]:
            return handler(request.override(tool_call={**request.tool_call, "args": args}))

        return run_sync(
            nemo_relay.typed.tool_execute(
                name=tool_name, args=tool_args, func=_call, args_codec=codec, result_codec=codec, handle=parent
            )
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap an async LangChain agent tool call in NeMo Relay tool execution."""

        (parent, codec, tool_name, tool_args) = self._prepare_tool_call(request)

        async def _call(args: Any) -> ToolMessage | Command[Any]:
            return await handler(request.override(tool_call={**request.tool_call, "args": args}))

        return await nemo_relay.typed.tool_execute(
            name=tool_name, args=tool_args, func=_call, args_codec=codec, result_codec=codec, handle=parent
        )
