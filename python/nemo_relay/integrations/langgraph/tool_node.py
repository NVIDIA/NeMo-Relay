# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed NeMo Relay wrappers for standalone LangGraph ``ToolNode`` objects."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolInvocationError
from langgraph.types import Command

import nemo_relay
from nemo_relay.typed import Codec
from nemo_relay.utils import run_sync


class _ToolNodeResultCodec(nemo_relay.typed.BestEffortAnyCodec):
    """Restore ToolMessages nested in LangGraph Command updates."""

    _LIST_TAG = "__nemo_relay_langgraph_list__"

    def to_json(self, value: object) -> nemo_relay.Json:
        if isinstance(value, list):
            return {self._LIST_TAG: [self.to_json(item) for item in value]}
        return super().to_json(value)

    def from_json(self, data: nemo_relay.Json) -> object:
        if isinstance(data, dict) and isinstance(data.get(self._LIST_TAG), list):
            return [self.from_json(item) for item in data[self._LIST_TAG]]

        result = super().from_json(data)
        if not isinstance(result, Command):
            return result

        if isinstance(result.update, dict):
            messages = result.update.get("messages")
            if isinstance(messages, list):
                result.update["messages"] = _restore_tool_messages(messages)
        elif isinstance(result.update, list):
            return replace(result, update=_restore_tool_messages(result.update))
        return result


def _restore_tool_messages(messages: list[object]) -> list[object]:
    """Reconstruct serialized ToolMessages in a LangGraph command update."""
    return [ToolMessage.model_validate(message) if isinstance(message, dict) else message for message in messages]


_DEFAULT_HANDLE_TOOL_ERRORS = object()
_TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
_GRAPH_BUBBLE_RESULT = {"__nemo_relay_langgraph_graph_bubble_up__": True}


def _handle_tool_error(error: Exception, policy: object) -> str:
    """Preserve ToolNode error handling while allowing graph bubbles to escape."""
    if isinstance(error, GraphBubbleUp):
        raise error
    if policy is _DEFAULT_HANDLE_TOOL_ERRORS:
        if isinstance(error, ToolInvocationError):
            return error.message
        raise error
    if isinstance(policy, tuple):
        if not isinstance(error, policy):
            raise error
        return _TOOL_CALL_ERROR_TEMPLATE.format(error=repr(error))
    if isinstance(policy, type) and issubclass(policy, Exception):
        if not isinstance(error, policy):
            raise error
        return _TOOL_CALL_ERROR_TEMPLATE.format(error=repr(error))
    if policy is True:
        return _TOOL_CALL_ERROR_TEMPLATE.format(error=repr(error))
    if isinstance(policy, str):
        return policy
    if callable(policy):
        return cast(Callable[[Exception], str], policy)(error)
    raise ValueError(f"unexpected handle_tool_errors value: {policy}")


def _tool_details(request: ToolCallRequest) -> tuple[nemo_relay.ScopeHandle, str, dict[str, Any], str | None]:
    """Extract the model-controlled tool-call fields managed by Relay."""
    return (
        nemo_relay.scope.get_handle(),
        request.tool_call["name"],
        request.tool_call.get("args") or {},
        request.tool_call.get("id"),
    )


def wrap_tool_call(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Run one synchronous LangGraph tool call through NeMo Relay.

    Args:
        request: LangGraph's tool-call request.
        execute: LangGraph callback that invokes the requested tool.

    Returns:
        The LangGraph tool result after managed Relay execution.
    """
    parent, tool_name, tool_args, tool_call_id = _tool_details(request)
    args_codec = cast(Codec[dict[str, Any]], nemo_relay.typed.BestEffortAnyCodec())
    result_codec = cast(Codec[ToolMessage | Command[Any] | dict[str, bool]], _ToolNodeResultCodec())
    graph_bubble: GraphBubbleUp | None = None

    def _call(args: dict[str, Any]) -> nemo_relay.ToolExecutionResult[ToolMessage | Command[Any] | dict[str, bool]]:
        nonlocal graph_bubble
        try:
            result = execute(request.override(tool_call={**request.tool_call, "args": args}))
        except GraphBubbleUp as error:
            # Relay's native callback boundary cannot propagate arbitrary Python
            # exceptions. Preserve the original graph bubble locally and re-raise
            # it after managed execution returns.
            graph_bubble = error
            result = _GRAPH_BUBBLE_RESULT
        return nemo_relay.ToolExecutionResult(result)

    outcome = run_sync(
        nemo_relay.typed.tool_execute(
            name=tool_name,
            args=tool_args,
            func=_call,
            args_codec=args_codec,
            result_codec=result_codec,
            handle=parent,
            tool_call_id=tool_call_id,
        )
    )
    if graph_bubble is not None:
        raise graph_bubble
    return cast(ToolMessage | Command[Any], outcome.result)


async def awrap_tool_call(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """Run one asynchronous LangGraph tool call through NeMo Relay.

    Args:
        request: LangGraph's tool-call request.
        execute: Async LangGraph callback that invokes the requested tool.

    Returns:
        The LangGraph tool result after managed Relay execution.
    """
    parent, tool_name, tool_args, tool_call_id = _tool_details(request)
    args_codec = cast(Codec[dict[str, Any]], nemo_relay.typed.BestEffortAnyCodec())
    result_codec = cast(Codec[ToolMessage | Command[Any] | dict[str, bool]], _ToolNodeResultCodec())
    graph_bubble: GraphBubbleUp | None = None

    async def _call(
        args: dict[str, Any],
    ) -> nemo_relay.ToolExecutionResult[ToolMessage | Command[Any] | dict[str, bool]]:
        nonlocal graph_bubble
        try:
            result = await execute(request.override(tool_call={**request.tool_call, "args": args}))
        except GraphBubbleUp as error:
            # See the synchronous wrapper: retain the original exception instead
            # of letting the native callback boundary convert it to RuntimeError.
            graph_bubble = error
            result = _GRAPH_BUBBLE_RESULT
        return nemo_relay.ToolExecutionResult(result)

    outcome = await nemo_relay.typed.tool_execute(
        name=tool_name,
        args=tool_args,
        func=_call,
        args_codec=args_codec,
        result_codec=result_codec,
        handle=parent,
        tool_call_id=tool_call_id,
    )
    if graph_bubble is not None:
        raise graph_bubble
    return cast(ToolMessage | Command[Any], outcome.result)


def create_tool_node(
    tools: Sequence[BaseTool | Callable[..., Any]],
    **tool_node_kwargs: Any,
) -> ToolNode:
    """Create a LangGraph ``ToolNode`` whose tool calls use managed Relay execution.

    For custom LangGraph wrapper composition, construct ``ToolNode`` directly
    and pass :func:`wrap_tool_call` and :func:`awrap_tool_call` explicitly.

    Args:
        tools: LangGraph tools available to the returned node.
        **tool_node_kwargs: Remaining native ``ToolNode`` constructor options,
            excluding the Relay-managed wrapper options.

    Returns:
        A native ``ToolNode`` configured with Relay's sync and async wrappers.
    """
    configured_wrappers = {"wrap_tool_call", "awrap_tool_call"}.intersection(tool_node_kwargs)
    if configured_wrappers:
        names = ", ".join(sorted(configured_wrappers))
        raise ValueError(f"create_tool_node configures {names}; construct ToolNode directly to compose custom wrappers")
    error_policy = tool_node_kwargs.pop("handle_tool_errors", _DEFAULT_HANDLE_TOOL_ERRORS)
    if error_policy is False:
        error_handler: bool | Callable[[Exception], str] = False
    else:

        def error_handler(error: Exception) -> str:
            return _handle_tool_error(error, error_policy)

    return ToolNode(
        tools,
        wrap_tool_call=wrap_tool_call,
        awrap_tool_call=awrap_tool_call,
        handle_tool_errors=error_handler,
        **tool_node_kwargs,
    )


__all__ = ["awrap_tool_call", "create_tool_node", "wrap_tool_call"]
