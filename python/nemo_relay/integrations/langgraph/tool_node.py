# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed NeMo Relay wrappers for standalone LangGraph ``ToolNode`` objects."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

import nemo_relay
from nemo_relay.typed import Codec
from nemo_relay.utils import run_sync


class _ToolNodeResultCodec(nemo_relay.typed.BestEffortAnyCodec):
    """Restore ToolMessages nested in LangGraph Command updates."""

    def from_json(self, data: nemo_relay.Json) -> object:
        result = super().from_json(data)
        if not isinstance(result, Command) or not isinstance(result.update, dict):
            return result

        messages = result.update.get("messages")
        if not isinstance(messages, list):
            return result
        result.update["messages"] = [
            ToolMessage.model_validate(message) if isinstance(message, dict) else message for message in messages
        ]
        return result


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
    result_codec = cast(Codec[ToolMessage | Command[Any]], _ToolNodeResultCodec())

    def _call(args: dict[str, Any]) -> nemo_relay.ToolExecutionResult[ToolMessage | Command[Any]]:
        return nemo_relay.ToolExecutionResult(execute(request.override(tool_call={**request.tool_call, "args": args})))

    return run_sync(
        nemo_relay.typed.tool_execute(
            name=tool_name,
            args=tool_args,
            func=_call,
            args_codec=args_codec,
            result_codec=result_codec,
            handle=parent,
            tool_call_id=tool_call_id,
        )
    ).result


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
    result_codec = cast(Codec[ToolMessage | Command[Any]], _ToolNodeResultCodec())

    async def _call(args: dict[str, Any]) -> nemo_relay.ToolExecutionResult[ToolMessage | Command[Any]]:
        return nemo_relay.ToolExecutionResult(
            await execute(request.override(tool_call={**request.tool_call, "args": args}))
        )

    return (
        await nemo_relay.typed.tool_execute(
            name=tool_name,
            args=tool_args,
            func=_call,
            args_codec=args_codec,
            result_codec=result_codec,
            handle=parent,
            tool_call_id=tool_call_id,
        )
    ).result


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
    return ToolNode(
        tools,
        wrap_tool_call=wrap_tool_call,
        awrap_tool_call=awrap_tool_call,
        **tool_node_kwargs,
    )


__all__ = ["awrap_tool_call", "create_tool_node", "wrap_tool_call"]
