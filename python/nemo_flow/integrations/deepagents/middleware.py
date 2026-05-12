# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents middleware for NeMo Flow observability."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from nemo_flow.integrations.deepagents._events import emit_mark, mark_base_name, tool_event_data, tool_kind
from nemo_flow.integrations.langchain.middleware import NemoFlowMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command


class NemoFlowDeepAgentsMiddleware(NemoFlowMiddleware):
    """Route Deep Agents model/tool calls through NeMo Flow and emit semantic marks.

    Deep Agents is built on LangChain ``AgentMiddleware`` and LangGraph. This
    middleware keeps the existing NeMo Flow LangChain wrapping behavior, then
    adds stable mark events for Deep Agents-specific tools such as ``task``,
    async subagent tools, filesystem tools, and sandbox execution.
    """

    def __init__(
        self,
        *,
        name: str = "NemoFlowDeepAgentsMiddleware",
        agent_name: str | None = None,
        skills: Sequence[str] | None = None,
        subagents: Sequence[Mapping[str, Any]] | None = None,
        backend_name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._agent_name = agent_name
        self._skills = list(skills) if skills is not None else None
        self._subagents = list(subagents) if subagents is not None else None
        self._backend_name = backend_name

    def before_agent(self, state: Any, runtime: Any) -> None:
        """Emit run configuration metadata for sync Deep Agents runs."""
        self._emit_agent_configuration()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        """Emit run configuration metadata for async Deep Agents runs."""
        self._emit_agent_configuration()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap a sync Deep Agents tool call with NeMo Flow tool execution and marks."""
        tool_name, tool_args, kind = self._tool_context(request)
        self._emit_tool_mark(kind, "start", tool_name, tool_args)

        try:
            result = super().wrap_tool_call(request, handler)
        except Exception as error:
            self._emit_tool_mark(kind, "error", tool_name, tool_args, error=error)
            raise

        self._emit_tool_mark(kind, "end", tool_name, tool_args, result=result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap an async Deep Agents tool call with NeMo Flow tool execution and marks."""
        tool_name, tool_args, kind = self._tool_context(request)
        self._emit_tool_mark(kind, "start", tool_name, tool_args)

        try:
            result = await super().awrap_tool_call(request, handler)
        except Exception as error:
            self._emit_tool_mark(kind, "error", tool_name, tool_args, error=error)
            raise

        self._emit_tool_mark(kind, "end", tool_name, tool_args, result=result)
        return result

    def _emit_agent_configuration(self) -> None:
        data: dict[str, Any] = {}
        if self._agent_name is not None:
            data["agent_name"] = self._agent_name

        if self._skills is not None:
            data["skills"] = list(self._skills)

        if self._subagents is not None:
            data["subagents"] = list(self._subagents)

        if self._backend_name is not None:
            data["backend"] = self._backend_name

        if data:
            emit_mark(
                mark_base_name("skill"),
                "skill",
                "configured",
                data,
                metadata={"agent_name": self._agent_name} if self._agent_name is not None else None,
            )

    def _tool_context(self, request: ToolCallRequest) -> tuple[str, Mapping[str, Any], str | None]:
        tool_name = request.tool_call["name"]
        raw_args = request.tool_call.get("args") or {}

        if isinstance(raw_args, Mapping):
            tool_args = raw_args
        else:
            tool_args = {"value": raw_args}

        return tool_name, tool_args, tool_kind(tool_name)

    def _emit_tool_mark(
        self,
        kind: str | None,
        phase: str,
        tool_name: str,
        tool_args: Mapping[str, Any],
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if kind is not None:
            emit_mark(
                mark_base_name(kind),
                kind,
                phase,
                tool_event_data(tool_name, tool_args, result=result, error=error),
                metadata={"agent_name": self._agent_name} if self._agent_name is not None else None,
            )
