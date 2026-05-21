# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents middleware for NeMo Relay observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_relay.integrations.deepagents._events import emit_mark, event_base_name
from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware


class NemoRelayDeepAgentsMiddleware(NemoRelayMiddleware):
    """Route Deep Agents model/tool calls through NeMo Relay and emit semantic events.

    Deep Agents is built on LangChain ``AgentMiddleware`` and LangGraph. This
    middleware keeps the existing NeMo Relay LangChain wrapping behavior, then
    emits Deep Agents configuration marks.
    """

    def __init__(
        self,
        *,
        name: str = "NemoRelayDeepAgentsMiddleware",
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
                event_base_name("skill"),
                "skill",
                "configured",
                data,
                metadata={"agent_name": self._agent_name} if self._agent_name is not None else None,
            )
