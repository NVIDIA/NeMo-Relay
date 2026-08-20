# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents callback handler for NeMo Relay observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from nemo_relay.integrations.deepagents._events import emit_mark, event_base_name
from nemo_relay.integrations.langgraph.callbacks import NemoRelayCallbackHandler as LangGraphNemoRelayCallbackHandler

_GraphEventKey = tuple[str | None, str | None, tuple[str, ...]]


class NemoRelayDeepAgentsCallbackHandler(LangGraphNemoRelayCallbackHandler):
    """Bridge semantic Deep Agents runs and LangGraph lifecycle events to NeMo Relay."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hitl_interrupts: set[_GraphEventKey] = set()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Push scopes for Deep Agents orchestrators and subagents, not graph nodes."""
        agent_name = self._semantic_agent_name(kwargs.get("name"), metadata)
        if agent_name is None:
            return None

        scope_metadata = dict(metadata or {})
        scope_metadata.update(
            {
                "integration": "deepagents",
                "deepagents_agent_name": agent_name,
                "deepagents_agent_role": "orchestrator" if parent_run_id is None else "subagent",
            }
        )
        scope_kwargs = dict(kwargs)
        scope_kwargs["name"] = agent_name
        return super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=scope_metadata,
            **scope_kwargs,
        )

    @staticmethod
    def _semantic_agent_name(name: Any, metadata: Mapping[str, Any] | None) -> str | None:
        if metadata is None:
            return None

        versions = metadata.get("lc_versions")
        if not isinstance(versions, Mapping) or "deepagents" not in versions:
            return None

        langgraph_node = metadata.get("langgraph_node")
        configured_name = metadata.get("lc_agent_name")
        if not isinstance(configured_name, str) or not configured_name:
            configured_name = None

        integration = metadata.get("ls_integration")
        if langgraph_node is not None:
            if integration == "langchain_create_agent" and name == configured_name:
                return configured_name
            if langgraph_node == name:
                return None
            return None

        if configured_name is not None and name == configured_name:
            return configured_name

        if integration == "deepagents":
            return "DeepAgent"

        return None

    def _emit_graph_mark(self, name: str, data: dict[str, Any]) -> None:
        key = self._graph_event_key(data)
        if name == "Graph Interrupt" and self._has_hitl_interrupt(data):
            self._hitl_interrupts.add(key)
            self._emit_human_in_the_loop_mark(name, "interrupt", data)
            return

        if name == "Graph Resume" and key in self._hitl_interrupts:
            self._hitl_interrupts.discard(key)
            self._emit_human_in_the_loop_mark(name, "resume", data)
            return

        super()._emit_graph_mark(name, data)

    def _emit_human_in_the_loop_mark(self, name: str, phase: str, data: dict[str, Any]) -> None:
        emit_mark(
            event_base_name("human_in_the_loop"),
            "human_in_the_loop",
            phase,
            data,
            metadata={"langgraph_event": name},
        )

    @staticmethod
    def _graph_event_key(data: Mapping[str, Any]) -> _GraphEventKey:
        run_id = NemoRelayDeepAgentsCallbackHandler._string_or_none(data.get("run_id"))
        checkpoint_id = NemoRelayDeepAgentsCallbackHandler._string_or_none(data.get("checkpoint_id"))
        checkpoint_ns = data.get("checkpoint_ns")
        if not isinstance(checkpoint_ns, Sequence) or isinstance(checkpoint_ns, str | bytes | bytearray):
            return (run_id, checkpoint_id, ())
        return (run_id, checkpoint_id, tuple(str(part) for part in checkpoint_ns))

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _has_hitl_interrupt(data: Mapping[str, Any]) -> bool:
        interrupts = data.get("interrupts")
        if not isinstance(interrupts, Sequence) or isinstance(interrupts, str | bytes | bytearray):
            return False
        return any(NemoRelayDeepAgentsCallbackHandler._is_hitl_interrupt_payload(interrupt) for interrupt in interrupts)

    @staticmethod
    def _is_hitl_interrupt_payload(interrupt: Any) -> bool:
        if not isinstance(interrupt, Mapping):
            return False
        return NemoRelayDeepAgentsCallbackHandler._is_hitl_request(interrupt.get("value"))

    @staticmethod
    def _is_hitl_request(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        action_requests = value.get("action_requests")
        review_configs = value.get("review_configs")
        return NemoRelayDeepAgentsCallbackHandler._is_mapping_sequence(
            action_requests
        ) and NemoRelayDeepAgentsCallbackHandler._is_mapping_sequence(review_configs)

    @staticmethod
    def _is_mapping_sequence(value: Any) -> bool:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
            return False
        return all(isinstance(item, Mapping) for item in value)


__all__ = ["NemoRelayDeepAgentsCallbackHandler"]
