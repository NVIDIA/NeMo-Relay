# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents callback handler for NeMo Flow observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_flow.integrations.deepagents._events import emit_mark, event_base_name
from nemo_flow.integrations.langgraph.callbacks import NemoFlowCallbackHandler as LangGraphNemoFlowCallbackHandler

_GraphEventKey = tuple[str | None, str | None, tuple[str, ...]]


class NemoFlowDeepAgentsCallbackHandler(LangGraphNemoFlowCallbackHandler):
    """Bridge Deep Agents LangGraph lifecycle events to NeMo Flow marks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hitl_interrupts: set[_GraphEventKey] = set()

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
        run_id = NemoFlowDeepAgentsCallbackHandler._string_or_none(data.get("run_id"))
        checkpoint_id = NemoFlowDeepAgentsCallbackHandler._string_or_none(data.get("checkpoint_id"))
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
        return any(NemoFlowDeepAgentsCallbackHandler._is_hitl_interrupt_payload(interrupt) for interrupt in interrupts)

    @staticmethod
    def _is_hitl_interrupt_payload(interrupt: Any) -> bool:
        if not isinstance(interrupt, Mapping):
            return False
        return NemoFlowDeepAgentsCallbackHandler._is_hitl_request(interrupt.get("value"))

    @staticmethod
    def _is_hitl_request(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        action_requests = value.get("action_requests")
        review_configs = value.get("review_configs")
        return NemoFlowDeepAgentsCallbackHandler._is_mapping_sequence(
            action_requests
        ) and NemoFlowDeepAgentsCallbackHandler._is_mapping_sequence(review_configs)

    @staticmethod
    def _is_mapping_sequence(value: Any) -> bool:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
            return False
        return all(isinstance(item, Mapping) for item in value)


__all__ = ["NemoFlowDeepAgentsCallbackHandler"]
