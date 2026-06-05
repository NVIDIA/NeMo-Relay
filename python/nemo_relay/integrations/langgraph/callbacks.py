# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph callback handler that reuses the LangChain NeMo Relay integration."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

import nemo_relay
from nemo_relay.integrations.langchain._serialization import _prepare_lc_payloads
from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler as LangChainNemoRelayCallbackHandler

_logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> nemo_relay.Json:
    """Return a conservative JSON-compatible representation for mark payloads."""
    try:
        value = _prepare_lc_payloads(value)
    except Exception:
        pass

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return repr(value)


def _interrupt_to_payload(interrupt: Any) -> dict[str, nemo_relay.Json]:
    return {
        "id": _json_safe(getattr(interrupt, "id", None)),
        "value": _json_safe(getattr(interrupt, "value", interrupt)),
    }


class NemoRelayCallbackHandler(LangChainNemoRelayCallbackHandler, GraphCallbackHandler):
    """
    Bridge LangChain and LangGraph runs to NeMo Relay using public callback APIs.

    This handler inherits the existing LangChain callback integration, so normal
    runnable scopes from LangGraph and LangChain are recorded by the same code
    path. It also implements LangGraph's public lifecycle callback hooks for
    interrupt and resume marks.
    """

    def on_interrupt(self, event: GraphInterruptEvent) -> Any:
        """Emit a NeMo Relay mark for a LangGraph interrupt lifecycle event."""
        self._emit_graph_mark(
            "Graph Interrupt",
            {
                "run_id": str(event.run_id) if event.run_id is not None else None,
                "status": event.status,
                "checkpoint_id": event.checkpoint_id,
                "checkpoint_ns": list(event.checkpoint_ns),
                "interrupts": [_interrupt_to_payload(interrupt) for interrupt in event.interrupts],
            },
        )
        return None

    def on_resume(self, event: GraphResumeEvent) -> Any:
        """Emit a NeMo Relay mark for a LangGraph resume lifecycle event."""
        self._emit_graph_mark(
            "Graph Resume",
            {
                "run_id": str(event.run_id) if event.run_id is not None else None,
                "status": event.status,
                "checkpoint_id": event.checkpoint_id,
                "checkpoint_ns": list(event.checkpoint_ns),
            },
        )
        return None

    def _emit_graph_mark(self, name: str, data: dict[str, Any]) -> None:
        try:
            nemo_relay.scope.event(
                name,
                data=_json_safe(data),
                metadata={"integration": "langgraph"},
            )
        except Exception:
            _logger.debug("NeMo Relay: LangGraph mark emission failed", exc_info=True)


__all__ = ["NemoRelayCallbackHandler"]
