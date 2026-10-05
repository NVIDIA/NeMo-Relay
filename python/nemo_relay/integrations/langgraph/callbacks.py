# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph callback handler that reuses the LangChain NeMo Relay integration."""

from __future__ import annotations

import logging
from typing import Any, Protocol, TypeVar

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

import nemo_relay
from nemo_relay.integrations.langchain._serialization import _prepare_lc_payloads
from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler as LangChainNemoRelayCallbackHandler

_logger = logging.getLogger(__name__)


class _GraphWithConfig(Protocol):
    config: RunnableConfig | None


_GraphT = TypeVar("_GraphT", bound=_GraphWithConfig)


def _interrupt_to_payload(interrupt: Any) -> dict[str, nemo_relay.Json]:
    return {
        "id": _prepare_lc_payloads(getattr(interrupt, "id", None)),
        "value": _prepare_lc_payloads(getattr(interrupt, "value", interrupt)),
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
                data=_prepare_lc_payloads(data),
                metadata={"integration": "langgraph"},
            )
        except Exception:
            _logger.debug("NeMo Relay: LangGraph mark emission failed", exc_info=True)


def configure_graph(graph: _GraphT) -> _GraphT:
    """Install Relay lifecycle callbacks in a compiled graph's base config.

    Keeping the callback in the graph config preserves it when a LangGraph
    server does not accept callback objects from an invocation request. The
    graph is updated in place so it remains a compiled graph that the server
    can load.

    Args:
        graph: Compiled LangGraph graph to configure.

    Returns:
        The same graph with a ``NemoRelayCallbackHandler`` in its base config.
    """
    config = graph.config if graph.config is not None else RunnableConfig()
    callbacks = config.get("callbacks")
    handlers = (
        [*callbacks.handlers, *callbacks.inheritable_handlers]
        if isinstance(callbacks, BaseCallbackManager)
        else callbacks or []
    )
    if not any(isinstance(handler, NemoRelayCallbackHandler) for handler in handlers):
        graph.config = merge_configs(config, {"callbacks": [NemoRelayCallbackHandler()]})
    return graph


__all__ = ["NemoRelayCallbackHandler", "configure_graph"]
