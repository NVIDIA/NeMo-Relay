# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for applying NeMo Flow callbacks to compiled LangGraph graphs."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

import nemo_flow
from nemo_flow.integrations.langgraph.callbacks import NemoFlowCallbackHandler

_logger = logging.getLogger(__name__)


def with_nemo_flow_callbacks(
    config: RunnableConfig | None = None,
    *,
    callback_handler: NemoFlowCallbackHandler | None = None,
) -> RunnableConfig:
    """Return a LangChain runnable config with a LangGraph NeMo Flow callback.

    The returned config is a shallow copy. Existing callbacks are preserved and
    a handler is added only when a NeMo Flow LangGraph handler is not already
    present.
    """
    next_config: RunnableConfig = dict(config or {})
    next_config["callbacks"] = _append_callback(
        next_config.get("callbacks"),
        callback_handler or NemoFlowCallbackHandler(),
    )
    return next_config


def instrument_graph(
    graph: Any,
    *,
    callback_handler: NemoFlowCallbackHandler | None = None,
) -> "NemoFlowGraph":
    """Wrap a compiled LangGraph graph so invocations include NeMo Flow callbacks."""
    return NemoFlowGraph(graph, callback_handler=callback_handler)


class NemoFlowGraph:
    """Thin proxy that injects NeMo Flow callbacks into graph invocations."""

    def __init__(
        self,
        graph: Any,
        *,
        callback_handler: NemoFlowCallbackHandler | None = None,
    ) -> None:
        self._graph = graph
        self._callback_handler = callback_handler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> "NemoFlowGraph":
        """Return an instrumented copy of the graph with updated LangChain config."""
        return NemoFlowGraph(
            self._graph.with_config(config, **kwargs),
            callback_handler=self._callback_handler,
        )

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the wrapped graph with NeMo Flow callbacks installed."""
        return self._graph.invoke(
            input,
            self._config(config),
            **kwargs,
        )

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously invoke the wrapped graph with NeMo Flow callbacks installed."""
        return await self._graph.ainvoke(
            input,
            self._config(config),
            **kwargs,
        )

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream from the wrapped graph and mark public task/checkpoint events."""
        for chunk in self._graph.stream(
            input,
            self._config(config),
            **kwargs,
        ):
            _emit_public_stream_mark(chunk)
            yield chunk

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronously stream from the wrapped graph and mark public events."""
        async for chunk in self._graph.astream(
            input,
            self._config(config),
            **kwargs,
        ):
            _emit_public_stream_mark(chunk)
            yield chunk

    def _config(self, config: RunnableConfig | None) -> RunnableConfig:
        return with_nemo_flow_callbacks(config, callback_handler=self._callback_handler)


def _append_callback(callbacks: Any, handler: BaseCallbackHandler) -> Any:
    if callbacks is None:
        return [handler]

    if isinstance(callbacks, NemoFlowCallbackHandler):
        return callbacks

    if isinstance(callbacks, BaseCallbackManager):
        if _manager_has_nemo_flow_handler(callbacks):
            return callbacks
        manager = callbacks.copy()
        manager.add_handler(handler, inherit=True)
        return manager

    if isinstance(callbacks, BaseCallbackHandler):
        if isinstance(callbacks, NemoFlowCallbackHandler):
            return callbacks
        return [callbacks, handler]

    if isinstance(callbacks, Sequence) and not isinstance(callbacks, str | bytes):
        callback_list = list(callbacks)
        if any(isinstance(callback, NemoFlowCallbackHandler) for callback in callback_list):
            return callback_list
        return [*callback_list, handler]

    return callbacks


def _manager_has_nemo_flow_handler(manager: BaseCallbackManager) -> bool:
    handlers = [*manager.handlers, *manager.inheritable_handlers]
    return any(isinstance(handler, NemoFlowCallbackHandler) for handler in handlers)


def _emit_public_stream_mark(chunk: Any) -> None:
    mode, payload = _stream_mode_payload(chunk)
    if mode == "tasks":
        _emit_task_mark(payload)
    elif mode == "checkpoints":
        _emit_mark("Checkpoint Save", payload)
    elif mode == "debug":
        _emit_debug_mark(payload)


def _stream_mode_payload(chunk: Any) -> tuple[str | None, Any]:
    if isinstance(chunk, dict) and isinstance(chunk.get("type"), str) and "data" in chunk:
        return chunk["type"], chunk["data"]
    if isinstance(chunk, tuple):
        if len(chunk) == 2 and isinstance(chunk[0], str):
            return chunk[0], chunk[1]
        if len(chunk) == 3 and isinstance(chunk[1], str):
            return chunk[1], chunk[2]
    if isinstance(chunk, dict):
        if "type" in chunk and "payload" in chunk:
            return "debug", chunk
        if {"id", "name", "triggers"}.issubset(chunk):
            return "tasks", chunk
        if {"id", "name", "result"}.issubset(chunk) or {"id", "name", "error"}.issubset(chunk):
            return "tasks", chunk
        if {"config", "metadata", "values", "next", "tasks"}.issubset(chunk):
            return "checkpoints", chunk
    return None, None


def _emit_debug_mark(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    event_type = payload.get("type")
    event_payload = payload.get("payload")
    if event_type == "task":
        _emit_mark("Graph Task Start", event_payload)
    elif event_type == "task_result":
        _emit_mark("Graph Task End", event_payload)
    elif event_type == "checkpoint":
        _emit_mark("Checkpoint Save", event_payload)


def _emit_task_mark(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    if "triggers" in payload:
        _emit_mark("Graph Task Start", payload)
    elif "result" in payload or "error" in payload:
        _emit_mark("Graph Task End", payload)


def _emit_mark(name: str, payload: Any) -> None:
    try:
        nemo_flow.scope.event(
            name,
            data=_json_safe(payload),
            metadata={"integration": "langgraph"},
        )
    except Exception:
        _logger.debug("NeMo Flow: LangGraph stream mark emission failed", exc_info=True)


def _json_safe(value: Any) -> nemo_flow.Json:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return repr(value)


__all__ = ["NemoFlowGraph", "instrument_graph", "with_nemo_flow_callbacks"]
