# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes Agent callback handler that maps lifecycle events to NeMo Relay scopes."""

from __future__ import annotations

import datetime
import logging
import threading
import typing

import nemo_relay

if typing.TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


class _CompletedScope(typing.NamedTuple):
    """A scope that has ended but cannot yet be popped (LIFO ordering)."""
    handle: nemo_relay.ScopeHandle
    output: nemo_relay.Json | None
    metadata: nemo_relay.Json | None
    ended_at: datetime.datetime


def _current_scope_handle() -> nemo_relay.ScopeHandle | None:
    try:
        if not nemo_relay.scope_stack_active():
            return None
        return nemo_relay.scope.get_handle()
    except Exception:
        _logger.debug("NeMo Relay: reading current scope failed", exc_info=True)
        return None


class NemoRelayHermesCallbackHandler:
    """Bridge Hermes Agent lifecycle events to NeMo Relay scopes.

    Maps the Hermes event hierarchy to NeMo Relay scope types:

    * **Session**  -> ScopeType.Run
    * **Turn**     -> ScopeType.Turn
    * **Tool**     -> ScopeType.Function
    * **LLM**      -> ScopeType.LLM
    * **Subagent** -> ScopeType.Agent
    * **Task**     -> ScopeType.Task
    """

    def __init__(self, agent_name: str = "hermes-agent") -> None:
        self._agent_name = agent_name
        self._lock = threading.Lock()
        self._active_scopes: dict[str, nemo_relay.ScopeHandle] = {}
        self._pending_completions: dict[str, _CompletedScope] = {}

    # -- public event methods --------------------------------------------------

    def on_session_start(
        self,
        session_id: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Open a Run scope for the session."""
        try:
            handle = nemo_relay.scope.push(
                f"session:{session_id}",
                nemo_relay.ScopeType.Run,
                data={"agent_name": self._agent_name, "session_id": session_id},
            )
            with self._lock:
                self._active_scopes[session_id] = handle
        except Exception:
            _logger.debug("on_session_start failed", exc_info=True)

    def on_session_end(self, session_id: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Close the Run scope for the session."""
        self._close_scope(session_id)

    def on_turn_start(
        self,
        turn_id: str,
        session_id: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Open a Turn scope as a child of the current top scope."""
        try:
            parent = self._active_scopes.get(session_id)
            handle = nemo_relay.scope.push(
                f"turn:{turn_id}",
                nemo_relay.ScopeType.Turn,
                handle=parent,
                data={"turn_id": turn_id},
            )
            with self._lock:
                self._active_scopes[turn_id] = handle
        except Exception:
            _logger.debug("on_turn_start failed", exc_info=True)

    def on_turn_end(self, turn_id: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._close_scope(turn_id)

    def on_tool_call_start(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, typing.Any] | None = None,
        *posargs: typing.Any,
        **kw: typing.Any,
    ) -> None:
        """Open a Function scope for a tool invocation."""
        try:
            handle = nemo_relay.scope.push(
                tool_name,
                nemo_relay.ScopeType.Function,
                data={"tool_call_id": tool_call_id, "tool_name": tool_name},
                input=args,
            )
            with self._lock:
                self._active_scopes[tool_call_id] = handle
        except Exception:
            _logger.debug("on_tool_call_start failed", exc_info=True)

    def on_tool_call_end(
        self,
        tool_call_id: str,
        result: typing.Any = None,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        self._close_scope(tool_call_id, output={"result": str(result)[:2048]} if result is not None else None)

    def on_llm_call_start(
        self,
        llm_call_id: str,
        model: str,
        request_data: dict[str, typing.Any] | None = None,
        *posargs: typing.Any,
        **kw: typing.Any,
    ) -> None:
        """Open an LLM scope for a model call."""
        try:
            handle = nemo_relay.scope.push(
                f"llm:{model}",
                nemo_relay.ScopeType.LLM,
                data={"llm_call_id": llm_call_id, "model": model},
                input=request_data,
            )
            with self._lock:
                self._active_scopes[llm_call_id] = handle
        except Exception:
            _logger.debug("on_llm_call_start failed", exc_info=True)

    def on_llm_call_end(
        self,
        llm_call_id: str,
        response_data: typing.Any = None,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        self._close_scope(llm_call_id, output=response_data if isinstance(response_data, dict) else None)

    def on_subagent_start(
        self,
        subagent_id: str,
        agent_name: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Open an Agent scope for a delegated sub-agent."""
        try:
            handle = nemo_relay.scope.push(
                f"subagent:{agent_name}",
                nemo_relay.ScopeType.Agent,
                data={"subagent_id": subagent_id, "agent_name": agent_name},
            )
            with self._lock:
                self._active_scopes[subagent_id] = handle
        except Exception:
            _logger.debug("on_subagent_start failed", exc_info=True)

    def on_subagent_end(self, subagent_id: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._close_scope(subagent_id)

    def on_task_start(
        self,
        task_id: str,
        task_name: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Open a Task scope for a cron/scheduled task."""
        try:
            handle = nemo_relay.scope.push(
                f"task:{task_name}",
                nemo_relay.ScopeType.Task,
                data={"task_id": task_id, "task_name": task_name},
            )
            with self._lock:
                self._active_scopes[task_id] = handle
        except Exception:
            _logger.debug("on_task_start failed", exc_info=True)

    def on_task_end(self, task_id: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._close_scope(task_id)

    # -- internal ---------------------------------------------------------------

    def _close_scope(
        self,
        event_id: str,
        output: nemo_relay.Json | None = None,
        metadata: nemo_relay.Json | None = None,
    ) -> None:
        """Close the scope for *event_id*, respecting LIFO ordering."""
        now = datetime.datetime.now(datetime.timezone.utc)
        with self._lock:
            handle = self._active_scopes.pop(event_id, None)
            if handle is None:
                return

            top = _current_scope_handle()
            if top is not None and top == handle:
                nemo_relay.scope.pop(handle, output=output, metadata=metadata)
                self._drain_pending()
            else:
                self._pending_completions[event_id] = _CompletedScope(
                    handle=handle, output=output, metadata=metadata, ended_at=now,
                )

    def _drain_pending(self) -> None:
        """Pop completed scopes that are now at the top of the stack."""
        drained = True
        while drained:
            drained = False
            top = _current_scope_handle()
            if top is None:
                break
            for eid, pending in list(self._pending_completions.items()):
                if pending.handle == top:
                    nemo_relay.scope.pop(
                        pending.handle,
                        output=pending.output,
                        metadata=pending.metadata,
                    )
                    del self._pending_completions[eid]
                    drained = True
                    break
