# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain callback handler that maps run hierarchy to NeMo Relay scopes."""

from __future__ import annotations

import datetime
import logging
import math
import threading
import typing

from langchain_core.callbacks.base import BaseCallbackHandler

import nemo_relay
from nemo_relay.integrations.langchain._serialization import _prepare_lc_payloads

if typing.TYPE_CHECKING:
    from uuid import UUID

_logger = logging.getLogger(__name__)


class _CompletedScope(typing.NamedTuple):
    """A run that has ended, waiting for its scope to reach the top of the stack.

    A scope stack closes strictly LIFO, but concurrent sibling runs finish in the order
    the graph chooses, so a run can end while scopes opened after it are still open.
    Recording the completion and closing it once it reaches the top preserves the stack's
    ordering without ever attempting a close the stack would reject.

    A completion is only ever closed when its scope is the top of the active stack, so
    ownership needs no separate check: a scope sits on exactly one stack, and its handle
    uuid can only be the current top on that stack. Comparing ``ScopeStack`` objects
    would not work anyway -- propagating a stack to a worker thread yields a different
    Python wrapper for the same stack, with no way to correlate the two.

    ``output`` is already serialized: callers own the mapping they hand to the callback
    and may mutate it afterwards, so it is snapshotted when the callback fires, for the
    same reason ``ended_at`` is.

    Entries are removed as their scopes reach the top, so the set is bounded by the runs
    a handler has open at once. The exception is a scope closed by something other than
    this handler: it can never return to the top, so its entry is retained for the life
    of the handler, holding the serialized output.
    """

    handle: nemo_relay.ScopeHandle
    output: nemo_relay.Json | None
    metadata: nemo_relay.Json | None
    ended_at: datetime.datetime


def _current_scope_handle() -> nemo_relay.ScopeHandle | None:
    """Return the scope on top of the active stack, or ``None`` if there is not one.

    ``get_handle`` creates a stack for the current context when it has none, so it is
    guarded by a status check: a run completing somewhere without a stack has nothing
    this handler could close, and should not leave one behind for having asked. The
    check consults the thread binding as well as the context, so a stack propagated to
    a worker thread still counts as active.
    """
    try:
        if not nemo_relay.scope_stack_active():
            return None
        return nemo_relay.scope.get_handle()
    except Exception:
        _logger.debug("NeMo Relay: reading the current scope failed", exc_info=True)
        return None


class NemoRelayCallbackHandler(BaseCallbackHandler):
    """Bridge LangChain chain run IDs to NeMo Relay Agent scopes."""

    # We need to run inline to ensure scopes are pushed and popped in the correct order.
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        # Start, completion, top-check and pop are multi-step transitions over the maps
        # below. A handler can be reused by callbacks delivered on worker threads, so the
        # transitions are serialized: a top-check is only meaningful if the pop it
        # authorizes cannot be overtaken by another thread pushing onto the same stack.
        # Reentrant because completing a run drains, and both take the lock.
        self._lock = threading.RLock()
        self._scope_handles: dict[UUID, nemo_relay.ScopeHandle] = {}
        self._completed: dict[str, _CompletedScope] = {}

    def on_chain_start(
        self,
        serialized: dict[str, typing.Any],
        inputs: dict[str, typing.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Push a NeMo Relay Agent scope for a LangChain chain run."""
        try:
            name = kwargs.get("name")

            if serialized is not None:
                name = name or serialized.get("name")
                if name is None:
                    id_list = serialized.get("id")
                    if isinstance(id_list, list) and len(id_list) > 0:
                        name = id_list[-1]

            if name is None:
                name = "Unknown"

            parent = None
            if parent_run_id is not None:
                parent = self._scope_handles.get(parent_run_id)

            scope_metadata = metadata.copy() if metadata else {}
            scope_metadata["langchain_run_id"] = str(run_id)
            prepared_inputs = _prepare_lc_payloads(inputs)
            handle = nemo_relay.scope.push(
                name,
                nemo_relay.ScopeType.Agent,
                handle=parent,
                input=prepared_inputs,
                metadata=scope_metadata,
            )
            with self._lock:
                self._scope_handles[run_id] = handle
        except Exception:
            _logger.error("NeMo Relay: on_chain_start failed", exc_info=True)

    def on_chain_end(
        self,
        outputs: dict[str, typing.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Pop the NeMo Relay scope associated with a LangChain chain run."""
        self._pop_scope(run_id, output=outputs, metadata={"otel.status_code": "OK"})

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Pop the NeMo Relay scope associated with a failed LangChain chain run."""
        self._pop_scope(
            run_id,
            output={"error": repr(error)},
            metadata={"otel.status_code": "ERROR", "otel.status_description": str(error)},
        )

    def _pop_scope(
        self, run_id: UUID, *, output: dict[str, typing.Any] | None = None, metadata: nemo_relay.Json | None = None
    ) -> None:
        # Serialized before the run mappings are touched. Serialization walks
        # caller-supplied data and can fail on its own -- a cyclic output raises
        # ``RecursionError`` -- and a run dropped from the maps with no completion
        # recorded is a scope nothing can ever close.
        prepared_output = _prepare_output(output)

        with self._lock:
            handle = self._scope_handles.pop(run_id, None)
            if handle is None:
                return

            self._completed[handle.uuid] = _CompletedScope(
                handle=handle,
                output=prepared_output,
                metadata=metadata,
                ended_at=datetime.datetime.now(datetime.timezone.utc),
            )
            self._close_completed_scopes_locked()

    def _close_completed_scopes(self) -> None:
        """Close finished scopes from the top of the active stack down."""
        with self._lock:
            self._close_completed_scopes_locked()

    def _close_completed_scopes_locked(self) -> None:
        """Close finished scopes from the top of ``stack`` down.

        Only the scope currently on top is ever closed, so the stack is never asked to
        accept a close it would reject. Draining stops as soon as the top is a run that
        is still going, or a scope this handler does not own; anything completed
        underneath it waits, which is what keeps the ordering intact.

        This is also what confines a shared handler to one stack at a time: a completion
        belonging to another stack cannot be the top here, so it is left alone.

        The caller must hold ``self._lock``: a top-check is only meaningful if the pop it
        authorizes cannot be overtaken by another thread pushing onto the same stack.
        """
        while self._completed:
            top = _current_scope_handle()
            if top is None:
                return
            completed = self._completed.get(top.uuid)
            if completed is None:
                return
            if top.name != completed.handle.name:
                # Same uuid, different scope. ``create_scope_stack_from_propagation``
                # rebuilds a parent from a captured context, so our uuid can appear on
                # another stack as a stand-in. Closing it would discard the completion
                # and strand the real scope. Handles cannot be told apart by identity --
                # reading the current one returns a fresh wrapper even on its own stack.
                return
            try:
                nemo_relay.scope.pop(
                    completed.handle,
                    output=completed.output,
                    metadata=completed.metadata,
                    timestamp=completed.ended_at,
                )
            except Exception:
                # The runtime can refuse before it mutates the stack, so the scope
                # may still be live and closable later. Keep the completion -- it is
                # the only record able to close it -- and stop rather than spin.
                _logger.error("NeMo Relay: scope.pop failed", exc_info=True)
                return
            # Only once the stack has actually accepted the close.
            del self._completed[top.uuid]


def _prepare_output(output: dict[str, typing.Any] | None) -> nemo_relay.Json | None:
    """Convert a callback output to Relay JSON, degrading to ``None`` on failure.

    Losing an output payload costs one scope's telemetry detail. Letting the failure
    escape would cost the scope itself, since the run has ended and nothing else will
    close it.
    """
    if output is None:
        return None
    try:
        prepared = _prepare_lc_payloads(output)
        _validate_json_value(prepared)
        return typing.cast(nemo_relay.Json, prepared)
    except Exception:
        _logger.error("NeMo Relay: preparing scope output failed", exc_info=True)
        return None


def _validate_json_value(value: typing.Any) -> None:
    """Raise when ``value`` is outside the JSON shape accepted by Relay."""
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int):
        if -(1 << 63) <= value < (1 << 64):
            return
        raise ValueError("integer is outside Relay's JSON range")
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError("non-finite float is not valid Relay JSON")
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Relay JSON object keys must be strings")
            _validate_json_value(item)
        return
    raise TypeError(f"{type(value).__name__} is not valid Relay JSON")
