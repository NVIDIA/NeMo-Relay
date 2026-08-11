# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain callback handler that maps run hierarchy to NeMo Relay scopes."""

from __future__ import annotations

import datetime
import enum
import logging
import typing

from langchain_core.callbacks.base import BaseCallbackHandler

import nemo_relay
from nemo_relay.integrations.langchain._serialization import _prepare_lc_payloads

if typing.TYPE_CHECKING:
    from uuid import UUID

_logger = logging.getLogger(__name__)

# The runtime reports a close that arrived out of LIFO order with this message (see
# ``pop_scope_inner`` in crates/core/src/api/scope.rs). It is the one pop failure worth
# retrying, and it is only distinguishable by text because the runtime raises a plain
# ``RuntimeError`` for every failure. ``test_relay_still_reports_out_of_order_closes``
# fails if that message ever changes, rather than letting the retry quietly stop working.
_OUT_OF_ORDER_CLOSE = "not at the top of the stack"


class _CloseOutcome(enum.Enum):
    """What the scope stack did with a close."""

    CLOSED = "closed"
    # Rejected because scopes opened later are still open; retry once they close.
    DEFERRED = "deferred"
    # Rejected for any other reason; retrying cannot help.
    FAILED = "failed"


class _PendingPop(typing.NamedTuple):
    """A scope close that is waiting for its handle to reach the top of the stack."""

    handle: typing.Any
    output: dict[str, typing.Any] | None
    metadata: nemo_relay.Json | None
    ended_at: datetime.datetime


class NemoRelayCallbackHandler(BaseCallbackHandler):
    """Bridge LangChain chain run IDs to NeMo Relay Agent scopes."""

    # We need to run inline to ensure scopes are pushed and popped in the correct order.
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._scope_handles: dict[UUID, typing.Any] = {}
        self._pending_pops: list[_PendingPop] = []

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
        handle = self._scope_handles.pop(run_id, None)
        if handle is None:
            return

        # A scope stack closes strictly LIFO, but concurrent sibling runs finish in the
        # order the graph chooses, so this run's scope may not be on top yet. Queue a
        # rejected close and replay it once the scopes above it have gone, rather than
        # abandoning a scope that is still live: an abandoned scope stays current for
        # everything that follows and makes the enclosing scope fail to close. The end
        # time is captured now so a deferred close still records when the run ended.
        pending = _PendingPop(
            handle=handle,
            output=output,
            metadata=metadata,
            ended_at=datetime.datetime.now(datetime.timezone.utc),
        )
        outcome = self._close_scope(pending)
        if outcome is _CloseOutcome.DEFERRED:
            self._pending_pops.append(pending)
        elif outcome is _CloseOutcome.CLOSED:
            # Closing this scope may have exposed one that was queued behind it.
            self._drain_pending_pops()

    def _drain_pending_pops(self) -> None:
        """Replay queued closes until none of them can make progress."""
        progressed = True
        while progressed and self._pending_pops:
            progressed = False
            for index, pending in enumerate(self._pending_pops):
                outcome = self._close_scope(pending)
                if outcome is _CloseOutcome.DEFERRED:
                    continue
                # Closed, or failed in a way retrying cannot fix: either way it leaves
                # the queue, so a scope that can never close cannot pin the queue open.
                del self._pending_pops[index]
                progressed = True
                break

    def _close_scope(self, pending: _PendingPop) -> _CloseOutcome:
        """Close one scope and report what the stack did with it."""
        try:
            prepared_outputs = _prepare_lc_payloads(pending.output) if pending.output is not None else None
            nemo_relay.scope.pop(
                pending.handle,
                output=prepared_outputs,
                metadata=pending.metadata,
                timestamp=pending.ended_at,
            )
        except Exception as error:
            if _OUT_OF_ORDER_CLOSE in str(error):
                # Routine while a scope opened later is still open, so not an error in
                # its own right; the close is retried as those scopes are popped.
                _logger.debug("NeMo Relay: scope.pop deferred", exc_info=True)
                return _CloseOutcome.DEFERRED
            _logger.error("NeMo Relay: scope.pop failed", exc_info=True)
            return _CloseOutcome.FAILED
        return _CloseOutcome.CLOSED
