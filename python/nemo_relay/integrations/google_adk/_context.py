# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share per-invocation scope state between the plugin and the recorded nodes."""

from __future__ import annotations

import contextlib
import contextvars
import datetime
import functools
import logging
import threading
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple, TypeAlias, TypeVar

import nemo_relay
from nemo_relay import Json, JsonObject
from nemo_relay._native import capture_thread_scope_stack, restore_thread_scope_stack

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from nemo_relay._native import _ThreadScopeStackBinding

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")

CURRENT_AGENT_HANDLE: Final[
    Annotated[
        contextvars.ContextVar[nemo_relay.ScopeHandle | None],
        "This variable holds the innermost invocation, agent, or node scope open in the current context.",
    ]
] = contextvars.ContextVar("nemo_relay_google_adk_agent_handle", default=None)


def parent_handle() -> nemo_relay.ScopeHandle:
    """Return the scope in `CURRENT_AGENT_HANDLE`, or the top of the current scope stack when that variable is None."""
    handle = CURRENT_AGENT_HANDLE.get()
    if handle is not None:
        return handle
    return nemo_relay.scope.get_handle()


def integration_metadata(**extra: Json) -> JsonObject:
    return {"integration": "google-adk", **extra}


OK_STATUS: Final[JsonObject] = {"otel.status_code": "OK"}


def error_status(description: str) -> JsonObject:
    return {"otel.status_code": "ERROR", "otel.status_description": description}


CANCELLED_STATUS: Final = error_status("invocation cancelled")


_RejectionName: TypeAlias = Literal[
    "adk.tool_arguments_rejected", "adk.tool_result_rejected", "adk.llm_request_rejected", "adk.llm_response_rejected"
]


@contextlib.contextmanager
def rejection_marked(name: _RejectionName, parent: nemo_relay.ScopeHandle, **call: Json) -> Iterator[None]:
    """Wrap a conversion of a managed call's value between ADK and Relay.

    When the conversion raises, emit the named mark at error severity under `parent` with the call fields and the
    error's type and message, and then re-raise the error.
    """
    try:
        yield
    except Exception as error:
        nemo_relay.scope.event(
            name,
            handle=parent,
            data={**call, "error_type": type(error).__name__, "error_message": str(error)},
            metadata=integration_metadata(),
            severity=nemo_relay.LogSeverity.Error,
        )
        raise


@functools.cache
def log_unrecorded_once(kind: str) -> None:
    _logger.debug(
        "NeMo Relay: a %s ran outside an ADK invocation observed by NemoRelayPlugin and is not recorded as a scope",
        kind,
    )


class ScopeContext(NamedTuple):
    """Hold the context and the thread binding that own an invocation's scope stack."""

    context: contextvars.Context
    binding: _ThreadScopeStackBinding

    @classmethod
    def capture(cls) -> ScopeContext:
        return cls(contextvars.copy_context(), capture_thread_scope_stack())

    def run(self, callback: Callable[[], _T]) -> _T:
        previous = capture_thread_scope_stack()
        try:
            restore_thread_scope_stack(self.binding)
            return self.context.run(callback)
        finally:
            restore_thread_scope_stack(previous)


class _PendingClose(NamedTuple):
    handle: nemo_relay.ScopeHandle
    output: JsonObject
    metadata: JsonObject
    ended_at: datetime.datetime


class DeferredCloser:
    """Close an invocation's scopes in stack order.

    A scope closes once it reaches the top of the stack, and `drain` closes each owned scope that is still open as
    cancelled.
    """

    def __init__(self, scope_context: ScopeContext, invocation_id: str) -> None:
        self.invocation_id = invocation_id
        self.recorded_nodes: Annotated[
            set[str], "This set holds the paths of the nodes that `wrap_node` recorded as scopes in the invocation."
        ] = set()
        self._scope_context = scope_context
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingClose] = {}
        self._owned: set[str] = set()

    def own(self, scope_uuid: str) -> None:
        with self._lock:
            self._owned.add(scope_uuid)

    def close(self, handle: nemo_relay.ScopeHandle, *, output: JsonObject, metadata: JsonObject) -> None:
        with self._lock:
            self._pending[handle.uuid] = _PendingClose(handle, output, metadata, datetime.datetime.now(datetime.UTC))
            self._scope_context.run(lambda: self._drain_locked(force=False))

    def drain(self) -> None:
        with self._lock:
            self._scope_context.run(lambda: self._drain_locked(force=True))

    def _drain_locked(self, *, force: bool) -> None:
        while self._pending or force:
            top = nemo_relay.scope.get_handle()
            entry = self._pending.get(top.uuid)
            if entry is None:
                if not force or top.uuid not in self._owned:
                    return
                nemo_relay.scope.pop(
                    top,
                    output={"status": "cancelled"},
                    metadata=CANCELLED_STATUS,
                )
                self._owned.discard(top.uuid)
                continue
            nemo_relay.scope.pop(entry.handle, output=entry.output, metadata=entry.metadata, timestamp=entry.ended_at)
            del self._pending[top.uuid]
            self._owned.discard(top.uuid)


CURRENT_CLOSER: Final[
    Annotated[
        contextvars.ContextVar[DeferredCloser | None],
        "This variable holds the closer of the ADK invocation that runs in the current context.",
    ]
] = contextvars.ContextVar("nemo_relay_google_adk_closer", default=None)
