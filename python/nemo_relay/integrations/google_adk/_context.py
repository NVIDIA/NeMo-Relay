# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-invocation scope state shared by plugin, recorded nodes, and root run observer."""

from __future__ import annotations

import asyncio
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
    from collections.abc import Callable, Generator

    from nemo_relay._native import _ThreadScopeStackBinding

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


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
def rejection_marked(name: _RejectionName, parent: nemo_relay.ScopeHandle, **call: Json) -> Generator[None, None, None]:
    """Wrap conversion of a managed call's value between ADK and Relay.

    On error, emit mark `name` at error severity under `parent`, with call fields plus error type and message, then
    re-raise.
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
    """Context and thread binding that own an invocation's scope stack."""

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


class InvocationScope(NamedTuple):
    name: str
    parent: nemo_relay.ScopeHandle
    input: JsonObject
    metadata: JsonObject
    started_at: datetime.datetime


class Invocation:
    """Own one ADK invocation's scope, and close its scopes in stack order.

    Invocation scope opens, with its start time, when a call first needs it. A scope closes once it reaches top of
    stack. `finish` closes each owned scope still open as cancelled.
    """

    def __init__(
        self,
        invocation_id: str,
        *,
        scope: InvocationScope,
        driver: asyncio.Task[object],
        previous_parent: _Parent,
        previous_invocation: Invocation | None,
        end: Callable[[JsonObject, JsonObject], None],
        abandon: Callable[[], None],
    ) -> None:
        self.invocation_id = invocation_id
        self.driver: Annotated[asyncio.Task[object], "Task running invocation's run hooks."] = driver
        self.previous_parent = previous_parent
        self.previous_invocation = previous_invocation
        self.end: Annotated[
            Callable[[JsonObject, JsonObject], None],
            "End invocation with output and status, as its run hooks do.",
        ] = end
        self.abandon: Annotated[Callable[[], None], "Close invocation scope as cancelled when a call opened it."] = (
            abandon
        )
        self.recorded_nodes: Annotated[set[str], "Paths of nodes `wrap_node` recorded as scopes in invocation."] = set()
        self.finished = False
        self._unrecorded = False
        self._scope = scope
        self._scope_context = ScopeContext.capture()
        self._lock = threading.Lock()
        self._handle: nemo_relay.ScopeHandle | None = None
        self._root_claimed = False
        self._pending: dict[str, _PendingClose] = {}
        self._owned: set[str] = set()
        driver.add_done_callback(self._driver_done)

    def handle(self) -> nemo_relay.ScopeHandle:
        """Return invocation scope, pushing it with its start time when no call opened it yet.

        A failed push leaves invocation unrecorded and finished. Later calls then take invocation's parent, and
        invocation's end records nothing.
        """
        with self._lock:
            if self._handle is None:
                scope = self._scope
                try:
                    self._handle = self._scope_context.run(
                        lambda: nemo_relay.scope.push(
                            scope.name,
                            nemo_relay.ScopeType.Agent,
                            handle=scope.parent,
                            input=scope.input,
                            metadata=scope.metadata,
                            timestamp=scope.started_at,
                        )
                    )
                except Exception:
                    self.finished = self._unrecorded = True
                    raise
                self._owned.add(self._handle.uuid)
            return self._handle

    def claim_root(self) -> bool:
        """Return true for first run claiming invocation's root, false for every later one."""
        with self._lock:
            claimed, self._root_claimed = self._root_claimed, True
            return not claimed

    def interrupt(self) -> None:
        """End invocation as cancelled when its driving task is being cancelled.

        ADK calls no run hook for a cancelled driving task. A run stopping for any other reason ends through ADK's run
        hooks.
        """
        if self.driver.cancelling():
            self.end({"status": "cancelled"}, CANCELLED_STATUS)

    def watch(self) -> None:
        """Interrupt invocation when current task, running a top-level run of it, ends cancelled.

        ADK's node runtime runs root node and each agent transfer in its own task, and cancels that task before driving
        task resumes. Done callback therefore still sees driving task's cancellation. Driving task itself needs no
        watch.
        """
        task = current_task()
        if task is not self.driver:
            task.add_done_callback(self._run_task_done)

    def _run_task_done(self, task: asyncio.Task[object]) -> None:
        if task.cancelled():
            self.interrupt()

    def _driver_done(self, task: asyncio.Task[object]) -> None:
        """Abandon invocation when driving task finishes.

        An invocation still open then ended without ADK's run hooks.
        """
        del task
        self.abandon()

    def own(self, scope_uuid: str) -> None:
        with self._lock:
            self._owned.add(scope_uuid)

    def close(self, handle: nemo_relay.ScopeHandle, *, output: JsonObject, metadata: JsonObject) -> None:
        with self._lock:
            self._pending[handle.uuid] = _PendingClose(handle, output, metadata, datetime.datetime.now(datetime.UTC))
            self._scope_context.run(lambda: self._drain_locked(force=False))

    def finish(self, *, output: JsonObject, metadata: JsonObject, record: bool) -> None:
        """Close invocation scope with output and status, and close each owned scope left open as cancelled.

        Open invocation scope first when `record` is true and no call opened it. Mark invocation finished before
        closing, so a call starting meanwhile no longer takes it as parent.
        """
        self.finished = True
        self.driver.remove_done_callback(self._driver_done)
        if not self._unrecorded and (record or self._handle is not None):
            self.close(self.handle(), output=output, metadata=metadata)
        with self._lock:
            self._scope_context.run(lambda: self._drain_locked(force=True))

    def release(self) -> None:
        """Restore pre-invocation context variables when invocation is current in this context."""
        if CURRENT_INVOCATION.get() is self:
            CURRENT_PARENT.set(self.previous_parent)
            CURRENT_INVOCATION.set(self.previous_invocation)

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


_Parent: TypeAlias = "nemo_relay.ScopeHandle | Invocation | None"

CURRENT_PARENT: Final[
    Annotated[
        contextvars.ContextVar[_Parent],
        "Innermost invocation, agent scope, or node scope in current context.",
    ]
] = contextvars.ContextVar("nemo_relay_google_adk_parent", default=None)
CURRENT_INVOCATION: Final[
    Annotated[
        contextvars.ContextVar[Invocation | None],
        "Innermost ADK invocation plugin observes in current context.",
    ]
] = contextvars.ContextVar("nemo_relay_google_adk_invocation", default=None)


def parent_handle() -> nemo_relay.ScopeHandle:
    """Return innermost open invocation, agent, or node scope in current context.

    Open invocation scope when a call first needs it. Skip finished invocations. Outside an invocation, return top of
    current scope stack.
    """
    parent = CURRENT_PARENT.get()
    while isinstance(parent, Invocation) and parent.finished:
        parent = parent.previous_parent
    if isinstance(parent, Invocation):
        return parent.handle()
    return nemo_relay.scope.get_handle() if parent is None else parent


def current_invocation() -> Invocation | None:
    """Return innermost unfinished invocation in current context, or None outside one."""
    invocation = CURRENT_INVOCATION.get()
    while invocation is not None and invocation.finished:
        invocation = invocation.previous_invocation
    return invocation


def current_task() -> asyncio.Task[object]:
    """Return running asyncio task. Raise `RuntimeError` outside one."""
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("NeMo Relay records Google ADK runs only inside an asyncio task")
    return task
