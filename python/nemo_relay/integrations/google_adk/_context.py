# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share per-invocation scope state between the plugin, the recorded nodes, and the root run observer."""

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
    from collections.abc import Callable, Iterator

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


class InvocationScope(NamedTuple):
    name: str
    parent: nemo_relay.ScopeHandle
    input: JsonObject
    metadata: JsonObject
    started_at: datetime.datetime


class Invocation:
    """Own one ADK invocation's scope, and close the invocation's scopes in stack order.

    The invocation scope opens with its start time when a call first needs it. A scope closes once it reaches the top
    of the stack, and `finish` closes each owned scope that is still open as cancelled.
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
        self.driver: Annotated[
            asyncio.Task[object], "This field holds the task that runs the invocation's run hooks."
        ] = driver
        self.previous_parent = previous_parent
        self.previous_invocation = previous_invocation
        self.end: Annotated[
            Callable[[JsonObject, JsonObject], None],
            "This callable ends the invocation with an output and status, as its run hooks do.",
        ] = end
        self.abandon: Annotated[
            Callable[[], None], "This callable closes the invocation's scope as cancelled when a call opened it."
        ] = abandon
        self.recorded_nodes: Annotated[
            set[str], "This set holds the paths of the nodes that `wrap_node` recorded as scopes in the invocation."
        ] = set()
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
        """Return the invocation scope, and push it with its start time when no call has opened it yet.

        A push that raises leaves the invocation unrecorded and finished, so later calls take the invocation's parent
        and the invocation's end records nothing.
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
        """Return true for the first run that claims the root of the invocation, and false for every later one."""
        with self._lock:
            claimed, self._root_claimed = self._root_claimed, True
            return not claimed

    def interrupt(self) -> None:
        """End the invocation as cancelled when its driving task is being cancelled.

        ADK calls no run hook for a cancelled driving task. A run that stops for any other reason ends through ADK's run
        hooks.
        """
        if self.driver.cancelling():
            self.end({"status": "cancelled"}, CANCELLED_STATUS)

    def watch(self) -> None:
        """Interrupt the invocation when the current task, which runs a top-level run of it, ends cancelled.

        ADK's node runtime runs the root node and each agent transfer in a task of its own, and cancels that task before
        the driving task resumes, so the done callback still sees the driving task's cancellation. The driving task
        itself needs no watch.
        """
        task = current_task()
        if task is not self.driver:
            task.add_done_callback(self._run_task_done)

    def _run_task_done(self, task: asyncio.Task[object]) -> None:
        if task.cancelled():
            self.interrupt()

    def _driver_done(self, task: asyncio.Task[object]) -> None:
        """Abandon the invocation, which ended without ADK's run hooks when its driving task finishes before it."""
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
        """Close the invocation scope with the output and status, and close each owned scope left open as cancelled.

        Open the invocation scope first when `record` is true and no call opened it. Mark the invocation finished
        before closing, so a call that starts meanwhile no longer takes it as a parent.
        """
        self.finished = True
        self.driver.remove_done_callback(self._driver_done)
        if not self._unrecorded and (record or self._handle is not None):
            self.close(self.handle(), output=output, metadata=metadata)
        with self._lock:
            self._scope_context.run(lambda: self._drain_locked(force=True))

    def release(self) -> None:
        """Restore the context variables from before the invocation when the invocation is current in this context."""
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
        "This variable holds the innermost invocation, agent scope, or node scope in the current context.",
    ]
] = contextvars.ContextVar("nemo_relay_google_adk_parent", default=None)
CURRENT_INVOCATION: Final[
    Annotated[
        contextvars.ContextVar[Invocation | None],
        "This variable holds the innermost ADK invocation that the plugin observes in the current context.",
    ]
] = contextvars.ContextVar("nemo_relay_google_adk_invocation", default=None)


def parent_handle() -> nemo_relay.ScopeHandle:
    """Return the innermost open invocation, agent, or node scope in the current context.

    Open the invocation scope when a call first needs it, and skip invocations that already finished. Outside an
    invocation, return the top of the current scope stack.
    """
    parent = CURRENT_PARENT.get()
    while isinstance(parent, Invocation) and parent.finished:
        parent = parent.previous_parent
    if isinstance(parent, Invocation):
        return parent.handle()
    return nemo_relay.scope.get_handle() if parent is None else parent


def current_invocation() -> Invocation | None:
    """Return the innermost unfinished invocation in the current context, or None outside one."""
    invocation = CURRENT_INVOCATION.get()
    while invocation is not None and invocation.finished:
        invocation = invocation.previous_invocation
    return invocation


def current_task() -> asyncio.Task[object]:
    """Return the running asyncio task, and raise `RuntimeError` outside one."""
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("NeMo Relay records Google ADK runs only inside an asyncio task")
    return task
