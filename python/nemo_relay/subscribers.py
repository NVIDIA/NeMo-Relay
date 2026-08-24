# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global event subscriber registration.

Subscribers observe all lifecycle events emitted by the current process,
including scope, tool, LLM, and mark events. They are typically used for
logging, metrics, tracing, and custom observability pipelines.

Example::

    import nemo_relay

    def log_event(event):
        print(f"{event.kind}: {event.name}")

    nemo_relay.subscribers.register("logger", log_event)
    try:
        with nemo_relay.scope.scope("demo", nemo_relay.ScopeType.Agent):
            nemo_relay.scope.event("started")
    finally:
        nemo_relay.subscribers.deregister("logger")
"""

import asyncio
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from nemo_relay._event_sanitizer_context import callback_active as _publication_callback_active
from nemo_relay._native import (
    deregister_subscriber as _native_deregister,
)
from nemo_relay._native import (
    flush_subscribers as _native_flush,
)
from nemo_relay._native import (
    register_subscriber as _native_register,
)
from nemo_relay._native import (
    subscriber_dispatcher_after_fork_child as _native_after_fork_child,
)
from nemo_relay._native import (
    subscriber_dispatcher_after_fork_parent as _native_after_fork_parent,
)
from nemo_relay._native import (
    subscriber_dispatcher_before_fork as _native_before_fork,
)

if TYPE_CHECKING:
    from nemo_relay import Event


def _finish_flush(completed: asyncio.Future[None], error: BaseException | None) -> None:
    if completed.done():
        return
    if error is None:
        completed.set_result(None)
    else:
        completed.set_exception(error)


class _FlushBridge:
    """Coalesce asynchronous flush barriers onto one native-wait thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = {}
        self._next_token = 0
        self._thread: threading.Thread | None = None

    def submit(
        self,
        loop: asyncio.AbstractEventLoop,
        completed: asyncio.Future[None],
    ) -> None:
        thread_to_start: threading.Thread | None = None
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._pending[token] = (loop, completed)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="nemo-relay-flush",
                    daemon=True,
                )
                thread_to_start = self._thread
            self._wake.set()
        completed.add_done_callback(lambda _future: self._discard(token))
        if thread_to_start is not None:
            thread_to_start.start()

    def _discard(self, token: int) -> None:
        with self._lock:
            self._pending.pop(token, None)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                batch = list(self._pending.values())
                self._pending.clear()
                self._wake.clear()
            if not batch:
                continue
            try:
                _native_flush()
            except BaseException as error:
                result = error
            else:
                result = None
            for loop, completed in batch:
                try:
                    loop.call_soon_threadsafe(_finish_flush, completed, result)
                except RuntimeError:
                    pass


_flush_bridge = _FlushBridge()


def _after_fork_child() -> None:
    global _flush_bridge
    _native_after_fork_child()
    # Never inspect the inherited bridge: its lock may have been held by a
    # thread that does not exist in the child.
    _flush_bridge = _FlushBridge()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_native_before_fork,
        after_in_parent=_native_after_fork_parent,
        after_in_child=_after_fork_child,
    )


def register(name: str, callback: "Callable[[Event], None]") -> None:
    """Register a global event subscriber.

    Args:
        name: Unique subscriber name.
        callback: Callable invoked as ``callback(event)`` for every emitted
            lifecycle event.

    Returns:
        None: This function returns after the subscriber is registered.

    Raises:
        RuntimeError: If a subscriber with the same name already exists.

    Example::

        import nemo_relay

        nemo_relay.subscribers.register("printer", lambda event: print(event.kind))
    """
    return _native_register(name, callback)


def deregister(name: str) -> bool:
    """Remove a previously registered global subscriber.

    Args:
        name: Subscriber name passed to ``register()``.

    Returns:
        ``True`` if a subscriber was removed, otherwise ``False``.

    Notes:
        Deregistering a subscriber affects only future event delivery. Events
        already emitted before removal carry a subscriber snapshot, so queued
        callbacks from that snapshot may still run.

    Example::

        import nemo_relay

        nemo_relay.subscribers.register("printer", lambda event: None)
        removed = nemo_relay.subscribers.deregister("printer")
        assert removed is True
    """
    return _native_deregister(name)


def flush() -> None:
    """Wait for queued callbacks and registered managed terminal publications.

    Native NeMo Relay event APIs enqueue subscriber callbacks and return without
    waiting for observer work. Use this barrier in tests and shutdown paths when
    captured subscriber output must be complete before continuing. The barrier
    also waits for tool, LLM, and guardrail-scope terminal events from managed
    work that started before the flush.

    Call this function outside subscribers, event sanitizers, conditional
    guardrails, and request or execution intercepts. A queued tool or LLM
    observability sanitizer may call it, but the call returns without waiting
    for its own publication. From an ``asyncio`` task, await
    :func:`flush_async` instead.

    Raises:
        RuntimeError: If called while an ``asyncio`` event loop is running on
            the current thread.

    Returns:
        None: All eligible queued work has completed before this returns.
    """
    if _publication_callback_active():
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "subscribers.flush() cannot block a running asyncio event loop; use 'await subscribers.flush_async()'"
        )
    return _native_flush()


async def flush_async() -> None:
    """Wait asynchronously for queued callbacks and managed terminal events.

    Use this barrier from an ``asyncio`` task. A process-local daemon bridge
    thread coalesces concurrent barriers and waits for the native dispatcher
    without blocking the Python event loop. Managed tool, LLM, and guardrail
    work registered before the flush is included through its terminal event.

    Returns:
        None: All eligible queued work has completed when the awaitable
        resolves.
    """
    if _publication_callback_active():
        return None
    loop = asyncio.get_running_loop()
    completed: asyncio.Future[None] = loop.create_future()
    _flush_bridge.submit(loop, completed)
    await completed


__all__ = ["deregister", "flush", "flush_async", "register"]
