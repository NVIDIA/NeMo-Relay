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

from nemo_relay._event_sanitizer_context import callback_active as _event_sanitizer_callback_active
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

if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_native_before_fork,
        after_in_parent=_native_after_fork_parent,
        after_in_child=_native_after_fork_child,
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
    """Wait for subscriber callbacks already queued by native event emission.

    Native NeMo Relay event APIs enqueue subscriber callbacks and return without
    waiting for observer work. Use this barrier in tests and shutdown paths when
    captured subscriber output must be complete before continuing.

    Call this function outside subscriber and queued publication sanitizer
    callbacks. A re-entrant call returns without waiting to avoid blocking the
    dispatcher. Publication middleware must not move such a call to an unmarked
    worker thread. From an ``asyncio`` task, await :func:`flush_async` instead.

    Raises:
        RuntimeError: If called while an ``asyncio`` event loop is running on
            the current thread.
    """
    if _event_sanitizer_callback_active():
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
    """Wait asynchronously for subscriber callbacks already queued by Relay.

    Use this barrier from an ``asyncio`` task. A daemon bridge thread waits for
    the native dispatcher without blocking the Python event loop or process
    shutdown when this coroutine is cancelled.
    """
    if _event_sanitizer_callback_active():
        return None
    loop = asyncio.get_running_loop()
    completed: asyncio.Future[None] = loop.create_future()

    def finish(error: BaseException | None) -> None:
        if completed.done():
            return
        if error is None:
            completed.set_result(None)
        else:
            completed.set_exception(error)

    def wait_for_dispatcher() -> None:
        try:
            _native_flush()
        except BaseException as error:
            result = error
        else:
            result = None
        try:
            loop.call_soon_threadsafe(finish, result)
        except RuntimeError:
            pass

    threading.Thread(
        target=wait_for_dispatcher,
        name="nemo-relay-flush",
        daemon=True,
    ).start()
    await completed


__all__ = ["deregister", "flush", "flush_async", "register"]
