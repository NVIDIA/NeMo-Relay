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

from collections.abc import Callable
from typing import TYPE_CHECKING

from nemo_relay._native import (
    deregister_subscriber as _native_deregister,
)
from nemo_relay._native import (
    flush_subscribers as _native_flush,
)
from nemo_relay._native import (
    register_subscriber as _native_register,
)

if TYPE_CHECKING:
    from nemo_relay import Event


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
    """
    return _native_flush()


__all__ = ["deregister", "flush", "register"]
