# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global event subscriber registration.

Subscribers observe all lifecycle events emitted by the current process,
including scope, tool, LLM, and mark events. They are typically used for
logging, metrics, tracing, and custom observability pipelines.

Example:
    ```python
    import nemo_flow

    def log_event(event):
        print(f"{event.kind}: {event.name}")

    nemo_flow.subscribers.register("logger", log_event)
    try:
        with nemo_flow.scope.scope("demo", nemo_flow.ScopeType.Agent):
            nemo_flow.scope.event("started")
    finally:
        nemo_flow.subscribers.deregister("logger")
    ```
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from nemo_flow._native import (
    nemo_flow_deregister_subscriber as _native_deregister,
)
from nemo_flow._native import (
    nemo_flow_register_subscriber as _native_register,
)

if TYPE_CHECKING:
    from nemo_flow import Event


def register(name: str, callback: "Callable[[Event], None]") -> None:
    """Register a global event subscriber.

    Args:
        name: Unique subscriber name.
        callback: Callable invoked as ``callback(event)`` for every emitted
            lifecycle event.

    Raises:
        RuntimeError: If a subscriber with the same name already exists.

    Example:
        ```python
        import nemo_flow

        nemo_flow.subscribers.register("printer", lambda event: print(event.kind))
        ```
    """
    return _native_register(name, callback)


def deregister(name: str) -> bool:
    """Remove a previously registered global subscriber.

    Args:
        name: Subscriber name passed to ``register()``.

    Returns:
        ``True`` if a subscriber was removed, otherwise ``False``.

    Example:
        ```python
        import nemo_flow

        nemo_flow.subscribers.register("printer", lambda event: None)
        removed = nemo_flow.subscribers.deregister("printer")
        assert removed is True
        ```
    """
    return _native_deregister(name)


__all__ = ["register", "deregister"]
