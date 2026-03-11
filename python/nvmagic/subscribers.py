# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event subscriber registration.

Subscribers receive lifecycle events (scope start/end, tool start/end,
LLM start/end, marks) for observability, logging, or tracing.

Functions:
    register(name, callback)
        Register a subscriber. ``callback`` signature:
        ``(event: Event) -> None``. Raises ``RuntimeError`` if a subscriber
        with this name already exists.

    deregister(name)
        Remove a subscriber. Returns ``True`` if found and removed.

Example::

    import nvmagic

    def log_events(event):
        print(f"[{event.event_type}] {event.name} ({event.uuid})")

    nvmagic.subscribers.register("logger", log_events)
"""

from nvmagic._native import (
    nvmagic_deregister_subscriber as deregister,
)
from nvmagic._native import (
    nvmagic_register_subscriber as register,
)

__all__ = ["register", "deregister"]
