# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event subscriber registration.

Subscribers receive lifecycle events (scope start/end, tool start/end,
LLM start/end, marks) for observability, logging, or tracing.

Functions:
    register(name, callback)
        Register a subscriber. ``callback`` signature:
        ``(event: ScopeStartEvent | ScopeEndEvent | ToolStartEvent |``
        `` ToolEndEvent | LLMStartEvent | LLMEndEvent | MarkEvent) -> None``.
        Raises ``RuntimeError`` if a subscriber with this name already exists.

    deregister(name)
        Remove a subscriber. Returns ``True`` if found and removed.

Example::

    import nat_nexus

    def log_events(event):
        print(f"[{event.kind}] {event.name} ({event.uuid})")

    nat_nexus.subscribers.register("logger", log_events)
"""

from nat_nexus._native import (
    nat_nexus_deregister_subscriber as deregister,
)
from nat_nexus._native import (
    nat_nexus_register_subscriber as register,
)

__all__ = ["register", "deregister"]
