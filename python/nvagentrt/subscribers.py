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

    import nvagentrt

    def log_events(event):
        print(f"[{event.event_type}] {event.name} ({event.uuid})")

    nvagentrt.subscribers.register("logger", log_events)
"""

from nvagentrt._native import (
    nv_agentrt_deregister_subscriber as deregister,
)
from nvagentrt._native import (
    nv_agentrt_register_subscriber as register,
)

__all__ = ["register", "deregister"]
