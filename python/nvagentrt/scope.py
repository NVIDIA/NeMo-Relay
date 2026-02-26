"""Scope and handle operations.

The scope stack is a hierarchical structure that tracks execution context.
Each scope has a UUID, name, type, and optional attributes/data/metadata.

Functions:
    get_handle()
        Return the current (topmost) ``ScopeHandle`` from the task-local scope
        stack, or raise ``RuntimeError`` if the stack is empty.

    push(name, scope_type, *, handle=None, attributes=None)
        Push a new child scope. If *handle* is omitted, the scope is parented
        to the current top of stack. Returns the new ``ScopeHandle``.

    pop(handle)
        Remove a scope from the stack and emit an ``End`` event.

    event(name, *, handle=None, data=None, metadata=None)
        Emit a ``Mark`` event under the current or specified scope.

Example::

    import nvagentrt

    handle = nvagentrt.scope.push("my-agent", nvagentrt.ScopeType.Agent)
    nvagentrt.scope.event("checkpoint", data={"step": 1})
    nvagentrt.scope.pop(handle)
"""

from nvagentrt._native import (
    nv_agentrt_event as event,
)
from nvagentrt._native import (
    nv_agentrt_get_handle as get_handle,
)
from nvagentrt._native import (
    nv_agentrt_pop_scope as pop,
)
from nvagentrt._native import (
    nv_agentrt_push_scope as push,
)

__all__ = ["get_handle", "push", "pop", "event"]
