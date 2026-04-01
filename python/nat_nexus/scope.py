# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope and handle operations.

The scope stack is a hierarchical structure that tracks execution context.
Each scope has a UUID, name, type, and optional attributes/data/metadata.

Functions:
    get_handle()
        Return the current (topmost) ``ScopeHandle`` from the task-local scope
        stack, or raise ``RuntimeError`` if the stack is empty.

    push(name, scope_type, *, handle=None, attributes=None, data=None, metadata=None)
        Push a new child scope. If *handle* is omitted, the scope is parented
        to the current top of stack. Returns the new ``ScopeHandle``.

    pop(handle)
        Remove a scope from the stack and emit an ``End`` event.

    event(name, *, handle=None, data=None, metadata=None)
        Emit a ``Mark`` event under the current or specified scope.

    scope(name, scope_type, *, handle=None, attributes=None, data=None, metadata=None)
        Context manager that pushes a new child scope and ensures it is popped at the end.
Example::

    import nat_nexus

    # Using the scope context manager (recommended)
    with nat_nexus.scope.scope("my-agent", nat_nexus.ScopeType.Agent) as handle:
        nat_nexus.scope.event("checkpoint", data={"step": 1})

    # Manual push/pop
    handle = nat_nexus.scope.push("my-agent", nat_nexus.ScopeType.Agent)
    nat_nexus.scope.event("checkpoint", data={"step": 1})
    nat_nexus.scope.pop(handle)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from nat_nexus._native import (
    nat_nexus_event as _native_event,
)
from nat_nexus._native import (
    nat_nexus_get_handle as _native_get_handle,
)
from nat_nexus._native import (
    nat_nexus_pop_scope as _native_pop_scope,
)
from nat_nexus._native import (
    nat_nexus_push_scope as _native_push_scope,
)


def _ensure_scope_stack() -> None:
    """Ensure the current context has a scope stack available.

    If the Rust-side thread-local was explicitly set via
    ``set_thread_scope_stack()`` (e.g. by a worker thread), this is a
    no-op — the Rust thread-local is already correct.

    Otherwise, calls ``get_scope_stack()`` which creates a scope stack if
    needed (via the ContextVar) and syncs it to the Rust thread-local.
    """
    import nat_nexus

    if nat_nexus._native_scope_stack_active():
        return
    nat_nexus.get_scope_stack()


def get_handle() -> Any:
    """Return the current (topmost) ScopeHandle from the scope stack.

    Returns:
        The current ``ScopeHandle`` at the top of the scope stack.

    Raises:
        RuntimeError: If the scope stack is empty.
    """
    _ensure_scope_stack()
    return _native_get_handle()


def push(
    name: str, scope_type: Any, *, handle: Any = None, attributes: Any = None, data: Any = None, metadata: Any = None
) -> Any:
    """Push a new child scope onto the scope stack.

    If *handle* is omitted, the scope is parented to the current top of stack.

    Args:
        name: Human-readable scope name.
        scope_type: The kind of scope (e.g. ``ScopeType.Agent``).
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``ScopeAttributes`` bitflags.
        data: Optional JSON-serializable application data to attach to the scope.
        metadata: Optional JSON-serializable metadata to attach to the scope.

    Returns:
        The newly created ``ScopeHandle``.
    """
    _ensure_scope_stack()
    return _native_push_scope(name, scope_type, handle=handle, attributes=attributes, data=data, metadata=metadata)


def pop(handle: Any) -> None:
    """Remove a scope from the stack and emit an ``End`` event.

    Args:
        handle: The ``ScopeHandle`` returned by ``push()`` or ``scope()``.
    """
    _ensure_scope_stack()
    _native_pop_scope(handle)


def event(name: str, *, handle: Any = None, data: Any = None, metadata: Any = None) -> None:
    """Emit a ``Mark`` event under the current or specified scope.

    Args:
        name: Event name.
        handle: Optional parent scope handle. Defaults to the current top of stack.
        data: Optional JSON-serializable application data.
        metadata: Optional JSON-serializable metadata.
    """
    _ensure_scope_stack()
    _native_event(name, handle=handle, data=data, metadata=metadata)


@contextmanager
def scope(
    name: str, scope_type: Any, *, handle: Any = None, attributes: Any = None, data: Any = None, metadata: Any = None
) -> Any:
    """Context manager that pushes a new child scope and pops it on exit.

    If *handle* is omitted, the scope is parented to the current top of stack.
    The scope is automatically popped when the context manager exits, even if
    an exception is raised.

    Args:
        name: Human-readable scope name.
        scope_type: The kind of scope (e.g. ``ScopeType.Agent``).
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``ScopeAttributes`` bitflags.
        data: Optional JSON-serializable application data to attach to the scope.
        metadata: Optional JSON-serializable metadata to attach to the scope.

    Yields:
        The newly created ``ScopeHandle``.
    """
    _ensure_scope_stack()
    try:
        pushed_handle = _native_push_scope(
            name, scope_type, handle=handle, attributes=attributes, data=data, metadata=metadata
        )
        yield pushed_handle
    finally:
        _native_pop_scope(pushed_handle)


__all__ = ["event", "get_handle", "pop", "push", "scope"]
