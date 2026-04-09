# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope stack operations.

Scopes define the hierarchy that tool calls, LLM calls, and mark events attach
to. They are the main way to model agents, tasks, and nested units of work.

Example:
    ```python
    import nat_nexus

    with nat_nexus.scope.scope("demo-agent", nat_nexus.ScopeType.Agent) as handle:
        nat_nexus.scope.event("checkpoint", handle=handle, data={"step": 1})
    ```
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from nat_nexus import Json
from nat_nexus._native import (
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
)
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


def get_handle() -> ScopeHandle:
    """Return the current top-of-stack ``ScopeHandle``.

    Returns:
        ScopeHandle: The scope currently at the top of the active scope stack.

    Notes:
        If the current Python context does not yet have a scope stack, one is
        created automatically before the handle lookup.
    """
    _ensure_scope_stack()
    return _native_get_handle()


def push(
    name: str,
    scope_type: ScopeType,
    *,
    handle: ScopeHandle | None = None,
    attributes: ScopeAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> ScopeHandle:
    """Push a new child scope and return its handle.

    Args:
        name: Human-readable name for the new scope.
        scope_type: Semantic scope type, such as ``ScopeType.Agent`` or
            ``ScopeType.Function``.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        attributes: Optional native scope attributes attached to the emitted
            start event.
        data: Optional JSON payload recorded on the scope start event.
        metadata: Optional JSON metadata recorded on the scope start event.

    Returns:
        ScopeHandle: Handle for the newly pushed scope.

    Notes:
        A scope stack is created automatically if the current context does not
        yet have one.

    Example:
        ```python
        import nat_nexus

        with nat_nexus.scope.scope("parent", nat_nexus.ScopeType.Agent) as parent:
            handle = nat_nexus.scope.push(
                "worker",
                nat_nexus.ScopeType.Function,
                handle=parent,
                attributes=None,
                data={"step": 1},
                metadata={"source": "scope.push"},
            )
            nat_nexus.scope.pop(handle)
        ```
    """
    _ensure_scope_stack()
    return _native_push_scope(name, scope_type, handle=handle, attributes=attributes, data=data, metadata=metadata)


def pop(handle: ScopeHandle) -> None:
    """Pop a scope previously returned by ``push()`` or ``scope()``.

    Args:
        handle: Scope handle to close.

    Notes:
        The handle must correspond to an active scope in the current scope
        stack. Popping a scope also removes any scope-local registrations owned
        by that scope.
    """
    _ensure_scope_stack()
    _native_pop_scope(handle)


def event(
    name: str,
    *,
    handle: ScopeHandle | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> None:
    """Emit a ``Mark`` event under the current or provided scope.

    Args:
        name: Event name to emit.
        handle: Optional scope handle that should own the event. When omitted,
            the current top-of-stack scope is used.
        data: Optional JSON payload attached to the event.
        metadata: Optional JSON metadata attached to the event.
    """
    _ensure_scope_stack()
    _native_event(name, handle=handle, data=data, metadata=metadata)


@contextmanager
def scope(
    name: str,
    scope_type: ScopeType,
    *,
    handle: ScopeHandle | None = None,
    attributes: ScopeAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
) -> Iterator[ScopeHandle]:
    """Create a scope for the duration of a ``with`` block.

    Args:
        name: Human-readable name for the new scope.
        scope_type: Semantic scope type, such as ``ScopeType.Agent`` or
            ``ScopeType.Function``.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        attributes: Optional native scope attributes attached to the emitted
            start event.
        data: Optional JSON payload recorded on the scope start event.
        metadata: Optional JSON metadata recorded on the scope start event.

    Yields:
        ScopeHandle: Handle for the scope that remains active inside the
        ``with`` block.

    Notes:
        The scope is always popped when the ``with`` block exits, even if the
        body raises an exception.

    Example:
        ```python
        import nat_nexus

        with nat_nexus.scope.scope(
            "demo",
            nat_nexus.ScopeType.Agent,
            handle=None,
            attributes=None,
            data={"stage": "start"},
            metadata={"owner": "docs"},
        ) as handle:
            nat_nexus.scope.event("inside", handle=handle, data={"ok": True}, metadata={"step": 1})
        ```
    """
    _ensure_scope_stack()
    pushed_handle = None
    try:
        pushed_handle = _native_push_scope(
            name, scope_type, handle=handle, attributes=attributes, data=data, metadata=metadata
        )
        yield pushed_handle
    finally:
        if pushed_handle is not None:
            _native_pop_scope(pushed_handle)


__all__ = ["event", "get_handle", "pop", "push", "scope"]
