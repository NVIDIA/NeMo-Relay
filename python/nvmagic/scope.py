# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    import nvmagic

    handle = nvmagic.scope.push("my-agent", nvmagic.ScopeType.Agent)
    nvmagic.scope.event("checkpoint", data={"step": 1})
    nvmagic.scope.pop(handle)
"""

from __future__ import annotations

from typing import Any

from nvmagic._native import (
    nvmagic_event as _native_event,
)
from nvmagic._native import (
    nvmagic_get_handle as _native_get_handle,
)
from nvmagic._native import (
    nvmagic_pop_scope as _native_pop_scope,
)
from nvmagic._native import (
    nvmagic_push_scope as _native_push_scope,
)


def _ensure_scope_stack() -> None:
    """Ensure the Python-side scope stack contextvar is initialized and synced
    with the Rust thread-local.

    This must be called before any scope operation so that:
    1. The LangChain bridges see ``_scope_stack_var`` as set (``available()``).
    2. The Rust thread-local ``THREAD_SCOPE_STACK`` matches what Python sees.
    """
    import nvmagic

    nvmagic.get_scope_stack()


def get_handle() -> Any:
    """Return the current (topmost) ScopeHandle from the scope stack."""
    _ensure_scope_stack()
    return _native_get_handle()


def push(name: str, scope_type: Any, *, handle: Any = None, attributes: Any = None) -> Any:
    """Push a new child scope onto the scope stack.

    If *handle* is omitted, the scope is parented to the current top of stack.
    Returns the new ``ScopeHandle``.
    """
    _ensure_scope_stack()
    return _native_push_scope(name, scope_type, handle=handle, attributes=attributes)


def pop(handle: Any) -> None:
    """Remove a scope from the stack and emit an ``End`` event."""
    _ensure_scope_stack()
    _native_pop_scope(handle)


def event(name: str, *, handle: Any = None, data: Any = None, metadata: Any = None) -> None:
    """Emit a ``Mark`` event under the current or specified scope."""
    _ensure_scope_stack()
    _native_event(name, handle=handle, data=data, metadata=metadata)


__all__ = ["get_handle", "push", "pop", "event"]
