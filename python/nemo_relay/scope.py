# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope stack operations.

Scopes define the hierarchy that tool calls, LLM calls, and mark events attach
to. They are the main way to model agents, tasks, and nested units of work.

Example::

    import nemo_relay

    with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent) as handle:
        nemo_relay.scope.event("checkpoint", handle=handle, data={"step": 1})
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from nemo_relay import Json
from nemo_relay._context import ensure_scope_stack as _ensure_scope_stack
from nemo_relay._native import (
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
)
from nemo_relay._native import (
    event as _native_event,
)
from nemo_relay._native import (
    get_handle as _native_get_handle,
)
from nemo_relay._native import (
    pop_scope as _native_pop_scope,
)
from nemo_relay._native import (
    push_scope as _native_push_scope,
)


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
    input: Json | None = None,
    timestamp: datetime | None = None,
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
        data: Optional JSON application payload stored on the scope handle.
        metadata: Optional JSON metadata recorded on the scope start event.
        input: Optional JSON payload exported as the semantic scope input.
        timestamp: Optional timezone-aware ``datetime`` recorded as the handle
            start time and on the scope start event. When omitted, the current
            runtime time is used.

    Returns:
        ScopeHandle: Handle for the newly pushed scope.

    Notes:
        A scope stack is created automatically if the current context does not
        yet have one. ``timestamp`` must be a timezone-aware ``datetime``;
        strings and naive datetimes are rejected.

    Example::

        import nemo_relay

        with nemo_relay.scope.scope("parent", nemo_relay.ScopeType.Agent) as parent:
            handle = nemo_relay.scope.push(
                "worker",
                nemo_relay.ScopeType.Function,
                handle=parent,
                attributes=None,
                data={"step": 1},
                metadata={"source": "scope.push"},
            )
            nemo_relay.scope.pop(handle)
    """
    _ensure_scope_stack()
    return _native_push_scope(
        name,
        scope_type,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        input=input,
        timestamp=timestamp,
    )


def pop(
    handle: ScopeHandle, *, output: Json | None = None, metadata: Json | None = None, timestamp: datetime | None = None
) -> None:
    """Pop a scope previously returned by ``push()`` or ``scope()``.

    Args:
        handle: Scope handle to close.
        output: Optional JSON payload exported as the semantic scope output.
        metadata: Optional JSON metadata to append to the metadata set when the scope was created.
        timestamp: Optional timezone-aware ``datetime`` recorded on the scope
            end event. When omitted, the runtime default end timestamp is used.

    Returns:
        None: This function returns after the scope is closed successfully.

    Notes:
        The handle must correspond to an active scope in the current scope
        stack. Popping a scope also removes any scope-local registrations owned
        by that scope. ``timestamp`` must be a timezone-aware ``datetime``;
        strings and naive datetimes are rejected.
    """
    _ensure_scope_stack()
    _native_pop_scope(handle, output=output, metadata=metadata, timestamp=timestamp)


def event(
    name: str,
    *,
    handle: ScopeHandle | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Emit a ``Mark`` event under the current or provided scope.

    Args:
        name: Event name to emit.
        handle: Optional scope handle that should own the event. When omitted,
            the current top-of-stack scope is used.
        data: Optional JSON payload attached to the event.
        metadata: Optional JSON metadata attached to the event.
        timestamp: Optional timezone-aware ``datetime`` recorded on the mark
            event. When omitted, the current runtime time is used.

    Returns:
        None: This function returns after the event has been emitted.

    Notes:
        A scope stack is created automatically when needed before the event is
        emitted through the native runtime. ``timestamp`` must be a
        timezone-aware ``datetime``; strings and naive datetimes are rejected.
    """
    _ensure_scope_stack()
    _native_event(name, handle=handle, data=data, metadata=metadata, timestamp=timestamp)


@contextmanager
def scope(
    name: str,
    scope_type: ScopeType,
    *,
    handle: ScopeHandle | None = None,
    attributes: ScopeAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    input: Json | None = None,
    timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
) -> Iterator[ScopeHandle]:
    """Create a scope for the duration of a ``with`` block.

    OTEL status codes will be automatically recorded in the scope's metadata.

    Args:
        name: Human-readable name for the new scope.
        scope_type: Semantic scope type, such as ``ScopeType.Agent`` or
            ``ScopeType.Function``.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        attributes: Optional native scope attributes attached to the emitted
            start event.
        data: Optional JSON application payload stored on the scope handle.
        metadata: Optional JSON metadata recorded on the scope start event.
        input: Optional JSON payload exported as the semantic scope input.
        timestamp: Optional timezone-aware ``datetime`` recorded as the handle
            start time and on the scope start event.
        end_timestamp: Optional timezone-aware ``datetime`` recorded on the
            scope end event.

    Yields:
        ScopeHandle: Handle for the scope that remains active inside the
        ``with`` block.

    Notes:
        The scope is always popped when the ``with`` block exits, even if the
        body raises an exception. Timestamp arguments must be timezone-aware
        ``datetime`` objects; strings and naive datetimes are rejected.

    Example::

        import nemo_relay

        with nemo_relay.scope.scope(
            "demo",
            nemo_relay.ScopeType.Agent,
            handle=None,
            attributes=None,
            data={"stage": "start"},
            metadata={"owner": "docs"},
        ) as handle:
            nemo_relay.scope.event("inside", handle=handle, data={"ok": True}, metadata={"step": 1})
    """
    _ensure_scope_stack()
    pushed_handle = None
    status_code = "UNSET"
    status_message = None
    try:
        pushed_handle = _native_push_scope(
            name,
            scope_type,
            handle=handle,
            attributes=attributes,
            data=data,
            metadata=metadata,
            input=input,
            timestamp=timestamp,
        )
        yield pushed_handle
        status_code = "OK"
    except Exception as e:
        status_code = "ERROR"
        status_message = str(e)
        raise
    finally:
        if pushed_handle is not None:
            metadata = {"otel.status_code": status_code}
            if status_message is not None:
                metadata["otel.status_description"] = status_message
            pop(pushed_handle, metadata=metadata, timestamp=end_timestamp)


__all__ = ["event", "get_handle", "pop", "push", "scope"]
