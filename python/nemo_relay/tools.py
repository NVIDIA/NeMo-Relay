# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool lifecycle helpers.

Use this module when you want NeMo Relay to emit tool start and end events around a
piece of application logic.

``execute()`` is the usual entry point and runs the full middleware pipeline.
``call()`` and ``call_end()`` are the lower-level manual lifecycle APIs.

Example::

    import nemo_relay

    async def search(args):
        return nemo_relay.ToolExecutionResult({"result": args["query"].upper()})

    result = await nemo_relay.tools.execute("search", {"query": "hello"}, search)
    assert result.result == {"result": "HELLO"}
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from nemo_relay import Json
from nemo_relay._context import ensure_scope_stack
from nemo_relay._native import (
    ScopeHandle,
    ToolAttributes,
    ToolExecutionResult,
    ToolHandle,
)
from nemo_relay._native import (
    tool_call as _native_tool_call,
)
from nemo_relay._native import (
    tool_call_end as _native_tool_call_end,
)
from nemo_relay._native import (
    tool_call_execute as _native_tool_call_execute,
)
from nemo_relay._native import (
    tool_conditional_execution as _native_tool_conditional_execution,
)
from nemo_relay._native import (
    tool_request_intercepts as _native_tool_request_intercepts,
)


def call(
    name: str,
    args: Json,
    *,
    handle: ScopeHandle | None = None,
    attributes: ToolAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    tool_call_id: str | None = None,
    timestamp: datetime | None = None,
) -> ToolHandle:
    """Start a manual tool span and return its ``ToolHandle``.

    Args:
        name: Tool name recorded on emitted lifecycle events.
        args: JSON-compatible tool arguments to associate with the call.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native tool attributes attached to the start event.
        data: Optional JSON application payload stored on the tool handle.
        metadata: Optional JSON metadata recorded on the emitted start event.
        tool_call_id: Optional provider-specific tool call identifier to attach
            to the emitted events.
        timestamp: Optional timezone-aware ``datetime`` recorded as the handle
            start time and on the emitted start event. When omitted, the current
            runtime time is used.

    Returns:
        ToolHandle: Handle used to finish the manual span with ``call_end()``.

    Notes:
        This starts only the manual tool lifecycle span. It applies
        sanitize-request guardrails to the emitted start-event payload but does
        not run request or execution intercepts. ``timestamp`` must be a
        timezone-aware ``datetime``; strings and naive datetimes are rejected.

    Example::

        import nemo_relay

        handle = nemo_relay.tools.call(
            "search",
            {"query": "hello"},
            handle=None,
            attributes=None,
            data={"attempt": 1},
            metadata={"path": "manual"},
            tool_call_id="tool-call-1",
        )
        nemo_relay.tools.call_end(
            handle,
            nemo_relay.ToolExecutionResult({"result": "ok"}),
            data={"cached": False},
            metadata={"status": "success"},
        )
    """
    ensure_scope_stack()
    return _native_tool_call(
        name,
        args,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        tool_call_id=tool_call_id,
        timestamp=timestamp,
    )


def call_end(
    handle: ToolHandle,
    result: ToolExecutionResult[Json],
    *,
    data: Json | None = None,
    metadata: Json | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Finish a manual tool span started by ``call()``.

    Args:
        handle: Tool handle returned by ``call()``.
        result: Canonical ``ToolExecutionResult`` to record on the end event.
        data: Optional JSON payload used when the sanitized ``result`` is JSON null.
        metadata: Optional JSON metadata recorded on the emitted end event.
        timestamp: Optional timezone-aware ``datetime`` recorded on the emitted
            end event. When omitted, the runtime default end timestamp is used.

    Returns:
        None: This function returns after an immutable end-event snapshot and
        its middleware/subscriber chains have been queued for publication.

    Notes:
        ``call_end()`` remains synchronous. Sanitize-response guardrails,
        event sanitizers, and subscriber delivery run later on Relay's serial
        publication path. Callback failures are logged and fail closed; they
        cannot be raised by this call. The caller-owned ``result`` is not
        altered.
        ``timestamp`` must be a timezone-aware ``datetime``; strings and naive
        datetimes are rejected.
    """
    ensure_scope_stack()
    return _native_tool_call_end(handle, result, data=data, metadata=metadata, timestamp=timestamp)


def execute(
    name: str,
    args: Json,
    func: Callable[[Json], ToolExecutionResult[Json] | Awaitable[ToolExecutionResult[Json]]],
    *,
    handle: ScopeHandle | None = None,
    attributes: ToolAttributes | None = None,
    data: Json | None = None,
    metadata: Json | None = None,
    tool_call_id: str | None = None,
) -> Awaitable[ToolExecutionResult[Json]]:
    """Run a tool through the managed middleware pipeline.

    Pipeline order:

    1. tool conditional-execution guardrails
    2. tool request intercepts
    3. tool sanitize-request guardrails for emitted start events
    4. tool execution intercepts
    5. ``func(args)``
    6. tool sanitize-response guardrails for emitted end events

    Args:
        name: Tool name recorded on emitted lifecycle events.
        args: JSON-compatible arguments passed through the middleware pipeline.
        func: Tool implementation invoked as ``func(args)`` after guardrails and
            intercepts run. It must return ``ToolExecutionResult``.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native tool attributes attached to the start event.
        data: Optional JSON application payload stored on the managed tool handle.
        metadata: Optional JSON metadata recorded on the emitted start event.
        tool_call_id: Optional provider-specific tool call identifier recorded
            on the emitted start and end events.

    Returns:
        ToolExecutionResult: The canonical result returned by ``func`` or an
        execution intercept.

    Notes:
        Sanitize guardrails affect emitted event payloads only. They do not
        mutate the arguments passed to ``func`` or the value returned to the
        caller.

    Example::

        import nemo_relay

        async def local_tool(args):
            return nemo_relay.ToolExecutionResult({"count": len(args["items"])})

        result = await nemo_relay.tools.execute(
            "count",
            {"items": [1, 2, 3]},
            local_tool,
            handle=None,
            attributes=None,
            data={"source": "example"},
            metadata={"request_id": "req-1"},
        )
        assert result.result["count"] == 3
    """
    ensure_scope_stack()
    return _native_tool_call_execute(
        name,
        args,
        func,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        tool_call_id=tool_call_id,
    )


def request_intercepts(name: str, args: Json) -> Json | Awaitable[Json]:
    """Apply global tool request intercepts to ``args``.

    Args:
        name: Tool name used when evaluating the registered intercept chain.
        args: JSON-compatible tool arguments to pass through the intercepts.

    Returns:
        Json | Awaitable[Json]: The arguments produced by the final request
        intercept. Outside a running event loop this is returned directly.
        Inside an event loop, await the returned value.

    Notes:
        This runs only the request-intercept chain. It does not execute
        conditional guardrails, sanitize guardrails, or the tool callback.
    """
    ensure_scope_stack()
    return _native_tool_request_intercepts(name, args)


def conditional_execution(name: str, args: Json) -> Awaitable[None] | None:
    """Run tool conditional-execution guardrails for ``args``.

    Args:
        name: Tool name used when evaluating registered guardrails.
        args: JSON-compatible tool arguments to validate.

    Returns:
        Awaitable[None] | None: ``None`` when execution is allowed, returned
        directly outside an event loop or through an awaitable inside one.

    Notes:
        This helper evaluates only the conditional-execution guardrail chain
        and does not invoke request intercepts or tool execution.

    Raises:
        RuntimeError: If a guardrail rejects the call or an asynchronous
        guardrail is registered when called outside an event loop.
    """
    ensure_scope_stack()
    return _native_tool_conditional_execution(name, args)


__all__ = ["call", "call_end", "execute", "request_intercepts", "conditional_execution"]
