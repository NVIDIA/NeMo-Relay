# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool lifecycle helpers.

Use this module when you want Nexus to emit tool start and end events around a
piece of application logic.

``execute()`` is the usual entry point and runs the full middleware pipeline.
``call()`` and ``call_end()`` are the lower-level manual lifecycle APIs.

Example:
    ```python
    import nat_nexus

    async def search(args):
        return {"result": args["query"].upper()}

    result = await nat_nexus.tools.execute("search", {"query": "hello"}, search)
    assert result == {"result": "HELLO"}
    ```
"""

from nat_nexus._native import (
    nat_nexus_tool_call as _native_tool_call,
)
from nat_nexus._native import (
    nat_nexus_tool_call_end as _native_tool_call_end,
)
from nat_nexus._native import (
    nat_nexus_tool_call_execute as _native_tool_call_execute,
)
from nat_nexus._native import (
    nat_nexus_tool_conditional_execution as _native_tool_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_tool_request_intercepts as _native_tool_request_intercepts,
)


def call(name, args, *, handle=None, attributes=None, data=None, metadata=None, tool_call_id=None):
    """Start a manual tool span and return its ``ToolHandle``.

    Args:
        name: Tool name recorded on emitted lifecycle events.
        args: JSON-compatible tool arguments to associate with the call.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native tool attributes attached to the start event.
        data: Optional JSON payload recorded on the emitted start event.
        metadata: Optional JSON metadata recorded on the emitted start event.
        tool_call_id: Optional provider-specific tool call identifier to attach
            to the emitted events.

    Returns:
        ToolHandle: Handle used to finish the manual span with ``call_end()``.

    Example:
        ```python
        import nat_nexus

        handle = nat_nexus.tools.call(
            "search",
            {"query": "hello"},
            handle=None,
            attributes=None,
            data={"attempt": 1},
            metadata={"path": "manual"},
            tool_call_id="tool-call-1",
        )
        nat_nexus.tools.call_end(
            handle,
            {"result": "ok"},
            data={"cached": False},
            metadata={"status": "success"},
        )
        ```
    """
    return _native_tool_call(
        name, args, handle=handle, attributes=attributes, data=data, metadata=metadata, tool_call_id=tool_call_id
    )


def call_end(handle, result, *, data=None, metadata=None):
    """Finish a manual tool span started by ``call()``.

    Args:
        handle: Tool handle returned by ``call()``.
        result: JSON-compatible tool result to record on the end event.
        data: Optional JSON payload recorded on the emitted end event.
        metadata: Optional JSON metadata recorded on the emitted end event.
    """
    return _native_tool_call_end(handle, result, data=data, metadata=metadata)


def execute(name, args, func, *, handle=None, attributes=None, data=None, metadata=None):
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
            intercepts run.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native tool attributes attached to the start event.
        data: Optional JSON payload recorded on the emitted start event.
        metadata: Optional JSON metadata recorded on the emitted start event.

    Returns:
        Json: The raw result returned by ``func`` or by an execution intercept.

    Notes:
        Sanitize guardrails affect emitted event payloads only. They do not
        mutate the arguments passed to ``func`` or the value returned to the
        caller.

    Example:
        ```python
        import nat_nexus

        async def local_tool(args):
            return {"count": len(args["items"])}

        result = await nat_nexus.tools.execute(
            "count",
            {"items": [1, 2, 3]},
            local_tool,
            handle=None,
            attributes=None,
            data={"source": "example"},
            metadata={"request_id": "req-1"},
        )
        assert result["count"] == 3
        ```
    """
    return _native_tool_call_execute(
        name, args, func, handle=handle, attributes=attributes, data=data, metadata=metadata
    )


def request_intercepts(name, args):
    """Apply global tool request intercepts to ``args``.

    Args:
        name: Tool name used when evaluating the registered intercept chain.
        args: JSON-compatible tool arguments to pass through the intercepts.

    Returns:
        Json: The arguments produced by the final request intercept.
    """
    return _native_tool_request_intercepts(name, args)


def conditional_execution(name, args):
    """Run tool conditional-execution guardrails for ``args``.

    Args:
        name: Tool name used when evaluating registered guardrails.
        args: JSON-compatible tool arguments to validate.

    Returns:
        str | None: A rejection message if execution should be blocked,
        otherwise ``None``.
    """
    return _native_tool_conditional_execution(name, args)


__all__ = ["call", "call_end", "execute", "request_intercepts", "conditional_execution"]
