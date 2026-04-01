# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool lifecycle operations.

Provides both manual and managed tool-call workflows.

Functions:
    call(name, args, *, handle=None, attributes=None, data=None, metadata=None, tool_call_id=None)
        Begin a tool call manually. Returns a ``ToolHandle`` that must later
        be passed to ``call_end``. Emits a ``Start`` event. The optional
        ``tool_call_id`` is an external correlation ID propagated to events
        for ATIF trajectory linking.

    call_end(handle, result, *, data=None, metadata=None)
        End a manual tool call. Records the result and emits an ``End`` event.

    execute(name, args, func, *, handle=None, attributes=None, data=None, metadata=None)
        Execute a tool call through the full middleware pipeline (conditional-
        execution guardrails on raw args → request intercepts →
        sanitize-request guardrails → execution intercepts → *func* →
        response intercepts → sanitize-response guardrails). On rejection,
        only a standalone Mark event is emitted and ``GuardrailRejected`` is
        raised. Returns an awaitable of the final result.

    request_intercepts(name, args)
        Run the registered tool request intercept chain on the given arguments.
        Returns the transformed arguments.

    conditional_execution(name, args)
        Run the registered tool conditional execution guardrail chain.
        Raises ``RuntimeError`` if any guardrail rejects.

    response_intercepts(name, result)
        Run the registered tool response intercept chain on the given result.
        Returns the transformed result.

Example::

    import nat_nexus

    # Managed execution (recommended)
    result = await nat_nexus.tools.execute("search", {"q": "hello"}, my_search)

    # Manual lifecycle
    handle = nat_nexus.tools.call("search", {"q": "hello"})
    result = my_search({"q": "hello"})
    nat_nexus.tools.call_end(handle, result)
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
from nat_nexus._native import (
    nat_nexus_tool_response_intercepts as _native_tool_response_intercepts,
)


def call(name, args, *, handle=None, attributes=None, data=None, metadata=None, tool_call_id=None):
    """Begin a tool call manually.

    Emits a ``Start`` event and returns a ``ToolHandle`` that must later be
    passed to ``call_end()`` to complete the tool call lifecycle.

    Args:
        name: Tool name identifier.
        args: JSON-serializable tool arguments.
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``ToolAttributes`` bitflags.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.
        tool_call_id: Optional external correlation ID for ATIF trajectory linking.

    Returns:
        A ``ToolHandle`` for use with ``call_end()``.
    """
    return _native_tool_call(
        name, args, handle=handle, attributes=attributes, data=data, metadata=metadata, tool_call_id=tool_call_id
    )


def call_end(handle, result, *, data=None, metadata=None):
    """End a manual tool call.

    Records the result and emits an ``End`` event.

    Args:
        handle: The ``ToolHandle`` returned by ``call()``.
        result: JSON-serializable tool result.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.
    """
    return _native_tool_call_end(handle, result, data=data, metadata=metadata)


def execute(name, args, func, *, handle=None, attributes=None, data=None, metadata=None):
    """Execute a tool call through the full middleware pipeline.

    Runs conditional-execution guardrails (on raw args) -> request intercepts ->
    sanitize-request guardrails -> execution intercepts -> *func* -> response
    intercepts -> sanitize-response guardrails. On rejection, only a standalone
    ``Mark`` event is emitted (no ``Start``/``End`` pair) and
    ``GuardrailRejected`` is raised.

    Args:
        name: Tool name identifier.
        args: JSON-serializable tool arguments.
        func: Async callable ``(args) -> result`` that performs the tool work.
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``ToolAttributes`` bitflags.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.

    Returns:
        An awaitable that resolves to the (possibly transformed) tool result.

    Raises:
        GuardrailRejected: If a conditional-execution guardrail rejects the call.
    """
    return _native_tool_call_execute(
        name, args, func, handle=handle, attributes=attributes, data=data, metadata=metadata
    )


def request_intercepts(name, args):
    """Run the registered tool request intercept chain.

    Applies all registered tool request intercepts in priority order to
    the given arguments.

    Args:
        name: Tool name identifier.
        args: JSON-serializable tool arguments to transform.

    Returns:
        The transformed arguments after all intercepts have been applied.
    """
    return _native_tool_request_intercepts(name, args)


def conditional_execution(name, args):
    """Run the registered tool conditional-execution guardrail chain.

    Evaluates all registered conditional-execution guardrails in priority
    order against the given arguments.

    Args:
        name: Tool name identifier.
        args: JSON-serializable tool arguments to evaluate.

    Raises:
        RuntimeError: If any guardrail rejects the tool call.
    """
    return _native_tool_conditional_execution(name, args)


def response_intercepts(name, result):
    """Run the registered tool response intercept chain.

    Applies all registered tool response intercepts in priority order to
    the given result.

    Args:
        name: Tool name identifier.
        result: JSON-serializable tool result to transform.

    Returns:
        The transformed result after all intercepts have been applied.
    """
    return _native_tool_response_intercepts(name, result)


__all__ = ["call", "call_end", "execute", "request_intercepts", "conditional_execution", "response_intercepts"]
