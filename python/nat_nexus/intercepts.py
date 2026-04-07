# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intercept registration for tools and LLMs.

Intercepts transform requests or replace execution functions entirely.
They are priority-ordered (ascending) and registered by name. Request
intercepts accept a ``break_chain`` flag — when ``True``, no
lower-priority intercepts run after this one.

**Tool intercepts** — callback signatures:

    register_tool_request(name, priority, break_chain, fn)
        ``fn(tool_name: str, args: Json) -> Json`` — transform tool arguments.

    register_tool_execution(name, priority, fn)
        ``fn`` is ``async (tool_name: str, args: Json, next) -> Json`` —
        middleware intercept. Call ``await next(args)`` to invoke the next
        intercept or original implementation; skip calling ``next`` to
        short-circuit.

**LLM intercepts** — callback signatures:

    register_llm_request(name, priority, break_chain, fn)
        ``fn(name: str, request: LLMRequest) -> LLMRequest`` — transform the
        LLM request.

    register_llm_execution(name, priority, fn)
        ``fn`` is ``async (name: str, request: LLMRequest, next) -> Json`` —
        middleware intercept. Call ``await next(request)`` to continue the
        chain.

    register_llm_stream_execution(name, priority, fn)
        ``fn`` is ``async (request: LLMRequest, next) -> AsyncIterator[Json]``
        — middleware intercept. Call ``await next(request)`` to continue.

Each ``register_*`` has a corresponding ``deregister_*`` that takes the name
and returns ``True`` if an intercept was found and removed.

Example::

    import nat_nexus
    from nat_nexus import LLMRequest

    def add_header(name, request):
        request.headers["X-Extra"] = "injected"
        return request

    nat_nexus.intercepts.register_llm_request("extra", 1, False, add_header)
"""

from typing import Any, AsyncIterator, Awaitable, Callable

from nat_nexus._native import LLMRequest
from nat_nexus._native import (
    nat_nexus_deregister_llm_execution_intercept as _native_deregister_llm_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_llm_request_intercept as _native_deregister_llm_request,
)
from nat_nexus._native import (
    nat_nexus_deregister_llm_stream_execution_intercept as _native_deregister_llm_stream_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_execution_intercept as _native_deregister_tool_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_request_intercept as _native_deregister_tool_request,
)
from nat_nexus._native import (
    nat_nexus_register_llm_execution_intercept as _native_register_llm_execution,
)
from nat_nexus._native import (
    nat_nexus_register_llm_request_intercept as _native_register_llm_request,
)
from nat_nexus._native import (
    nat_nexus_register_llm_stream_execution_intercept as _native_register_llm_stream_execution,
)
from nat_nexus._native import (
    nat_nexus_register_tool_execution_intercept as _native_register_tool_execution,
)
from nat_nexus._native import (
    nat_nexus_register_tool_request_intercept as _native_register_tool_request,
)

Json = Any
"""Type alias for JSON-serializable Python objects (dicts, lists, strings, numbers, etc.)."""

ToolRequestIntercept = Callable[[str, Json], Json]
ToolExecutionIntercept = Callable[[str, Json, Callable[[Json], Awaitable[Json]]], Json | Awaitable[Json]]
LlmRequestIntercept = Callable[[str, LLMRequest], LLMRequest]
LlmExecutionIntercept = Callable[
    [str, LLMRequest, Callable[[LLMRequest], Awaitable[Json]]],
    Json | Awaitable[Json],
]
LlmStreamExecutionIntercept = Callable[
    [LLMRequest, Callable[[LLMRequest], Awaitable[AsyncIterator[Json]]]],
    AsyncIterator[Json] | Awaitable[AsyncIterator[Json]],
]

# ---------------------------------------------------------------------------
# Tool intercepts
# ---------------------------------------------------------------------------


def register_tool_request(name: str, priority: int, break_chain: bool, fn: ToolRequestIntercept) -> None:
    """Register a tool request intercept.

    The intercept callback receives the tool name and arguments and returns
    transformed arguments that replace the originals in the pipeline.

    Args:
        name: Unique intercept name.
        priority: Priority (ascending order; lower runs first).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        fn: Callable ``(tool_name: str, args: Json) -> Json``.

    Raises:
        RuntimeError: If an intercept with this name already exists.
    """
    return _native_register_tool_request(name, priority, break_chain, fn)


def deregister_tool_request(name: str) -> bool:
    """Remove a tool request intercept.

    Args:
        name: Name of the intercept to remove.

    Returns:
        ``True`` if an intercept with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_tool_request(name)


def register_tool_execution(name: str, priority: int, fn: ToolExecutionIntercept) -> None:
    """Register a tool execution intercept (middleware chain pattern).

    The intercept receives the tool name, arguments, and a ``next`` function.
    Call ``await next(args)`` to invoke the next intercept or original
    implementation. Skip calling ``next`` to short-circuit the chain.

    Args:
        name: Unique intercept name.
        priority: Priority (ascending order; lower runs first).
        fn: Async callable ``(tool_name: str, args: Json, next: (Json) -> Awaitable[Json]) -> Awaitable[Json]``.

    Raises:
        RuntimeError: If an intercept with this name already exists.
    """
    return _native_register_tool_execution(name, priority, fn)


def deregister_tool_execution(name: str) -> bool:
    """Remove a tool execution intercept.

    Args:
        name: Name of the intercept to remove.

    Returns:
        ``True`` if an intercept with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_tool_execution(name)


# ---------------------------------------------------------------------------
# LLM intercepts
# ---------------------------------------------------------------------------


def register_llm_request(name: str, priority: int, break_chain: bool, fn: LlmRequestIntercept) -> None:
    """Register an LLM request intercept.

    The intercept callback receives the intercept name and ``LLMRequest`` and
    returns a transformed ``LLMRequest`` that replaces the original in the
    pipeline.

    Args:
        name: Unique intercept name.
        priority: Priority (ascending order; lower runs first).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        fn: Callable ``(name: str, request: LLMRequest) -> LLMRequest``.

    Raises:
        RuntimeError: If an intercept with this name already exists.
    """
    return _native_register_llm_request(name, priority, break_chain, fn)


def deregister_llm_request(name: str) -> bool:
    """Remove an LLM request intercept.

    Args:
        name: Name of the intercept to remove.

    Returns:
        ``True`` if an intercept with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_llm_request(name)


def register_llm_execution(name: str, priority: int, fn: LlmExecutionIntercept) -> None:
    """Register an LLM execution intercept (middleware chain pattern).

    The intercept receives the intercept name, ``LLMRequest``, and a ``next``
    function. Call ``await next(request)`` to invoke the next intercept or
    original implementation. Skip calling ``next`` to short-circuit the chain.

    Args:
        name: Unique intercept name.
        priority: Priority (ascending order; lower runs first).
        fn: Async callable
            ``(name: str, request: LLMRequest, next: (LLMRequest) -> Awaitable[Json]) -> Awaitable[Json]``.

    Raises:
        RuntimeError: If an intercept with this name already exists.
    """
    return _native_register_llm_execution(name, priority, fn)


def deregister_llm_execution(name: str) -> bool:
    """Remove an LLM execution intercept.

    Args:
        name: Name of the intercept to remove.

    Returns:
        ``True`` if an intercept with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_llm_execution(name)


def register_llm_stream_execution(
    name: str,
    priority: int,
    fn: LlmStreamExecutionIntercept,
) -> None:
    """Register an LLM stream-execution intercept (middleware chain pattern).

    The intercept receives the ``LLMRequest`` and a ``next`` function. Call
    ``await next(request)`` to invoke the next intercept or original streaming
    implementation. Skip calling ``next`` to short-circuit the chain.

    Args:
        name: Unique intercept name.
        priority: Priority (ascending order; lower runs first).
        fn: Async callable
            ``(request: LLMRequest, next: (LLMRequest) -> Awaitable[AsyncIterator[Json]])
            -> Awaitable[AsyncIterator[Json]]``.

    Raises:
        RuntimeError: If an intercept with this name already exists.
    """
    return _native_register_llm_stream_execution(name, priority, fn)


def deregister_llm_stream_execution(name: str) -> bool:
    """Remove an LLM stream-execution intercept.

    Args:
        name: Name of the intercept to remove.

    Returns:
        ``True`` if an intercept with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_llm_stream_execution(name)


__all__ = [
    "Json",
    "ToolRequestIntercept",
    "ToolExecutionIntercept",
    "LlmRequestIntercept",
    "LlmExecutionIntercept",
    "LlmStreamExecutionIntercept",
    "register_tool_request",
    "deregister_tool_request",
    "register_tool_execution",
    "deregister_tool_execution",
    "register_llm_request",
    "deregister_llm_request",
    "register_llm_execution",
    "deregister_llm_execution",
    "register_llm_stream_execution",
    "deregister_llm_stream_execution",
]
