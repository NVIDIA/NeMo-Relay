# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global middleware intercept registration for tools and LLMs.

Request intercepts transform inputs before execution. Execution intercepts wrap
the downstream callable and can observe, modify, or replace the result.

Example::

    import nemo_relay

    def add_header(
        name: str,
        request: nemo_relay.LLMRequest,
        annotated: nemo_relay.AnnotatedLLMRequest | None
    ) -> tuple[nemo_relay.LLMRequest, nemo_relay.AnnotatedLLMRequest | None]:
        # The request object is immutable, however we can return a new instance with updated headers.
        headers = request.headers.copy()
        headers["X-Trace"] = "demo"
        return nemo_relay.LLMRequest(headers=headers, content=request.content), annotated

    nemo_relay.intercepts.register_llm_request("trace-header", 10, False, add_header)
"""

from nemo_relay import (
    LlmExecutionIntercept,
    LlmRequestIntercept,
    LlmStreamExecutionIntercept,
    ToolExecutionIntercept,
    ToolRequestIntercept,
)
from nemo_relay._native import (
    deregister_llm_execution_intercept as _native_deregister_llm_execution,
)
from nemo_relay._native import (
    deregister_llm_request_intercept as _native_deregister_llm_request,
)
from nemo_relay._native import (
    deregister_llm_stream_execution_intercept as _native_deregister_llm_stream_execution,
)
from nemo_relay._native import (
    deregister_tool_execution_intercept as _native_deregister_tool_execution,
)
from nemo_relay._native import (
    deregister_tool_request_intercept as _native_deregister_tool_request,
)
from nemo_relay._native import (
    register_llm_execution_intercept as _native_register_llm_execution,
)
from nemo_relay._native import (
    register_llm_request_intercept as _native_register_llm_request,
)
from nemo_relay._native import (
    register_llm_stream_execution_intercept as _native_register_llm_stream_execution,
)
from nemo_relay._native import (
    register_tool_execution_intercept as _native_register_tool_execution,
)
from nemo_relay._native import (
    register_tool_request_intercept as _native_register_tool_request,
)

# ---------------------------------------------------------------------------
# Tool intercepts
# ---------------------------------------------------------------------------


def register_tool_request(name: str, priority: int, break_chain: bool, fn: ToolRequestIntercept) -> None:
    """Register an intercept that rewrites tool arguments before execution.

    Args:
        name: Unique intercept name used for later replacement or removal.
        priority: Execution order for the intercept. Lower values run first.
        break_chain: Whether to stop applying lower-priority request intercepts
            after this intercept runs.
        fn: Callable invoked as ``fn(tool_name, args)`` that returns the
            rewritten tool arguments.

    Returns:
        None: This function returns after the intercept is registered.

    Notes:
        Request intercepts run after conditional-execution guardrails and
        before sanitize-request guardrails or execution intercepts.

    Example::

        import nemo_relay

        def add_trace_id(tool_name, args):
            return {**args, "trace_id": "req-123"}

        nemo_relay.intercepts.register_tool_request(
            "trace-id",
            10,
            False,
            add_trace_id,
        )
    """
    return _native_register_tool_request(name, priority, break_chain, fn)


def deregister_tool_request(name: str) -> bool:
    """Remove a previously registered tool request intercept.

    Args:
        name: Intercept name previously passed to ``register_tool_request()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removal affects only future executions. In-flight calls continue using
        the intercept chain they already resolved.
    """
    return _native_deregister_tool_request(name)


def register_tool_execution(name: str, priority: int, fn: ToolExecutionIntercept) -> None:
    """Register middleware around tool execution.

    Args:
        name: Unique intercept name used for later replacement or removal.
        priority: Execution order for the intercept. Lower values run first.
        fn: Callable invoked as ``fn(tool_name, args, next_call)``. The
            callback may await or call ``next_call(args)`` to continue the
            chain, modify the result, or bypass downstream execution entirely.

    Returns:
        None: This function returns after the intercept is registered.

    Notes:
        Execution intercepts wrap the downstream tool callback. They are the
        right place for timing, retries, short-circuiting, or result shaping.
    """
    return _native_register_tool_execution(name, priority, fn)


def deregister_tool_execution(name: str) -> bool:
    """Remove a previously registered tool execution intercept.

    Args:
        name: Intercept name previously passed to
            ``register_tool_execution()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removal affects only future executions. In-flight calls continue using
        the execution chain they already resolved.
    """
    return _native_deregister_tool_execution(name)


# ---------------------------------------------------------------------------
# LLM intercepts
# ---------------------------------------------------------------------------


def register_llm_request(name: str, priority: int, break_chain: bool, fn: LlmRequestIntercept) -> None:
    """Register an intercept that rewrites an ``LLMRequest`` before execution.

    Args:
        name: Unique intercept name used for later replacement or removal.
        priority: Execution order for the intercept. Lower values run first.
        break_chain: Whether to stop applying lower-priority request intercepts
            after this intercept runs.
        fn: Callable invoked as ``fn(name, request, annotated)`` that returns a
            tuple of ``(request, annotated)`` for the next intercept or the
            provider callback.

    Returns:
        None: This function returns after the intercept is registered.

    Notes:
        ``annotated`` is ``None`` unless a request codec was supplied to the
        managed LLM call. Intercepts should preserve both values when they do
        not need to mutate them.

    Example::

        import nemo_relay

        def add_header(
            name: str, request: nemo_relay.LLMRequest,
            annotated: nemo_relay.AnnotatedLLMRequest | None
        ) -> tuple[nemo_relay.LLMRequest, nemo_relay.AnnotatedLLMRequest | None]:
            headers = request.headers.copy()
            headers["X-Trace"] = "req-123"
            return nemo_relay.LLMRequest(headers=headers, content=request.content), annotated

        nemo_relay.intercepts.register_llm_request(
            "trace-header",
            10,
            False,
            add_header,
        )
    """
    return _native_register_llm_request(name, priority, break_chain, fn)


def deregister_llm_request(name: str) -> bool:
    """Remove a previously registered LLM request intercept.

    Args:
        name: Intercept name previously passed to ``register_llm_request()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removal affects only future executions. In-flight calls continue using
        the intercept chain they already resolved.
    """
    return _native_deregister_llm_request(name)


def register_llm_execution(name: str, priority: int, fn: LlmExecutionIntercept) -> None:
    """Register middleware around non-streaming LLM execution.

    Args:
        name: Unique intercept name used for later replacement or removal.
        priority: Execution order for the intercept. Lower values run first.
        fn: Callable invoked as ``fn(name, request, next_call)``. The callback
            may call ``next_call(request)`` to continue execution, modify the
            result, or short-circuit the provider call.

    Returns:
        None: This function returns after the intercept is registered.

    Notes:
        Execution intercepts wrap only non-streaming LLM execution. Use
        ``register_llm_stream_execution()`` for streaming callbacks.
    """
    return _native_register_llm_execution(name, priority, fn)


def deregister_llm_execution(name: str) -> bool:
    """Remove a previously registered LLM execution intercept.

    Args:
        name: Intercept name previously passed to
            ``register_llm_execution()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removal affects only future executions. In-flight calls continue using
        the execution chain they already resolved.
    """
    return _native_deregister_llm_execution(name)


def register_llm_stream_execution(
    name: str,
    priority: int,
    fn: LlmStreamExecutionIntercept,
) -> None:
    """Register middleware around streaming LLM execution.

    Args:
        name: Unique intercept name used for later replacement or removal.
        priority: Execution order for the intercept. Lower values run first.
        fn: Callable invoked as ``fn(request, next_call)`` that returns an
            async iterator of JSON chunks, either by delegating to
            ``next_call(request)`` or by replacing the stream entirely.

    Returns:
        None: This function returns after the intercept is registered.

    Notes:
        Streaming execution intercepts wrap chunk production only. They do not
        replace the separate collector or finalizer callbacks.
    """
    return _native_register_llm_stream_execution(name, priority, fn)


def deregister_llm_stream_execution(name: str) -> bool:
    """Remove a previously registered streaming LLM execution intercept.

    Args:
        name: Intercept name previously passed to
            ``register_llm_stream_execution()``.

    Returns:
        bool: ``True`` if an intercept was removed, otherwise ``False``.

    Notes:
        Removal affects only future executions. In-flight streams continue
        using the execution chain they already resolved.
    """
    return _native_deregister_llm_stream_execution(name)


__all__ = [
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
