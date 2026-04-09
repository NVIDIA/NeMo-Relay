# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for the Nexus runtime.

This package exposes the runtime's scope stack, lifecycle events, middleware
registries, typed wrappers, and optimizer helpers from Python.

The main entry points are:

- ``nat_nexus.scope`` for creating and nesting scopes
- ``nat_nexus.tools`` for tool lifecycle management
- ``nat_nexus.llm`` for non-streaming and streaming LLM lifecycle management
- ``nat_nexus.guardrails`` and ``nat_nexus.intercepts`` for global middleware
- ``nat_nexus.scope_local`` for middleware scoped to a specific ``ScopeHandle``
- ``nat_nexus.typed`` for codec-based typed wrappers
- ``nat_nexus.optimizer`` for optimizer configuration and plugin registration

Example:
    ```python
    import asyncio

    import nat_nexus

    def redact_args(tool_name, args):
        return {**args, "api_key": "***"}

    def add_header(name, request, annotated):
        request.headers["Authorization"] = "Bearer test-token"
        return request, annotated

    async def tool_impl(args):
        return {"echo": args["query"]}

    async def llm_impl(request):
        return {"messages": request.content["messages"], "ok": True}

    async def main():
        nat_nexus.guardrails.register_tool_sanitize_request("redact", 10, redact_args)
        nat_nexus.intercepts.register_llm_request("auth", 10, False, add_header)

        with nat_nexus.scope.scope("demo-agent", nat_nexus.ScopeType.Agent):
            tool_result = await nat_nexus.tools.execute("search", {"query": "hello"}, tool_impl)
            llm_result = await nat_nexus.llm.execute(
                "demo-model",
                nat_nexus.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
                llm_impl,
            )

            print(tool_result, llm_result)

    asyncio.run(main())
    ```
"""

from __future__ import annotations

import contextvars
import typing
from typing import AsyncIterator, Awaitable, Callable, Literal, Optional, TypeAlias

from nat_nexus._native import (
    AnnotatedLLMRequest,
    AnnotatedLLMResponse,
    # ATIF exporter
    AtifExporter,
    LLMAttributes,
    LLMEndEvent,
    LLMHandle,
    LLMRequest,
    LLMStartEvent,
    MarkEvent,
    OpenInferenceConfig,
    OpenInferenceSubscriber,
    OpenTelemetryConfig,
    OpenTelemetrySubscriber,
    ScopeAttributes,
    # Types (always at top level)
    ScopeEndEvent,
    ScopeHandle,
    ScopeStartEvent,
    ScopeType,
    ToolAttributes,
    ToolEndEvent,
    ToolHandle,
    ToolStartEvent,
)
from nat_nexus._native import ScopeStack as _ScopeStack
from nat_nexus._native import create_scope_stack as _create_scope_stack
from nat_nexus._native import scope_stack_active as _native_scope_stack_active
from nat_nexus._native import set_thread_scope_stack as _set_thread_scope_stack
from nat_nexus._native import sync_thread_scope_stack as _sync_thread_scope_stack

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Json: TypeAlias = JsonValue
UnsupportedBehavior: TypeAlias = Literal["ignore", "warn", "error"]

ToolSanitizeGuardrail: TypeAlias = Callable[[str, Json], Json]
ToolConditionalExecutionGuardrail: TypeAlias = Callable[[str, Json], Optional[str]]
LlmSanitizeRequestGuardrail: TypeAlias = Callable[[LLMRequest], LLMRequest]
LlmSanitizeResponseGuardrail: TypeAlias = Callable[[JsonObject], JsonObject]
LlmConditionalExecutionGuardrail: TypeAlias = Callable[[LLMRequest], Optional[str]]
ToolRequestIntercept: TypeAlias = Callable[[str, Json], Json]
ToolExecutionIntercept: TypeAlias = Callable[
    [str, Json, Callable[[Json], Awaitable[Json]]],
    Json | Awaitable[Json],
]
LlmRequestIntercept: TypeAlias = Callable[
    [str, LLMRequest, AnnotatedLLMRequest | None],
    tuple[LLMRequest, AnnotatedLLMRequest | None],
]
LlmExecutionIntercept: TypeAlias = Callable[
    [str, LLMRequest, Callable[[LLMRequest], Awaitable[Json]]],
    Json | Awaitable[Json],
]
LlmStreamExecutionIntercept: TypeAlias = Callable[
    [LLMRequest, Callable[[LLMRequest], Awaitable[AsyncIterator[Json]]]],
    AsyncIterator[Json] | Awaitable[AsyncIterator[Json]],
]

from nat_nexus import (  # noqa: E402
    codecs,
    guardrails,
    intercepts,
    llm,
    optimizer,
    scope,
    scope_local,
    subscribers,
    tools,
    typed,
)

_scope_stack_var: contextvars.ContextVar[_ScopeStack] = contextvars.ContextVar("nat_nexus_scope_stack")


def get_scope_stack() -> _ScopeStack:
    """Return the current task's active scope stack.

    If the current async context does not yet own a scope stack, this function
    creates one and synchronizes it into the Rust thread-local storage used by
    the native runtime. Most callers do not need to invoke this directly
    because higher-level helpers such as ``nat_nexus.scope.push()`` do it
    automatically.

    Returns:
        ScopeStack: The scope stack associated with the current Python context.

    Notes:
        Calling this function synchronizes the Python ``ContextVar`` state into
        the native thread-local slot so subsequent native runtime calls observe
        the same scope hierarchy.

    Example:
        ```python
        import nat_nexus

        stack = nat_nexus.get_scope_stack()
        assert stack is not None
        ```
    """
    stack = _scope_stack_var.get(None)
    if stack is None:
        stack = _create_scope_stack()
        _scope_stack_var.set(stack)
    # Keep the Rust thread-local in sync so that native calls (which read
    # from THREAD_SCOPE_STACK / TASK_SCOPE_STACK) see the same scope stack.
    # Uses sync (not set) to avoid marking this thread as explicitly active.
    _sync_thread_scope_stack(stack)
    return stack


def scope_stack_active() -> bool:
    """Report whether the current context already owns a scope stack.

    Returns:
        bool: ``True`` when the current Python context already has an active
        stack, either because it was created in this context or because a stack
        was explicitly installed for the current thread.

    Notes:
        This function does not create a scope stack. It is a pure status check
        used to decide whether scope propagation work is required.

    Example:
        ```python
        import nat_nexus

        assert nat_nexus.scope_stack_active() is False
        nat_nexus.get_scope_stack()
        assert nat_nexus.scope_stack_active() is True
        ```
    """
    if _scope_stack_var.get(None) is not None:
        return True
    return _native_scope_stack_active()


def propagate_scope_to_thread() -> _ScopeStack:
    """Capture the active scope stack for use in another thread.

    The returned stack can be passed to ``set_thread_scope_stack()`` inside a
    worker thread so that the worker emits events into the same scope hierarchy
    as the parent context.

    Returns:
        ScopeStack: The active stack from the current context.

    Raises:
        RuntimeError: If the current context does not yet have an active scope
            stack to propagate.

    Notes:
        This function does not clone the scope hierarchy. It shares the current
        stack reference with the target thread, which is appropriate when the
        worker should contribute events to the same logical trace.

    Example:
        ```python
        from concurrent.futures import ThreadPoolExecutor

        import nat_nexus

        with nat_nexus.scope.scope("parent", nat_nexus.ScopeType.Agent) as handle:
            stack = nat_nexus.propagate_scope_to_thread()

            def worker() -> None:
                nat_nexus.set_thread_scope_stack(stack)
                nat_nexus.scope.event(
                    "worker-ran",
                    handle=handle,
                    data={"source": "thread"},
                    metadata={"thread": "pool-1"},
                )

            with ThreadPoolExecutor() as pool:
                pool.submit(worker).result()
        ```
    """
    if not scope_stack_active():
        raise RuntimeError(
            "no active scope stack in current context; call nat_nexus.get_scope_stack() or nat_nexus.scope.push() first"
        )
    # Return the ContextVar value directly if available, to avoid
    # calling get_scope_stack() which would sync to the Rust thread-local.
    stack = _scope_stack_var.get(None)
    if stack is not None:
        return stack
    # Rust-side explicit flag is set. Return via get_scope_stack() which
    # will create a ContextVar entry and sync.
    return get_scope_stack()


ScopeStack = _ScopeStack
create_scope_stack = _create_scope_stack
set_thread_scope_stack = _set_thread_scope_stack
Event = typing.Union[
    ScopeStartEvent,
    ScopeEndEvent,
    ToolStartEvent,
    ToolEndEvent,
    LLMStartEvent,
    LLMEndEvent,
    MarkEvent,
]


__all__ = [
    # Submodules
    "scope",
    "tools",
    "llm",
    "guardrails",
    "intercepts",
    "subscribers",
    "scope_local",
    "codecs",
    "typed",
    "optimizer",
    # Scope stack isolation
    "ScopeStack",
    "create_scope_stack",
    "get_scope_stack",
    "scope_stack_active",
    "propagate_scope_to_thread",
    "set_thread_scope_stack",
    # Types
    "ScopeAttributes",
    "ToolAttributes",
    "LLMAttributes",
    "ScopeType",
    "ScopeStartEvent",
    "ScopeEndEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "LLMStartEvent",
    "LLMEndEvent",
    "MarkEvent",
    "ScopeHandle",
    "ToolHandle",
    "LLMHandle",
    "LLMRequest",
    "Event",
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "AtifExporter",
    "OpenInferenceConfig",
    "OpenInferenceSubscriber",
    "OpenTelemetryConfig",
    "OpenTelemetrySubscriber",
    "JsonPrimitive",
    "JsonValue",
    "JsonObject",
    "Json",
    "UnsupportedBehavior",
    "ToolSanitizeGuardrail",
    "ToolConditionalExecutionGuardrail",
    "LlmSanitizeRequestGuardrail",
    "LlmSanitizeResponseGuardrail",
    "LlmConditionalExecutionGuardrail",
    "ToolRequestIntercept",
    "ToolExecutionIntercept",
    "LlmRequestIntercept",
    "LlmExecutionIntercept",
    "LlmStreamExecutionIntercept",
]
