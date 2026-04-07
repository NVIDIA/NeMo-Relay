# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Agent Toolkit Nexus - Agent Runtime with scope/handle management, guardrails, and intercepts.

NeMo Agent Toolkit Nexus provides execution scope management, lifecycle event tracking, and
configurable middleware pipelines (guardrails and intercepts) for tool and LLM
calls. The core is written in Rust; this package exposes the full API to Python.

**Quick start**::

    import asyncio

    import nat_nexus

    def sanitizer(tool_name, args):
        # Sanitize PII from tool arguments
        return {k: "***" if "ssn" in k else v for k, v in args.items()}

    def add_auth(request):
        request.headers["Authorization"] = "Bearer XYZ"
        return request

    async def amain():
        # Define your tool and LLM functions
        my_tool_fn = lambda args: {**args, "result": "ok"}
        my_llm_fn = lambda request: {**request.content, "response": "ok"}

        # Register guardrails and intercepts
        nat_nexus.guardrails.register_tool_sanitize_request("pii-filter", 10, sanitizer)
        nat_nexus.intercepts.register_llm_request("auth-header", 1, False, add_auth)

        # Subscribe to lifecycle events
        nat_nexus.subscribers.register("logger", lambda event: print(event.name))

        # Create a scope for your agent
        with nat_nexus.scope.scope("my-agent", nat_nexus.ScopeType.Agent) as handle:
            # Execute a tool through the middleware pipeline
            result = await nat_nexus.tools.execute("search", {"q": "hello"}, my_tool_fn)

            # Execute an LLM call through the middleware pipeline
            native = {"messages": [{"role": "user", "content": "hello"}]}
            resp = await nat_nexus.llm.execute("gpt-4", nat_nexus.LLMRequest({}, native), my_llm_fn)

    asyncio.run(amain())


Submodules:
    scope       - Scope stack operations (push, pop, get_handle, event).
    tools       - Tool call lifecycle (call, call_end, execute).
    llm         - LLM call lifecycle (call, call_end, execute, stream_execute).
    guardrails  - Guardrail registration for tools and LLMs.
    intercepts  - Intercept registration for tools and LLMs.
    subscribers - Event subscriber registration.
    scope_local - Scope-local guardrail, intercept, and subscriber registration.
    proxy       - Proxy types, backends, and declarative proxy API.

Types (available at top level):
    ScopeAttributes, ToolAttributes, LLMAttributes,
    ScopeType, EventType,
    ScopeHandle, ToolHandle, LLMHandle,
    LLMRequest, Event, AtifExporter,
    OpenTelemetryConfig, OpenTelemetrySubscriber
"""

import contextvars

from nat_nexus import guardrails, intercepts, llm, proxy, scope, scope_local, subscribers, tools, typed
from nat_nexus._native import (
    # ATIF exporter
    AtifExporter,
    Event,
    EventType,
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    OpenTelemetryConfig,
    OpenTelemetrySubscriber,
    # Types (always at top level)
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
    ToolAttributes,
    ToolHandle,
)
from nat_nexus._native import ScopeStack as _ScopeStack
from nat_nexus._native import create_scope_stack as _create_scope_stack
from nat_nexus._native import scope_stack_active as _native_scope_stack_active
from nat_nexus._native import set_thread_scope_stack as _set_thread_scope_stack
from nat_nexus._native import sync_thread_scope_stack as _sync_thread_scope_stack

_scope_stack_var: contextvars.ContextVar[_ScopeStack] = contextvars.ContextVar("nat_nexus_scope_stack")


def get_scope_stack() -> _ScopeStack:
    """Get the current task's scope stack, creating one if needed.

    Also syncs the scope stack to the Rust-side thread-local storage so that
    native API calls on this thread use the same scope stack.

    .. note::
        This uses ``sync_thread_scope_stack`` (not ``set_thread_scope_stack``)
        so the internal sync does not set the ``scope_stack_active()`` flag.
        Only explicit user calls to ``set_thread_scope_stack()`` mark the
        thread as having an active scope stack.
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
    """Return whether the current context has an explicitly-initialized scope stack.

    Returns ``True`` when:
    - The Python-side ``contextvars.ContextVar`` has been set (e.g. via
      ``get_scope_stack()``), **or**
    - The Rust-side thread-local has been explicitly set via
      ``set_thread_scope_stack()``.

    Returns ``False`` when only the auto-created default is present.

    This replaces the ``nat_nexus._scope_stack_var.get(None) is not None``
    pattern used in integrations.
    """
    if _scope_stack_var.get(None) is not None:
        return True
    return _native_scope_stack_active()


def propagate_scope_to_thread() -> _ScopeStack:
    """Capture the current scope stack for propagation to a worker thread.

    Returns the current ``ScopeStack`` handle. Call
    ``set_thread_scope_stack()`` with the returned value inside the worker
    thread before making any Nexus API calls.

    Example::

        stack = nat_nexus.propagate_scope_to_thread()

        def worker():
            nat_nexus.set_thread_scope_stack(stack)
            # All Nexus calls on this thread now use the captured stack
            ...

        with ThreadPoolExecutor() as pool:
            pool.submit(worker).result()

    Raises:
        RuntimeError: If no scope stack has been explicitly initialized in
            the current context (i.e., ``scope_stack_active()`` returns
            ``False``).
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


__all__ = [
    # Submodules
    "scope",
    "tools",
    "llm",
    "guardrails",
    "intercepts",
    "subscribers",
    "scope_local",
    "typed",
    "proxy",
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
    "EventType",
    "ScopeHandle",
    "ToolHandle",
    "LLMHandle",
    "LLMRequest",
    "Event",
    "AtifExporter",
    "OpenTelemetryConfig",
    "OpenTelemetrySubscriber",
]
