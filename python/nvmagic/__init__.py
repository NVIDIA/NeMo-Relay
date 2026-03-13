# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVMagic - Agent Runtime with scope/handle management, guardrails, and intercepts.

NVMagic provides execution scope management, lifecycle event tracking, and
configurable middleware pipelines (guardrails and intercepts) for tool and LLM
calls. The core is written in Rust; this package exposes the full API to Python.

**Quick start**::

    import nvmagic

    # Push a scope for your agent
    handle = nvmagic.scope.push("my-agent", nvmagic.ScopeType.Agent)

    # Execute a tool through the middleware pipeline
    result = await nvmagic.tools.execute("search", {"q": "hello"}, my_tool_fn)

    # Execute an LLM call through the middleware pipeline
    native = {"messages": [{"role": "user", "content": "hello"}]}
    resp = await nvmagic.llm.execute("gpt-4", native, my_llm_fn)

    # Register guardrails and intercepts
    nvmagic.guardrails.register_tool_sanitize_request("pii-filter", 10, sanitizer)
    nvmagic.intercepts.register_llm_request("auth-header", 1, False, add_auth)

    # Subscribe to lifecycle events
    nvmagic.subscribers.register("logger", lambda event: print(event.name))

    nvmagic.scope.pop(handle)

Submodules:
    scope       - Scope stack operations (push, pop, get_handle, event).
    tools       - Tool call lifecycle (call, call_end, execute).
    llm         - LLM call lifecycle (call, call_end, execute, stream_execute).
    guardrails  - Guardrail registration for tools and LLMs.
    intercepts  - Intercept registration for tools and LLMs.
    subscribers - Event subscriber registration.

Types (available at top level):
    ScopeAttributes, ToolAttributes, LLMAttributes,
    ScopeType, EventType,
    ScopeHandle, ToolHandle, LLMHandle,
    LLMRequest, Event, AtifExporter
"""

import contextvars

from nvmagic import guardrails, intercepts, llm, scope, subscribers, tools, typed
from nvmagic._native import (
    # ATIF exporter
    AtifExporter,
    Event,
    EventType,
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    # Types (always at top level)
    ScopeAttributes,
    ScopeHandle,
    ScopeType,
    ToolAttributes,
    ToolHandle,
)
from nvmagic._native import ScopeStack as _ScopeStack
from nvmagic._native import create_scope_stack as _create_scope_stack

_scope_stack_var: contextvars.ContextVar[_ScopeStack] = contextvars.ContextVar("nvmagic_scope_stack")


def get_scope_stack() -> _ScopeStack:
    """Get the current task's scope stack, creating one if needed."""
    stack = _scope_stack_var.get(None)
    if stack is None:
        stack = _create_scope_stack()
        _scope_stack_var.set(stack)
    return stack


ScopeStack = _ScopeStack
create_scope_stack = _create_scope_stack

__all__ = [
    # Submodules
    "scope",
    "tools",
    "llm",
    "guardrails",
    "intercepts",
    "subscribers",
    "typed",
    # Scope stack isolation
    "ScopeStack",
    "create_scope_stack",
    "get_scope_stack",
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
]
