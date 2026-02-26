"""NVAgentRT - Agent Runtime with scope/handle management, guardrails, and intercepts.

NVAgentRT provides execution scope management, lifecycle event tracking, and
configurable middleware pipelines (guardrails and intercepts) for tool and LLM
calls. The core is written in Rust; this package exposes the full API to Python.

**Quick start**::

    import nvagentrt

    # Push a scope for your agent
    handle = nvagentrt.scope.push("my-agent", nvagentrt.ScopeType.Agent)

    # Execute a tool through the middleware pipeline
    result = await nvagentrt.tools.execute("search", {"q": "hello"}, my_tool_fn)

    # Execute an LLM call through the middleware pipeline
    req = nvagentrt.LLMRequest("POST", "https://api.example.com/chat", {}, body)
    resp = await nvagentrt.llm.execute("gpt-4", req, my_llm_fn)

    # Register guardrails and intercepts
    nvagentrt.guardrails.register_tool_sanitize_request("pii-filter", 10, sanitizer)
    nvagentrt.intercepts.register_llm_request("auth-header", 1, False, add_auth)

    # Subscribe to lifecycle events
    nvagentrt.subscribers.register("logger", lambda event: print(event.name))

    nvagentrt.scope.pop(handle)

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
    LLMRequest, Event
"""

import contextvars

from nvagentrt import guardrails, intercepts, llm, scope, subscribers, tools
from nvagentrt._native import (
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
from nvagentrt._native import ScopeStack as _ScopeStack
from nvagentrt._native import create_scope_stack as _create_scope_stack

_scope_stack_var: contextvars.ContextVar[_ScopeStack] = contextvars.ContextVar("nvagentrt_scope_stack")


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
]
