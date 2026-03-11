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

    import nvmagic

    # Managed execution (recommended)
    result = await nvmagic.tools.execute("search", {"q": "hello"}, my_search)

    # Manual lifecycle
    handle = nvmagic.tools.call("search", {"q": "hello"})
    result = my_search({"q": "hello"})
    nvmagic.tools.call_end(handle, result)
"""

from nvmagic._native import (
    nvmagic_tool_call as call,
)
from nvmagic._native import (
    nvmagic_tool_call_end as call_end,
)
from nvmagic._native import (
    nvmagic_tool_call_execute as execute,
)
from nvmagic._native import (
    nvmagic_tool_conditional_execution as conditional_execution,
)
from nvmagic._native import (
    nvmagic_tool_request_intercepts as request_intercepts,
)
from nvmagic._native import (
    nvmagic_tool_response_intercepts as response_intercepts,
)

__all__ = ["call", "call_end", "execute", "request_intercepts", "conditional_execution", "response_intercepts"]
