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
    nat_nexus_tool_call as call,
)
from nat_nexus._native import (
    nat_nexus_tool_call_end as call_end,
)
from nat_nexus._native import (
    nat_nexus_tool_call_execute as execute,
)
from nat_nexus._native import (
    nat_nexus_tool_conditional_execution as conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_tool_request_intercepts as request_intercepts,
)
from nat_nexus._native import (
    nat_nexus_tool_response_intercepts as response_intercepts,
)

__all__ = ["call", "call_end", "execute", "request_intercepts", "conditional_execution", "response_intercepts"]
