# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool lifecycle operations.

Provides both manual and managed tool-call workflows.

Functions:
    call(name, args, *, handle=None, attributes=None, data=None, metadata=None)
        Begin a tool call manually. Returns a ``ToolHandle`` that must later
        be passed to ``call_end``. Emits a ``Start`` event.

    call_end(handle, result, *, data=None, metadata=None)
        End a manual tool call. Records the result and emits an ``End`` event.

    execute(name, args, func, *, handle=None, attributes=None, data=None, metadata=None)
        Execute a tool call through the full middleware pipeline (request
        intercepts → sanitize-request guardrails → conditional-execution
        guardrails → execution intercepts → *func* → response intercepts →
        sanitize-response guardrails). Returns an awaitable of the final result.

Example::

    import nvagentrt

    # Managed execution (recommended)
    result = await nvagentrt.tools.execute("search", {"q": "hello"}, my_search)

    # Manual lifecycle
    handle = nvagentrt.tools.call("search", {"q": "hello"})
    result = my_search({"q": "hello"})
    nvagentrt.tools.call_end(handle, result)
"""

from nvagentrt._native import (
    nvagentrt_tool_call as call,
)
from nvagentrt._native import (
    nvagentrt_tool_call_end as call_end,
)
from nvagentrt._native import (
    nvagentrt_tool_call_execute as execute,
)

__all__ = ["call", "call_end", "execute"]
