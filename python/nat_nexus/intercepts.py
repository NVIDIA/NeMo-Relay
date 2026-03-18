# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intercept registration for tools and LLMs.

Intercepts transform requests or replace execution functions entirely.
They are priority-ordered (ascending) and registered by name. Request
intercepts accept a ``break_chain`` flag — when ``True``, no
lower-priority intercepts run after this one.

**Tool intercepts** — callback signatures:

    register_tool_request(name, priority, break_chain, fn)
        ``fn(tool_name: str, args: Any) -> Any`` — transform tool arguments.

    register_tool_response(name, priority, break_chain, fn)
        ``fn(tool_name: str, result: Any) -> Any`` — transform tool result.

    register_tool_execution(name, priority, fn)
        ``fn`` is ``async (args: Any, next) -> Any`` — middleware
        intercept. Call ``await next(args)`` to invoke the next intercept or
        original implementation; skip calling ``next`` to short-circuit.

**LLM intercepts** — callback signatures:

    register_llm_request(name, priority, break_chain, fn)
        ``fn(request: LLMRequest) -> LLMRequest`` — transform the LLM request.

    register_llm_execution(name, priority, fn)
        ``fn`` is ``async (request: LLMRequest, next) -> Any`` — middleware
        intercept. Call ``await next(request)`` to continue the chain.

    register_llm_stream_execution(name, priority, fn)
        ``fn`` is ``async (request: LLMRequest, next) -> AsyncIterator[Any]``
        — middleware intercept. Call ``await next(request)`` to continue.

Each ``register_*`` has a corresponding ``deregister_*`` that takes the name
and returns ``True`` if an intercept was found and removed.

Example::

    import nat_nexus
    from nat_nexus import LLMRequest

    def add_header(request):
        request.headers["X-Extra"] = "injected"
        return request

    nat_nexus.intercepts.register_llm_request("extra", 1, False, add_header)
"""

from nat_nexus._native import (
    nat_nexus_deregister_llm_execution_intercept as deregister_llm_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_llm_request_intercept as deregister_llm_request,
)
from nat_nexus._native import (
    nat_nexus_deregister_llm_stream_execution_intercept as deregister_llm_stream_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_execution_intercept as deregister_tool_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_request_intercept as deregister_tool_request,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_response_intercept as deregister_tool_response,
)
from nat_nexus._native import (
    nat_nexus_register_llm_execution_intercept as register_llm_execution,
)
from nat_nexus._native import (
    # LLM intercepts
    nat_nexus_register_llm_request_intercept as register_llm_request,
)
from nat_nexus._native import (
    nat_nexus_register_llm_stream_execution_intercept as register_llm_stream_execution,
)
from nat_nexus._native import (
    nat_nexus_register_tool_execution_intercept as register_tool_execution,
)
from nat_nexus._native import (
    # Tool intercepts
    nat_nexus_register_tool_request_intercept as register_tool_request,
)
from nat_nexus._native import (
    nat_nexus_register_tool_response_intercept as register_tool_response,
)

__all__ = [
    "register_tool_request",
    "deregister_tool_request",
    "register_tool_response",
    "deregister_tool_response",
    "register_tool_execution",
    "deregister_tool_execution",
    "register_llm_request",
    "deregister_llm_request",
    "register_llm_execution",
    "deregister_llm_execution",
    "register_llm_stream_execution",
    "deregister_llm_stream_execution",
]
