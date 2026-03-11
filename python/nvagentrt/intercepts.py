# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intercept registration for tools and LLMs.

Intercepts transform requests/responses or replace execution functions entirely.
They are priority-ordered (ascending) and registered by name. Request and
response intercepts accept a ``break_chain`` flag — when ``True``, no
lower-priority intercepts run after this one.

**Tool intercepts** — callback signatures:

    register_tool_request(name, priority, break_chain, fn)
        ``fn(tool_name: str, args: Any) -> Any`` — transform tool arguments.

    register_tool_response(name, priority, break_chain, fn)
        ``fn(tool_name: str, result: Any) -> Any`` — transform tool result.

    register_tool_execution(name, priority, conditional, fn)
        ``conditional(tool_name: str, args: Any) -> bool`` — return ``True``
        to activate. ``fn`` is ``async (args: Any, next) -> Any`` — middleware
        intercept. Call ``await next(args)`` to invoke the next intercept or
        original implementation; skip calling ``next`` to short-circuit.

**LLM intercepts** — callback signatures:

    register_llm_request(name, priority, break_chain, fn)
        ``fn(native: Any) -> Any`` — transform the opaque native payload.

    register_llm_response(name, priority, break_chain, fn)
        ``fn(response: LLMResponse) -> LLMResponse`` — transform LLM response.

    register_llm_stream_response(name, priority, break_chain, fn)
        ``fn(chunk: Any) -> Any`` — transform each Json chunk.

    register_llm_execution(name, priority, conditional, fn)
        ``conditional(native: Any) -> bool``.
        ``fn`` is ``async (native: Any, next) -> Any`` — middleware
        intercept. Call ``await next(native)`` to continue the chain.

    register_llm_stream_execution(name, priority, conditional, fn)
        ``conditional(native: Any) -> bool``.
        ``fn`` is ``async (native: Any, next) -> AsyncIterator[Any]``
        — middleware intercept. Call ``await next(native)`` to continue.

Each ``register_*`` has a corresponding ``deregister_*`` that takes the name
and returns ``True`` if an intercept was found and removed.

Example::

    import nvagentrt

    def add_field(native):
        native["extra"] = "injected"
        return native

    nvagentrt.intercepts.register_llm_request("extra", 1, False, add_field)
"""

from nvagentrt._native import (
    nvagentrt_deregister_llm_execution_intercept as deregister_llm_execution,
)
from nvagentrt._native import (
    nvagentrt_deregister_llm_request_intercept as deregister_llm_request,
)
from nvagentrt._native import (
    nvagentrt_deregister_llm_response_intercept as deregister_llm_response,
)
from nvagentrt._native import (
    nvagentrt_deregister_llm_stream_execution_intercept as deregister_llm_stream_execution,
)
from nvagentrt._native import (
    nvagentrt_deregister_llm_stream_response_intercept as deregister_llm_stream_response,
)
from nvagentrt._native import (
    nvagentrt_deregister_tool_execution_intercept as deregister_tool_execution,
)
from nvagentrt._native import (
    nvagentrt_deregister_tool_request_intercept as deregister_tool_request,
)
from nvagentrt._native import (
    nvagentrt_deregister_tool_response_intercept as deregister_tool_response,
)
from nvagentrt._native import (
    nvagentrt_register_llm_execution_intercept as register_llm_execution,
)
from nvagentrt._native import (
    # LLM intercepts
    nvagentrt_register_llm_request_intercept as register_llm_request,
)
from nvagentrt._native import (
    nvagentrt_register_llm_response_intercept as register_llm_response,
)
from nvagentrt._native import (
    nvagentrt_register_llm_stream_execution_intercept as register_llm_stream_execution,
)
from nvagentrt._native import (
    nvagentrt_register_llm_stream_response_intercept as register_llm_stream_response,
)
from nvagentrt._native import (
    nvagentrt_register_tool_execution_intercept as register_tool_execution,
)
from nvagentrt._native import (
    # Tool intercepts
    nvagentrt_register_tool_request_intercept as register_tool_request,
)
from nvagentrt._native import (
    nvagentrt_register_tool_response_intercept as register_tool_response,
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
    "register_llm_response",
    "deregister_llm_response",
    "register_llm_stream_response",
    "deregister_llm_stream_response",
    "register_llm_execution",
    "deregister_llm_execution",
    "register_llm_stream_execution",
    "deregister_llm_stream_execution",
]
