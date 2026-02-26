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
        to activate. ``fn`` is ``async (args: Any) -> Any`` — replacement
        execution function.

**LLM intercepts** — callback signatures:

    register_llm_request(name, priority, break_chain, fn)
        ``fn(request: LLMRequest) -> LLMRequest`` — transform LLM request.

    register_llm_response(name, priority, break_chain, fn)
        ``fn(response: Any) -> Any`` — transform LLM response.

    register_llm_stream_response(name, priority, break_chain, fn)
        ``fn(event: SseEvent) -> SseEvent`` — transform each SSE event.

    register_llm_execution(name, priority, conditional, fn)
        ``conditional(request: LLMRequest) -> bool``.
        ``fn`` is ``async (request: LLMRequest) -> Any``.

    register_llm_stream_execution(name, priority, conditional, fn)
        ``conditional(request: LLMRequest) -> bool``.
        ``fn`` is ``async (request: LLMRequest) -> AsyncIterator[str]``.

Each ``register_*`` has a corresponding ``deregister_*`` that takes the name
and returns ``True`` if an intercept was found and removed.

Example::

    import nvagentrt

    def add_auth_header(request):
        request.headers["Authorization"] = "Bearer ..."
        return request

    nvagentrt.intercepts.register_llm_request("auth", 1, False, add_auth_header)
"""

from nvagentrt._native import (
    nv_agentrt_deregister_llm_execution_intercept as deregister_llm_execution,
)
from nvagentrt._native import (
    nv_agentrt_deregister_llm_request_intercept as deregister_llm_request,
)
from nvagentrt._native import (
    nv_agentrt_deregister_llm_response_intercept as deregister_llm_response,
)
from nvagentrt._native import (
    nv_agentrt_deregister_llm_stream_execution_intercept as deregister_llm_stream_execution,
)
from nvagentrt._native import (
    nv_agentrt_deregister_llm_stream_response_intercept as deregister_llm_stream_response,
)
from nvagentrt._native import (
    nv_agentrt_deregister_tool_execution_intercept as deregister_tool_execution,
)
from nvagentrt._native import (
    nv_agentrt_deregister_tool_request_intercept as deregister_tool_request,
)
from nvagentrt._native import (
    nv_agentrt_deregister_tool_response_intercept as deregister_tool_response,
)
from nvagentrt._native import (
    nv_agentrt_register_llm_execution_intercept as register_llm_execution,
)
from nvagentrt._native import (
    # LLM intercepts
    nv_agentrt_register_llm_request_intercept as register_llm_request,
)
from nvagentrt._native import (
    nv_agentrt_register_llm_response_intercept as register_llm_response,
)
from nvagentrt._native import (
    nv_agentrt_register_llm_stream_execution_intercept as register_llm_stream_execution,
)
from nvagentrt._native import (
    nv_agentrt_register_llm_stream_response_intercept as register_llm_stream_response,
)
from nvagentrt._native import (
    nv_agentrt_register_tool_execution_intercept as register_tool_execution,
)
from nvagentrt._native import (
    # Tool intercepts
    nv_agentrt_register_tool_request_intercept as register_tool_request,
)
from nvagentrt._native import (
    nv_agentrt_register_tool_response_intercept as register_tool_response,
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
