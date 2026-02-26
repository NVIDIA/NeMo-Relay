"""LLM lifecycle operations.

Provides both manual and managed LLM-call workflows, including streaming.

Functions:
    call(name, request, *, handle=None, attributes=None, data=None, metadata=None)
        Begin an LLM call manually. Returns an ``LLMHandle`` that must later
        be passed to ``call_end``. Emits a ``Start`` event.

    call_end(handle, response, *, data=None, metadata=None)
        End a manual LLM call. Records the response and emits an ``End`` event.

    execute(name, request, func, *, handle=None, attributes=None, data=None, metadata=None)
        Execute an LLM call through the full middleware pipeline (request
        intercepts → sanitize-request guardrails → conditional-execution
        guardrails → execution intercepts → *func* → response intercepts →
        sanitize-response guardrails). Returns an awaitable of the final response.

    stream_execute(name, request, func, *, handle=None, attributes=None, data=None, metadata=None)
        Like ``execute`` but the execution function returns an async iterator
        of SSE text chunks. Returns an awaitable ``LlmStream`` that can be
        iterated with ``async for``. Stream-response intercepts are applied
        to each SSE event in flight.

Example::

    import nvagentrt

    req = nvagentrt.LLMRequest("POST", "https://api.example.com/chat", {}, body)

    # Non-streaming
    resp = await nvagentrt.llm.execute("gpt-4", req, my_llm_fn)

    # Streaming
    stream = await nvagentrt.llm.stream_execute("gpt-4", req, my_stream_fn)
    async for chunk in stream:
        print(chunk, end="")
"""

from nvagentrt._native import (
    nv_agentrt_llm_call as call,
)
from nvagentrt._native import (
    nv_agentrt_llm_call_end as call_end,
)
from nvagentrt._native import (
    nv_agentrt_llm_call_execute as execute,
)
from nvagentrt._native import (
    nv_agentrt_llm_stream_call_execute as stream_execute,
)

__all__ = ["call", "call_end", "execute", "stream_execute"]
