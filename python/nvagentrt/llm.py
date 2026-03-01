# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM lifecycle operations.

Provides both manual and managed LLM-call workflows, including streaming.

Functions:
    call(name, request, *, handle=None, attributes=None, data=None, metadata=None, model_name=None)
        Begin an LLM call manually. Returns an ``LLMHandle`` that must later
        be passed to ``call_end``. Emits a ``Start`` event. The optional
        ``model_name`` identifies the LLM model and is propagated to events
        for ATIF trajectory export.

    call_end(handle, response, *, data=None, metadata=None)
        End a manual LLM call. Records the response and emits an ``End`` event.

    execute(name, request, func, *, handle=None, attributes=None, data=None, metadata=None, model_name=None)
        Execute an LLM call through the full middleware pipeline (request
        intercepts -> sanitize-request guardrails -> conditional-execution
        guardrails -> execution intercepts -> *func* -> response intercepts ->
        sanitize-response guardrails). Returns an awaitable of the final response.
        The optional ``model_name`` is propagated to events for ATIF trajectory export.

    stream_execute(name, request, func, collector, finalizer, *, handle=None,
            attributes=None, data=None, metadata=None, model_name=None)
        Like ``execute`` but the execution function returns an async iterator
        of SSE text chunks. The ``collector`` callable is invoked with each
        intercepted chunk (after stream response intercepts). The ``finalizer``
        callable is invoked once when the stream is exhausted and returns the
        aggregated response as a JSON-serializable value. Returns an awaitable
        ``LlmStream`` that can be iterated with ``async for``.
        Stream-response intercepts are applied to each SSE event in flight.

    request_intercepts(request)
        Run the registered LLM request intercept chain on the given request.
        Returns the transformed ``LLMRequest``.

    conditional_execution(request)
        Run the registered LLM conditional execution guardrail chain.
        Raises ``RuntimeError`` if any guardrail rejects.

    response_intercepts(response)
        Run the registered LLM response intercept chain on the given response.
        Returns the transformed response.

Example::

    import nvagentrt

    req = nvagentrt.LLMRequest("POST", "https://api.example.com/chat", {}, body)

    # Non-streaming
    resp = await nvagentrt.llm.execute("gpt-4", req, my_llm_fn)

    # Streaming with collector/finalizer
    chunks = []
    def collect(chunk: str) -> None:
        chunks.append(chunk)
    def finalize() -> dict:
        return {"content": "".join(chunks)}

    stream = await nvagentrt.llm.stream_execute(
        "gpt-4", req, my_stream_fn, collect, finalize,
    )
    async for chunk in stream:
        print(chunk, end="")
"""

from nvagentrt._native import (
    nvagentrt_llm_call as call,
)
from nvagentrt._native import (
    nvagentrt_llm_call_end as call_end,
)
from nvagentrt._native import (
    nvagentrt_llm_call_execute as execute,
)
from nvagentrt._native import (
    nvagentrt_llm_conditional_execution as conditional_execution,
)
from nvagentrt._native import (
    nvagentrt_llm_request_intercepts as request_intercepts,
)
from nvagentrt._native import (
    nvagentrt_llm_response_intercepts as response_intercepts,
)
from nvagentrt._native import (
    nvagentrt_llm_stream_call_execute as stream_execute,
)

__all__ = [
    "call",
    "call_end",
    "execute",
    "stream_execute",
    "request_intercepts",
    "conditional_execution",
    "response_intercepts",
]
