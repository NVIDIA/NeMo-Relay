# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM lifecycle operations.

Provides both manual and managed LLM-call workflows, including streaming.

Functions:
    call(name, native, *, handle=None, attributes=None, data=None, metadata=None, model_name=None)
        Begin an LLM call manually.

        Returns an ``LLMHandle`` that must later be passed to ``call_end``. Emits a ``Start`` event.
        The optional ``model_name`` identifies the LLM model and is propagated to events for ATIF trajectory export.

    call_end(handle, response, *, data=None, metadata=None)
        End a manual LLM call. Records the response and emits an ``End`` event.

    execute(name, native, func, *, handle=None, attributes=None, data=None, metadata=None, model_name=None)
        Execute an LLM call through the full middleware pipeline:

        - conditional-execution guardrails (on formal request derived via converter)
        - request intercepts (on opaque native Json)
        - sanitize-request guardrails (on formal request)
        - execution intercepts
        - *func*
        - sanitize-response guardrails

        On rejection, only a standalone Mark event is emitted (no Start/End
        pair) and ``GuardrailRejected`` is raised.

        Returns an awaitable of the final response. The optional ``model_name`` is propagated to events
        for ATIF trajectory export.

    stream_execute(name, native, func, collector, finalizer, *, handle=None,
            attributes=None, data=None, metadata=None, model_name=None)
        Like ``execute``, conditional-execution guardrails run first on the
        formal request derived via the converter. The execution function returns
        an async iterator of Json chunks.

        The ``collector`` callable is invoked with each Json chunk.

        The ``finalizer`` callable is invoked once when the stream is exhausted and returns the
        aggregated response as a JSON-serializable value.

        Returns an awaitable ``LlmStream`` that can be iterated with ``async for``.

    request_intercepts(native)
        Run the registered LLM request intercept chain on the given native payload.
        Returns the transformed native Json.

    conditional_execution(native)
        Run the registered LLM conditional execution guardrail chain.
        Raises ``RuntimeError`` if any guardrail rejects.

Example::

    import nvmagic

    native = {"messages": [{"role": "user", "content": "hello"}]}

    # Non-streaming
    resp = await nvmagic.llm.execute("gpt-4", native, my_llm_fn)

    # Streaming with collector/finalizer
    chunks = []
    def collect(chunk) -> None:
        chunks.append(chunk)
    def finalize() -> dict:
        return {"chunks": chunks}

    stream = await nvmagic.llm.stream_execute(
        "gpt-4", native, my_stream_fn, collect, finalize,
    )
    async for chunk in stream:
        process(chunk)
"""

from nvmagic._native import (
    nvmagic_llm_call as call,
)
from nvmagic._native import (
    nvmagic_llm_call_end as call_end,
)
from nvmagic._native import (
    nvmagic_llm_call_execute as execute,
)
from nvmagic._native import (
    nvmagic_llm_conditional_execution as conditional_execution,
)
from nvmagic._native import (
    nvmagic_llm_request_intercepts as request_intercepts,
)
from nvmagic._native import (
    nvmagic_llm_stream_call_execute as stream_execute,
)

__all__ = [
    "call",
    "call_end",
    "execute",
    "stream_execute",
    "request_intercepts",
    "conditional_execution",
]
