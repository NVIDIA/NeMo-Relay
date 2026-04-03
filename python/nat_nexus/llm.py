# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM lifecycle operations.

Provides both manual and managed LLM-call workflows, including streaming.

Functions:
    call(name, request, *, handle=None, attributes=None, data=None, metadata=None, model_name=None)
        Begin an LLM call manually.

        Returns an ``LLMHandle`` that must later be passed to ``call_end``. Emits a ``Start`` event.
        The optional ``model_name`` identifies the LLM model and is propagated to events for ATIF trajectory export.

    call_end(handle, response, *, data=None, metadata=None)
        End a manual LLM call. Records the response and emits an ``End`` event.

    execute(name, request, func, *, handle=None, attributes=None, data=None, metadata=None, model_name=None)
        Execute an LLM call through the full middleware pipeline:

        - conditional-execution guardrails (on ``LLMRequest``)
        - request intercepts (on ``LLMRequest``)
        - sanitize-request guardrails (for the emitted ``Start`` event payload)
        - execution intercepts
        - *func*
        - sanitize-response guardrails (for the emitted ``End`` event payload)

        On rejection, only a standalone Mark event is emitted (no Start/End
        pair) and ``GuardrailRejected`` is raised.

        Returns an awaitable of the final response. The optional ``model_name`` is propagated to events
        for ATIF trajectory export.

    stream_execute(name, request, func, collector, finalizer, *, handle=None,
            attributes=None, data=None, metadata=None, model_name=None)
        Like ``execute``, conditional-execution guardrails run first on the
        ``LLMRequest``. The execution function returns an async iterator of Json chunks.

        The ``collector`` callable is invoked with each Json chunk.

        The ``finalizer`` callable is invoked once when the stream is exhausted and returns the
        aggregated response as a JSON-serializable value.

        Returns an awaitable ``LlmStream`` that can be iterated with ``async for``.

    request_intercepts(request)
        Run the registered LLM request intercept chain on the given ``LLMRequest``.
        Returns the transformed ``LLMRequest``.

    conditional_execution(request)
        Run the registered LLM conditional execution guardrail chain.
        Raises ``RuntimeError`` if any guardrail rejects.

Example::

    import nat_nexus
    from nat_nexus import LLMRequest

    request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})

    # Non-streaming
    resp = await nat_nexus.llm.execute("gpt-4", request, my_llm_fn)

    # Streaming with collector/finalizer
    chunks = []
    def collect(chunk) -> None:
        chunks.append(chunk)
    def finalize() -> dict:
        return {"chunks": chunks}

    stream = await nat_nexus.llm.stream_execute(
        "gpt-4", request, my_stream_fn, collect, finalize,
    )
    async for chunk in stream:
        process(chunk)
"""

from nat_nexus._native import (
    nat_nexus_llm_call as _native_llm_call,
)
from nat_nexus._native import (
    nat_nexus_llm_call_end as _native_llm_call_end,
)
from nat_nexus._native import (
    nat_nexus_llm_call_execute as _native_llm_call_execute,
)
from nat_nexus._native import (
    nat_nexus_llm_conditional_execution as _native_llm_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_llm_request_intercepts as _native_llm_request_intercepts,
)
from nat_nexus._native import (
    nat_nexus_llm_stream_call_execute as _native_llm_stream_call_execute,
)


def call(name, request, *, handle=None, attributes=None, data=None, metadata=None, model_name=None):
    """Begin an LLM call manually.

    Emits a ``Start`` event and returns an ``LLMHandle`` that must later be
    passed to ``call_end()`` to complete the LLM call lifecycle.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.
        model_name: Optional LLM model identifier propagated to events for ATIF export.

    Returns:
        An ``LLMHandle`` for use with ``call_end()``.
    """
    return _native_llm_call(
        name, request, handle=handle, attributes=attributes, data=data, metadata=metadata, model_name=model_name
    )


def call_end(handle, response, *, data=None, metadata=None):
    """End a manual LLM call.

    Records the response and emits an ``End`` event.

    Args:
        handle: The ``LLMHandle`` returned by ``call()``.
        response: JSON-serializable LLM response.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.
    """
    return _native_llm_call_end(handle, response, data=data, metadata=metadata)


def execute(name, request, func, *, handle=None, attributes=None, data=None, metadata=None, model_name=None):
    """Execute an LLM call through the full middleware pipeline.

    Runs conditional-execution guardrails -> request intercepts ->
    sanitize-request guardrails for the emitted ``Start`` event ->
    execution intercepts -> *func* -> sanitize-response guardrails for the
    emitted ``End`` event. On rejection, only a standalone ``Mark`` event is
    emitted (no ``Start``/``End`` pair) and ``GuardrailRejected`` is raised.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        func: Async callable ``(LLMRequest) -> response`` that performs the LLM call.
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.
        model_name: Optional LLM model identifier propagated to events for ATIF export.

    Returns:
        An awaitable that resolves to the execution response after intercepts.
        Sanitize guardrails only affect recorded event payloads.

    Raises:
        GuardrailRejected: If a conditional-execution guardrail rejects the call.
    """
    return _native_llm_call_execute(
        name, request, func, handle=handle, attributes=attributes, data=data, metadata=metadata, model_name=model_name
    )


def stream_execute(
    name,
    request,
    func,
    collector,
    finalizer,
    *,
    handle=None,
    attributes=None,
    data=None,
    metadata=None,
    model_name=None,
):
    """Execute a streaming LLM call through the full middleware pipeline.

    Like ``execute()``, conditional-execution guardrails run first on the
    ``LLMRequest``. The execution function returns an async iterator of Json
    chunks. The ``collector`` callable is invoked with each chunk. The
    ``finalizer`` callable is invoked once when the stream is exhausted and
    returns the aggregated response used for the emitted ``End`` event.

    Args:
        name: Model/provider name.
        request: An ``LLMRequest`` object with headers and content.
        func: Async callable ``(LLMRequest) -> AsyncIterator[Json]`` returning chunks.
        collector: Callable ``(chunk: Json) -> None`` invoked with each intercepted chunk.
        finalizer: Callable ``() -> Any`` invoked once when the stream ends; returns
            the aggregated response.
        handle: Optional parent scope handle. Defaults to the current top of stack.
        attributes: Optional ``LLMAttributes`` bitflags.
        data: Optional JSON-serializable application data payload.
        metadata: Optional JSON-serializable metadata payload.
        model_name: Optional LLM model identifier propagated to events for ATIF export.

    Returns:
        An awaitable ``LlmStream`` async iterator of Json chunks.

    Raises:
        GuardrailRejected: If a conditional-execution guardrail rejects the call.
    """
    return _native_llm_stream_call_execute(
        name,
        request,
        func,
        collector,
        finalizer,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        model_name=model_name,
    )


def request_intercepts(name, request):
    """Run the registered LLM request intercept chain.

    Applies all registered LLM request intercepts in priority order to
    the given request.

    Args:
        name: LLM name identifier.
        request: An ``LLMRequest`` object to transform.

    Returns:
        The transformed ``LLMRequest`` after all intercepts have been applied.
    """
    return _native_llm_request_intercepts(name, request)


def conditional_execution(request):
    """Run the registered LLM conditional-execution guardrail chain.

    Evaluates all registered conditional-execution guardrails in priority
    order against the given request.

    Args:
        request: An ``LLMRequest`` object to evaluate.

    Raises:
        RuntimeError: If any guardrail rejects the LLM call.
    """
    return _native_llm_conditional_execution(request)


__all__ = [
    "call",
    "call_end",
    "execute",
    "stream_execute",
    "request_intercepts",
    "conditional_execution",
]
