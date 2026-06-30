# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM lifecycle helpers for non-streaming and streaming calls.

This module is the LLM analogue of ``nemo_relay.tools``. It manages emitted
events, global middleware, optional request codecs for annotated intercepts,
and optional response codecs for structured end-event annotations.

Example::

    import nemo_relay

    request = nemo_relay.LLMRequest(
        {},
        {"messages": [{"role": "user", "content": "hello"}], "model": "demo-model"},
    )

    async def impl(req):
        return {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    result = await nemo_relay.llm.execute(
        "demo-provider",
        request,
        impl,
        response_codec=nemo_relay.codecs.OpenAIChatCodec(),
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from nemo_relay._native import (
    LLMRequest,
    LlmStream,
)
from nemo_relay._native import (
    llm_call as _native_llm_call,
)
from nemo_relay._native import (
    llm_call_end as _native_llm_call_end,
)
from nemo_relay._native import (
    llm_call_execute as _native_llm_call_execute,
)
from nemo_relay._native import (
    llm_conditional_execution as _native_llm_conditional_execution,
)
from nemo_relay._native import (
    llm_request_intercepts as _native_llm_request_intercepts,
)
from nemo_relay._native import (
    llm_stream_call_execute as _native_llm_stream_call_execute,
)

if TYPE_CHECKING:
    from nemo_relay import Json
    from nemo_relay._native import AnnotatedLLMResponse
    from nemo_relay.codecs import LlmCodec, LlmResponseCodec


def call(
    name: str,
    request: LLMRequest,
    *,
    handle=None,
    attributes=None,
    data=None,
    metadata=None,
    model_name: str | None = None,
    timestamp: datetime | None = None,
):
    """Start a manual LLM span and return its ``LLMHandle``.

    Args:
        name: Provider or logical call name recorded on emitted events.
        request: Raw ``LLMRequest`` to associate with the call.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native LLM attributes attached to the start event.
        data: Optional JSON application payload stored on the LLM handle.
        metadata: Optional JSON metadata recorded on the emitted start event.
        model_name: Optional normalized model name to record separately from the
            provider-specific request payload.
        timestamp: Optional timezone-aware ``datetime`` recorded as the handle
            start time and on the emitted start event. When omitted, the current
            runtime time is used.

    Returns:
        LLMHandle: Handle used to finish the manual span with ``call_end()``.

    Notes:
        This starts only the manual LLM lifecycle span. It applies
        sanitize-request guardrails to the emitted start-event payload but does
        not run request or execution intercepts. ``timestamp`` must be a
        timezone-aware ``datetime``; strings and naive datetimes are rejected.

    Example::

        import nemo_relay

        request = nemo_relay.LLMRequest({}, {"messages": [], "model": "demo-model"})
        handle = nemo_relay.llm.call(
            "demo-provider",
            request,
            handle=None,
            attributes=None,
            data={"attempt": 1},
            metadata={"path": "manual"},
            model_name="demo-model",
        )
        nemo_relay.llm.call_end(
            handle,
            {"ok": True},
            data={"cached": False},
            metadata={"status": "success"},
        )
    """
    return _native_llm_call(
        name,
        request,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        model_name=model_name,
        timestamp=timestamp,
    )


def call_end(
    handle,
    response,
    *,
    data=None,
    metadata=None,
    annotated_response: AnnotatedLLMResponse | Mapping[str, Json] | None = None,
    response_codec: LlmResponseCodec | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Finish a manual LLM span started by ``call()``.

    Args:
        handle: LLM handle returned by ``call()``.
        response: Raw JSON-compatible response to record on the end event.
        data: Optional JSON payload used when the sanitized ``response`` is JSON null.
        metadata: Optional JSON metadata recorded on the emitted end event.
        annotated_response: Optional normalized response annotation attached to
            the emitted end event. Accepts an ``AnnotatedLLMResponse`` returned
            by a codec, or a JSON-compatible mapping matching that schema.
        response_codec: Optional response codec used to derive
            ``annotated_response`` from the sanitized end-event payload for
            observability. Ignored when ``annotated_response`` is provided.
        timestamp: Optional timezone-aware ``datetime`` recorded on the emitted
            end event. When omitted, the runtime default end timestamp is used.

    Returns:
        None: This function returns after the end event has been recorded.

    Notes:
        ``call_end()`` applies sanitize-response guardrails to the emitted
        end-event payload. ``response_codec`` and ``annotated_response`` enrich
        observability output only and do not rewrite the recorded response.
        Response codec failures are raised after the end event is emitted
        without an annotation.
        ``timestamp`` must be a timezone-aware ``datetime``; strings and naive
        datetimes are rejected.
    """
    return _native_llm_call_end(
        handle,
        response,
        data=data,
        metadata=metadata,
        annotated_response=annotated_response,
        response_codec=response_codec,
        timestamp=timestamp,
    )


def execute(
    name: str,
    request: LLMRequest,
    func,
    *,
    handle=None,
    attributes=None,
    data=None,
    metadata=None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
):
    """Run an LLM call through the managed middleware pipeline.

    Pipeline order:

    1. LLM conditional-execution guardrails
    2. LLM request intercepts
    3. LLM sanitize-request guardrails for emitted start events
    4. LLM execution intercepts
    5. ``func(request)``
    6. LLM sanitize-response guardrails for emitted end events

    Args:
        name: Provider or logical call name recorded on emitted events.
        request: Raw ``LLMRequest`` passed through guardrails, intercepts, and
            then into ``func``.
        func: Provider callback invoked as ``func(request)`` after middleware
            has finished processing the request.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native LLM attributes attached to the start event.
        data: Optional JSON application payload stored on the managed LLM handle.
        metadata: Optional JSON metadata recorded on the emitted start event.
        model_name: Optional normalized model name to record separately from the
            provider-specific request payload.
        codec: Optional request codec used to provide
            ``AnnotatedLLMRequest`` values to request intercepts.
        response_codec: Optional response codec used to attach a normalized
            response to the emitted ``LLMEnd`` event for observability.

    Returns:
        Json: The raw JSON-compatible value returned by ``func`` or by an
        execution intercept.

    Notes:
        ``codec`` enables annotated request intercepts. ``response_codec``
        decodes the raw response for observability only and does not change the
        value returned to the caller.

    Example::

        import nemo_relay

        request = nemo_relay.LLMRequest(
            {},
            {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"},
        )

        async def impl(req):
            return {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "hello"}}]}

        result = await nemo_relay.llm.execute(
            "demo-provider",
            request,
            impl,
            handle=None,
            attributes=None,
            data={"path": "managed"},
            metadata={"request_id": "req-1"},
            model_name="demo-model",
            codec=None,
            response_codec=nemo_relay.codecs.OpenAIChatCodec(),
        )
    """
    return _native_llm_call_execute(
        name,
        request,
        func,
        handle=handle,
        attributes=attributes,
        data=data,
        metadata=metadata,
        model_name=model_name,
        codec=codec,
        response_codec=response_codec,
    )


def stream_execute(
    name: str,
    request: LLMRequest,
    func,
    collector,
    finalizer,
    *,
    handle=None,
    attributes=None,
    data=None,
    metadata=None,
    model_name: str | None = None,
    codec: LlmCodec | None = None,
    response_codec: LlmResponseCodec | None = None,
) -> LlmStream:
    """Run a streaming LLM call through the managed middleware pipeline.

    Args:
        name: Provider or logical call name recorded on emitted events.
        request: Raw ``LLMRequest`` passed through guardrails and intercepts.
        func: Provider callback invoked as ``func(request)`` that yields raw
            JSON chunks.
        collector: Callback invoked for each chunk after streaming intercepts
            run. It typically accumulates state for ``finalizer``.
        finalizer: Callback invoked after the stream completes to build the
            final JSON-compatible response recorded on the ``LLMEnd`` event.
        handle: Optional parent scope handle. When omitted, the current scope
            becomes the parent.
        attributes: Optional native LLM attributes attached to the start event.
        data: Optional JSON application payload stored on the managed LLM handle.
        metadata: Optional JSON metadata recorded on the emitted start event.
        model_name: Optional normalized model name to record separately from the
            provider-specific request payload.
        codec: Optional request codec used to provide
            ``AnnotatedLLMRequest`` values to request intercepts.
        response_codec: Optional response codec used to attach a normalized
            final response to the emitted ``LLMEnd`` event for observability.

    Returns:
        LlmStream: Async iterator that yields the streamed JSON chunks.

    Notes:
        ``collector`` observes the post-intercept chunk values. ``finalizer``
        runs once at stream completion and should return a representation of
        the full response, not the final chunk.

    Example::

        import nemo_relay

        request = nemo_relay.LLMRequest(
            {},
            {"messages": [{"role": "user", "content": "hi"}], "model": "demo-model"},
        )
        collected = []

        async def impl(req):
            yield {"token": "hel"}
            yield {"token": "lo"}

        def collect(chunk):
            collected.append(chunk)

        def finalize():
            return {"text": "".join(chunk["token"] for chunk in collected)}

        stream = await nemo_relay.llm.stream_execute(
            "demo-provider",
            request,
            impl,
            collect,
            finalize,
            handle=None,
            attributes=None,
            data={"path": "stream"},
            metadata={"request_id": "req-2"},
            model_name="demo-model",
            codec=None,
            response_codec=None,
        )
        async for chunk in stream:
            print(chunk)
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
        codec=codec,
        response_codec=response_codec,
    )


def request_intercepts(name, request):
    """Apply global LLM request intercepts to ``request``.

    Args:
        name: Provider or logical call name used when evaluating intercepts.
        request: Raw ``LLMRequest`` to pass through the registered request
            intercept chain.

    Returns:
        LLMRequestInterceptOutcome: The complete request, annotation, and
        pending-mark outcome produced by the intercept chain.

    Notes:
        This runs only the request-intercept chain. It does not execute
        guardrails, codecs, provider callbacks, or stream handling.
    """
    return _native_llm_request_intercepts(name, request)


def conditional_execution(request):
    """Run LLM conditional-execution guardrails for ``request``.

    Args:
        request: Raw ``LLMRequest`` to validate against registered
            conditional-execution guardrails.

    Returns:
        str | None: A rejection message if execution should be blocked,
        otherwise ``None``.

    Notes:
        This helper evaluates only conditional-execution guardrails and does
        not invoke request intercepts, codecs, or provider execution.
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
