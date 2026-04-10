# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global guardrail registration for tools and LLMs.

Guardrails can either sanitize data recorded on lifecycle events or reject a
call before it runs.

In managed ``tools.execute()`` and ``llm.execute()`` flows, sanitize
guardrails are observability-only: they change the payload written to emitted
events, not the value passed to the user callback or returned to the caller.

Example:
    ```python
    import nemo_flow

    def redact(tool_name, args):
        return {**args, "api_key": "***"}

    nemo_flow.guardrails.register_tool_sanitize_request("redact", 10, redact)
    ```
"""

from nemo_flow import (
    LlmConditionalExecutionGuardrail,
    LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail,
    ToolConditionalExecutionGuardrail,
    ToolSanitizeGuardrail,
)
from nemo_flow._native import (
    deregister_llm_conditional_execution_guardrail as _native_deregister_llm_conditional_execution,
)
from nemo_flow._native import (
    deregister_llm_sanitize_request_guardrail as _native_deregister_llm_sanitize_request,
)
from nemo_flow._native import (
    deregister_llm_sanitize_response_guardrail as _native_deregister_llm_sanitize_response,
)
from nemo_flow._native import (
    deregister_tool_conditional_execution_guardrail as _native_deregister_tool_conditional_execution,
)
from nemo_flow._native import (
    deregister_tool_sanitize_request_guardrail as _native_deregister_tool_sanitize_request,
)
from nemo_flow._native import (
    deregister_tool_sanitize_response_guardrail as _native_deregister_tool_sanitize_response,
)
from nemo_flow._native import (
    register_llm_conditional_execution_guardrail as _native_register_llm_conditional_execution,
)
from nemo_flow._native import (
    register_llm_sanitize_request_guardrail as _native_register_llm_sanitize_request,
)
from nemo_flow._native import (
    register_llm_sanitize_response_guardrail as _native_register_llm_sanitize_response,
)
from nemo_flow._native import (
    register_tool_conditional_execution_guardrail as _native_register_tool_conditional_execution,
)
from nemo_flow._native import (
    register_tool_sanitize_request_guardrail as _native_register_tool_sanitize_request,
)
from nemo_flow._native import (
    register_tool_sanitize_response_guardrail as _native_register_tool_sanitize_response,
)

# ---------------------------------------------------------------------------
# Tool guardrails
# ---------------------------------------------------------------------------


def register_tool_sanitize_request(name: str, priority: int, guardrail: ToolSanitizeGuardrail) -> None:
    """Register a guardrail that sanitizes tool inputs for emitted start events.

    Args:
        name: Unique guardrail name used for later replacement or removal.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(tool_name, args)`` that must
            return the sanitized payload to record on the emitted start event.

    Notes:
        In managed ``nemo_flow.tools.execute()`` flows, sanitize guardrails are
        observability-only. They change the payload written to events, not the
        arguments passed to the tool callback.

    Example:
        ```python
        import nemo_flow

        def redact(tool_name, args):
            return {**args, "api_key": "***"}

        nemo_flow.guardrails.register_tool_sanitize_request("redact", 10, redact)
        ```
    """
    return _native_register_tool_sanitize_request(name, priority, guardrail)


def deregister_tool_sanitize_request(name: str) -> bool:
    """Remove a previously registered tool sanitize-request guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_tool_sanitize_request()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    return _native_deregister_tool_sanitize_request(name)


def register_tool_sanitize_response(name: str, priority: int, guardrail: ToolSanitizeGuardrail) -> None:
    """Register a guardrail that sanitizes tool outputs for emitted end events.

    Args:
        name: Unique guardrail name used for later replacement or removal.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(tool_name, result)`` that
            must return the sanitized payload to record on emitted end events.

    Notes:
        This guardrail affects event payloads only. The caller still receives
        the original tool result.
    """
    return _native_register_tool_sanitize_response(name, priority, guardrail)


def deregister_tool_sanitize_response(name: str) -> bool:
    """Remove a previously registered tool sanitize-response guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_tool_sanitize_response()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    return _native_deregister_tool_sanitize_response(name)


def register_tool_conditional_execution(name: str, priority: int, guardrail: ToolConditionalExecutionGuardrail) -> None:
    """Register a guardrail that can reject a tool call before execution.

    Args:
        name: Unique guardrail name used for later replacement or removal.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(tool_name, args)``. Return
            ``None`` to allow execution or a rejection message to block it.

    Notes:
        Conditional-execution guardrails run before request intercepts and
        before the tool callback is invoked.
    """
    return _native_register_tool_conditional_execution(name, priority, guardrail)


def deregister_tool_conditional_execution(name: str) -> bool:
    """Remove a previously registered tool conditional-execution guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_tool_conditional_execution()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    return _native_deregister_tool_conditional_execution(name)


# ---------------------------------------------------------------------------
# LLM guardrails
# ---------------------------------------------------------------------------


def register_llm_sanitize_request(name: str, priority: int, guardrail: LlmSanitizeRequestGuardrail) -> None:
    """Register a guardrail that sanitizes LLM requests for emitted start events.

    Args:
        name: Unique guardrail name used for later replacement or removal.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(request)`` that must return
            the sanitized request recorded on the emitted start event.

    Notes:
        In managed ``nemo_flow.llm.execute()`` and
        ``nemo_flow.llm.stream_execute()`` flows, this is observability-only
        and does not mutate the request forwarded to the provider callback.

    Example:
        ```python
        import nemo_flow

        def strip_auth(request):
            headers = {k: v for k, v in request.headers.items() if k.lower() != "authorization"}
            return nemo_flow.LLMRequest(headers, request.content)

        nemo_flow.guardrails.register_llm_sanitize_request("strip-auth", 10, strip_auth)
        ```
    """
    return _native_register_llm_sanitize_request(name, priority, guardrail)


def deregister_llm_sanitize_request(name: str) -> bool:
    """Remove a previously registered LLM sanitize-request guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_llm_sanitize_request()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    return _native_deregister_llm_sanitize_request(name)


def register_llm_sanitize_response(name: str, priority: int, guardrail: LlmSanitizeResponseGuardrail) -> None:
    """Register a guardrail that sanitizes LLM outputs for emitted end events.

    Args:
        name: Unique guardrail name used for later replacement or removal.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(response)`` that must return
            the sanitized payload recorded on the emitted end event.

    Notes:
        This guardrail changes only event payloads. The raw provider response
        returned to the caller is left unchanged.
    """
    return _native_register_llm_sanitize_response(name, priority, guardrail)


def deregister_llm_sanitize_response(name: str) -> bool:
    """Remove a previously registered LLM sanitize-response guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_llm_sanitize_response()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    return _native_deregister_llm_sanitize_response(name)


def register_llm_conditional_execution(name: str, priority: int, guardrail: LlmConditionalExecutionGuardrail) -> None:
    """Register a guardrail that can reject an LLM call before execution.

    Args:
        name: Unique guardrail name used for later replacement or removal.
        priority: Execution order for the guardrail. Lower values run first.
        guardrail: Callable invoked as ``guardrail(request)``. Return ``None``
            to allow execution or a rejection message to block the call.

    Notes:
        Conditional-execution guardrails run before request intercepts, codecs,
        and provider execution.
    """
    return _native_register_llm_conditional_execution(name, priority, guardrail)


def deregister_llm_conditional_execution(name: str) -> bool:
    """Remove a previously registered LLM conditional-execution guardrail.

    Args:
        name: Guardrail name previously passed to
            ``register_llm_conditional_execution()``.

    Returns:
        bool: ``True`` if a guardrail was removed, otherwise ``False``.
    """
    return _native_deregister_llm_conditional_execution(name)


__all__ = [
    "register_tool_sanitize_request",
    "deregister_tool_sanitize_request",
    "register_tool_sanitize_response",
    "deregister_tool_sanitize_response",
    "register_tool_conditional_execution",
    "deregister_tool_conditional_execution",
    "register_llm_sanitize_request",
    "deregister_llm_sanitize_request",
    "register_llm_sanitize_response",
    "deregister_llm_sanitize_response",
    "register_llm_conditional_execution",
    "deregister_llm_conditional_execution",
]
