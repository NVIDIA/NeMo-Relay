# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrail registration for tools and LLMs.

Guardrails run inside the middleware pipeline and can sanitize or gate requests
and responses. They are priority-ordered (ascending) and registered by name.

Sanitize guardrails are currently observability-oriented in the managed
``execute`` APIs: they affect the payload captured on lifecycle events, but
they do not rewrite the values passed to the user callable or returned to the
caller.

**Tool guardrails** — callback signatures:

    register_tool_sanitize_request(name, priority, fn)
        ``fn(tool_name: str, args: Any) -> Any`` — sanitize tool arguments.

    register_tool_sanitize_response(name, priority, fn)
        ``fn(tool_name: str, result: Any) -> Any`` — sanitize tool result.

    register_tool_conditional_execution(name, priority, fn)
        ``fn(tool_name: str, args: Any) -> Optional[str]`` — return ``None``
        to allow, or a rejection reason to block.

**LLM guardrails** — callback signatures:

    register_llm_sanitize_request(name, priority, fn)
        ``fn(request: LLMRequest) -> LLMRequest`` — sanitize the LLM request.

    register_llm_sanitize_response(name, priority, fn)
        ``fn(response: dict) -> dict`` — sanitize the LLM response.

    register_llm_conditional_execution(name, priority, fn)
        ``fn(request: LLMRequest) -> Optional[str]`` — return ``None``
        to allow, or a rejection reason to block.

Each ``register_*`` has a corresponding ``deregister_*`` that takes the name
and returns ``True`` if a guardrail was found and removed.

Example::

    import nat_nexus

    def redact_pii(tool_name, args):
        # Sanitize PII from tool arguments
        return {k: "***" if "ssn" in k else v for k, v in args.items()}

    nat_nexus.guardrails.register_tool_sanitize_request("pii-filter", 10, redact_pii)
"""

from typing import Any, Callable, Optional

from nat_nexus._native import LLMRequest
from nat_nexus._native import (
    nat_nexus_deregister_llm_conditional_execution_guardrail as _native_deregister_llm_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_llm_sanitize_request_guardrail as _native_deregister_llm_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_deregister_llm_sanitize_response_guardrail as _native_deregister_llm_sanitize_response,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_conditional_execution_guardrail as _native_deregister_tool_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_sanitize_request_guardrail as _native_deregister_tool_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_deregister_tool_sanitize_response_guardrail as _native_deregister_tool_sanitize_response,
)
from nat_nexus._native import (
    nat_nexus_register_llm_conditional_execution_guardrail as _native_register_llm_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_register_llm_sanitize_request_guardrail as _native_register_llm_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_register_llm_sanitize_response_guardrail as _native_register_llm_sanitize_response,
)
from nat_nexus._native import (
    nat_nexus_register_tool_conditional_execution_guardrail as _native_register_tool_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_register_tool_sanitize_request_guardrail as _native_register_tool_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_register_tool_sanitize_response_guardrail as _native_register_tool_sanitize_response,
)

# ---------------------------------------------------------------------------
# Tool guardrails
# ---------------------------------------------------------------------------

ToolSanitizeGuardrail = Callable[[str, Any], Any]
ToolConditionalExecutionGuardrail = Callable[[str, Any], Optional[str]]
LlmSanitizeRequestGuardrail = Callable[[LLMRequest], LLMRequest]
LlmSanitizeResponseGuardrail = Callable[[dict], dict]
LlmConditionalExecutionGuardrail = Callable[[LLMRequest], Optional[str]]


def register_tool_sanitize_request(name: str, priority: int, guardrail: ToolSanitizeGuardrail) -> None:
    """Register a tool sanitize-request guardrail.

    The guardrail callback receives the tool name and arguments and returns
    sanitized arguments for the emitted ``Start`` event payload.

    Args:
        name: Unique guardrail name.
        priority: Priority (ascending order; lower runs first).
        guardrail: Callable ``(tool_name: str, args: Any) -> Any``.

    Raises:
        RuntimeError: If a guardrail with this name already exists.
    """
    return _native_register_tool_sanitize_request(name, priority, guardrail)


def deregister_tool_sanitize_request(name: str) -> bool:
    """Remove a tool sanitize-request guardrail.

    Args:
        name: Name of the guardrail to remove.

    Returns:
        ``True`` if a guardrail with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_tool_sanitize_request(name)


def register_tool_sanitize_response(name: str, priority: int, guardrail: ToolSanitizeGuardrail) -> None:
    """Register a tool sanitize-response guardrail.

    The guardrail callback receives the tool name and result and returns
    a sanitized result for the emitted ``End`` event payload.

    Args:
        name: Unique guardrail name.
        priority: Priority (ascending order; lower runs first).
        guardrail: Callable ``(tool_name: str, result: Any) -> Any``.

    Raises:
        RuntimeError: If a guardrail with this name already exists.
    """
    return _native_register_tool_sanitize_response(name, priority, guardrail)


def deregister_tool_sanitize_response(name: str) -> bool:
    """Remove a tool sanitize-response guardrail.

    Args:
        name: Name of the guardrail to remove.

    Returns:
        ``True`` if a guardrail with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_tool_sanitize_response(name)


def register_tool_conditional_execution(name: str, priority: int, guardrail: ToolConditionalExecutionGuardrail) -> None:
    """Register a tool conditional-execution guardrail.

    The guardrail callback receives the tool name and arguments and returns
    ``None`` to allow execution or a rejection reason string to block it.

    Args:
        name: Unique guardrail name.
        priority: Priority (ascending order; lower runs first).
        guardrail: Callable ``(tool_name: str, args: Any) -> Optional[str]``.

    Raises:
        RuntimeError: If a guardrail with this name already exists.
    """
    return _native_register_tool_conditional_execution(name, priority, guardrail)


def deregister_tool_conditional_execution(name: str) -> bool:
    """Remove a tool conditional-execution guardrail.

    Args:
        name: Name of the guardrail to remove.

    Returns:
        ``True`` if a guardrail with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_tool_conditional_execution(name)


# ---------------------------------------------------------------------------
# LLM guardrails
# ---------------------------------------------------------------------------


def register_llm_sanitize_request(name: str, priority: int, guardrail: LlmSanitizeRequestGuardrail) -> None:
    """Register an LLM sanitize-request guardrail.

    The guardrail callback receives the LLM request and returns a sanitized
    ``LLMRequest`` for the emitted ``Start`` event payload.

    Args:
        name: Unique guardrail name.
        priority: Priority (ascending order; lower runs first).
        guardrail: Callable ``(request: LLMRequest) -> LLMRequest``.

    Raises:
        RuntimeError: If a guardrail with this name already exists.
    """
    return _native_register_llm_sanitize_request(name, priority, guardrail)


def deregister_llm_sanitize_request(name: str) -> bool:
    """Remove an LLM sanitize-request guardrail.

    Args:
        name: Name of the guardrail to remove.

    Returns:
        ``True`` if a guardrail with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_llm_sanitize_request(name)


def register_llm_sanitize_response(name: str, priority: int, guardrail: LlmSanitizeResponseGuardrail) -> None:
    """Register an LLM sanitize-response guardrail.

    The guardrail callback receives the LLM response dict and returns a
    sanitized dict for the emitted ``End`` event payload.

    Args:
        name: Unique guardrail name.
        priority: Priority (ascending order; lower runs first).
        guardrail: Callable ``(response: dict) -> dict``.

    Raises:
        RuntimeError: If a guardrail with this name already exists.
    """
    return _native_register_llm_sanitize_response(name, priority, guardrail)


def deregister_llm_sanitize_response(name: str) -> bool:
    """Remove an LLM sanitize-response guardrail.

    Args:
        name: Name of the guardrail to remove.

    Returns:
        ``True`` if a guardrail with the given name was found and removed,
        ``False`` otherwise.
    """
    return _native_deregister_llm_sanitize_response(name)


def register_llm_conditional_execution(name: str, priority: int, guardrail: LlmConditionalExecutionGuardrail) -> None:
    """Register an LLM conditional-execution guardrail.

    The guardrail callback receives the LLM request and returns ``None``
    to allow execution or a rejection reason string to block it.

    Args:
        name: Unique guardrail name.
        priority: Priority (ascending order; lower runs first).
        guardrail: Callable ``(request: LLMRequest) -> Optional[str]``.

    Raises:
        RuntimeError: If a guardrail with this name already exists.
    """
    return _native_register_llm_conditional_execution(name, priority, guardrail)


def deregister_llm_conditional_execution(name: str) -> bool:
    """Remove an LLM conditional-execution guardrail.

    Args:
        name: Name of the guardrail to remove.

    Returns:
        ``True`` if a guardrail with the given name was found and removed,
        ``False`` otherwise.
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
