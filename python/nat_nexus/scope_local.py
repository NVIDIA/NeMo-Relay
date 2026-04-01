# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope-local guardrail, intercept, and subscriber registration.

Scope-local registrations are scoped to a particular scope handle and are
automatically cleaned up when that scope is popped. This allows middleware
to be registered per-request or per-agent without polluting the global
registry.

All ``register_*`` functions take a ``ScopeHandle`` as the first argument.
The scope handle's ``.uuid`` is used to look up the scope in the scope stack.

**Tool guardrails** (scope-local):

    register_tool_sanitize_request(scope_handle, name, priority, fn)
    register_tool_sanitize_response(scope_handle, name, priority, fn)
    register_tool_conditional_execution(scope_handle, name, priority, fn)

**Tool intercepts** (scope-local):

    register_tool_request(scope_handle, name, priority, break_chain, fn)
    register_tool_response(scope_handle, name, priority, break_chain, fn)
    register_tool_execution(scope_handle, name, priority, fn)

**LLM guardrails** (scope-local):

    register_llm_sanitize_request(scope_handle, name, priority, fn)
    register_llm_sanitize_response(scope_handle, name, priority, fn)
    register_llm_conditional_execution(scope_handle, name, priority, fn)

**LLM intercepts** (scope-local):

    register_llm_request(scope_handle, name, priority, break_chain, fn)
    register_llm_execution(scope_handle, name, priority, fn)
    register_llm_stream_execution(scope_handle, name, priority, fn)

**Subscribers** (scope-local):

    register_subscriber(scope_handle, name, callback)

Each ``register_*`` has a corresponding ``deregister_*`` that takes the
scope handle and name, and returns ``True`` if found and removed.

Example::

    import nat_nexus

    def redact_pii(tool_name, args):
        return {k: "***" if "ssn" in k else v for k, v in args.items()}

    with nat_nexus.scope.scope("my-agent", nat_nexus.ScopeType.Agent) as handle:
        nat_nexus.scope_local.register_tool_sanitize_request(handle, "pii-filter", 10, redact_pii)
        # ... guardrail is active only within this scope ...
"""

from nat_nexus._native import (
    nat_nexus_scope_deregister_llm_conditional_execution_guardrail as _deregister_llm_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_llm_execution_intercept as _deregister_llm_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_llm_request_intercept as _deregister_llm_request,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_llm_sanitize_request_guardrail as _deregister_llm_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_llm_sanitize_response_guardrail as _deregister_llm_sanitize_response,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_llm_stream_execution_intercept as _deregister_llm_stream_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_subscriber as _deregister_subscriber,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_tool_conditional_execution_guardrail as _deregister_tool_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_tool_execution_intercept as _deregister_tool_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_tool_request_intercept as _deregister_tool_request,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_tool_response_intercept as _deregister_tool_response,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_tool_sanitize_request_guardrail as _deregister_tool_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_scope_deregister_tool_sanitize_response_guardrail as _deregister_tool_sanitize_response,
)
from nat_nexus._native import (
    nat_nexus_scope_register_llm_conditional_execution_guardrail as _register_llm_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_register_llm_execution_intercept as _register_llm_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_register_llm_request_intercept as _register_llm_request,
)
from nat_nexus._native import (
    nat_nexus_scope_register_llm_sanitize_request_guardrail as _register_llm_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_scope_register_llm_sanitize_response_guardrail as _register_llm_sanitize_response,
)
from nat_nexus._native import (
    nat_nexus_scope_register_llm_stream_execution_intercept as _register_llm_stream_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_register_subscriber as _register_subscriber,
)
from nat_nexus._native import (
    nat_nexus_scope_register_tool_conditional_execution_guardrail as _register_tool_conditional_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_register_tool_execution_intercept as _register_tool_execution,
)
from nat_nexus._native import (
    nat_nexus_scope_register_tool_request_intercept as _register_tool_request,
)
from nat_nexus._native import (
    nat_nexus_scope_register_tool_response_intercept as _register_tool_response,
)
from nat_nexus._native import (
    nat_nexus_scope_register_tool_sanitize_request_guardrail as _register_tool_sanitize_request,
)
from nat_nexus._native import (
    nat_nexus_scope_register_tool_sanitize_response_guardrail as _register_tool_sanitize_response,
)

# ---------------------------------------------------------------------------
# Tool guardrails (scope-local)
# ---------------------------------------------------------------------------


def register_tool_sanitize_request(scope_handle, name, priority, guardrail):
    """Register a scope-local tool sanitize-request guardrail.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name: str, args: Any) -> Any``.
    """
    return _register_tool_sanitize_request(scope_handle.uuid, name, priority, guardrail)


def deregister_tool_sanitize_request(scope_handle, name):
    """Remove a scope-local tool sanitize-request guardrail. Returns ``True`` if found."""
    return _deregister_tool_sanitize_request(scope_handle.uuid, name)


def register_tool_sanitize_response(scope_handle, name, priority, guardrail):
    """Register a scope-local tool sanitize-response guardrail.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name: str, result: Any) -> Any``.
    """
    return _register_tool_sanitize_response(scope_handle.uuid, name, priority, guardrail)


def deregister_tool_sanitize_response(scope_handle, name):
    """Remove a scope-local tool sanitize-response guardrail. Returns ``True`` if found."""
    return _deregister_tool_sanitize_response(scope_handle.uuid, name)


def register_tool_conditional_execution(scope_handle, name, priority, guardrail):
    """Register a scope-local tool conditional-execution guardrail.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(tool_name: str, args: Any) -> Optional[str]``.
    """
    return _register_tool_conditional_execution(scope_handle.uuid, name, priority, guardrail)


def deregister_tool_conditional_execution(scope_handle, name):
    """Remove a scope-local tool conditional-execution guardrail. Returns ``True`` if found."""
    return _deregister_tool_conditional_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# Tool intercepts (scope-local)
# ---------------------------------------------------------------------------


def register_tool_request(scope_handle, name, priority, break_chain, fn):
    """Register a scope-local tool request intercept.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        fn: ``(tool_name: str, args: Any) -> Any``.
    """
    return _register_tool_request(scope_handle.uuid, name, priority, break_chain, fn)


def deregister_tool_request(scope_handle, name):
    """Remove a scope-local tool request intercept. Returns ``True`` if found."""
    return _deregister_tool_request(scope_handle.uuid, name)


def register_tool_response(scope_handle, name, priority, break_chain, fn):
    """Register a scope-local tool response intercept.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        fn: ``(tool_name: str, result: Any) -> Any``.
    """
    return _register_tool_response(scope_handle.uuid, name, priority, break_chain, fn)


def deregister_tool_response(scope_handle, name):
    """Remove a scope-local tool response intercept. Returns ``True`` if found."""
    return _deregister_tool_response(scope_handle.uuid, name)


def register_tool_execution(scope_handle, name, priority, fn):
    """Register a scope-local tool execution intercept (middleware chain pattern).

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        fn: ``async (tool_name: str, args: Any, next) -> Any``.
    """
    return _register_tool_execution(scope_handle.uuid, name, priority, fn)


def deregister_tool_execution(scope_handle, name):
    """Remove a scope-local tool execution intercept. Returns ``True`` if found."""
    return _deregister_tool_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# LLM guardrails (scope-local)
# ---------------------------------------------------------------------------


def register_llm_sanitize_request(scope_handle, name, priority, guardrail):
    """Register a scope-local LLM sanitize-request guardrail.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(request: LLMRequest) -> LLMRequest``.
    """
    return _register_llm_sanitize_request(scope_handle.uuid, name, priority, guardrail)


def deregister_llm_sanitize_request(scope_handle, name):
    """Remove a scope-local LLM sanitize-request guardrail. Returns ``True`` if found."""
    return _deregister_llm_sanitize_request(scope_handle.uuid, name)


def register_llm_sanitize_response(scope_handle, name, priority, guardrail):
    """Register a scope-local LLM sanitize-response guardrail.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(response: dict) -> dict``.
    """
    return _register_llm_sanitize_response(scope_handle.uuid, name, priority, guardrail)


def deregister_llm_sanitize_response(scope_handle, name):
    """Remove a scope-local LLM sanitize-response guardrail. Returns ``True`` if found."""
    return _deregister_llm_sanitize_response(scope_handle.uuid, name)


def register_llm_conditional_execution(scope_handle, name, priority, guardrail):
    """Register a scope-local LLM conditional-execution guardrail.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique guardrail name.
        priority: Priority (ascending order).
        guardrail: ``(request: LLMRequest) -> Optional[str]``.
    """
    return _register_llm_conditional_execution(scope_handle.uuid, name, priority, guardrail)


def deregister_llm_conditional_execution(scope_handle, name):
    """Remove a scope-local LLM conditional-execution guardrail. Returns ``True`` if found."""
    return _deregister_llm_conditional_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# LLM intercepts (scope-local)
# ---------------------------------------------------------------------------


def register_llm_request(scope_handle, name, priority, break_chain, fn):
    """Register a scope-local LLM request intercept.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        break_chain: If ``True``, no lower-priority intercepts run after this one.
        fn: ``(name: str, request: LLMRequest) -> LLMRequest``.
    """
    return _register_llm_request(scope_handle.uuid, name, priority, break_chain, fn)


def deregister_llm_request(scope_handle, name):
    """Remove a scope-local LLM request intercept. Returns ``True`` if found."""
    return _deregister_llm_request(scope_handle.uuid, name)


def register_llm_execution(scope_handle, name, priority, fn):
    """Register a scope-local LLM execution intercept (middleware chain pattern).

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        fn: ``async (name: str, request: LLMRequest, next) -> Any``.
    """
    return _register_llm_execution(scope_handle.uuid, name, priority, fn)


def deregister_llm_execution(scope_handle, name):
    """Remove a scope-local LLM execution intercept. Returns ``True`` if found."""
    return _deregister_llm_execution(scope_handle.uuid, name)


def register_llm_stream_execution(scope_handle, name, priority, fn):
    """Register a scope-local LLM stream-execution intercept (middleware chain pattern).

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique intercept name.
        priority: Priority (ascending order).
        fn: ``async (request: LLMRequest, next) -> AsyncIterator[Any]``.
    """
    return _register_llm_stream_execution(scope_handle.uuid, name, priority, fn)


def deregister_llm_stream_execution(scope_handle, name):
    """Remove a scope-local LLM stream-execution intercept. Returns ``True`` if found."""
    return _deregister_llm_stream_execution(scope_handle.uuid, name)


# ---------------------------------------------------------------------------
# Subscribers (scope-local)
# ---------------------------------------------------------------------------


def register_subscriber(scope_handle, name, callback):
    """Register a scope-local event subscriber.

    Args:
        scope_handle: The ``ScopeHandle`` to register under.
        name: Unique subscriber name.
        callback: ``(event: Event) -> None``.
    """
    return _register_subscriber(scope_handle.uuid, name, callback)


def deregister_subscriber(scope_handle, name):
    """Remove a scope-local event subscriber. Returns ``True`` if found."""
    return _deregister_subscriber(scope_handle.uuid, name)


__all__ = [
    # Tool guardrails
    "register_tool_sanitize_request",
    "deregister_tool_sanitize_request",
    "register_tool_sanitize_response",
    "deregister_tool_sanitize_response",
    "register_tool_conditional_execution",
    "deregister_tool_conditional_execution",
    # Tool intercepts
    "register_tool_request",
    "deregister_tool_request",
    "register_tool_response",
    "deregister_tool_response",
    "register_tool_execution",
    "deregister_tool_execution",
    # LLM guardrails
    "register_llm_sanitize_request",
    "deregister_llm_sanitize_request",
    "register_llm_sanitize_response",
    "deregister_llm_sanitize_response",
    "register_llm_conditional_execution",
    "deregister_llm_conditional_execution",
    # LLM intercepts
    "register_llm_request",
    "deregister_llm_request",
    "register_llm_execution",
    "deregister_llm_execution",
    "register_llm_stream_execution",
    "deregister_llm_stream_execution",
    # Subscribers
    "register_subscriber",
    "deregister_subscriber",
]
