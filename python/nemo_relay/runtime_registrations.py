# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic eligibility controls for global runtime registrations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from nemo_relay._native import (
    deregister_conditional_middleware_guardrail as _native_deregister,
)
from nemo_relay._native import list_runtime_registrations as _native_list
from nemo_relay._native import (
    register_conditional_middleware_guardrail as _native_register,
)


class RuntimeRegistrationKind(StrEnum):
    """Global runtime registration surfaces that can be gated."""

    SUBSCRIBER = "subscriber"
    EVENT_METADATA_INJECTOR = "event_metadata_injector"
    MARK_SANITIZE_GUARDRAIL = "mark_sanitize_guardrail"
    SCOPE_SANITIZE_START_GUARDRAIL = "scope_sanitize_start_guardrail"
    SCOPE_SANITIZE_END_GUARDRAIL = "scope_sanitize_end_guardrail"
    TOOL_SANITIZE_REQUEST_GUARDRAIL = "tool_sanitize_request_guardrail"
    TOOL_SANITIZE_RESPONSE_GUARDRAIL = "tool_sanitize_response_guardrail"
    TOOL_CONDITIONAL_EXECUTION_GUARDRAIL = "tool_conditional_execution_guardrail"
    TOOL_REQUEST_INTERCEPT = "tool_request_intercept"
    TOOL_EXECUTION_INTERCEPT = "tool_execution_intercept"
    LLM_SANITIZE_REQUEST_GUARDRAIL = "llm_sanitize_request_guardrail"
    LLM_SANITIZE_RESPONSE_GUARDRAIL = "llm_sanitize_response_guardrail"
    LLM_CONDITIONAL_EXECUTION_GUARDRAIL = "llm_conditional_execution_guardrail"
    LLM_REQUEST_INTERCEPT = "llm_request_intercept"
    LLM_EXECUTION_INTERCEPT = "llm_execution_intercept"
    LLM_STREAM_EXECUTION_INTERCEPT = "llm_stream_execution_intercept"


class RuntimeRegistrationOwnerKind(StrEnum):
    """Owner categories reported by runtime registration discovery."""

    CORE = "core"
    GLOBAL_API = "global_api"
    PLUGIN = "plugin"


@dataclass(frozen=True)
class RuntimeRegistrationOwner:
    """Owner metadata for a discovered runtime registration."""

    kind: RuntimeRegistrationOwnerKind
    plugin_kind: str | None
    component_ordinal: int | None


@dataclass(frozen=True)
class RuntimeRegistrationIdentity:
    """Structured identity for a global gateable registration."""

    kind: RuntimeRegistrationKind
    local_name: str
    effective_name: str
    owner: RuntimeRegistrationOwner


ConditionalMiddlewareGuardrail = Callable[[set[RuntimeRegistrationKind], str], str | None]


def register_conditional_middleware_guardrail(
    name: str,
    kinds: set[RuntimeRegistrationKind],
    registration_name: str,
    guardrail: ConditionalMiddlewareGuardrail,
) -> None:
    """Register a global gate for registrations matching kind and effective name.

    Args:
        name: Unique name for this gate registration.
        kinds: Registration kinds to which this gate applies.
        registration_name: Effective registration name to match.
        guardrail: Callback returning a rejection message or ``None`` to allow
            the registration.

    Returns:
        None: The gate is registered for subsequent matching registrations.
    """

    def wrapped(received_kinds: set[str], effective_name: str) -> str | None:
        return guardrail({RuntimeRegistrationKind(kind) for kind in received_kinds}, effective_name)

    _native_register(name, {kind.value for kind in kinds}, registration_name, wrapped)


def deregister_conditional_middleware_guardrail(name: str) -> bool:
    """Remove a named global registration gate.

    Args:
        name: Gate name previously passed to
            ``register_conditional_middleware_guardrail``.

    Returns:
        bool: ``True`` if a gate was removed; otherwise ``False``.
    """
    return _native_deregister(name)


def list_runtime_registrations(
    kinds: Iterable[RuntimeRegistrationKind] | None = None,
) -> list[RuntimeRegistrationIdentity]:
    """List a deterministic snapshot of global gateable registrations.

    Args:
        kinds: Optional registration kinds to include. When omitted, includes
            every gateable global registration.

    Returns:
        list[RuntimeRegistrationIdentity]: Ordered registration identities.
    """
    selected = None if kinds is None else {kind.value for kind in kinds}
    registrations = _native_list(selected)
    return [
        RuntimeRegistrationIdentity(
            kind=RuntimeRegistrationKind(registration["kind"]),
            local_name=registration["local_name"],
            effective_name=registration["effective_name"],
            owner=RuntimeRegistrationOwner(
                kind=RuntimeRegistrationOwnerKind(registration["owner"]["kind"]),
                plugin_kind=registration["owner"]["plugin_kind"],
                component_ordinal=registration["owner"]["component_ordinal"],
            ),
        )
        for registration in registrations
    ]


__all__ = [
    "ConditionalMiddlewareGuardrail",
    "RuntimeRegistrationIdentity",
    "RuntimeRegistrationKind",
    "RuntimeRegistrationOwner",
    "RuntimeRegistrationOwnerKind",
    "deregister_conditional_middleware_guardrail",
    "list_runtime_registrations",
    "register_conditional_middleware_guardrail",
]
