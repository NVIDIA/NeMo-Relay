# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic plugin configuration and registration helpers.

This module exposes the top-level plugin system used to validate and activate
adaptive and custom plugin components. Component registration names are scoped
per component by the runtime, so end users do not provide instance ids.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypedDict, cast

from nemo_relay import (
    EventMetadataInjectorCallback,
    EventSanitizeGuardrail,
    Json,
    JsonObject,
    LlmConditionalExecutionGuardrail,
    LlmExecutionIntercept,
    LlmRequestIntercept,
    LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail,
    LlmStreamExecutionIntercept,
    ToolConditionalExecutionGuardrail,
    ToolExecutionIntercept,
    ToolRequestIntercept,
    ToolSanitizeGuardrail,
    UnsupportedBehavior,
)
from nemo_relay._native import _PluginHostActivation as _NativePluginHostActivation
from nemo_relay._native import (
    deregister_plugin as _deregister_plugin,
)
from nemo_relay._native import (
    initialize as _initialize,
)
from nemo_relay._native import (
    list_plugin_kinds as _list_plugin_kinds,
)
from nemo_relay._native import (
    register_plugin as _register_plugin,
)
from nemo_relay._native import (
    validate as _validate,
)
from nemo_relay._native import (
    validate_exact as _validate_exact,
)
from nemo_relay.runtime_registrations import ConditionalMiddlewareGuardrail, RuntimeRegistrationKind

if TYPE_CHECKING:
    from types import TracebackType

    from nemo_relay import Event


class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One plugin validation diagnostic."""

    component: str
    field: str


class _RuntimeDiagnosticRequired(TypedDict):
    code: str
    component: str
    message: str
    count: int


class RuntimeDiagnostic(_RuntimeDiagnosticRequired, total=False):
    """One aggregated runtime failure from an active plugin."""

    field: str
    session_id: str


class ConfigReport(TypedDict):
    """Validation or activation report for a plugin config."""

    diagnostics: list[ConfigDiagnostic]
    runtime_diagnostics: list[RuntimeDiagnostic]


DynamicPluginCheckState = Literal["unknown", "valid", "invalid"]


class _DynamicPluginValidationStatusRequired(TypedDict):
    """Per-phase validation state for one dynamic plugin."""

    manifest: DynamicPluginCheckState
    compatibility: DynamicPluginCheckState
    integrity: DynamicPluginCheckState
    environment: DynamicPluginCheckState
    authenticity: DynamicPluginCheckState
    policy_satisfied: DynamicPluginCheckState


class DynamicPluginValidationStatus(_DynamicPluginValidationStatusRequired, total=False):
    checked_at: str | None
    message: str | None


class DynamicPluginFailure(TypedDict):
    """Actionable reason a dynamic plugin could not be trusted or loaded."""

    phase: str
    code: str
    message: str


class _DynamicPluginValidationReportRequired(TypedDict):
    """Validation result for one authored dynamic plugin declaration."""

    plugin_id: str
    manifest_ref: str
    kind: DynamicPluginKind
    status: DynamicPluginValidationStatus
    selected: bool


class DynamicPluginValidationReport(_DynamicPluginValidationReportRequired, total=False):
    failure: DynamicPluginFailure | None


class PluginHostReport(TypedDict):
    """Static and dynamic validation results owned by one plugin host."""

    config: ConfigReport
    dynamic_plugins: list[DynamicPluginValidationReport]


DynamicPluginKind = Literal["rust_dynamic", "worker"]
"""Execution lane for a dynamically loaded plugin."""


class PluginContext(Protocol):
    """Component-scoped registration context passed to custom plugin handlers."""

    def register_subscriber(self, name: str, callback: Callable[[Event], None]) -> None:
        """Register an infallible event subscriber for this component."""
        ...

    def register_conditional_middleware_guardrail(
        self,
        name: str,
        kinds: set[RuntimeRegistrationKind],
        registration_name: str,
        guardrail: ConditionalMiddlewareGuardrail,
    ) -> None:
        """Register an activation-owned gate for a global runtime registration."""
        ...

    def register_event_metadata_injector(
        self, name: str, priority: int, callback: EventMetadataInjectorCallback
    ) -> None:
        """Register an event metadata injector for this component."""
        ...

    def register_mark_sanitize_guardrail(self, name: str, priority: int, callback: EventSanitizeGuardrail) -> None:
        """Register a mark event sanitizer for this component."""
        ...

    def register_scope_sanitize_start_guardrail(
        self, name: str, priority: int, callback: EventSanitizeGuardrail
    ) -> None:
        """Register a scope-start event sanitizer for this component."""
        ...

    def register_scope_sanitize_end_guardrail(self, name: str, priority: int, callback: EventSanitizeGuardrail) -> None:
        """Register a scope-end event sanitizer for this component."""
        ...

    def register_tool_sanitize_request_guardrail(
        self, name: str, priority: int, callback: ToolSanitizeGuardrail
    ) -> None:
        """Register a tool sanitize-request guardrail for this component."""
        ...

    def register_tool_sanitize_response_guardrail(
        self, name: str, priority: int, callback: ToolSanitizeGuardrail
    ) -> None:
        """Register a tool sanitize-response guardrail for this component."""
        ...

    def register_tool_conditional_execution_guardrail(
        self, name: str, priority: int, callback: ToolConditionalExecutionGuardrail
    ) -> None:
        """Register a tool conditional-execution guardrail for this component."""
        ...

    def register_llm_sanitize_request_guardrail(
        self, name: str, priority: int, callback: LlmSanitizeRequestGuardrail
    ) -> None:
        """Register an LLM sanitize-request guardrail for this component."""
        ...

    def register_llm_sanitize_response_guardrail(
        self, name: str, priority: int, callback: LlmSanitizeResponseGuardrail
    ) -> None:
        """Register an LLM sanitize-response guardrail for this component."""
        ...

    def register_llm_conditional_execution_guardrail(
        self, name: str, priority: int, callback: LlmConditionalExecutionGuardrail
    ) -> None:
        """Register an LLM conditional-execution guardrail for this component."""
        ...

    def register_llm_request_intercept(
        self, name: str, priority: int, break_chain: bool, callback: LlmRequestIntercept
    ) -> None:
        """Register an LLM request intercept for this component."""
        ...

    def register_llm_execution_intercept(self, name: str, priority: int, callback: LlmExecutionIntercept) -> None:
        """Register an LLM execution intercept for this component."""
        ...

    def register_llm_stream_execution_intercept(
        self, name: str, priority: int, callback: LlmStreamExecutionIntercept
    ) -> None:
        """Register an LLM streaming execution intercept for this component."""
        ...

    def register_tool_request_intercept(
        self, name: str, priority: int, break_chain: bool, callback: ToolRequestIntercept
    ) -> None:
        """Register a tool request intercept for this component."""
        ...

    def register_tool_execution_intercept(self, name: str, priority: int, callback: ToolExecutionIntercept) -> None:
        """Register a tool execution intercept for this component."""
        ...


class Plugin(Protocol):
    """Custom plugin callback contract."""

    def validate(self, plugin_config: JsonObject) -> list[ConfigDiagnostic] | None:
        """Validate one component-local config object.

        Args:
            plugin_config: The `config` object from a single component.

        Returns:
            A list of diagnostics, or `None` for no diagnostics.

        Behavior:
            Error diagnostics block `initialize(...)`.
        """
        ...

    def register(self, plugin_config: JsonObject, context: PluginContext) -> None:
        """Install middleware and subscribers for one component instance.

        Args:
            plugin_config: The `config` object from a single component.
            context: Component-scoped registration context used to install
                middleware and subscribers.

        Returns:
            `None`.

        Behavior:
            Any exception aborts the current initialization and triggers
            rollback of partial registrations.
        """
        ...


class _SupportsToDict(Protocol):
    def to_dict(self) -> JsonObject: ...


def _normalize(value: object, *, preserve_nulls: bool = False) -> Json:
    if hasattr(value, "to_dict"):
        return cast(_SupportsToDict, value).to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: _normalize(field_value, preserve_nulls=preserve_nulls)
            for field_info in fields(value)
            for field_value in [getattr(value, field_info.name)]
            if preserve_nulls or field_value is not None
        }
    if isinstance(value, list):
        return [_normalize(item, preserve_nulls=preserve_nulls) for item in value]
    if isinstance(value, dict):
        return {
            cast(str, key): _normalize(val, preserve_nulls=preserve_nulls or key == "config")
            for key, val in value.items()
            if preserve_nulls or val is not None
        }
    return cast(Json, value)


def _normalize_object(value: object) -> JsonObject:
    return cast(JsonObject, _normalize(value))


def _normalize_component_config(value: object) -> JsonObject:
    return cast(JsonObject, _normalize(value, preserve_nulls=True))


@dataclass(slots=True)
class ConfigPolicy:
    """Policy for unsupported plugin configuration.

    Args:
        unknown_component: How to handle unknown component kinds.
        unknown_field: How to handle unknown fields inside known components.
        unsupported_value: How to handle known fields with unsupported values.

    Behavior:
        `"warn"` emits a warning diagnostic, `"error"` emits an error
        diagnostic that blocks initialization, and `"ignore"` suppresses the
        diagnostic entirely.
    """

    unknown_component: UnsupportedBehavior = "warn"
    unknown_field: UnsupportedBehavior = "warn"
    unsupported_value: UnsupportedBehavior = "error"

    def to_dict(self) -> JsonObject:
        """Serialize this policy to the canonical JSON object shape."""
        return {
            "unknown_component": self.unknown_component,
            "unknown_field": self.unknown_field,
            "unsupported_value": self.unsupported_value,
        }


@dataclass(slots=True)
class ComponentSpec:
    """One top-level custom plugin component.

    Args:
        kind: Registered plugin kind string.
        enabled: Whether the component should be activated.
        config: Component-local JSON config object.

    Behavior:
        Disabled components are still validated but skipped during runtime
        registration.
    """

    kind: str
    enabled: bool = True
    config: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """Serialize this component to the canonical JSON object shape."""
        return {
            "kind": self.kind,
            "enabled": self.enabled,
            "config": _normalize_component_config(self.config),
        }


@dataclass(slots=True)
class PluginConfig:
    """Canonical plugin configuration document.

    Args:
        version: Plugin config schema version.
        components: Ordered list of top-level components. This may mix
            `plugin.ComponentSpec(...)` and `adaptive.ComponentSpec(...)`.
        policy: Plugin-level unsupported-config policy.

    Behavior:
        Component order is preserved during initialization.
    """

    version: int = 1
    components: list[object] = field(default_factory=list)
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> JsonObject:
        """Serialize this config to the canonical JSON document shape."""
        return {
            "version": self.version,
            "components": [_normalize(component) for component in self.components],
            "policy": self.policy.to_dict(),
        }


class PluginHostActivation:
    """Owned lifetime for one process-wide static and dynamic plugin host.

    Keep this object alive while agent code may invoke callbacks from the
    loaded plugins. Prefer ``async with`` or call :meth:`close` explicitly.
    Native finalization performs best-effort cleanup when an object is dropped.
    """

    __slots__ = ("_native",)

    def __init__(self, native: _NativePluginHostActivation) -> None:
        self._native = native

    @property
    def report(self) -> PluginHostReport:
        """Return the validation report captured during activation."""
        return cast(PluginHostReport, self._native.report)

    @property
    def is_active(self) -> bool:
        """Return whether this activation handle has not begun teardown.

        Failed teardown leaves the handle active so :meth:`close` can retry.
        """
        return self._native.is_active

    async def close(self) -> None:
        """Clear callbacks and unload plugins; repeated calls are safe."""
        await self._native.close()

    async def __aenter__(self) -> Self:
        """Return this active host when entering an async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the host when leaving an async context."""
        del exc_type, exc_value, traceback
        await self.close()


async def initialize(
    config: PluginConfig | JsonObject,
    additional_plugins_toml: str | os.PathLike[str] | None = None,
) -> PluginHostActivation:
    """Initialize the core-owned static and dynamic plugin host.

    Programmatic configuration is the lowest-precedence layer. An optional
    explicit ``plugins.toml`` replaces user-file discovery, and the system
    file overlays either source. The returned handle owns every activated
    plugin.

    Args:
        config: Lowest-precedence programmatic plugin configuration.
        additional_plugins_toml: Optional explicit configuration layer.

    Returns:
        An owned activation whose report includes static and dynamic results.
    """
    path = os.fspath(additional_plugins_toml) if additional_plugins_toml is not None else None
    return PluginHostActivation(await _initialize(_normalize_object(config), path))


@asynccontextmanager
async def activate(
    config: PluginConfig | JsonObject,
    additional_plugins_toml: str | os.PathLike[str] | None = None,
) -> AsyncIterator[PluginHostActivation]:
    """Initialize and close the plugin host around an async context.

    Use :func:`initialize` instead when the activation lifetime must extend
    beyond one ``async with`` block.

    Args:
        config: Lowest-precedence programmatic plugin configuration.
        additional_plugins_toml: Optional explicit configuration layer.

    Returns:
        An async context manager that yields the owned activation.
    """
    activation = await initialize(config, additional_plugins_toml)
    try:
        yield activation
    finally:
        await activation.close()


def validate(
    config: PluginConfig | JsonObject,
    additional_plugins_toml: str | os.PathLike[str] | None = None,
) -> PluginHostReport:
    """Validate the core-owned static and dynamic plugin host.

    This resolves the same configuration layers as :func:`initialize` without
    loading plugin code or acquiring the process-wide activation lease.

    Args:
        config: Lowest-precedence programmatic plugin configuration.
        additional_plugins_toml: Optional explicit configuration layer.

    Returns:
        A static configuration report and selected dynamic validation reports.
    """
    path = os.fspath(additional_plugins_toml) if additional_plugins_toml is not None else None
    return cast(PluginHostReport, _validate(_normalize_object(config), path))


def validate_exact(config: PluginConfig | JsonObject) -> PluginHostReport:
    """Validate only the supplied static plugin configuration.

    Unlike :func:`validate`, this does not discover or merge ``plugins.toml``
    files. Use it for component-specific validation when ``config`` is the
    complete document to check.

    Args:
        config: Complete static plugin configuration.

    Returns:
        The static validation report with no dynamic plugin results.
    """
    return cast(PluginHostReport, _validate_exact(_normalize_object(config)))


def list_kinds() -> list[str]:
    """List registered custom plugin kinds.

    Returns:
        A sorted list of plugin kind strings known to the plugin registry.

    Behavior:
        This reports available plugin kinds, not the currently active
        component set.
    """
    return _list_plugin_kinds()


def register(plugin_kind: str, plugin: Plugin) -> None:
    """Register a custom plugin implementation.

    Args:
        plugin_kind: Unique top-level component kind string.
        plugin: Custom plugin implementation.

    Returns:
        `None`.

    Behavior:
        Registering the same kind twice raises an error.
    """
    _register_plugin(plugin_kind, plugin)


def deregister(plugin_kind: str) -> bool:
    """Deregister a custom plugin kind.

    Args:
        plugin_kind: Kind string to remove from the plugin registry.

    Returns:
        `True` if a plugin was removed, otherwise `False`.

    Behavior:
        This affects future validation and initialization only. Active runtime
        registrations remain until the owning :class:`PluginHostActivation`
        closes.
    """
    return _deregister_plugin(plugin_kind)


__all__ = [
    "ComponentSpec",
    "ConfigDiagnostic",
    "RuntimeDiagnostic",
    "ConfigPolicy",
    "ConfigReport",
    "DynamicPluginCheckState",
    "DynamicPluginFailure",
    "DynamicPluginKind",
    "DynamicPluginValidationReport",
    "DynamicPluginValidationStatus",
    "PluginConfig",
    "PluginContext",
    "PluginHostActivation",
    "PluginHostReport",
    "Plugin",
    "activate",
    "initialize",
    "deregister",
    "list_kinds",
    "register",
    "validate",
    "validate_exact",
]
