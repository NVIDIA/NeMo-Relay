# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic plugin configuration and registration helpers.

This module exposes the top-level plugin system used to validate and activate
adaptive and custom plugin components. Component registration names are scoped
per component by the runtime, so end users do not provide instance ids.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Callable, Literal, Protocol, TypedDict, cast

from nemo_relay import (
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
    subscribers,
)
from nemo_relay._config_normalize import normalize, normalize_object
from nemo_relay._native import (
    active_plugin_report as _active_plugin_report,
)
from nemo_relay._native import (
    clear_plugin_configuration as _clear_plugin_configuration,
)
from nemo_relay._native import (
    deregister_plugin as _deregister_plugin,
)
from nemo_relay._native import (
    initialize_plugins as _initialize_plugins,
)
from nemo_relay._native import (
    list_plugin_kinds as _list_plugin_kinds,
)
from nemo_relay._native import (
    register_plugin as _register_plugin,
)
from nemo_relay._native import (
    validate_plugin_config as _validate_plugin_config,
)

if TYPE_CHECKING:
    from nemo_relay import Event


class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One plugin validation diagnostic."""

    component: str
    field: str


class ConfigReport(TypedDict):
    """Validation or activation report for a plugin config."""

    diagnostics: list[ConfigDiagnostic]


class PluginContext(Protocol):
    """Component-scoped registration context passed to custom plugin handlers."""

    def register_subscriber(self, name: str, callback: Callable[[Event], None]) -> None:
        """Register an infallible event subscriber for this component."""
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
            "config": cast(JsonObject, normalize_object(self.config)),
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
            "components": [normalize(component) for component in self.components],
            "policy": self.policy.to_dict(),
        }


def validate(config: PluginConfig | JsonObject) -> ConfigReport:
    """Validate a plugin configuration without changing runtime state.

    Args:
        config: `PluginConfig` or an equivalent JSON object.

    Returns:
        The validation report for the supplied config.

    Behavior:
        Validation checks plugin-level compatibility, unknown component kinds,
        multiplicity rules, and per-plugin validation logic.
    """
    return cast(ConfigReport, _validate_plugin_config(cast(JsonObject, normalize_object(config))))


async def initialize(config: PluginConfig | JsonObject) -> ConfigReport:
    """Validate and activate a plugin configuration.

    Args:
        config: `PluginConfig` or an equivalent JSON object.

    Returns:
        The report for the successfully activated configuration.

    Behavior:
        Initialization replaces the current active plugin configuration. Partial
        registration is rolled back on failure, and the previous configuration
        is restored when possible.
    """
    return cast(ConfigReport, await _initialize_plugins(cast(JsonObject, normalize_object(config))))


def clear() -> None:
    """Clear the active plugin configuration.

    Returns:
        `None`.

    Behavior:
        This removes active component registrations but leaves the plugin kind
        registry intact for future validation or initialization.
    """
    _clear_plugin_configuration()


@asynccontextmanager
async def plugin(config: PluginConfig | JsonObject, *, clear_on_exit: bool = True) -> AsyncIterator[ConfigReport]:
    """Context manager for plugin initialization and cleanup.

    Args:
        config: `PluginConfig` or an equivalent JSON object.
        clear_on_exit: Whether to clear the plugin configuration on exit.

    Yields:
        The `ConfigReport` for the initialized configuration.

    Behavior:
        This context manager initializes the plugin configuration on entry and clears it on exit.
    """
    report_ = await initialize(config)
    try:
        yield report_
    finally:
        subscribers.flush()
        if clear_on_exit:
            clear()


def report() -> ConfigReport | None:
    """Return the last successful plugin report.

    Returns:
        The active `ConfigReport`, or `None` when no plugin configuration is
        currently active.

    Behavior:
        This reports the last successfully activated configuration snapshot. It
        does not revalidate plugin state or inspect pending registrations.
    """
    return cast(ConfigReport | None, _active_plugin_report())


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
        registrations remain until `clear()` or the next successful
        `initialize(...)`.
    """
    return _deregister_plugin(plugin_kind)


__all__ = [
    "ComponentSpec",
    "ConfigDiagnostic",
    "ConfigPolicy",
    "ConfigReport",
    "PluginConfig",
    "PluginContext",
    "Plugin",
    "clear",
    "initialize",
    "deregister",
    "list_kinds",
    "register",
    "report",
    "validate",
]
