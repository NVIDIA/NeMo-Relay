# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer configuration, validation, and runtime registration helpers.

This module contains the Python-side config model used to configure the native
optimizer runtime, along with plugin registration helpers for Python-hosted
optimizer components.

Example:
    ```python
    from nemo_flow.optimizer import BackendSpec, OptimizerConfig, OptimizerRuntime, StateConfig

    runtime = OptimizerRuntime(
        OptimizerConfig(
            state=StateConfig(backend=BackendSpec.in_memory()),
        )
    )

    await runtime.register()
    runtime.deregister()
    await runtime.shutdown()
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Callable, Literal, Protocol, TypedDict, cast

from nemo_flow import (
    Json,
    JsonObject,
    LlmExecutionIntercept,
    LlmRequestIntercept,
    LlmStreamExecutionIntercept,
    ToolExecutionIntercept,
    ToolRequestIntercept,
    UnsupportedBehavior,
)
from nemo_flow._native import (
    OptimizerRuntime as _NativeOptimizerRuntime,
)
from nemo_flow._native import (
    deregister_optimizer_plugin as _deregister_optimizer_plugin,
)
from nemo_flow._native import (
    register_optimizer_plugin as _register_optimizer_plugin,
)
from nemo_flow._native import (
    set_latency_sensitivity,
)
from nemo_flow._native import (
    validate_optimizer_config as _validate_optimizer_config,
)

if TYPE_CHECKING:
    from nemo_flow import Event


class _ConfigDiagnosticRequired(TypedDict):
    """Required fields for ConfigDiagnostic."""

    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """Single validation diagnostic produced by optimizer config checks.

    Fields:
        level: Diagnostic severity, either ``"warning"`` or ``"error"``.
        code: Stable machine-readable diagnostic code.
        component: Config component associated with the diagnostic.
        field: Specific field associated with the diagnostic when available.
        message: Human-readable explanation of the validation result.
    """

    component: str
    field: str


class ConfigReport(TypedDict):
    """Validation report returned by ``validate_optimizer_config()``.

    Fields:
        diagnostics: Ordered list of validation diagnostics.
    """

    diagnostics: list[ConfigDiagnostic]


class OptimizerPluginContext(Protocol):
    """Registration surface exposed to Python optimizer plugins.

    Plugin handlers receive this context during registration and use it to add
    subscribers or middleware on behalf of the optimizer runtime.
    """

    def register_subscriber(self, name: str, callback: Callable[[Event], None]) -> None:
        """Register a subscriber owned by the optimizer runtime.

        Args:
            name: Unique subscriber name within the runtime.
            callback: Callable invoked as ``callback(event)`` for each emitted
                event.
        """
        ...

    def register_llm_request_intercept(
        self, name: str, priority: int, break_chain: bool, callback: LlmRequestIntercept
    ) -> None:
        """Register an LLM request intercept owned by the optimizer runtime.

        Args:
            name: Unique intercept name within the runtime.
            priority: Execution order for the intercept. Lower values run first.
            break_chain: Whether to stop lower-priority request intercepts after
                this one runs.
            callback: Intercept callback invoked as
                ``callback(name, request, annotated)``.
        """
        ...

    def register_llm_execution_intercept(self, name: str, priority: int, callback: LlmExecutionIntercept) -> None:
        """Register a non-streaming LLM execution intercept.

        Args:
            name: Unique intercept name within the runtime.
            priority: Execution order for the intercept. Lower values run first.
            callback: Intercept callback invoked as
                ``callback(name, request, next_call)``.
        """
        ...

    def register_llm_stream_execution_intercept(
        self, name: str, priority: int, callback: LlmStreamExecutionIntercept
    ) -> None:
        """Register a streaming LLM execution intercept.

        Args:
            name: Unique intercept name within the runtime.
            priority: Execution order for the intercept. Lower values run first.
            callback: Intercept callback invoked as
                ``callback(request, next_call)``.
        """
        ...

    def register_tool_request_intercept(
        self, name: str, priority: int, break_chain: bool, callback: ToolRequestIntercept
    ) -> None:
        """Register a tool request intercept owned by the optimizer runtime.

        Args:
            name: Unique intercept name within the runtime.
            priority: Execution order for the intercept. Lower values run first.
            break_chain: Whether to stop lower-priority request intercepts after
                this one runs.
            callback: Intercept callback invoked as ``callback(tool_name, args)``.
        """
        ...

    def register_tool_execution_intercept(self, name: str, priority: int, callback: ToolExecutionIntercept) -> None:
        """Register a tool execution intercept owned by the optimizer runtime.

        Args:
            name: Unique intercept name within the runtime.
            priority: Execution order for the intercept. Lower values run first.
            callback: Intercept callback invoked as
                ``callback(tool_name, args, next_call)``.
        """
        ...


class OptimizerPluginHandler(Protocol):
    """Protocol implemented by Python-hosted optimizer plugins."""

    def validate(self, instance_id: str, plugin_config: JsonObject) -> list[ConfigDiagnostic] | None:
        """Validate plugin configuration before runtime registration.

        Args:
            instance_id: Unique instance identifier from the optimizer config.
            plugin_config: Plugin-specific JSON config document.

        Returns:
            list[ConfigDiagnostic] | None: Additional diagnostics to merge into
            the validation report, or ``None`` if the config is valid.
        """
        ...

    def register(self, instance_id: str, plugin_config: JsonObject, context: OptimizerPluginContext) -> None:
        """Register plugin-owned subscribers or intercepts with the runtime.

        Args:
            instance_id: Unique instance identifier from the optimizer config.
            plugin_config: Plugin-specific JSON config document.
            context: Registration surface used to attach subscribers and
                intercepts to the runtime.
        """
        ...


class _SupportsToDict(Protocol):
    def to_dict(self) -> JsonObject: ...


def _normalize(value: object) -> Json:
    if isinstance(value, _ComponentHelper):
        return _normalize(value.to_component())
    if hasattr(value, "to_dict"):
        return cast(_SupportsToDict, value).to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: _normalize(field_value)
            for field_info in fields(value)
            if (field_value := getattr(value, field_info.name)) is not None
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {cast(str, key): _normalize(val) for key, val in value.items() if val is not None}
    return cast(Json, value)


def _normalize_object(value: object) -> JsonObject:
    normalized = _normalize(value)
    return cast(JsonObject, normalized)


@dataclass(slots=True)
class ConfigPolicy:
    """Policy controlling how unknown config values are treated.

    Args:
        unknown_component: Behavior when the config references a component kind
            the runtime does not recognize.
        unknown_field: Behavior when a known component includes an unknown
            field.
        unsupported_value: Behavior when a field is known but its value is not
            supported by the runtime.
    """

    unknown_component: UnsupportedBehavior = "warn"
    unknown_field: UnsupportedBehavior = "warn"
    unsupported_value: UnsupportedBehavior = "error"

    def to_dict(self) -> JsonObject:
        """Convert the policy to a JSON-compatible dictionary.

        Returns:
            JsonObject: Serialized policy document suitable for native runtime
            calls.
        """
        return {
            "unknown_component": self.unknown_component,
            "unknown_field": self.unknown_field,
            "unsupported_value": self.unsupported_value,
        }


@dataclass(slots=True)
class BackendSpec:
    """Storage backend definition used by ``StateConfig``.

    Args:
        kind: Backend kind understood by the optimizer runtime.
        config: Backend-specific JSON configuration.
    """

    kind: str
    config: JsonObject = field(default_factory=dict)

    @staticmethod
    def in_memory() -> "BackendSpec":
        """Create an in-memory state backend specification.

        Returns:
            BackendSpec: Backend definition for ephemeral process-local state.
        """
        return BackendSpec(kind="in_memory")

    @staticmethod
    def redis(url: str, key_prefix: str = "nemo_flow:") -> "BackendSpec":
        """Create a Redis-backed state backend specification.

        Args:
            url: Redis connection URL.
            key_prefix: Prefix applied to all optimizer keys stored in Redis.

        Returns:
            BackendSpec: Backend definition for Redis-backed state storage.
        """
        return BackendSpec(kind="redis", config={"url": url, "key_prefix": key_prefix})

    def to_dict(self) -> JsonObject:
        """Convert the backend spec to a JSON-compatible dictionary.

        Returns:
            JsonObject: Serialized backend definition suitable for native
            runtime calls.
        """
        return {
            "kind": self.kind,
            "config": _normalize_object(self.config),
        }


@dataclass(slots=True)
class StateConfig:
    """Top-level optimizer state configuration.

    Args:
        backend: Storage backend used for optimizer state.
    """

    backend: BackendSpec

    def to_dict(self) -> JsonObject:
        """Convert the state config to a JSON-compatible dictionary.

        Returns:
            JsonObject: Serialized state configuration document.
        """
        return {"backend": _normalize_object(self.backend)}


@dataclass(slots=True)
class ComponentSpec:
    """Generic optimizer component definition.

    Args:
        kind: Component kind understood by the optimizer runtime.
        enabled: Whether the component should be activated.
        config: Component-specific JSON configuration.
    """

    kind: str
    enabled: bool = True
    config: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """Convert the component spec to a JSON-compatible dictionary.

        Returns:
            JsonObject: Serialized component definition.
        """
        return {
            "kind": self.kind,
            "enabled": self.enabled,
            "config": _normalize_object(self.config),
        }


class _ComponentHelper:
    def to_component(self) -> ComponentSpec:
        raise NotImplementedError


@dataclass(slots=True)
class TelemetryComponent(_ComponentHelper):
    """Helper for the built-in telemetry component.

    Args:
        subscriber_name: Optional subscriber name that receives optimizer
            telemetry output.
        learners: Optional list of learner names to enable.
    """

    subscriber_name: str | None = None
    learners: list[str] = field(default_factory=list)

    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``.

        Returns:
            ComponentSpec: Serialized telemetry component definition.
        """
        return ComponentSpec(
            kind="telemetry",
            config=_normalize_object(
                {
                    "subscriber_name": self.subscriber_name,
                    "learners": self.learners,
                }
            ),
        )


@dataclass(slots=True)
class DynamoHintsComponent(_ComponentHelper):
    """Helper for the built-in Dynamo hints component.

    Args:
        priority: Request intercept priority for hint injection.
        break_chain: Whether the request intercept should stop lower-priority
            request intercepts after it runs.
        inject_header: Whether to write hints into request headers.
        inject_body_path: Body field path that should receive injected hints.
    """

    priority: int = 100
    break_chain: bool = False
    inject_header: bool = True
    inject_body_path: str = "nvext.agent_hints"

    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``.

        Returns:
            ComponentSpec: Serialized Dynamo hints component definition.
        """
        return ComponentSpec(
            kind="dynamo_hints",
            config=_normalize_object(
                {
                    "priority": self.priority,
                    "break_chain": self.break_chain,
                    "inject_header": self.inject_header,
                    "inject_body_path": self.inject_body_path,
                }
            ),
        )


@dataclass(slots=True)
class ToolParallelismComponent(_ComponentHelper):
    """Helper for the built-in tool parallelism component.

    Args:
        priority: Intercept priority used by the component.
        mode: Component behavior mode. ``"observe_only"`` records signals
            without changing behavior, while the other modes inject hints or
            scheduling decisions.
    """

    priority: int = 100
    mode: Literal["observe_only", "inject_hints", "schedule"] = "observe_only"

    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``.

        Returns:
            ComponentSpec: Serialized tool parallelism component definition.
        """
        return ComponentSpec(
            kind="tool_parallelism",
            config=_normalize_object({"priority": self.priority, "mode": self.mode}),
        )


@dataclass(slots=True)
class ExternalComponent(_ComponentHelper):
    """Helper for plugin-backed optimizer components.

    Args:
        plugin_kind: Registered plugin kind to instantiate.
        instance_id: Unique instance identifier passed back to the plugin
            handler.
        plugin_config: Plugin-specific JSON configuration.
    """

    plugin_kind: str
    instance_id: str
    plugin_config: JsonObject = field(default_factory=dict)

    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``.

        Returns:
            ComponentSpec: Serialized external plugin component definition.
        """
        return ComponentSpec(
            kind="external_component",
            config=_normalize_object(
                {
                    "plugin_kind": self.plugin_kind,
                    "instance_id": self.instance_id,
                    "plugin_config": self.plugin_config,
                }
            ),
        )


@dataclass(slots=True)
class OptimizerConfig:
    """Top-level optimizer configuration document.

    Args:
        version: Config schema version expected by the runtime.
        agent_id: Optional logical agent identifier used for optimizer state.
        state: Optional state backend configuration.
        components: Ordered list of optimizer components or helper objects.
        policy: Validation policy for unknown or unsupported config values.

    Example:
        ```python
        from nemo_flow.optimizer import (
            BackendSpec,
            ConfigPolicy,
            OptimizerConfig,
            StateConfig,
            TelemetryComponent,
        )

        config = OptimizerConfig(
            version=1,
            agent_id="agent-1",
            state=StateConfig(backend=BackendSpec.in_memory()),
            components=[TelemetryComponent(learners=["latency_sensitivity"])],
            policy=ConfigPolicy(
                unknown_component="warn",
                unknown_field="warn",
                unsupported_value="error",
            ),
        )
        ```
    """

    version: int = 1
    agent_id: str | None = None
    state: StateConfig | None = None
    components: list[ComponentSpec | _ComponentHelper] = field(default_factory=list)
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> JsonObject:
        """Convert the optimizer config to a JSON-compatible dictionary.

        Returns:
            JsonObject: Serialized config document suitable for validation or
            runtime registration.
        """
        return {
            "version": self.version,
            "agent_id": self.agent_id,
            "state": _normalize(self.state),
            "components": [_normalize(component) for component in self.components],
            "policy": self.policy.to_dict(),
        }


class OptimizerRuntime:
    """Runtime wrapper used to register optimizer components with NeMo Flow.

    Args:
        config: Optimizer configuration, either as ``OptimizerConfig`` or an
            equivalent JSON object.

    Example:
        ```python
        from nemo_flow.optimizer import BackendSpec, OptimizerConfig, OptimizerRuntime, StateConfig

        runtime = OptimizerRuntime(
            OptimizerConfig(
                version=1,
                agent_id="agent-1",
                state=StateConfig(backend=BackendSpec.in_memory()),
            )
        )

        await runtime.register()
        runtime.deregister()
        await runtime.shutdown()
        ```
    """

    def __init__(self, config: OptimizerConfig | JsonObject) -> None:
        self._config = _normalize_object(config)
        self._native = _NativeOptimizerRuntime(self._config)

    async def register(self) -> None:
        """Register the configured optimizer components with the runtime.

        Notes:
            Registration may create subscribers and middleware in the current
            process. Call ``deregister()`` to remove them when they are no
            longer needed.
        """
        await self._native.register()

    def deregister(self) -> None:
        """Remove optimizer-owned registrations from the runtime."""
        self._native.deregister()

    async def shutdown(self) -> None:
        """Shut down native optimizer resources owned by this runtime."""
        await self._native.shutdown()

    def report(self) -> ConfigReport:
        """Return the runtime's current validation and registration report.

        Returns:
            ConfigReport: Diagnostics reported by the native optimizer runtime.
        """
        return cast(ConfigReport, self._native.report())


def validate_optimizer_config(config: OptimizerConfig | JsonObject) -> ConfigReport:
    """Validate an optimizer config without registering it.

    Args:
        config: Optimizer configuration to validate, either as
            ``OptimizerConfig`` or an equivalent JSON object.

    Returns:
        ConfigReport: Validation diagnostics produced by the native runtime.

    Example:
        ```python
        from nemo_flow.optimizer import ComponentSpec, OptimizerConfig, validate_optimizer_config

        report = validate_optimizer_config(
            OptimizerConfig(components=[ComponentSpec(kind="future_component")])
        )
        print(report["diagnostics"])
        ```
    """
    return cast(ConfigReport, _validate_optimizer_config(_normalize_object(config)))


def register_optimizer_plugin(plugin_kind: str, handler: OptimizerPluginHandler) -> None:
    """Register a Python-hosted optimizer plugin handler.

    Args:
        plugin_kind: Plugin kind string referenced by ``ExternalComponent``.
        handler: Python plugin handler implementing validation and registration
            hooks for that plugin kind.

    Notes:
        Register the handler before validating or registering optimizer configs
        that reference the plugin kind.
    """
    _register_optimizer_plugin(plugin_kind, handler)


def deregister_optimizer_plugin(plugin_kind: str) -> bool:
    """Remove a previously registered Python optimizer plugin handler.

    Args:
        plugin_kind: Plugin kind previously passed to
            ``register_optimizer_plugin()``.

    Returns:
        bool: ``True`` if a plugin handler was removed, otherwise ``False``.
    """
    return _deregister_optimizer_plugin(plugin_kind)


__all__ = [
    "BackendSpec",
    "ComponentSpec",
    "ConfigDiagnostic",
    "ConfigPolicy",
    "ConfigReport",
    "DynamoHintsComponent",
    "ExternalComponent",
    "OptimizerConfig",
    "OptimizerPluginContext",
    "OptimizerPluginHandler",
    "OptimizerRuntime",
    "StateConfig",
    "TelemetryComponent",
    "ToolParallelismComponent",
    "deregister_optimizer_plugin",
    "register_optimizer_plugin",
    "UnsupportedBehavior",
    "set_latency_sensitivity",
    "validate_optimizer_config",
]
