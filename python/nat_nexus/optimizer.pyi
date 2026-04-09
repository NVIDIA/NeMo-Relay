# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal, Protocol, TypedDict

from nat_nexus import (
    Event,
    JsonObject,
    LlmExecutionIntercept,
    LlmRequestIntercept,
    LlmStreamExecutionIntercept,
    ToolExecutionIntercept,
    ToolRequestIntercept,
)

UnsupportedBehavior = Literal["ignore", "warn", "error"]

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
    """Registration surface exposed to Python optimizer plugins."""

    def register_subscriber(self, name: str, callback: Callable[[Event], None]) -> None:
        """Register a subscriber owned by the optimizer runtime.

        Args:
            name: Unique subscriber name within the runtime.
            callback: Callable invoked as ``callback(event)`` for each emitted
                event.
        """
        ...

    def register_llm_request_intercept(
        self,
        name: str,
        priority: int,
        break_chain: bool,
        callback: LlmRequestIntercept,
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

    def register_llm_execution_intercept(
        self,
        name: str,
        priority: int,
        callback: LlmExecutionIntercept,
    ) -> None:
        """Register a non-streaming LLM execution intercept.

        Args:
            name: Unique intercept name within the runtime.
            priority: Execution order for the intercept. Lower values run first.
            callback: Intercept callback invoked as
                ``callback(name, request, next_call)``.
        """
        ...

    def register_llm_stream_execution_intercept(
        self,
        name: str,
        priority: int,
        callback: LlmStreamExecutionIntercept,
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
        self,
        name: str,
        priority: int,
        break_chain: bool,
        callback: ToolRequestIntercept,
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

    def register_tool_execution_intercept(
        self,
        name: str,
        priority: int,
        callback: ToolExecutionIntercept,
    ) -> None:
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

    def register(
        self,
        instance_id: str,
        plugin_config: JsonObject,
        context: OptimizerPluginContext,
    ) -> None:
        """Register plugin-owned subscribers or intercepts with the runtime.

        Args:
            instance_id: Unique instance identifier from the optimizer config.
            plugin_config: Plugin-specific JSON config document.
            context: Registration surface used to attach subscribers and
                intercepts to the runtime.
        """
        ...

class ConfigPolicy:
    """Policy controlling how unknown config values are treated.

    Args:
        unknown_component: Behavior when the config references an unknown
            component kind.
        unknown_field: Behavior when a known component includes an unknown
            field.
        unsupported_value: Behavior when a field is known but its value is not
            supported by the runtime.
    """

    unknown_component: UnsupportedBehavior
    unknown_field: UnsupportedBehavior
    unsupported_value: UnsupportedBehavior

    def __init__(
        self,
        unknown_component: UnsupportedBehavior = "warn",
        unknown_field: UnsupportedBehavior = "warn",
        unsupported_value: UnsupportedBehavior = "error",
    ) -> None: ...
    def to_dict(self) -> JsonObject:
        """Convert the policy to a JSON-compatible dictionary."""
        ...

class BackendSpec:
    """Storage backend definition used by ``StateConfig``.

    Args:
        kind: Backend kind understood by the optimizer runtime.
        config: Backend-specific JSON configuration.
    """

    kind: str
    config: JsonObject

    def __init__(self, kind: str, config: JsonObject = ...) -> None: ...
    @staticmethod
    def in_memory() -> BackendSpec:
        """Create an in-memory state backend specification."""
        ...

    @staticmethod
    def redis(url: str, key_prefix: str = "nexus:") -> BackendSpec:
        """Create a Redis-backed state backend specification.

        Args:
            url: Redis connection URL.
            key_prefix: Prefix applied to all optimizer keys stored in Redis.
        """
        ...

    def to_dict(self) -> JsonObject:
        """Convert the backend spec to a JSON-compatible dictionary."""
        ...

class StateConfig:
    """Top-level optimizer state configuration.

    Args:
        backend: Storage backend used for optimizer state.
    """

    backend: BackendSpec

    def __init__(self, backend: BackendSpec) -> None: ...
    def to_dict(self) -> JsonObject:
        """Convert the state config to a JSON-compatible dictionary."""
        ...

class ComponentSpec:
    """Generic optimizer component definition.

    Args:
        kind: Component kind understood by the optimizer runtime.
        enabled: Whether the component should be activated.
        config: Component-specific JSON configuration.
    """

    kind: str
    enabled: bool
    config: JsonObject

    def __init__(self, kind: str, enabled: bool = True, config: JsonObject = ...) -> None: ...
    def to_dict(self) -> JsonObject:
        """Convert the component spec to a JSON-compatible dictionary."""
        ...

class TelemetryComponent:
    """Helper for the built-in telemetry component.

    Args:
        subscriber_name: Optional subscriber name that receives optimizer
            telemetry output.
        learners: Optional list of learner names to enable.
    """

    subscriber_name: str | None
    learners: list[str]

    def __init__(
        self,
        subscriber_name: str | None = None,
        learners: list[str] = ...,
    ) -> None: ...
    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``."""
        ...

class DynamoHintsComponent:
    """Helper for the built-in Dynamo hints component.

    Args:
        priority: Request intercept priority for hint injection.
        break_chain: Whether the request intercept should stop lower-priority
            request intercepts after it runs.
        inject_header: Whether to write hints into request headers.
        inject_body_path: Body field path that should receive injected hints.
    """

    priority: int
    break_chain: bool
    inject_header: bool
    inject_body_path: str

    def __init__(
        self,
        priority: int = 100,
        break_chain: bool = False,
        inject_header: bool = True,
        inject_body_path: str = "nvext.agent_hints",
    ) -> None: ...
    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``."""
        ...

class ToolParallelismComponent:
    """Helper for the built-in tool parallelism component.

    Args:
        priority: Intercept priority used by the component.
        mode: Component behavior mode.
    """

    priority: int
    mode: Literal["observe_only", "inject_hints", "schedule"]

    def __init__(
        self,
        priority: int = 100,
        mode: Literal["observe_only", "inject_hints", "schedule"] = "observe_only",
    ) -> None: ...
    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``."""
        ...

class ExternalComponent:
    """Helper for plugin-backed optimizer components.

    Args:
        plugin_kind: Registered plugin kind to instantiate.
        instance_id: Unique instance identifier passed back to the plugin
            handler.
        plugin_config: Plugin-specific JSON configuration.
    """

    plugin_kind: str
    instance_id: str
    plugin_config: JsonObject

    def __init__(
        self,
        plugin_kind: str,
        instance_id: str,
        plugin_config: JsonObject = ...,
    ) -> None: ...
    def to_component(self) -> ComponentSpec:
        """Convert this helper into a generic ``ComponentSpec``."""
        ...

OptimizerComponentLike = (
    ComponentSpec | TelemetryComponent | DynamoHintsComponent | ToolParallelismComponent | ExternalComponent
)

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
        from nat_nexus.optimizer import (
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

    version: int
    agent_id: str | None
    state: StateConfig | None
    components: list[OptimizerComponentLike]
    policy: ConfigPolicy

    def __init__(
        self,
        version: int = 1,
        agent_id: str | None = None,
        state: StateConfig | None = None,
        components: list[OptimizerComponentLike] = ...,
        policy: ConfigPolicy = ...,
    ) -> None: ...
    def to_dict(self) -> JsonObject:
        """Convert the optimizer config to a JSON-compatible dictionary."""
        ...

class OptimizerRuntime:
    """Runtime wrapper used to register optimizer components with Nexus.

    Args:
        config: Optimizer configuration, either as ``OptimizerConfig`` or an
            equivalent JSON object.
    """

    def __init__(self, config: OptimizerConfig | JsonObject) -> None: ...
    async def register(self) -> None:
        """Register the configured optimizer components with the runtime."""
        ...

    def deregister(self) -> None:
        """Remove optimizer-owned registrations from the runtime."""
        ...

    async def shutdown(self) -> None:
        """Shut down native optimizer resources owned by this runtime."""
        ...

    def report(self) -> ConfigReport:
        """Return the runtime's current validation and registration report."""
        ...

def validate_optimizer_config(config: OptimizerConfig | JsonObject) -> ConfigReport:
    """Validate an optimizer config without registering it.

    Args:
        config: Optimizer configuration to validate, either as
            ``OptimizerConfig`` or an equivalent JSON object.

    Returns:
        ConfigReport: Validation diagnostics produced by the native runtime.
    """
    ...

def register_optimizer_plugin(plugin_kind: str, handler: OptimizerPluginHandler) -> None:
    """Register a Python-hosted optimizer plugin handler.

    Args:
        plugin_kind: Plugin kind string referenced by ``ExternalComponent``.
        handler: Python plugin handler implementing validation and registration
            hooks for that plugin kind.
    """
    ...

def deregister_optimizer_plugin(plugin_kind: str) -> bool:
    """Remove a previously registered Python optimizer plugin handler.

    Args:
        plugin_kind: Plugin kind previously passed to
            ``register_optimizer_plugin()``.

    Returns:
        bool: ``True`` if a plugin handler was removed, otherwise ``False``.
    """
    ...

def set_latency_sensitivity(value: int) -> None:
    """Set the current latency sensitivity hint for the optimizer runtime.

    Args:
        value: Integer latency sensitivity level to forward to the native
            optimizer runtime.
    """
    ...
