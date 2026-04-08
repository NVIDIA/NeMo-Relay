# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

UnsupportedBehavior = Literal["ignore", "warn", "error"]

class ConfigPolicy:
    unknown_component: UnsupportedBehavior
    unknown_field: UnsupportedBehavior
    unsupported_value: UnsupportedBehavior

    def __init__(
        self,
        unknown_component: UnsupportedBehavior = "warn",
        unknown_field: UnsupportedBehavior = "warn",
        unsupported_value: UnsupportedBehavior = "error",
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class BackendSpec:
    kind: str
    config: dict[str, Any]

    def __init__(self, kind: str, config: dict[str, Any] = ...) -> None: ...
    @staticmethod
    def in_memory() -> BackendSpec: ...
    @staticmethod
    def redis(url: str, key_prefix: str = "nexus:") -> BackendSpec: ...
    def to_dict(self) -> dict[str, Any]: ...

class StateConfig:
    backend: BackendSpec

    def __init__(self, backend: BackendSpec) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ComponentSpec:
    kind: str
    enabled: bool
    config: dict[str, Any]

    def __init__(self, kind: str, enabled: bool = True, config: dict[str, Any] = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class TelemetryComponent:
    subscriber_name: str | None
    learners: list[str]

    def __init__(
        self,
        subscriber_name: str | None = None,
        learners: list[str] = ...,
    ) -> None: ...
    def to_component(self) -> ComponentSpec: ...

class DynamoHintsComponent:
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
    def to_component(self) -> ComponentSpec: ...

class ToolParallelismComponent:
    priority: int
    mode: Literal["observe_only", "inject_hints", "schedule"]

    def __init__(
        self,
        priority: int = 100,
        mode: Literal["observe_only", "inject_hints", "schedule"] = "observe_only",
    ) -> None: ...
    def to_component(self) -> ComponentSpec: ...

class ExternalComponent:
    plugin_kind: str
    instance_id: str
    plugin_config: dict[str, Any]

    def __init__(
        self,
        plugin_kind: str,
        instance_id: str,
        plugin_config: dict[str, Any] = ...,
    ) -> None: ...
    def to_component(self) -> ComponentSpec: ...

OptimizerComponentLike = (
    ComponentSpec | TelemetryComponent | DynamoHintsComponent | ToolParallelismComponent | ExternalComponent
)

class OptimizerConfig:
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
    def to_dict(self) -> dict[str, Any]: ...

class OptimizerRuntime:
    def __init__(self, config: OptimizerConfig | dict[str, Any]) -> None: ...
    async def register(self) -> None: ...
    def deregister(self) -> None: ...
    async def shutdown(self) -> None: ...
    def report(self) -> dict[str, Any]: ...

def validate_optimizer_config(config: OptimizerConfig | dict[str, Any]) -> dict[str, Any]: ...
def register_optimizer_plugin(plugin_kind: str, handler: Any) -> None: ...
def deregister_optimizer_plugin(plugin_kind: str) -> bool: ...
def set_latency_sensitivity(value: int) -> None: ...
