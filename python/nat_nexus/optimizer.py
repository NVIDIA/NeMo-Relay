# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer runtime config helpers and lifecycle wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from nat_nexus._native import (
    OptimizerRuntime as _NativeOptimizerRuntime,
)
from nat_nexus._native import (
    deregister_optimizer_plugin as _deregister_optimizer_plugin,
)
from nat_nexus._native import (
    register_optimizer_plugin as _register_optimizer_plugin,
)
from nat_nexus._native import (
    set_latency_sensitivity,
)
from nat_nexus._native import (
    validate_optimizer_config as _validate_optimizer_config,
)

UnsupportedBehavior = Literal["ignore", "warn", "error"]


def _normalize(value: Any) -> Any:
    if isinstance(value, _ComponentHelper):
        return _normalize(value.to_component())
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return {key: _normalize(val) for key, val in asdict(value).items() if val is not None}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items() if val is not None}
    return value


@dataclass(slots=True)
class ConfigPolicy:
    unknown_component: UnsupportedBehavior = "warn"
    unknown_field: UnsupportedBehavior = "warn"
    unsupported_value: UnsupportedBehavior = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown_component": self.unknown_component,
            "unknown_field": self.unknown_field,
            "unsupported_value": self.unsupported_value,
        }


@dataclass(slots=True)
class BackendSpec:
    kind: str
    config: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def in_memory() -> "BackendSpec":
        return BackendSpec(kind="in_memory")

    @staticmethod
    def redis(url: str, key_prefix: str = "nexus:") -> "BackendSpec":
        return BackendSpec(kind="redis", config={"url": url, "key_prefix": key_prefix})

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config": _normalize(self.config),
        }


@dataclass(slots=True)
class StateConfig:
    backend: BackendSpec

    def to_dict(self) -> dict[str, Any]:
        return {"backend": _normalize(self.backend)}


@dataclass(slots=True)
class ComponentSpec:
    kind: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "enabled": self.enabled,
            "config": _normalize(self.config),
        }


class _ComponentHelper:
    def to_component(self) -> ComponentSpec:
        raise NotImplementedError


@dataclass(slots=True)
class TelemetryComponent(_ComponentHelper):
    subscriber_name: str | None = None
    learners: list[str] = field(default_factory=list)

    def to_component(self) -> ComponentSpec:
        return ComponentSpec(
            kind="telemetry",
            config=_normalize(
                {
                    "subscriber_name": self.subscriber_name,
                    "learners": self.learners,
                }
            ),
        )


@dataclass(slots=True)
class DynamoHintsComponent(_ComponentHelper):
    priority: int = 100
    break_chain: bool = False
    inject_header: bool = True
    inject_body_path: str = "nvext.agent_hints"

    def to_component(self) -> ComponentSpec:
        return ComponentSpec(
            kind="dynamo_hints",
            config=_normalize(
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
    priority: int = 100
    mode: Literal["observe_only", "inject_hints", "schedule"] = "observe_only"

    def to_component(self) -> ComponentSpec:
        return ComponentSpec(
            kind="tool_parallelism",
            config=_normalize({"priority": self.priority, "mode": self.mode}),
        )


@dataclass(slots=True)
class ExternalComponent(_ComponentHelper):
    plugin_kind: str
    instance_id: str
    plugin_config: dict[str, Any] = field(default_factory=dict)

    def to_component(self) -> ComponentSpec:
        return ComponentSpec(
            kind="external_component",
            config=_normalize(
                {
                    "plugin_kind": self.plugin_kind,
                    "instance_id": self.instance_id,
                    "plugin_config": self.plugin_config,
                }
            ),
        )


@dataclass(slots=True)
class OptimizerConfig:
    version: int = 1
    agent_id: str | None = None
    state: StateConfig | None = None
    components: list[ComponentSpec | _ComponentHelper] = field(default_factory=list)
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "agent_id": self.agent_id,
            "state": _normalize(self.state),
            "components": [_normalize(component) for component in self.components],
            "policy": self.policy.to_dict(),
        }


class OptimizerRuntime:
    def __init__(self, config: OptimizerConfig | dict[str, Any]) -> None:
        self._config = _normalize(config)
        self._native = _NativeOptimizerRuntime(self._config)

    async def register(self) -> None:
        await self._native.register()

    def deregister(self) -> None:
        self._native.deregister()

    async def shutdown(self) -> None:
        await self._native.shutdown()

    def report(self) -> dict[str, Any]:
        return self._native.report()


def validate_optimizer_config(config: OptimizerConfig | dict[str, Any]) -> dict[str, Any]:
    return _validate_optimizer_config(_normalize(config))


def register_optimizer_plugin(plugin_kind: str, handler: Any) -> None:
    _register_optimizer_plugin(plugin_kind, handler)


def deregister_optimizer_plugin(plugin_kind: str) -> bool:
    return _deregister_optimizer_plugin(plugin_kind)


__all__ = [
    "BackendSpec",
    "ComponentSpec",
    "ConfigPolicy",
    "DynamoHintsComponent",
    "ExternalComponent",
    "OptimizerConfig",
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
