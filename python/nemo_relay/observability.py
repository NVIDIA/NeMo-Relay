# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability plugin configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Protocol, cast

from nemo_relay import Json, JsonObject, UnsupportedBehavior


class _SupportsToDict(Protocol):
    def to_dict(self) -> JsonObject: ...


def _normalize(value: object) -> Json:
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
    return cast(JsonObject, _normalize(value))


@dataclass(slots=True)
class ConfigPolicy:
    """Policy for unsupported observability configuration."""

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
class AtofConfig:
    """Filesystem-backed raw ATOF JSONL export settings."""

    enabled: bool = False
    output_directory: str | None = None
    filename: str | None = None
    mode: Literal["append", "overwrite"] = "append"

    def to_dict(self) -> JsonObject:
        """Serialize this ATOF config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "enabled": self.enabled,
                "output_directory": self.output_directory,
                "filename": self.filename,
                "mode": self.mode,
            }
        )


@dataclass(slots=True)
class AtifConfig:
    """Per-top-level-agent ATIF file export settings."""

    enabled: bool = False
    agent_name: str = "NeMo Relay"
    agent_version: str | None = None
    model_name: str = "unknown"
    tool_definitions: list[JsonObject] | None = None
    extra: JsonObject | None = None
    output_directory: str | None = None
    filename_template: str = "nemo-relay-atif-{session_id}.json"

    def to_dict(self) -> JsonObject:
        """Serialize this ATIF config to the canonical JSON object shape."""
        value = {
            "enabled": self.enabled,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model_name": self.model_name,
            "tool_definitions": self.tool_definitions,
            "extra": self.extra,
            "output_directory": self.output_directory,
            "filename_template": self.filename_template,
        }
        if value["agent_version"] is None:
            value.pop("agent_version")
        return _normalize_object(value)


@dataclass(slots=True)
class OtlpConfig:
    """Shared OpenTelemetry/OpenInference OTLP export settings."""

    enabled: bool = False
    transport: Literal["http_binary", "grpc"] = "http_binary"
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)
    service_name: str = "nemo-relay"
    service_namespace: str | None = None
    service_version: str | None = None
    instrumentation_scope: str | None = None
    timeout_millis: int = 3000

    def to_dict(self) -> JsonObject:
        """Serialize this OTLP config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "enabled": self.enabled,
                "transport": self.transport,
                "endpoint": self.endpoint,
                "headers": self.headers,
                "resource_attributes": self.resource_attributes,
                "service_name": self.service_name,
                "service_namespace": self.service_namespace,
                "service_version": self.service_version,
                "instrumentation_scope": self.instrumentation_scope,
                "timeout_millis": self.timeout_millis,
            }
        )


@dataclass(slots=True)
class ObservabilityConfig:
    """Canonical config document for the top-level observability component."""

    version: int = 1
    atof: AtofConfig | None = None
    atif: AtifConfig | None = None
    opentelemetry: OtlpConfig | None = None
    openinference: OtlpConfig | None = None
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> JsonObject:
        """Serialize this observability config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "version": self.version,
                "atof": self.atof,
                "atif": self.atif,
                "opentelemetry": self.opentelemetry,
                "openinference": self.openinference,
                "policy": self.policy,
            }
        )


OBSERVABILITY_PLUGIN_KIND = "observability"


@dataclass(slots=True)
class ComponentSpec:
    """Top-level observability component wrapper."""

    config: ObservabilityConfig | JsonObject
    enabled: bool = True

    def to_dict(self) -> JsonObject:
        """Serialize this component to the canonical plugin shape."""
        return {
            "kind": OBSERVABILITY_PLUGIN_KIND,
            "enabled": self.enabled,
            "config": _normalize_object(self.config),
        }


__all__ = [
    "ConfigPolicy",
    "AtofConfig",
    "AtifConfig",
    "OtlpConfig",
    "ObservabilityConfig",
    "OBSERVABILITY_PLUGIN_KIND",
    "ComponentSpec",
]
