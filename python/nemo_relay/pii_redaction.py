# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PII redaction plugin configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Protocol, TypedDict, cast

from nemo_relay import Json, JsonObject, UnsupportedBehavior
from nemo_relay import plugin as plugin_module


class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One PII redaction validation diagnostic."""

    component: str
    field: str


class ConfigReport(TypedDict):
    """Validation report for PII redaction configuration."""

    diagnostics: list[ConfigDiagnostic]


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
    """Policy for unsupported PII redaction configuration."""

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
class BuiltinConfig:
    """Deterministic built-in redaction backend settings."""

    action: Literal["remove", "redact", "regex_replace", "hash", "mask"] = "remove"
    target_paths: list[str] = field(default_factory=list)
    pattern: str | None = None
    detector: str | None = None
    replacement: str | None = None
    mask_char: str | None = None
    unmasked_prefix: int | None = None
    unmasked_suffix: int | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this built-in backend config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "action": self.action,
                "target_paths": self.target_paths,
                "pattern": self.pattern,
                "detector": self.detector,
                "replacement": self.replacement,
                "mask_char": self.mask_char,
                "unmasked_prefix": self.unmasked_prefix,
                "unmasked_suffix": self.unmasked_suffix,
            }
        )


@dataclass(slots=True)
class LocalModelConfig:
    """Future local-model backend seam settings."""

    backend: str | None = None
    model_id: str | None = None
    detector_profile: str | None = None
    allow_network: bool | None = None
    max_latency_ms: int | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this local-model config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "backend": self.backend,
                "model_id": self.model_id,
                "detector_profile": self.detector_profile,
                "allow_network": self.allow_network,
                "max_latency_ms": self.max_latency_ms,
            }
        )


@dataclass(slots=True)
class PiiRedactionConfig:
    """Canonical config document for the top-level PII redaction component."""

    version: int = 1
    mode: Literal["builtin", "local_model"] = "builtin"
    input: bool = True
    output: bool = True
    tool_input: bool = True
    tool_output: bool = True
    priority: int = 100
    codec: Literal["openai_chat", "openai_responses", "anthropic_messages"] | str | None = None
    builtin: BuiltinConfig | None = None
    local: LocalModelConfig | None = None
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> JsonObject:
        """Serialize this PII redaction config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "version": self.version,
                "mode": self.mode,
                "input": self.input,
                "output": self.output,
                "tool_input": self.tool_input,
                "tool_output": self.tool_output,
                "priority": self.priority,
                "codec": self.codec,
                "builtin": self.builtin,
                "local": self.local,
                "policy": self.policy,
            }
        )


PII_REDACTION_PLUGIN_KIND = "pii_redaction"


@dataclass(slots=True)
class ComponentSpec:
    """Top-level PII redaction component wrapper."""

    config: PiiRedactionConfig | JsonObject
    enabled: bool = True

    def to_dict(self) -> JsonObject:
        """Serialize this component to the canonical plugin shape."""
        return {
            "kind": PII_REDACTION_PLUGIN_KIND,
            "enabled": self.enabled,
            "config": _normalize_object(self.config),
        }


def validate_config(config: PiiRedactionConfig | JsonObject) -> ConfigReport:
    """Validate a PII redaction config document without activating it."""
    report = plugin_module.validate(
        plugin_module.PluginConfig(
            components=[ComponentSpec(config)],
        )
    )
    return cast(ConfigReport, report)


__all__ = [
    "BuiltinConfig",
    "ComponentSpec",
    "ConfigDiagnostic",
    "ConfigPolicy",
    "ConfigReport",
    "LocalModelConfig",
    "PII_REDACTION_PLUGIN_KIND",
    "PiiRedactionConfig",
    "validate_config",
]
