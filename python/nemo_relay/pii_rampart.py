# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Rampart PII plugin configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict, cast

from nemo_relay import JsonObject
from nemo_relay import plugin as plugin_module
from nemo_relay.plugin import ConfigDiagnostic, ConfigPolicy


class ConfigReport(TypedDict):
    """Validation report for Rampart PII configuration."""

    diagnostics: list[ConfigDiagnostic]


@dataclass(slots=True)
class RampartPiiConfig:
    """Canonical config for in-process Rampart PII redaction."""

    model_path: str
    version: int = 1
    input: bool = True
    output: bool = True
    mark: bool = True
    tool_input: bool = True
    tool_output: bool = True
    priority: int = 100
    codec: Literal["openai_chat", "openai_responses", "anthropic_messages"] | str | None = None
    target_paths: list[str] = field(default_factory=list)
    target_path_patterns: list[str] = field(default_factory=list)
    min_score: float = 0.4
    excluded_labels: list[str] = field(default_factory=list)
    replacement: str = "[REDACTED]"
    max_windows_per_payload: int = 128
    inference_batch_size: int = 16
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> JsonObject:
        """Serialize this config to the canonical JSON object shape."""
        value: JsonObject = {
            "version": self.version,
            "model_path": self.model_path,
            "input": self.input,
            "output": self.output,
            "mark": self.mark,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "priority": self.priority,
            "target_paths": self.target_paths,
            "target_path_patterns": self.target_path_patterns,
            "min_score": self.min_score,
            "excluded_labels": self.excluded_labels,
            "replacement": self.replacement,
            "max_windows_per_payload": self.max_windows_per_payload,
            "inference_batch_size": self.inference_batch_size,
            "policy": self.policy.to_dict(),
        }
        if self.codec is not None:
            value["codec"] = self.codec
        return value


RAMPART_PII_PLUGIN_KIND = "pii_rampart"
RAMPART_MODEL_ID = "nationaldesignstudio/rampart"
RAMPART_MODEL_REVISION = "b1993e4e68b082835b80ffc65acc03325ea2e501"


@dataclass(slots=True)
class ComponentSpec:
    """Top-level Rampart PII component wrapper."""

    config: RampartPiiConfig | JsonObject
    enabled: bool = True

    def to_dict(self) -> JsonObject:
        """Serialize this component to the canonical plugin shape."""
        config = self.config.to_dict() if isinstance(self.config, RampartPiiConfig) else self.config
        return {
            "kind": RAMPART_PII_PLUGIN_KIND,
            "enabled": self.enabled,
            "config": config,
        }


def validate_config(config: RampartPiiConfig | JsonObject) -> ConfigReport:
    """Validate Rampart PII configuration without loading the model."""
    report = plugin_module.validate(plugin_module.PluginConfig(components=[ComponentSpec(config)]))
    return cast(ConfigReport, report)


__all__ = [
    "RAMPART_MODEL_ID",
    "RAMPART_MODEL_REVISION",
    "RAMPART_PII_PLUGIN_KIND",
    "ComponentSpec",
    "ConfigReport",
    "RampartPiiConfig",
    "validate_config",
]
