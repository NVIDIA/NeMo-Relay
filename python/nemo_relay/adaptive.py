# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive plugin configuration helpers.

Adaptive is configured as a single flat top-level plugin component. Hosted
plugins remain separate top-level components managed through ``nemo_relay.plugin``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Protocol, TypedDict, cast

from nemo_relay import Json, JsonObject, UnsupportedBehavior
from nemo_relay._native import AdaptiveRuntime as AdaptiveRuntime
from nemo_relay._native import build_cache_telemetry_event as _build_cache_telemetry_event
from nemo_relay._native import set_latency_sensitivity as _set_latency_sensitivity
from nemo_relay._native import validate_adaptive_config as _validate_adaptive_config


class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One adaptive validation diagnostic."""

    component: str
    field: str


class ConfigReport(TypedDict):
    """Validation report for adaptive configuration."""

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
    """Policy for unsupported adaptive configuration.

    Args:
        unknown_component: How to handle unknown component kinds.
        unknown_field: How to handle unknown adaptive config fields.
        unsupported_value: How to handle known fields with unsupported values.
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
class BackendSpec:
    """Adaptive state backend selection.

    Args:
        kind: Backend kind string such as ``"in_memory"`` or ``"redis"``.
        config: Backend-specific JSON object.
    """

    kind: str
    config: JsonObject = field(default_factory=dict)

    @staticmethod
    def in_memory() -> "BackendSpec":
        """Return an in-memory adaptive backend spec."""
        return BackendSpec(kind="in_memory")

    @staticmethod
    def redis(url: str, key_prefix: str = "nemo_relay:") -> "BackendSpec":
        """Return a Redis adaptive backend spec."""
        return BackendSpec(kind="redis", config={"url": url, "key_prefix": key_prefix})

    def to_dict(self) -> JsonObject:
        """Serialize this backend spec to the canonical JSON object shape."""
        return {"kind": self.kind, "config": _normalize_object(self.config)}


@dataclass(slots=True)
class StateConfig:
    """Adaptive state configuration.

    Args:
        backend: State backend selection for adaptive features that persist or
            learn over time.
    """

    backend: BackendSpec

    def to_dict(self) -> JsonObject:
        """Serialize this state config to the canonical JSON object shape."""
        return {"backend": _normalize_object(self.backend)}


@dataclass(slots=True)
class TelemetryConfig:
    """Built-in adaptive telemetry subscriber settings.

    Args:
        subscriber_name: Optional subscriber registration name override.
        learners: Enabled learner identifiers.
    """

    subscriber_name: str | None = None
    learners: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonObject:
        """Serialize this telemetry config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "subscriber_name": self.subscriber_name,
                "learners": self.learners,
            }
        )


@dataclass(slots=True)
class AdaptiveHintsConfig:
    """Built-in adaptive hints injection settings.

    Args:
        priority: Intercept priority. Lower values run first.
        break_chain: Whether to stop later request intercepts after this one.
        inject_header: Whether to inject the adaptive hints HTTP header.
        inject_body_path: JSON body path used when injecting request-body hints.
    """

    priority: int = 100
    break_chain: bool = False
    inject_header: bool = True
    inject_body_path: str = "nvext.agent_hints"

    def to_dict(self) -> JsonObject:
        """Serialize this adaptive-hints config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "priority": self.priority,
                "break_chain": self.break_chain,
                "inject_header": self.inject_header,
                "inject_body_path": self.inject_body_path,
            }
        )


@dataclass(slots=True)
class AgentContextConfig:
    """Built-in agent context propagation settings.

    Args:
        priority: Intercept priority. Lower values run first.
        break_chain: Whether to stop later request intercepts after this one.
        inject_body_path: JSON body path used when injecting request-body
            agent context.
    """

    priority: int = 100
    break_chain: bool = False
    inject_body_path: str = "nvext.agent_context"

    def to_dict(self) -> JsonObject:
        """Serialize this agent-context config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "priority": self.priority,
                "break_chain": self.break_chain,
                "inject_body_path": self.inject_body_path,
            }
        )


@dataclass(slots=True)
class ToolParallelismConfig:
    """Built-in adaptive tool scheduling settings.

    Args:
        priority: Intercept priority. Lower values run first.
        mode: Scheduling mode. ``"observe_only"`` records signals without
            changing behavior, while other modes enable stronger adaptive
            scheduling behavior.
    """

    priority: int = 100
    mode: Literal["observe_only", "inject_hints", "schedule"] = "observe_only"

    def to_dict(self) -> JsonObject:
        """Serialize this tool-parallelism config to the canonical JSON object shape."""
        return _normalize_object({"priority": self.priority, "mode": self.mode})


@dataclass(slots=True)
class AcgStabilityThresholds:
    """Prompt-stability classification thresholds for ACG.

    Args:
        stable_threshold: Minimum effective score classified as stable.
        semi_stable_threshold: Minimum effective score classified as semi-stable.
        min_observations_for_full_confidence: Observation count required to
            reach full confidence.
    """

    stable_threshold: float = 0.95
    semi_stable_threshold: float = 0.50
    min_observations_for_full_confidence: int = 20

    def to_dict(self) -> JsonObject:
        """Serialize these ACG stability thresholds to the canonical JSON object shape."""
        return _normalize_object(
            {
                "stable_threshold": self.stable_threshold,
                "semi_stable_threshold": self.semi_stable_threshold,
                "min_observations_for_full_confidence": self.min_observations_for_full_confidence,
            }
        )


@dataclass(slots=True)
class AcgConfig:
    """Adaptive Cache Governor settings.

    Args:
        provider: Provider cache plugin name.
        observation_window: Rolling PromptIR observation window size.
        priority: LLM execution intercept priority.
        stability_thresholds: Prompt-stability classification thresholds.
    """

    provider: Literal["anthropic", "openai", "passthrough"] = "passthrough"
    observation_window: int = 100
    priority: int = 50
    stability_thresholds: AcgStabilityThresholds | None = field(default_factory=AcgStabilityThresholds)

    def to_dict(self) -> JsonObject:
        """Serialize this ACG config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "provider": self.provider,
                "observation_window": self.observation_window,
                "priority": self.priority,
                "stability_thresholds": _normalize(self.stability_thresholds),
            }
        )


@dataclass(slots=True)
class AdaptiveConfig:
    """Canonical config document for the top-level adaptive component.

    Args:
        version: Adaptive config schema version.
        agent_id: Optional explicit agent identifier for learned state.
        state: Adaptive state backend configuration.
        telemetry: Built-in adaptive telemetry settings.
        adaptive_hints: Built-in LLM hint-injection settings.
        agent_context: Built-in agent context propagation settings.
        tool_parallelism: Built-in tool scheduling settings.
        acg: Adaptive Cache Governor settings.
        policy: Unsupported-config policy applied within the adaptive config.

    Behavior:
        This document configures only the adaptive component. Plugins are
        configured separately through top-level plugin components.
    """

    version: int = 1
    agent_id: str | None = None
    state: StateConfig | None = None
    telemetry: TelemetryConfig | None = None
    adaptive_hints: AdaptiveHintsConfig | None = None
    agent_context: AgentContextConfig | None = None
    tool_parallelism: ToolParallelismConfig | None = None
    acg: AcgConfig | None = None
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)

    def to_dict(self) -> JsonObject:
        """Serialize this adaptive config to the canonical JSON object shape."""
        return {
            "version": self.version,
            "agent_id": self.agent_id,
            "state": _normalize(self.state),
            "telemetry": _normalize(self.telemetry),
            "adaptive_hints": _normalize(self.adaptive_hints),
            "agent_context": _normalize(self.agent_context),
            "tool_parallelism": _normalize(self.tool_parallelism),
            "acg": _normalize(self.acg),
            "policy": self.policy.to_dict(),
        }


ADAPTIVE_PLUGIN_KIND = "adaptive"


@dataclass(slots=True)
class ComponentSpec:
    """Top-level adaptive component wrapper.

    Args:
        config: ``AdaptiveConfig`` or an equivalent JSON object.
        enabled: Whether the adaptive component should be activated.

    Behavior:
        The component kind is always ``"adaptive"``.
    """

    config: AdaptiveConfig | JsonObject
    enabled: bool = True

    def to_dict(self) -> JsonObject:
        """Serialize this component to the canonical plugin shape."""
        return {
            "kind": ADAPTIVE_PLUGIN_KIND,
            "enabled": self.enabled,
            "config": _normalize_object(self.config),
        }


def validate_config(config: AdaptiveConfig | JsonObject) -> ConfigReport:
    """Validate an adaptive config document without constructing a runtime."""
    return cast(ConfigReport, _validate_adaptive_config(_normalize_object(config)))


def build_cache_telemetry_event(
    *,
    provider: str,
    request_id: str,
    usage: JsonObject | None = None,
    request_facts: JsonObject | None = None,
    agent_id: str,
    template_version: str,
    toolset_hash: str,
    model_family: str,
    tenant_scope: str,
    timestamp: str | None = None,
) -> JsonObject | None:
    """Build one canonical cache telemetry event from usage plus request facts."""
    return cast(
        JsonObject | None,
        _build_cache_telemetry_event(
            provider=provider,
            request_id=request_id,
            usage=usage,
            request_facts=request_facts,
            agent_id=agent_id,
            template_version=template_version,
            toolset_hash=toolset_hash,
            model_family=model_family,
            tenant_scope=tenant_scope,
            timestamp=timestamp,
        ),
    )


def set_latency_sensitivity(level: int) -> None:
    """Set a request-local latency-sensitivity hint.

    Args:
        level: Positive integer sensitivity value for the current execution
            context.

    Returns:
        `None`.

    Behavior:
        This is an execution-time hint for the current request/scope context,
        not persistent adaptive configuration. The native adaptive layer stores
        this as a positive integer.
    """
    _set_latency_sensitivity(level)


__all__ = [
    "AcgConfig",
    "AcgStabilityThresholds",
    "AdaptiveConfig",
    "AdaptiveHintsConfig",
    "AgentContextConfig",
    "ADAPTIVE_PLUGIN_KIND",
    "BackendSpec",
    "ConfigDiagnostic",
    "ConfigPolicy",
    "ConfigReport",
    "ComponentSpec",
    "StateConfig",
    "TelemetryConfig",
    "ToolParallelismConfig",
    "set_latency_sensitivity",
    "UnsupportedBehavior",
]
