# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for ``nemo_relay.adaptive``.

This module exposes the canonical adaptive configuration helpers, the
``AdaptiveRuntime`` bridge used by external integrations, and cache-telemetry
helpers that summarize ACG observations into structured JSON payloads.
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from nemo_relay import JsonObject, ScopeHandle, UnsupportedBehavior

__all__: list[str]

class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str

class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One adaptive configuration diagnostic."""

    component: str
    field: str

class ConfigReport(TypedDict):
    """Validation report returned by adaptive configuration helpers."""

    diagnostics: list[ConfigDiagnostic]

@dataclass(slots=True)
class ConfigPolicy:
    """Policy for unsupported adaptive configuration.

    Args:
        unknown_component: How to handle unknown component kinds.
        unknown_field: How to handle unknown adaptive config fields.
        unsupported_value: How to handle known fields with unsupported values.
    """

    unknown_component: UnsupportedBehavior = ...
    unknown_field: UnsupportedBehavior = ...
    unsupported_value: UnsupportedBehavior = ...

    def to_dict(self) -> JsonObject:
        """Serialize this policy to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class BackendSpec:
    """Adaptive state backend selection.

    Args:
        kind: Backend kind string such as ``"in_memory"`` or ``"redis"``.
        config: Backend-specific JSON object.
    """

    kind: str
    config: JsonObject = ...

    @staticmethod
    def in_memory() -> "BackendSpec":
        """Return an in-memory adaptive backend spec."""
        ...

    @staticmethod
    def redis(url: str, key_prefix: str = ...) -> "BackendSpec":
        """Return a Redis adaptive backend spec."""
        ...

    def to_dict(self) -> JsonObject:
        """Serialize this backend spec to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class StateConfig:
    """Adaptive state configuration.

    Args:
        backend: Backend used to persist learned adaptive state.
    """

    backend: BackendSpec

    def to_dict(self) -> JsonObject:
        """Serialize this state config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class TelemetryConfig:
    """Built-in adaptive telemetry subscriber settings.

    Args:
        subscriber_name: Optional subscriber registration name override.
        learners: Enabled learner identifiers.
    """

    subscriber_name: str | None = ...
    learners: list[str] = ...

    def to_dict(self) -> JsonObject:
        """Serialize this telemetry config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class GovernorConfig:
    """Topology-aware load-shedding settings for adaptive hints."""

    enabled: bool = ...
    epsilon: float = ...

    def to_dict(self) -> JsonObject:
        """Serialize this governor config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class DriftConfig:
    """Topology-aware drift detection settings for tool plans."""

    enabled: bool = ...
    threshold: float = ...

    def to_dict(self) -> JsonObject:
        """Serialize this drift config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class ConvergenceConfig:
    """Topological convergence detection settings."""

    enabled: bool = ...
    epsilon: float = ...
    stability_window: int = ...

    def to_dict(self) -> JsonObject:
        """Serialize this convergence config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class AdaptiveHintsConfig:
    """Built-in adaptive hints injection settings.

    Args:
        priority: Intercept priority. Lower values run first.
        break_chain: Whether to stop later request intercepts after this one.
        inject_header: Whether to inject the adaptive hints HTTP header.
        inject_body_path: JSON body path used when injecting request-body hints.
        governor: Optional topology-aware load-shedding settings.
    """

    priority: int = ...
    break_chain: bool = ...
    inject_header: bool = ...
    inject_body_path: str = ...
    governor: GovernorConfig | None = ...

    def to_dict(self) -> JsonObject:
        """Serialize this adaptive-hints config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class ToolParallelismConfig:
    """Built-in adaptive tool scheduling settings.

    Args:
        priority: Intercept priority. Lower values run first.
        mode: Scheduling mode. ``"observe_only"`` records signals without
            changing behavior, while stronger modes allow adaptive scheduling.
        drift: Optional topology-aware stale-plan invalidation settings.
    """

    priority: int = ...
    mode: Literal["observe_only", "inject_hints", "schedule"] = ...
    drift: DriftConfig | None = ...

    def to_dict(self) -> JsonObject:
        """Serialize this tool-parallelism config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class AcgStabilityThresholds:
    """Prompt-stability classification thresholds for ACG.

    Args:
        stable_threshold: Minimum effective score classified as stable.
        semi_stable_threshold: Minimum effective score classified as semi-stable.
        min_observations_for_full_confidence: Observation count required to
            reach full confidence.
    """

    stable_threshold: float = ...
    semi_stable_threshold: float = ...
    min_observations_for_full_confidence: int = ...

    def to_dict(self) -> JsonObject:
        """Serialize these thresholds to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class AcgConfig:
    """Adaptive Cache Governor settings.

    Args:
        provider: Provider cache plugin name.
        observation_window: Rolling PromptIR observation window size.
        priority: LLM execution intercept priority used by ACG.
        stability_thresholds: Prompt-stability classification thresholds.
        convergence: Optional component-scoped topological convergence settings.
    """

    provider: Literal["anthropic", "openai", "passthrough"] = ...
    observation_window: int = ...
    priority: int = ...
    stability_thresholds: AcgStabilityThresholds | None = ...
    convergence: ConvergenceConfig | None = ...

    def to_dict(self) -> JsonObject:
        """Serialize this ACG config to the canonical JSON object shape."""
        ...

@dataclass(slots=True)
class AdaptiveConfig:
    """Canonical config document for the top-level adaptive component.

    Args:
        version: Adaptive config schema version.
        agent_id: Optional explicit agent identifier for learned state.
        state: Adaptive state backend configuration.
        telemetry: Built-in adaptive telemetry subscriber settings.
        adaptive_hints: Built-in adaptive request-hints configuration.
        tool_parallelism: Built-in adaptive tool-scheduling configuration.
        acg: Adaptive Cache Governor configuration.
        convergence: Global topological convergence settings.
        policy: Policy for unsupported adaptive configuration.
    """

    version: int = ...
    agent_id: str | None = ...
    state: StateConfig | None = ...
    telemetry: TelemetryConfig | None = ...
    adaptive_hints: AdaptiveHintsConfig | None = ...
    tool_parallelism: ToolParallelismConfig | None = ...
    acg: AcgConfig | None = ...
    convergence: ConvergenceConfig | None = ...
    policy: ConfigPolicy = ...

    def to_dict(self) -> JsonObject:
        """Serialize this adaptive config to the canonical JSON object shape."""
        ...

ADAPTIVE_PLUGIN_KIND: Literal["adaptive"]
"""Registered plugin kind string for the top-level adaptive component."""

@dataclass(slots=True)
class ComponentSpec:
    """Plugin-ready wrapper for one adaptive component.

    Args:
        config: Adaptive component config document.
        enabled: Whether the component should be activated.
    """

    config: AdaptiveConfig | JsonObject
    enabled: bool = ...

    def to_dict(self) -> JsonObject:
        """Serialize this component wrapper to the core plugin JSON shape."""
        ...

class AdaptiveRuntime:
    """Hosted adaptive runtime wrapper used by external framework integrations.

    ``AdaptiveRuntime`` validates and stores one adaptive config, registers the
    configured adaptive features into the shared NeMo Relay runtime, and exposes
    helpers that depend on the runtime's hot cache and registered agent
    identity.
    """

    def __init__(self, config: AdaptiveConfig | JsonObject) -> None:
        """Construct a pending adaptive runtime from configuration."""
        ...

    async def register(self) -> None:
        """Register the configured adaptive features with NeMo Relay."""
        ...

    def deregister(self) -> None:
        """Deregister previously registered adaptive features."""
        ...

    async def shutdown(self) -> None:
        """Deregister the runtime and release its owned resources."""
        ...

    def wait_for_idle(self) -> None:
        """Block until the adaptive telemetry drain has processed pending work."""
        ...

    def report(self) -> ConfigReport:
        """Return the validation report associated with this runtime."""
        ...

    def bind_scope(self, scope_handle: ScopeHandle) -> None:
        """Bind this runtime's ACG request rewriting to an active scope.

        After binding, requests emitted under ``scope_handle`` can explicitly
        call ``nemo_relay.llm.request_intercepts(...)`` to apply the runtime's
        provider-native ACG rewrite path.
        """
        ...

    def build_cache_request_facts(
        self,
        *,
        provider: str,
        request_id: str,
        annotated_request: object,
        agent_id: str,
        timestamp: str | None = ...,
    ) -> JsonObject | None:
        """Build cache-diagnostics facts for an annotated request.

        Args:
            provider: Logical provider name associated with the request.
            request_id: Stable request UUID.
            annotated_request: ``AnnotatedLLMRequest`` or equivalent mapping.
            agent_id: Agent identifier associated with the request.
            timestamp: Optional RFC 3339 timestamp override.

        Returns:
            dict | None: Derived cache facts when enough hot-cache state exists,
            otherwise ``None``.
        """
        ...

def validate_config(config: AdaptiveConfig | JsonObject) -> ConfigReport:
    """Validate adaptive configuration without constructing a runtime."""
    ...

def build_cache_telemetry_event(
    *,
    provider: str,
    request_id: str,
    usage: JsonObject | None = ...,
    request_facts: JsonObject | None = ...,
    agent_id: str,
    template_version: str,
    toolset_hash: str,
    model_family: str,
    tenant_scope: str,
    timestamp: str | None = ...,
) -> JsonObject | None:
    """Build one normalized cache-telemetry event payload.

    Returns ``None`` when the supplied inputs do not produce a valid telemetry
    event for the selected provider.
    """
    ...

def set_latency_sensitivity(level: int) -> None:
    """Set the process-local manual latency-sensitivity override."""
    ...
