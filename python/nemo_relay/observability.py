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
class AtofStreamSinkConfig:
    """Stream sink for raw ATOF events."""

    url: str
    transport: Literal["http_post", "websocket", "ndjson"] = "http_post"
    headers: dict[str, str] = field(default_factory=dict)
    header_env: dict[str, str] = field(default_factory=dict)
    timeout_millis: int = 3000
    field_name_policy: Literal["preserve", "replace_dots"] = "preserve"
    name: str | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this ATOF stream sink to the canonical JSON object shape."""
        return _normalize_object(
            {
                "type": "stream",
                "name": self.name,
                "url": self.url,
                "transport": self.transport,
                "headers": self.headers,
                "header_env": self.header_env,
                "timeout_millis": self.timeout_millis,
                "field_name_policy": self.field_name_policy,
            }
        )


@dataclass(slots=True)
class AtofConfig:
    """Multi-sink raw ATOF export settings."""

    enabled: bool = False
    sinks: list[AtofFileSinkConfig | AtofStreamSinkConfig] | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this ATOF config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "enabled": self.enabled,
                "sinks": self.sinks,
            }
        )


@dataclass(slots=True)
class AtofFileSinkConfig:
    """Filesystem destination for raw ATOF JSONL events."""

    output_directory: str | None = None
    filename: str | None = None
    mode: Literal["append", "overwrite"] = "append"

    def to_dict(self) -> JsonObject:
        return _normalize_object(
            {
                "type": "file",
                "output_directory": self.output_directory,
                "filename": self.filename,
                "mode": self.mode,
            }
        )


# Compatibility alias for the former plugin helper name.
AtofEndpointConfig = AtofStreamSinkConfig


@dataclass(slots=True)
class S3StorageConfig:
    """S3-compatible remote storage settings for ATIF trajectory upload.

    Every connection field is optional. Unset fields fall back to the matching
    ``AWS_*`` environment variable. Secret credentials are referenced by env
    var *name* (the ``_var`` suffix), validated at plugin initialization time,
    so multiple destinations can each carry their own credentials without
    leaking secret material into the config.
    """

    bucket: str = ""
    key_prefix: str | None = None
    access_key_id: str | None = None
    secret_access_key_var: str | None = None
    session_token_var: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    allow_http: bool | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this S3 storage config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "type": "s3",
                "bucket": self.bucket,
                "key_prefix": self.key_prefix,
                "access_key_id": self.access_key_id,
                "secret_access_key_var": self.secret_access_key_var,
                "session_token_var": self.session_token_var,
                "region": self.region,
                "endpoint_url": self.endpoint_url,
                "allow_http": self.allow_http,
            }
        )


@dataclass(slots=True)
class HttpStorageConfig:
    """HTTP endpoint settings for ATIF trajectory upload."""

    endpoint: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    header_env: dict[str, str] = field(default_factory=dict)
    timeout_millis: int = 3000

    def to_dict(self) -> JsonObject:
        """Serialize this HTTP storage config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "type": "http",
                "endpoint": self.endpoint,
                "headers": self.headers,
                "header_env": self.header_env,
                "timeout_millis": self.timeout_millis,
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
    storage: list[S3StorageConfig | HttpStorageConfig] | None = None

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
            "storage": self.storage,
        }
        if value["agent_version"] is None:
            value.pop("agent_version")
        return _normalize_object(value)


@dataclass(slots=True)
class OpenTelemetryEndpointConfig:
    """One typed OpenTelemetry OTLP destination."""

    type: Literal["full", "gen_ai", "openinference"]
    endpoint: str
    mark_projection: Literal["inherit", "event", "tool"] = "inherit"
    mark_exclude_names: list[str] = field(default_factory=lambda: ["llm.chunk"])
    attribute_mappings: list[dict[str, str]] = field(default_factory=list)
    transport: Literal["http_binary", "grpc"] = "http_binary"
    service_name: str = "unknown_service"
    service_namespace: str | None = None
    service_version: str | None = None
    instrumentation_scope: str = "opentelemetry"
    timeout_millis: int = 3000
    headers: dict[str, str] = field(default_factory=dict)
    header_env: dict[str, str] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)
    max_queue_size: int | None = None
    max_export_batch_size: int | None = None
    scheduled_delay_millis: int | None = None
    completed_span_context_ttl_millis: int | None = None
    promote_metadata_prefixes: list[str] = field(default_factory=list)
    promote_resource_metadata_prefixes: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonObject:
        """Serialize this endpoint to the canonical plugin shape."""
        return _normalize_object(
            {
                "type": self.type,
                "endpoint": self.endpoint,
                "mark_projection": self.mark_projection,
                "mark_exclude_names": self.mark_exclude_names,
                "attribute_mappings": self.attribute_mappings,
                "promote_metadata_prefixes": self.promote_metadata_prefixes,
                "promote_resource_metadata_prefixes": self.promote_resource_metadata_prefixes,
                "transport": self.transport,
                "service_name": self.service_name,
                "service_namespace": self.service_namespace,
                "service_version": self.service_version,
                "instrumentation_scope": self.instrumentation_scope,
                "timeout_millis": self.timeout_millis,
                "max_queue_size": self.max_queue_size,
                "max_export_batch_size": self.max_export_batch_size,
                "scheduled_delay_millis": self.scheduled_delay_millis,
                "completed_span_context_ttl_millis": self.completed_span_context_ttl_millis,
                "headers": self.headers,
                "header_env": self.header_env,
                "resource_attributes": self.resource_attributes,
            }
        )


@dataclass(slots=True)
class OpenTelemetrySignalEndpointConfig:
    """One signal-specific OTLP destination for logs or metrics."""

    endpoint: str
    transport: Literal["http_binary", "grpc"] = "http_binary"
    headers: dict[str, str] = field(default_factory=dict)
    header_env: dict[str, str] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)
    service_name: str = "unknown_service"
    service_namespace: str | None = None
    service_version: str | None = None
    instrumentation_scope: str = "opentelemetry"
    timeout_millis: int = 3000

    def to_dict(self) -> JsonObject:
        """Serialize this signal endpoint to the canonical plugin shape."""
        return _normalize_object(
            {
                "endpoint": self.endpoint,
                "transport": self.transport,
                "headers": self.headers,
                "header_env": self.header_env,
                "resource_attributes": self.resource_attributes,
                "service_name": self.service_name,
                "service_namespace": self.service_namespace,
                "service_version": self.service_version,
                "instrumentation_scope": self.instrumentation_scope,
                "timeout_millis": self.timeout_millis,
            }
        )


@dataclass(slots=True)
class OpenTelemetryLogSectionConfig:
    """OTLP log pipeline settings sourced from non-metric marks."""

    enabled: bool = False
    endpoints: list[OpenTelemetrySignalEndpointConfig] | None = None
    minimum_severity: Literal["trace", "debug", "info", "warn", "warning", "error"] = "info"
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    scheduled_delay_millis: int = 1000
    completed_span_context_ttl_millis: int = 60000

    def to_dict(self) -> JsonObject:
        """Serialize this log section to the canonical plugin shape."""
        return _normalize_object(
            {
                "enabled": self.enabled,
                "endpoints": self.endpoints,
                "minimum_severity": self.minimum_severity,
                "max_queue_size": self.max_queue_size,
                "max_export_batch_size": self.max_export_batch_size,
                "scheduled_delay_millis": self.scheduled_delay_millis,
            }
        )


@dataclass(slots=True)
class OpenTelemetryMetricSectionConfig:
    """OTLP metric pipeline settings sourced from metric marks."""

    enabled: bool = False
    endpoints: list[OpenTelemetrySignalEndpointConfig] | None = None
    export_interval_millis: int = 60_000
    temporality: Literal["cumulative", "delta", "low_memory"] = "cumulative"
    max_instruments: int = 256
    cardinality_limit: int = 2000

    def to_dict(self) -> JsonObject:
        """Serialize this metric section to the canonical plugin shape."""
        return _normalize_object(
            {
                "enabled": self.enabled,
                "endpoints": self.endpoints,
                "export_interval_millis": self.export_interval_millis,
                "temporality": self.temporality,
                "max_instruments": self.max_instruments,
                "cardinality_limit": self.cardinality_limit,
            }
        )


@dataclass(slots=True)
class OpenTelemetrySectionConfig:
    """Multi-endpoint OpenTelemetry plugin settings."""

    enabled: bool = False
    endpoints: list[OpenTelemetryEndpointConfig] = field(default_factory=list)
    logs: OpenTelemetryLogSectionConfig | None = None
    metrics: OpenTelemetryMetricSectionConfig | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this section to the canonical plugin shape."""
        return _normalize_object(
            {
                "enabled": self.enabled,
                "endpoints": self.endpoints,
                "logs": self.logs,
                "metrics": self.metrics,
            }
        )


@dataclass(slots=True)
class ObservabilityConfig:
    """Canonical config document for the top-level observability component.

    ``enable_full_payloads`` retains complete sanitized request data on every
    LLM start event.
    """

    version: int = 4
    atof: AtofConfig | None = None
    atif: AtifConfig | None = None
    opentelemetry: OpenTelemetrySectionConfig | None = None
    policy: ConfigPolicy = field(default_factory=ConfigPolicy)
    enable_full_payloads: bool = False

    def to_dict(self) -> JsonObject:
        """Serialize this observability config to the canonical JSON object shape."""
        return _normalize_object(
            {
                "version": self.version,
                "atof": self.atof,
                "atif": self.atif,
                "opentelemetry": self.opentelemetry,
                "policy": self.policy,
                "enable_full_payloads": self.enable_full_payloads,
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
    "AtofEndpointConfig",
    "AtofFileSinkConfig",
    "AtofStreamSinkConfig",
    "AtofConfig",
    "AtifConfig",
    "HttpStorageConfig",
    "S3StorageConfig",
    "OpenTelemetryEndpointConfig",
    "OpenTelemetrySignalEndpointConfig",
    "OpenTelemetryLogSectionConfig",
    "OpenTelemetryMetricSectionConfig",
    "OpenTelemetrySectionConfig",
    "ObservabilityConfig",
    "OBSERVABILITY_PLUGIN_KIND",
    "ComponentSpec",
]
