// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in observability plugin component.
//!
//! This module packages NeMo Relay's first-party observability exporters behind
//! the shared plugin configuration system. Each exporter section is opt-in:
//! omitted sections and sections with `enabled = false` validate but do not
//! register subscribers or construct exporters.
//!
//! The plugin intentionally infers subscriber names from the component namespace
//! so configuration remains portable across bindings. Agent Trajectory
//! Observability Format (ATOF) registers one global subscriber when enabled.
//! Typed OpenTelemetry endpoints share one global fan-out subscriber. Agent
//! Trajectory Interchange Format (ATIF) uses a global dispatcher that detects
//! top-level agent or turn scopes and creates one scope-local exporter for each
//! trajectory run. Coding-agent turns that need bounded traces carry role
//! metadata; their declared scope type is preserved in the exported event
//! stream.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::net::IpAddr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
#[cfg(feature = "object-store")]
use std::sync::atomic::AtomicU8;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use uuid::Uuid;

use super::private_file::atomic_private_write;
use crate::api::event::{Event, LogSeverity, ScopeCategory, ValidatedMetricMeasurement};
use crate::api::runtime::{EventSubscriberFn, current_scope_stack, global_context};
use crate::api::scope::ScopeType;
use crate::api::subscriber::{
    flush_subscribers, scope_deregister_subscriber, try_scope_deregister_subscriber,
    try_scope_register_subscriber,
};
use crate::config_editor::{
    EditorConfig, EditorFieldKind, EditorFieldSpec, EditorListItemSpec, EditorSchema,
    EditorTaggedUnionSpec, EditorVariantSpec,
};
use crate::error::FlowError;
use crate::observability::atif::{AtifAgentInfo, AtifExporter};
use crate::observability::atof::{
    AtofEndpointFieldNamePolicy, AtofEndpointTransport, AtofExporter,
    AtofExporterConfig as CoreAtofExporterConfig, AtofExporterMode, AtofFileSinkConfig,
    AtofSinkConfig as CoreAtofSinkConfig, AtofStreamSinkConfig,
};
use crate::observability::otel::{
    OpenTelemetryConfig as CoreOpenTelemetryConfig, OpenTelemetrySubscriber, OtlpTransport,
    resolve_http_trace_endpoint,
};
use crate::observability::otel_logs::{
    OpenTelemetryLogConfig as CoreOpenTelemetryLogConfig, OpenTelemetryLogSubscriber,
    resolve_http_log_endpoint,
};
use crate::observability::otel_metrics::{
    MetricTemporality, OpenTelemetryMetricConfig as CoreOpenTelemetryMetricConfig,
    OpenTelemetryMetricSubscriber, resolve_http_metric_endpoint,
};
use crate::observability::otel_signal::{
    MetricMarkClassification, classify_metric_mark, validate_signal_headers,
};
use crate::observability::{
    MarkProjection, OpenTelemetryType, OtlpAttributeMapping, default_mark_exclude_names,
    validate_attribute_mappings, validate_metadata_promotion_prefixes,
};
use crate::plugin::{
    ATIF_RUNTIME_DELIVERY_FAILURE_MARKER, ConfigDiagnostic, ConfigPolicy, DiagnosticLevel,
    OTEL_RUNTIME_DELIVERY_FAILURE_MARKER, Plugin, PluginComponentSpec, PluginError,
    PluginRegistration, PluginRegistrationCleanupOutcome, PluginRegistrationContext,
    Result as PluginResult, UnsupportedBehavior, apply_global_config_policy, deregister_plugin,
    register_builtin_plugin,
};
use crate::plugin::{RuntimeDiagnostic, record_active_plugin_runtime_diagnostic};

/// The plugin kind registered by the core crate.
pub const OBSERVABILITY_PLUGIN_KIND: &str = "observability";
/// Top-level observability component wrapper.
///
/// Use this wrapper when constructing a [`PluginComponentSpec`] from Rust
/// instead of hand-writing the generic plugin component shape. The component
/// kind is always [`OBSERVABILITY_PLUGIN_KIND`].
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the observability component should be activated.
    pub enabled: bool,
    /// Observability config for this top-level component.
    pub config: ObservabilityConfig,
}

impl ComponentSpec {
    /// Creates an enabled observability component spec.
    ///
    /// The returned component can be converted into the generic plugin config
    /// entry with `PluginComponentSpec::from(...)`.
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(value.config)
            .expect("observability config should serialize to object")
        else {
            unreachable!("observability config must serialize to object");
        };

        PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

/// Canonical config document for the observability plugin component.
///
/// Every section is optional. A missing section has the same activation
/// behavior as a section with `enabled = false`: it contributes no runtime
/// subscribers and performs no export work.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ObservabilityConfig {
    /// Observability config schema version.
    #[serde(default = "default_observability_config_version")]
    pub version: u32,
    /// Filesystem-backed raw ATOF JSONL export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atof: Option<AtofSectionConfig>,
    /// Per-top-level-agent ATIF trajectory export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atif: Option<AtifSectionConfig>,
    /// OpenTelemetry trace export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opentelemetry: Option<OpenTelemetrySectionConfig>,
    /// Observability-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
    /// Whether LLM start events retain complete sanitized request payloads.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub enable_full_payloads: bool,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            version: default_observability_config_version(),
            atof: None,
            atif: None,
            opentelemetry: None,
            policy: ConfigPolicy::default(),
            enable_full_payloads: false,
        }
    }
}

/// Multi-endpoint OpenTelemetry export settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenTelemetrySectionConfig {
    /// Whether OpenTelemetry export is active.
    #[serde(default)]
    pub enabled: bool,
    /// Independently configured OTLP destinations.
    #[serde(
        default,
        rename = "traces",
        alias = "endpoints",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub endpoints: Vec<OpenTelemetryEndpointConfig>,
    /// Optional OTLP log pipeline sourced from non-metric marks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logs: Option<OpenTelemetryLogSectionConfig>,
    /// Optional OTLP metric pipeline sourced from Relay metric marks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<OpenTelemetryMetricSectionConfig>,
}

/// Signal-common OTLP destination fields used by logs and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenTelemetrySignalEndpointConfig {
    /// Required OTLP endpoint. Bare HTTP authorities gain the signal path.
    pub endpoint: String,
    /// OTLP transport: `http_binary` or `grpc`.
    #[serde(default = "default_otlp_transport")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "otlp_transport_schema"))]
    pub transport: String,
    /// Extra exporter headers or metadata.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Exporter headers mapped to environment variable names.
    #[serde(default)]
    pub header_env: HashMap<String, String>,
    /// Extra resource attributes.
    #[serde(default)]
    pub resource_attributes: HashMap<String, String>,
    /// `service.name` resource attribute.
    #[serde(default = "default_otel_service_name")]
    pub service_name: String,
    /// Optional `service.namespace` resource attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_version: Option<String>,
    /// Instrumentation scope name.
    #[serde(default = "default_otel_instrumentation_scope")]
    pub instrumentation_scope: String,
    /// OTLP request timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
}

/// OTLP log pipeline settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenTelemetryLogSectionConfig {
    /// Whether log export is active.
    #[serde(default)]
    pub enabled: bool,
    /// Explicit log destinations. When omitted, destinations derive from trace endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoints: Option<Vec<OpenTelemetrySignalEndpointConfig>>,
    /// Minimum exported telemetry-log severity.
    #[serde(default = "default_otel_log_minimum_severity")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "log_severity_schema"))]
    pub minimum_severity: String,
    /// Maximum queued log records.
    #[serde(default = "default_otel_log_max_queue_size")]
    pub max_queue_size: usize,
    /// Maximum log records in one batch.
    #[serde(default = "default_otel_log_max_export_batch_size")]
    pub max_export_batch_size: usize,
    /// Maximum delay before exporting a non-full log batch.
    #[serde(default = "default_otel_log_scheduled_delay_millis")]
    pub scheduled_delay_millis: u64,
    /// Positive duration for retaining completed scope context for late logs.
    #[serde(default = "default_completed_span_context_ttl_millis")]
    pub completed_span_context_ttl_millis: u64,
}

fn default_completed_span_context_ttl_millis() -> u64 {
    60_000
}

impl Default for OpenTelemetryLogSectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoints: None,
            minimum_severity: default_otel_log_minimum_severity(),
            max_queue_size: default_otel_log_max_queue_size(),
            max_export_batch_size: default_otel_log_max_export_batch_size(),
            scheduled_delay_millis: default_otel_log_scheduled_delay_millis(),
            completed_span_context_ttl_millis: default_completed_span_context_ttl_millis(),
        }
    }
}

/// OTLP metric pipeline settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenTelemetryMetricSectionConfig {
    /// Whether metric export is active.
    #[serde(default)]
    pub enabled: bool,
    /// Explicit metric destinations. When omitted, destinations derive from trace endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoints: Option<Vec<OpenTelemetrySignalEndpointConfig>>,
    /// Collection and export interval in milliseconds.
    #[serde(default = "default_otel_metric_export_interval_millis")]
    pub export_interval_millis: u64,
    /// Preferred aggregation temporality.
    #[serde(default = "default_otel_metric_temporality")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "metric_temporality_schema")
    )]
    pub temporality: String,
    /// Maximum retained metric instruments per destination.
    #[serde(default = "default_otel_metric_max_instruments")]
    pub max_instruments: usize,
    /// Maximum series per instrument.
    #[serde(default = "default_otel_metric_cardinality_limit")]
    pub cardinality_limit: usize,
}

impl Default for OpenTelemetryMetricSectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoints: None,
            export_interval_millis: default_otel_metric_export_interval_millis(),
            temporality: default_otel_metric_temporality(),
            max_instruments: default_otel_metric_max_instruments(),
            cardinality_limit: default_otel_metric_cardinality_limit(),
        }
    }
}

/// One typed OTLP destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenTelemetryEndpointConfig {
    /// Semantic projection emitted by this endpoint.
    #[serde(rename = "type")]
    pub otel_type: OpenTelemetryType,
    /// Required OTLP endpoint.
    pub endpoint: String,
    /// Representation used for point-in-time marks.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mark_projection_schema"))]
    pub mark_projection: MarkProjection,
    /// Mark names excluded from tool projection.
    #[serde(default = "default_mark_exclude_names")]
    pub mark_exclude_names: Vec<String>,
    /// Projected attributes copied to aliases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attribute_mappings: Vec<OtlpAttributeMapping>,
    /// Literal Event metadata prefixes copied to top-level OTLP attributes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub promote_metadata_prefixes: Vec<String>,
    /// Literal root-scope Event metadata prefixes copied to OTLP resource attributes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub promote_resource_metadata_prefixes: Vec<String>,
    /// OTLP transport: `http_binary` or `grpc`.
    #[serde(default = "default_otlp_transport")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "otlp_transport_schema"))]
    pub transport: String,
    /// Extra exporter headers or metadata.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Exporter headers mapped to environment variable names.
    #[serde(default)]
    pub header_env: HashMap<String, String>,
    /// Extra resource attributes.
    #[serde(default)]
    pub resource_attributes: HashMap<String, String>,
    /// `service.name` resource attribute.
    #[serde(default = "default_otel_service_name")]
    pub service_name: String,
    /// Optional `service.namespace` resource attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_version: Option<String>,
    /// Instrumentation scope name.
    #[serde(default = "default_otel_instrumentation_scope")]
    pub instrumentation_scope: String,
    /// OTLP request timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
    /// Maximum completed spans buffered before the endpoint drops new spans.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_queue_size: Option<usize>,
    /// Maximum spans exported in one batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_export_batch_size: Option<usize>,
    /// Maximum delay before exporting a non-full batch, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduled_delay_millis: Option<u64>,
    /// How long completed scopes retain trace context for late marks, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_span_context_ttl_millis: Option<u64>,
}

/// Multi-sink ATOF JSONL exporter config.
///
/// When enabled, this section wraps
/// [`crate::observability::atof::AtofExporter`] and writes the raw ATOF event
/// stream to one or more explicitly configured file or stream sinks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AtofSectionConfig {
    /// Whether ATOF JSONL export is active.
    #[serde(default)]
    pub enabled: bool,
    /// Destinations that each receive every raw ATOF event.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sinks: Vec<AtofSinkSectionConfig>,
}

/// One plugin-managed destination for raw ATOF events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AtofSinkSectionConfig {
    /// A local JSONL file.
    File(AtofFileSinkSectionConfig),
    /// A remote stream.
    Stream(AtofStreamSinkSectionConfig),
}

/// File sink settings for the ATOF plugin section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AtofFileSinkSectionConfig {
    /// Directory containing the JSONL output file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    /// Output filename. Defaults to the native timestamped filename.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// File open mode: `append` or `overwrite`.
    #[serde(default = "default_atof_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "atof_mode_schema"))]
    pub mode: String,
}

/// Stream sink settings for the ATOF plugin section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AtofStreamSinkSectionConfig {
    /// Endpoint URL.
    pub url: String,
    /// Transport: `http_post`, `websocket`, or `ndjson`.
    #[serde(default = "default_atof_endpoint_transport")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "atof_endpoint_transport_schema")
    )]
    pub transport: String,
    /// Headers applied to endpoint requests or handshakes.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Header names mapped to environment variables containing their values.
    #[serde(default)]
    pub header_env: HashMap<String, String>,
    /// Per-endpoint timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
    /// Field name policy applied before sending events: `preserve` or `replace_dots`.
    #[serde(default = "default_atof_endpoint_field_name_policy")]
    pub field_name_policy: String,
    /// Optional stable name used by other components to reference this endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Per-trajectory ATIF exporter config.
///
/// When enabled, this section creates a dispatcher that opens a separate
/// [`crate::observability::atif::AtifExporter`] for each top-level agent or turn scope. The
/// historical `{session_id}` placeholder in [`AtifSectionConfig::filename_template`] expands to
/// the trajectory scope UUID so concurrent sibling trajectories cannot overwrite each other's
/// files.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AtifSectionConfig {
    /// Whether ATIF export is active.
    #[serde(default)]
    pub enabled: bool,
    /// Human-readable agent name.
    #[serde(default = "default_agent_name")]
    pub agent_name: String,
    /// Agent version string.
    #[serde(default = "default_agent_version")]
    pub agent_version: String,
    /// Default model name.
    #[serde(default = "default_model_name")]
    pub model_name: String,
    /// Tool definitions available to the agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_definitions: Option<Vec<Json>>,
    /// Extra ATIF agent metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
    /// Directory containing trajectory JSON files. Ignored when [`storage`] is non-empty.
    ///
    /// [`storage`]: Self::storage
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    /// Filename template. The historical `{session_id}` placeholder is always replaced with the
    /// top-level trajectory scope UUID. `{metadata.<path>:-fallback}` placeholders sanitize strings from the
    /// top-level scope metadata into path-safe filename fragments or use the optional literal
    /// fallback. When [`storage`] is non-empty, the rendered filename is appended to each
    /// backend's key prefix.
    /// [`storage`]: Self::storage
    #[serde(default = "default_atif_filename_template")]
    pub filename_template: String,
    /// Optional list of remote storage destinations. When non-empty, completed
    /// trajectories are uploaded to every configured backend instead of being
    /// written locally; the local file write at [`output_directory`] is
    /// skipped. Backends are independent: an upload failure on one destination
    /// is recorded against that destination and skipped on subsequent
    /// trajectories, while the other destinations continue to receive writes.
    ///
    /// [`output_directory`]: Self::output_directory
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub storage: Vec<AtifStorageConfig>,
}

impl Default for AtifSectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            agent_name: default_agent_name(),
            agent_version: default_agent_version(),
            model_name: default_model_name(),
            tool_definitions: None,
            extra: None,
            output_directory: None,
            filename_template: default_atif_filename_template(),
            storage: Vec::new(),
        }
    }
}

/// Remote storage destination for ATIF trajectory files.
///
/// When [`AtifSectionConfig::storage`] is non-empty, the ATIF dispatcher
/// uploads each completed trajectory to every configured backend instead of
/// writing it to the local filesystem. The shape is tagged with a `type`
/// discriminator so additional backends (for example, Azure Blob Storage) can
/// be added without breaking existing configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AtifStorageConfig {
    /// HTTP endpoint storage.
    Http(HttpStorageConfig),
    /// S3-compatible object storage.
    ///
    /// Non-secret connection settings (`region`, `endpoint_url`, `allow_http`)
    /// and the static `access_key_id` may be set directly. The secret
    /// credential fields (`secret_access_key_var`, `session_token_var`) must
    /// reference the *name* of an environment variable that holds the secret,
    /// so multiple S3 destinations can coexist in one config without writing
    /// secrets into checked-in files. Any field left unset falls back to the
    /// matching `AWS_*` environment variable (`AWS_ACCESS_KEY_ID`,
    /// `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`,
    /// `AWS_ENDPOINT_URL`, `AWS_ALLOW_HTTP`).
    S3(S3StorageConfig),
}

/// S3-compatible storage settings for ATIF trajectory upload.
///
/// Every connection field is optional. Unset fields fall back to the matching
/// `AWS_*` environment variable, preserving the env-driven workflow while
/// letting one config file fully describe a destination when needed. Secret
/// credentials are referenced by env var *name* (the `_var` suffix), so
/// multiple destinations can each carry their own credentials without leaking
/// secret material into the config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct S3StorageConfig {
    /// Destination bucket name. Must be non-empty.
    pub bucket: String,
    /// Optional key prefix applied to every uploaded object. A trailing `/` is
    /// inserted automatically when one is missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_prefix: Option<String>,
    /// Static AWS access key ID. When unset, `AWS_ACCESS_KEY_ID` is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_key_id: Option<String>,
    /// Name of the environment variable that holds the static secret access
    /// key. Validated to be non-empty and present at plugin initialization
    /// time. When unset, `AWS_SECRET_ACCESS_KEY` is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_access_key_var: Option<String>,
    /// Name of the environment variable that holds the optional STS session
    /// token. Validated to be non-empty and present at plugin initialization
    /// time. When unset, `AWS_SESSION_TOKEN` is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_token_var: Option<String>,
    /// AWS region for the bucket. When unset, `AWS_REGION` is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Endpoint URL override for S3-compatible storage (for example, MinIO).
    /// When unset, `AWS_ENDPOINT_URL` is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    /// Allow plain HTTP endpoints. When unset, `AWS_ALLOW_HTTP` is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_http: Option<bool>,
}

/// HTTP endpoint settings for ATIF trajectory upload.
///
/// Completed trajectories are uploaded with `POST` and an
/// `application/json` body. Inline `headers` are merged with values resolved
/// from `header_env`; `header_env` values are environment variable names, not
/// secret values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct HttpStorageConfig {
    /// Destination endpoint URL. Must use `http://` or `https://`.
    pub endpoint: String,
    /// Static request headers.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    /// Request headers whose values are read from environment variables.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub header_env: HashMap<String, String>,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
}

crate::editor_config! {
    impl ObservabilityConfig {
        version => {
            label: "version",
            kind: IntegerEnum,
            values: ["3", "4"],
        },
        atof => {
            label: "ATOF",
            kind: Section,
            optional: true,
            nested: AtofSectionConfig,
            default: AtofSectionConfig,
        },
        atif => {
            label: "ATIF",
            kind: Section,
            optional: true,
            nested: AtifSectionConfig,
            default: AtifSectionConfig,
        },
        opentelemetry => {
            label: "OpenTelemetry",
            kind: Section,
            optional: true,
            nested: OpenTelemetrySectionConfig,
            default: OpenTelemetrySectionConfig,
        },
        policy => {
            label: "policy",
            kind: Section,
            nested: ConfigPolicy,
            default: ConfigPolicy,
        },
        enable_full_payloads => { label: "enable_full_payloads", kind: Boolean },
    }
}

crate::editor_config! {
    impl OpenTelemetrySectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        endpoints => { label: "traces", kind: List, name: "traces", list: &OPENTELEMETRY_ENDPOINT_LIST },
        logs => {
            label: "logs",
            kind: Section,
            optional: true,
            nested: OpenTelemetryLogSectionConfig,
            default: OpenTelemetryLogSectionConfig,
        },
        metrics => {
            label: "metrics",
            kind: Section,
            optional: true,
            nested: OpenTelemetryMetricSectionConfig,
            default: OpenTelemetryMetricSectionConfig,
        },
    }
}

crate::editor_config! {
    impl OpenTelemetryLogSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        endpoints => {
            label: "endpoints",
            kind: List,
            optional: true,
            list: &OPENTELEMETRY_SIGNAL_ENDPOINT_LIST,
        },
        minimum_severity => {
            label: "minimum_severity",
            kind: Enum,
            values: ["trace", "debug", "info", "warn", "warning", "error"],
        },
        max_queue_size => { label: "max_queue_size", kind: Integer },
        max_export_batch_size => { label: "max_export_batch_size", kind: Integer },
        scheduled_delay_millis => { label: "scheduled_delay_millis", kind: Integer },
        completed_span_context_ttl_millis => { label: "completed_span_context_ttl_millis", kind: Integer },
    }
}

crate::editor_config! {
    impl OpenTelemetryMetricSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        endpoints => {
            label: "endpoints",
            kind: List,
            optional: true,
            list: &OPENTELEMETRY_SIGNAL_ENDPOINT_LIST,
        },
        export_interval_millis => { label: "export_interval_millis", kind: Integer },
        temporality => {
            label: "temporality",
            kind: Enum,
            values: ["cumulative", "delta", "low_memory"],
        },
        max_instruments => { label: "max_instruments", kind: Integer },
        cardinality_limit => { label: "cardinality_limit", kind: Integer },
    }
}

const fn otel_editor_field(
    name: &'static str,
    kind: EditorFieldKind,
    enum_values: &'static [&'static str],
    optional: bool,
) -> EditorFieldSpec {
    EditorFieldSpec {
        name,
        label: name,
        kind,
        enum_values,
        optional,
        nested_schema: None,
        nested_default: None,
        list_item: None,
        tagged_union: None,
    }
}

const fn otel_list_editor_field(
    name: &'static str,
    optional: bool,
    item: &'static EditorListItemSpec,
) -> EditorFieldSpec {
    EditorFieldSpec {
        name,
        label: name,
        kind: EditorFieldKind::List,
        enum_values: &[],
        optional,
        nested_schema: None,
        nested_default: None,
        list_item: Some(item),
        tagged_union: None,
    }
}

impl EditorConfig for OpenTelemetryEndpointConfig {
    fn editor_schema() -> &'static EditorSchema {
        static SCHEMA: EditorSchema = EditorSchema {
            fields: &[
                otel_editor_field(
                    "type",
                    EditorFieldKind::Enum,
                    &["full", "gen_ai", "openinference"],
                    false,
                ),
                otel_editor_field("endpoint", EditorFieldKind::String, &[], false),
                otel_editor_field(
                    "mark_projection",
                    EditorFieldKind::Enum,
                    &["inherit", "event", "tool"],
                    false,
                ),
                otel_list_editor_field(
                    "mark_exclude_names",
                    false,
                    &crate::config_editor::STRING_LIST_ITEM,
                ),
                otel_editor_field("attribute_mappings", EditorFieldKind::List, &[], false),
                otel_editor_field(
                    "promote_metadata_prefixes",
                    EditorFieldKind::List,
                    &[],
                    true,
                ),
                otel_editor_field(
                    "promote_resource_metadata_prefixes",
                    EditorFieldKind::List,
                    &[],
                    true,
                ),
                otel_editor_field(
                    "transport",
                    EditorFieldKind::Enum,
                    &["http_binary", "grpc"],
                    false,
                ),
                otel_editor_field("service_name", EditorFieldKind::String, &[], false),
                otel_editor_field("service_namespace", EditorFieldKind::String, &[], true),
                otel_editor_field("service_version", EditorFieldKind::String, &[], true),
                otel_editor_field("instrumentation_scope", EditorFieldKind::String, &[], false),
                otel_editor_field("timeout_millis", EditorFieldKind::Integer, &[], false),
                otel_editor_field("max_queue_size", EditorFieldKind::Integer, &[], true),
                otel_editor_field("max_export_batch_size", EditorFieldKind::Integer, &[], true),
                otel_editor_field(
                    "scheduled_delay_millis",
                    EditorFieldKind::Integer,
                    &[],
                    true,
                ),
                otel_editor_field(
                    "completed_span_context_ttl_millis",
                    EditorFieldKind::Integer,
                    &[],
                    true,
                ),
                otel_editor_field("headers", EditorFieldKind::StringMap, &[], false),
                otel_editor_field("header_env", EditorFieldKind::StringMap, &[], false),
                otel_editor_field(
                    "resource_attributes",
                    EditorFieldKind::StringMap,
                    &[],
                    false,
                ),
            ],
        };
        &SCHEMA
    }
}

impl EditorConfig for OpenTelemetrySignalEndpointConfig {
    fn editor_schema() -> &'static EditorSchema {
        static SCHEMA: EditorSchema = EditorSchema {
            fields: &[
                otel_editor_field("endpoint", EditorFieldKind::String, &[], false),
                otel_editor_field(
                    "transport",
                    EditorFieldKind::Enum,
                    &["http_binary", "grpc"],
                    false,
                ),
                otel_editor_field("service_name", EditorFieldKind::String, &[], false),
                otel_editor_field("service_namespace", EditorFieldKind::String, &[], true),
                otel_editor_field("service_version", EditorFieldKind::String, &[], true),
                otel_editor_field("instrumentation_scope", EditorFieldKind::String, &[], false),
                otel_editor_field("timeout_millis", EditorFieldKind::Integer, &[], false),
                otel_editor_field("headers", EditorFieldKind::StringMap, &[], false),
                otel_editor_field("header_env", EditorFieldKind::StringMap, &[], false),
                otel_editor_field(
                    "resource_attributes",
                    EditorFieldKind::StringMap,
                    &[],
                    false,
                ),
            ],
        };
        &SCHEMA
    }
}

fn default_opentelemetry_endpoint_editor_value() -> Json {
    serde_json::json!({
        "type": "full",
        "endpoint": "",
        "transport": "http_binary",
        "service_name": "unknown_service",
        "instrumentation_scope": "opentelemetry",
        "timeout_millis": 3000,
        "headers": {},
        "header_env": {},
        "resource_attributes": {},
    })
}

static OPENTELEMETRY_ENDPOINT_LIST: EditorListItemSpec = EditorListItemSpec {
    kind: EditorFieldKind::Section,
    schema: Some(<OpenTelemetryEndpointConfig as EditorConfig>::editor_schema),
    default: Some(default_opentelemetry_endpoint_editor_value),
    tagged_union: None,
    list_item: None,
};

fn default_opentelemetry_signal_endpoint_editor_value() -> Json {
    serde_json::json!({
        "endpoint": "",
        "transport": "http_binary",
        "service_name": "unknown_service",
        "instrumentation_scope": "opentelemetry",
        "timeout_millis": 3000,
        "headers": {},
        "header_env": {},
        "resource_attributes": {},
    })
}

static OPENTELEMETRY_SIGNAL_ENDPOINT_LIST: EditorListItemSpec = EditorListItemSpec {
    kind: EditorFieldKind::Section,
    schema: Some(<OpenTelemetrySignalEndpointConfig as EditorConfig>::editor_schema),
    default: Some(default_opentelemetry_signal_endpoint_editor_value),
    tagged_union: None,
    list_item: None,
};

crate::editor_config! {
    impl AtofSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        sinks => { label: "sinks", kind: List, list: &ATOF_SINK_LIST },
    }
}

crate::editor_config! {
    impl AtofFileSinkSectionConfig {
        output_directory => { label: "output_directory", kind: String, optional: true },
        filename => { label: "filename", kind: String, optional: true },
        mode => { label: "mode", kind: Enum, values: ["append", "overwrite"] },
    }
}

crate::editor_config! {
    impl AtofStreamSinkSectionConfig {
        url => { label: "url", kind: String },
        transport => { label: "transport", kind: Enum, values: ["http_post", "websocket", "ndjson"] },
        headers => { label: "headers", kind: StringMap },
        header_env => { label: "header_env", kind: StringMap },
        timeout_millis => { label: "timeout_millis", kind: Integer },
        field_name_policy => { label: "field_name_policy", kind: Enum, values: ["preserve", "replace_dots"] },
        name => { label: "name", kind: String, optional: true },
    }
}

crate::editor_config! {
    impl HttpStorageConfig {
        endpoint => { label: "endpoint", kind: String },
        headers => { label: "headers", kind: StringMap },
        header_env => { label: "header_env", kind: StringMap },
        timeout_millis => { label: "timeout_millis", kind: Integer },
    }
}

crate::editor_config! {
    impl S3StorageConfig {
        bucket => { label: "bucket", kind: String },
        key_prefix => { label: "key_prefix", kind: String, optional: true },
        access_key_id => { label: "access_key_id", kind: String, optional: true },
        secret_access_key_var => { label: "secret_access_key_var", kind: String, optional: true },
        session_token_var => { label: "session_token_var", kind: String, optional: true },
        region => { label: "region", kind: String, optional: true },
        endpoint_url => { label: "endpoint_url", kind: String, optional: true },
        allow_http => { label: "allow_http", kind: Boolean, optional: true },
    }
}

fn default_http_storage_editor_value() -> Json {
    serde_json::json!({"type": "http", "endpoint": "", "headers": {}, "header_env": {}, "timeout_millis": 3000})
}

fn default_s3_storage_editor_value() -> Json {
    serde_json::json!({"type": "s3", "bucket": ""})
}

static ATIF_STORAGE_VARIANTS: [EditorVariantSpec; 2] = [
    EditorVariantSpec {
        label: "HTTP",
        tag: "http",
        schema: <HttpStorageConfig as EditorConfig>::editor_schema,
        default: default_http_storage_editor_value,
    },
    EditorVariantSpec {
        label: "S3",
        tag: "s3",
        schema: <S3StorageConfig as EditorConfig>::editor_schema,
        default: default_s3_storage_editor_value,
    },
];

static ATIF_STORAGE_TAGGED_UNION: EditorTaggedUnionSpec = EditorTaggedUnionSpec {
    discriminator: "type",
    variants: &ATIF_STORAGE_VARIANTS,
};

static ATIF_STORAGE_LIST: EditorListItemSpec = EditorListItemSpec {
    kind: EditorFieldKind::Section,
    schema: None,
    default: None,
    tagged_union: Some(&ATIF_STORAGE_TAGGED_UNION),
    list_item: None,
};

crate::editor_config! {
    impl AtifSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        agent_name => { label: "agent_name", kind: String },
        agent_version => { label: "agent_version", kind: String },
        model_name => { label: "model_name", kind: String },
        tool_definitions => { label: "tool_definitions", kind: List, optional: true, list: &crate::config_editor::JSON_LIST_ITEM },
        extra => { label: "extra", kind: Json, optional: true },
        output_directory => { label: "output_directory", kind: String, optional: true },
        filename_template => { label: "filename_template", kind: String },
        storage => { label: "storage", kind: List, optional: true, list: &ATIF_STORAGE_LIST },
    }
}

fn default_atof_file_sink_editor_value() -> Json {
    serde_json::json!({"type": "file", "mode": "append"})
}

fn default_atof_stream_sink_editor_value() -> Json {
    serde_json::json!({
        "type": "stream",
        "url": "",
        "transport": "http_post",
        "headers": {},
        "header_env": {},
        "timeout_millis": 3000,
        "field_name_policy": "preserve",
    })
}

static ATOF_SINK_VARIANTS: [EditorVariantSpec; 2] = [
    EditorVariantSpec {
        label: "File",
        tag: "file",
        schema: <AtofFileSinkSectionConfig as EditorConfig>::editor_schema,
        default: default_atof_file_sink_editor_value,
    },
    EditorVariantSpec {
        label: "Stream",
        tag: "stream",
        schema: <AtofStreamSinkSectionConfig as EditorConfig>::editor_schema,
        default: default_atof_stream_sink_editor_value,
    },
];

static ATOF_SINK_TAGGED_UNION: EditorTaggedUnionSpec = EditorTaggedUnionSpec {
    discriminator: "type",
    variants: &ATOF_SINK_VARIANTS,
};

static ATOF_SINK_LIST: EditorListItemSpec = EditorListItemSpec {
    kind: EditorFieldKind::Section,
    schema: None,
    default: None,
    tagged_union: Some(&ATOF_SINK_TAGGED_UNION),
    list_item: None,
};

struct ObservabilityPlugin;

impl Plugin for ObservabilityPlugin {
    fn plugin_kind(&self) -> &str {
        OBSERVABILITY_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_observability_plugin_config(plugin_config)
    }

    fn validate_with_policy(
        &self,
        plugin_config: &Map<String, Json>,
        policy: &ConfigPolicy,
    ) -> Vec<ConfigDiagnostic> {
        validate_observability_plugin_config_with_policy(plugin_config, Some(policy))
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let config = parse_observability_config(&plugin_config)?;
            register_observability(config, ctx)
        })
    }
}

/// Registers the observability component kind in the core plugin registry.
///
/// Calling this function more than once is safe. The core plugin APIs call it
/// automatically before listing, looking up, validating, or initializing plugin
/// components, so applications normally do not need to invoke it directly.
pub fn register_observability_component() -> PluginResult<()> {
    register_builtin_plugin(Arc::new(ObservabilityPlugin))
}

/// Deregisters the observability component kind from the core plugin registry.
///
/// This helper exists primarily for tests and specialized embedding scenarios.
/// It removes the plugin kind from future registry lookups but does not clear an
/// already active plugin configuration.
pub fn deregister_observability_component() -> bool {
    deregister_plugin(OBSERVABILITY_PLUGIN_KIND)
}

/// Returns the JSON Schema for the observability component configuration.
#[cfg(feature = "schema")]
pub fn observability_config_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(ObservabilityConfig))
        .expect("observability config schema should serialize")
}

#[cfg(feature = "schema")]
fn atof_mode_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    string_enum_schema(generator, &["append", "overwrite"], Some("append"))
}

#[cfg(feature = "schema")]
fn atof_endpoint_transport_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["http_post", "websocket", "ndjson"],
        Some("http_post"),
    )
}

#[cfg(feature = "schema")]
fn otlp_transport_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(generator, &["http_binary", "grpc"], Some("http_binary"))
}

#[cfg(feature = "schema")]
fn mark_projection_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(generator, &["inherit", "event", "tool"], Some("inherit"))
}

#[cfg(feature = "schema")]
fn log_severity_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["trace", "debug", "info", "warn", "warning", "error"],
        Some("info"),
    )
}

#[cfg(feature = "schema")]
fn metric_temporality_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["cumulative", "delta", "low_memory"],
        Some("cumulative"),
    )
}

#[cfg(feature = "schema")]
fn string_enum_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
    values: &[&str],
    default: Option<&str>,
) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <String as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(
        values
            .iter()
            .map(|value| Json::String((*value).into()))
            .collect(),
    );
    if let Some(default) = default {
        schema.metadata().default = Some(Json::String(default.into()));
    }
    schema.into()
}

fn register_observability(
    config: ObservabilityConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    if !matches!(config.version, 3 | 4) {
        return Err(PluginError::InvalidConfig(format!(
            "observability config version {} is unsupported",
            config.version
        )));
    }
    if config.version == 3
        && config
            .opentelemetry
            .as_ref()
            .is_some_and(|section| section.logs.is_some() || section.metrics.is_some())
    {
        return Err(PluginError::InvalidConfig(
            "observability config version 3 is trace-only; use version 4 for OpenTelemetry logs or metrics"
                .to_string(),
        ));
    }
    register_full_payload_policy(config.enable_full_payloads, ctx)?;
    if let Some(atof) = config.atof.filter(|section| section.enabled) {
        register_atof_exporter(atof, ctx)?;
    }
    if let Some(atif) = config.atif.filter(|section| section.enabled) {
        register_atif_dispatcher(atif, ctx)?;
    }
    if let Some(otel) = config.opentelemetry.filter(|section| section.enabled) {
        register_opentelemetry(otel, ctx)?;
    }
    Ok(())
}

fn register_full_payload_policy(
    enabled: bool,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let context = global_context();
    context
        .write()
        .map_err(|error| PluginError::RegistrationFailed(error.to_string()))?
        .observability_full_payloads_enabled = enabled;
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("payload-policy"),
        Box::new(|| {
            global_context()
                .write()
                .map_err(|error| PluginError::RegistrationFailed(error.to_string()))?
                .observability_full_payloads_enabled = false;
            Ok(())
        }),
    ));
    Ok(())
}

fn register_atof_exporter(
    section: AtofSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let exporters = section
        .sinks
        .into_iter()
        .enumerate()
        .map(|(index, sink)| {
            let config = CoreAtofExporterConfig {
                sink: build_atof_sink_config(index, sink)?,
            };
            AtofExporter::new(config)
                .map(Arc::new)
                .map_err(observability_registration_error)
        })
        .collect::<PluginResult<Vec<_>>>()?;
    let subscribers = exporters
        .iter()
        .map(|exporter| exporter.subscriber())
        .collect::<Vec<_>>();
    let subscriber: EventSubscriberFn = Arc::new(move |event| {
        for subscriber in &subscribers {
            subscriber(event);
        }
    });

    ctx.register_subscriber("atof", subscriber)?;
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("atof.shutdown"),
        Box::new(move || {
            let mut first_error = None;
            for exporter in &exporters {
                if let Err(error) = exporter.shutdown() {
                    first_error.get_or_insert_with(|| observability_registration_error(error));
                }
            }
            first_error.map_or(Ok(()), Err)
        }),
    ));
    Ok(())
}

fn build_atof_sink_config(
    index: usize,
    sink: AtofSinkSectionConfig,
) -> PluginResult<CoreAtofSinkConfig> {
    match sink {
        AtofSinkSectionConfig::File(file) => {
            let mode = AtofExporterMode::parse(&file.mode).ok_or_else(|| {
                PluginError::InvalidConfig(format!(
                    "ATOF sinks[{index}].mode must be 'append' or 'overwrite'"
                ))
            })?;
            let mut sink = AtofFileSinkConfig::new();
            sink.mode = mode;
            if let Some(output_directory) = file.output_directory {
                sink.output_directory = output_directory;
            }
            if let Some(filename) = file.filename {
                sink.filename = filename;
            }
            Ok(CoreAtofSinkConfig::File(sink))
        }
        AtofSinkSectionConfig::Stream(stream) => {
            let transport = AtofEndpointTransport::parse(&stream.transport).ok_or_else(|| {
                PluginError::InvalidConfig(format!(
                    "ATOF sinks[{index}].transport must be 'http_post', 'websocket', or 'ndjson'"
                ))
            })?;
            let field_name_policy = AtofEndpointFieldNamePolicy::parse(&stream.field_name_policy)
                .ok_or_else(|| {
                PluginError::InvalidConfig(format!(
                    "ATOF sinks[{index}].field_name_policy must be 'preserve' or 'replace_dots'"
                ))
            })?;
            let mut config = AtofStreamSinkConfig::new(stream.url, transport)
                .with_timeout_millis(stream.timeout_millis)
                .with_field_name_policy(field_name_policy);
            for (key, value) in stream.headers {
                config = config.with_header(key, value);
            }
            for (key, variable) in stream.header_env {
                config = config.with_header_env(key, variable);
            }
            Ok(CoreAtofSinkConfig::Stream(config))
        }
    }
}

type AtifStorageList = Arc<Vec<Arc<AtifRemoteStorage>>>;

fn register_atif_dispatcher(
    section: AtifSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    validate_atif_filename_template(&section.filename_template)
        .map_err(PluginError::InvalidConfig)?;

    let mut storage_vec = Vec::with_capacity(section.storage.len());
    for (index, entry) in section.storage.iter().enumerate() {
        storage_vec.push(build_atif_storage(index, entry)?);
    }
    let storage: AtifStorageList = Arc::new(storage_vec);

    let manager = Arc::new(Mutex::new(AtifDispatcher::new(section)));
    let dispatcher = atif_dispatcher_subscriber(
        Arc::clone(&manager),
        ctx.qualify_name("atif-"),
        Arc::clone(&storage),
    );
    ctx.register_subscriber("atif", dispatcher)?;
    let shutdown_storage = Arc::clone(&storage);
    ctx.add_registration(PluginRegistration::new_with_outcome(
        "observability",
        ctx.qualify_name("atif.shutdown"),
        Box::new(move || {
            atif_shutdown_cleanup(Arc::clone(&manager), Arc::clone(&shutdown_storage))
        }),
    ));
    Ok(())
}

fn atif_shutdown_cleanup(
    manager: Arc<Mutex<AtifDispatcher>>,
    shutdown_storage: AtifStorageList,
) -> PluginRegistrationCleanupOutcome {
    let (work, deregistration_error) = match flush_atif_shutdown_work(&manager) {
        Ok(work) => work,
        Err(error) => return PluginRegistrationCleanupOutcome::NotRemoved(error),
    };
    if let Err(error) = write_atif_shutdown_exports(&manager, &shutdown_storage, work.exports) {
        return PluginRegistrationCleanupOutcome::RemovedWithError(error);
    }
    if let Some(error) = deregistration_error {
        return PluginRegistrationCleanupOutcome::NotRemoved(error);
    }
    match manager.lock() {
        Ok(guard) => match guard.last_error_result() {
            Ok(()) => PluginRegistrationCleanupOutcome::Removed,
            Err(error) => PluginRegistrationCleanupOutcome::RemovedWithError(
                observability_registration_error(error),
            ),
        },
        Err(error) => PluginRegistrationCleanupOutcome::RemovedWithError(PluginError::Internal(
            format!("ATIF dispatcher lock poisoned: {error}"),
        )),
    }
}

fn flush_atif_shutdown_work(
    manager: &Arc<Mutex<AtifDispatcher>>,
) -> PluginResult<(AtifFlushWork, Option<PluginError>)> {
    let work = {
        let mut guard = manager.lock().map_err(|err| {
            PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
        })?;
        guard.flush_open_agents()
    };
    let mut deregistration_error = None;
    for (scope_uuid, name) in &work.scope_subscribers {
        if let Err(error) = deregister_atif_shutdown_subscriber(scope_uuid, name)
            && deregistration_error.is_none()
        {
            deregistration_error = Some(error);
        }
    }
    Ok((work, deregistration_error))
}

fn write_atif_shutdown_exports(
    manager: &Arc<Mutex<AtifDispatcher>>,
    shutdown_storage: &AtifStorageList,
    exports: Vec<PendingAtifExport>,
) -> PluginResult<()> {
    let mut first_error = None;
    for export in exports {
        if let Err(error) = write_atif_shutdown_export(manager, shutdown_storage, &export)
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn write_atif_shutdown_export(
    manager: &Arc<Mutex<AtifDispatcher>>,
    shutdown_storage: &AtifStorageList,
    export: &PendingAtifExport,
) -> PluginResult<()> {
    let write = prepare_atif_shutdown_file(export, Arc::clone(manager))
        .map_err(observability_registration_error)?;
    let targets = manager
        .lock()
        .map_err(|err| PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}")))?
        .sink_targets();
    let results = write_atif(&write, shutdown_storage.as_slice(), &targets);
    let mut guard = manager
        .lock()
        .map_err(|err| PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}")))?;
    let _ = guard.complete_scope_write(write.agent_uuid, results);
    Ok(())
}

fn deregister_atif_shutdown_subscriber(scope_uuid: &Uuid, name: &str) -> PluginResult<()> {
    match scope_deregister_subscriber(scope_uuid, name) {
        Ok(_) | Err(FlowError::NotFound(_)) => Ok(()),
        Err(error) => Err(observability_registration_error(error)),
    }
}

#[cfg(feature = "object-store")]
fn build_atif_storage(
    index: usize,
    config: &AtifStorageConfig,
) -> PluginResult<Arc<AtifRemoteStorage>> {
    let storage = AtifRemoteStorage::from_config(index, config)
        .map(Arc::new)
        .map_err(observability_registration_error)?;
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_resource_access_pending",
        plugin_kind = OBSERVABILITY_PLUGIN_KIND,
        resource_kind = storage.resource_kind,
        resource_index = index,
        permission = "write";
        "Plugin resource access will be validated on first use"
    );
    Ok(storage)
}

#[cfg(not(feature = "object-store"))]
fn build_atif_storage(
    _index: usize,
    _config: &AtifStorageConfig,
) -> PluginResult<Arc<AtifRemoteStorage>> {
    Err(PluginError::InvalidConfig(
        "ATIF storage support is not enabled in this build".to_string(),
    ))
}

#[derive(Clone)]
enum OpenTelemetryResource<T> {
    Active(T),
    Skipped(String),
}

impl<T> OpenTelemetryResource<T> {
    fn as_active(&self) -> Option<&T> {
        match self {
            Self::Active(value) => Some(value),
            Self::Skipped(_) => None,
        }
    }
}

#[derive(Clone)]
struct IndexedOpenTelemetryResource<T> {
    index: usize,
    value: OpenTelemetryResource<T>,
}

struct ResolvedSignalEndpoints {
    endpoints: Vec<IndexedOpenTelemetryResource<OpenTelemetrySignalEndpointConfig>>,
}

fn register_opentelemetry(
    section: OpenTelemetrySectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let OpenTelemetrySectionConfig {
        endpoints,
        logs,
        metrics,
        ..
    } = section;
    let logs = logs.filter(|section| section.enabled);
    let metrics = metrics.filter(|section| section.enabled);
    if endpoints.is_empty() && logs.is_none() && metrics.is_none() {
        return Err(PluginError::InvalidConfig(
            "enabled OpenTelemetry section requires at least one endpoint or an enabled log/metric signal"
                .to_string(),
        ));
    }
    validate_distinct_opentelemetry_destinations(&endpoints)?;
    let log_resolution = logs
        .as_ref()
        .map(|section| resolve_signal_endpoints("logs", section.endpoints.as_ref(), &endpoints))
        .transpose()?;
    let metric_resolution = metrics
        .as_ref()
        .map(|section| resolve_signal_endpoints("metrics", section.endpoints.as_ref(), &endpoints))
        .transpose()?;
    if let Some(resolution) = &log_resolution {
        warn_skipped_signal_endpoints("logs", &resolution.endpoints);
    }
    if let Some(resolution) = &metric_resolution {
        warn_skipped_signal_endpoints("metrics", &resolution.endpoints);
    }
    let trace_subscribers = build_opentelemetry_subscribers(endpoints)?;
    let signal_subscribers = build_opentelemetry_signal_subscribers(
        logs,
        log_resolution,
        metrics,
        metric_resolution,
        &trace_subscribers,
    )?;
    let log_subscribers = signal_subscribers.logs;
    let metric_subscribers = signal_subscribers.metrics;
    if !has_active_opentelemetry_resource(&trace_subscribers)
        && !has_active_opentelemetry_resource(&log_subscribers)
        && !has_active_opentelemetry_resource(&metric_subscribers)
    {
        return Err(PluginError::InvalidConfig(
            "enabled OpenTelemetry section requires at least one valid trace, log, or metric endpoint"
                .to_string(),
        ));
    }
    log_opentelemetry_resource_access("traces", &trace_subscribers);
    log_opentelemetry_resource_access("logs", &log_subscribers);
    log_opentelemetry_resource_access("metrics", &metric_subscribers);
    let trace_callbacks = trace_subscribers
        .iter()
        .map(|subscriber| IndexedOpenTelemetryResource {
            index: subscriber.index,
            value: match &subscriber.value {
                OpenTelemetryResource::Active(value) => {
                    OpenTelemetryResource::Active(value.subscriber())
                }
                OpenTelemetryResource::Skipped(message) => {
                    OpenTelemetryResource::Skipped(message.clone())
                }
            },
        })
        .collect::<Vec<_>>();
    let log_callbacks = log_subscribers
        .iter()
        .map(|subscriber| IndexedOpenTelemetryResource {
            index: subscriber.index,
            value: match &subscriber.value {
                OpenTelemetryResource::Active(value) => {
                    OpenTelemetryResource::Active(value.subscriber())
                }
                OpenTelemetryResource::Skipped(message) => {
                    OpenTelemetryResource::Skipped(message.clone())
                }
            },
        })
        .collect::<Vec<_>>();
    let metric_callbacks = metric_subscribers
        .iter()
        .map(|subscriber| IndexedOpenTelemetryResource {
            index: subscriber.index,
            value: match &subscriber.value {
                OpenTelemetryResource::Active(value) => {
                    let value = Arc::clone(value);
                    OpenTelemetryResource::Active(Arc::new(
                        move |event: &Event, measurements: &[ValidatedMetricMeasurement]| {
                            value.process_validated(event, measurements)
                        },
                    ) as MetricEventCallback)
                }
                OpenTelemetryResource::Skipped(message) => {
                    OpenTelemetryResource::Skipped(message.clone())
                }
            },
        })
        .collect::<Vec<_>>();
    let metric_diagnostic_field = (!metric_callbacks.is_empty()).then_some("opentelemetry.metrics");
    // Retain the subscribers as long as the registered fan-out callback exists.
    // Their providers and exporter runtimes must outlive event delivery.
    let delivery_trace_subscribers = trace_subscribers.clone();
    let delivery_log_subscribers = log_subscribers.clone();
    let delivery_metric_subscribers = metric_subscribers.clone();
    let rejected_metric_marks = AtomicU64::new(0);
    ctx.add_registration(PluginRegistration::new_with_outcome(
        "observability",
        ctx.qualify_name("opentelemetry.shutdown"),
        Box::new(move || {
            match shutdown_all_opentelemetry_subscribers(
                &trace_subscribers,
                &log_subscribers,
                &metric_subscribers,
            ) {
                None => PluginRegistrationCleanupOutcome::Removed,
                Some(OpenTelemetryShutdownFailure::Delivery(error)) => {
                    PluginRegistrationCleanupOutcome::RemovedWithError(error)
                }
                Some(OpenTelemetryShutdownFailure::Other(error)) => {
                    PluginRegistrationCleanupOutcome::NotRemoved(error)
                }
            }
        }),
    ));
    ctx.register_subscriber(
        "opentelemetry",
        Arc::new(move |event| {
            let _keep_exporters_alive = (
                &delivery_trace_subscribers,
                &delivery_log_subscribers,
                &delivery_metric_subscribers,
            );
            deliver_opentelemetry_event(
                &trace_callbacks,
                &log_callbacks,
                &metric_callbacks,
                &rejected_metric_marks,
                metric_diagnostic_field,
                event,
            );
        }),
    )?;
    Ok(())
}

struct OpenTelemetrySignalSubscribers {
    logs: Vec<IndexedOpenTelemetryResource<Arc<OpenTelemetryLogSubscriber>>>,
    metrics: Vec<IndexedOpenTelemetryResource<Arc<OpenTelemetryMetricSubscriber>>>,
}

fn build_opentelemetry_signal_subscribers(
    logs: Option<OpenTelemetryLogSectionConfig>,
    log_resolution: Option<ResolvedSignalEndpoints>,
    metrics: Option<OpenTelemetryMetricSectionConfig>,
    metric_resolution: Option<ResolvedSignalEndpoints>,
    trace_subscribers: &[IndexedOpenTelemetryResource<Arc<OpenTelemetrySubscriber>>],
) -> PluginResult<OpenTelemetrySignalSubscribers> {
    let logs = match (logs, log_resolution) {
        (Some(section), Some(resolution)) => {
            build_opentelemetry_log_subscribers(section, resolution.endpoints).inspect_err(
                |_| {
                    let _ = shutdown_indexed_opentelemetry_providers(trace_subscribers);
                },
            )?
        }
        _ => Vec::new(),
    };
    let metrics = match (metrics, metric_resolution) {
        (Some(section), Some(resolution)) => {
            build_opentelemetry_metric_subscribers(section, resolution.endpoints).inspect_err(
                |_| {
                    let _ = shutdown_indexed_opentelemetry_providers(trace_subscribers);
                    for subscriber in &logs {
                        if let Some(value) = subscriber.value.as_active() {
                            let _ = value.shutdown_provider();
                        }
                    }
                },
            )?
        }
        _ => Vec::new(),
    };
    Ok(OpenTelemetrySignalSubscribers { logs, metrics })
}

fn has_active_opentelemetry_resource<T>(resources: &[IndexedOpenTelemetryResource<T>]) -> bool {
    resources
        .iter()
        .any(|resource| resource.value.as_active().is_some())
}

fn log_opentelemetry_resource_access<T>(
    signal: &str,
    subscribers: &[IndexedOpenTelemetryResource<T>],
) {
    for subscriber in subscribers {
        if subscriber.value.as_active().is_none() {
            continue;
        }
        log::info!(
            target: "nemo_relay.plugin",
            event = "plugin_resource_access_pending",
            plugin_kind = OBSERVABILITY_PLUGIN_KIND,
            resource_kind = "otlp_endpoint",
            exporter = "opentelemetry",
            signal,
            resource_index = subscriber.index,
            permission = "write";
            "Plugin resource access will be validated during export"
        );
    }
}

fn shutdown_all_opentelemetry_subscribers(
    traces: &[IndexedOpenTelemetryResource<Arc<OpenTelemetrySubscriber>>],
    logs: &[IndexedOpenTelemetryResource<Arc<OpenTelemetryLogSubscriber>>],
    metrics: &[IndexedOpenTelemetryResource<Arc<OpenTelemetryMetricSubscriber>>],
) -> Option<OpenTelemetryShutdownFailure> {
    let mut errors = Vec::new();
    if let Err(error) = flush_subscribers() {
        errors.push(OpenTelemetryShutdownIssue::other(
            crate::observability::otel::OpenTelemetryError::Core(error),
        ));
    }
    errors.extend(shutdown_indexed_opentelemetry_providers(traces));
    for subscriber in logs {
        let Some(value) = subscriber.value.as_active() else {
            continue;
        };
        if let Some(issue) =
            signal_shutdown_issue(value.shutdown_provider(), value.delivery_failure_summary())
        {
            errors.push(issue);
        }
    }
    for subscriber in metrics {
        let Some(value) = subscriber.value.as_active() else {
            continue;
        };
        if let Some(issue) =
            signal_shutdown_issue(value.shutdown_provider(), value.delivery_failure_summary())
        {
            errors.push(issue);
        }
    }
    shutdown_failure_from_errors(errors)
}

fn deliver_opentelemetry_event(
    trace_callbacks: &[IndexedOpenTelemetryResource<EventSubscriberFn>],
    log_callbacks: &[IndexedOpenTelemetryResource<EventSubscriberFn>],
    metric_callbacks: &[IndexedOpenTelemetryResource<MetricEventCallback>],
    rejected_metric_marks: &AtomicU64,
    metric_diagnostic_field: Option<&str>,
    event: &Event,
) {
    match classify_metric_mark(event) {
        MetricMarkClassification::NotMetric => {
            deliver_opentelemetry_callbacks(trace_callbacks, event);
            deliver_opentelemetry_callbacks(log_callbacks, event);
        }
        MetricMarkClassification::Valid(measurements) => {
            deliver_opentelemetry_metric_callbacks(metric_callbacks, event, &measurements);
        }
        MetricMarkClassification::Invalid(error) => {
            reject_opentelemetry_metric_mark(
                event,
                rejected_metric_marks,
                metric_diagnostic_field,
                &error,
            );
        }
    }
}

type MetricEventCallback = Arc<
    dyn for<'event, 'measurements> Fn(&'event Event, &'measurements [ValidatedMetricMeasurement])
        + Send
        + Sync,
>;

fn deliver_opentelemetry_callbacks(
    callbacks: &[IndexedOpenTelemetryResource<EventSubscriberFn>],
    event: &Event,
) {
    for callback in callbacks {
        let Some(value) = callback.value.as_active() else {
            continue;
        };
        if catch_unwind(AssertUnwindSafe(|| value(event))).is_err() {
            log::error!(
                target: "nemo_relay.plugin",
                event = "opentelemetry_endpoint_callback_panicked",
                plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                resource_kind = "otlp_endpoint",
                resource_index = callback.index;
                "OpenTelemetry endpoint callback panicked; delivery continued to remaining endpoints"
            );
        }
    }
}

fn deliver_opentelemetry_metric_callbacks(
    callbacks: &[IndexedOpenTelemetryResource<MetricEventCallback>],
    event: &Event,
    measurements: &[ValidatedMetricMeasurement],
) {
    for callback in callbacks {
        let Some(value) = callback.value.as_active() else {
            continue;
        };
        if catch_unwind(AssertUnwindSafe(|| value(event, measurements))).is_err() {
            log::error!(
                target: "nemo_relay.plugin",
                event = "opentelemetry_endpoint_callback_panicked",
                plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                resource_kind = "otlp_endpoint",
                resource_index = callback.index;
                "OpenTelemetry endpoint callback panicked; delivery continued to remaining endpoints"
            );
        }
    }
}

fn reject_opentelemetry_metric_mark(
    event: &Event,
    rejected_metric_marks: &AtomicU64,
    metric_diagnostic_field: Option<&str>,
    error: &str,
) {
    let rejection_count = rejected_metric_marks.fetch_add(1, Ordering::Relaxed) + 1;
    if rejection_count == 1 {
        log::warn!(
            target: "nemo_relay.observability",
            event = "otel_metric_mark_rejected",
            mark_name = event.name();
            "OpenTelemetry metric mark was dropped atomically: {error}"
        );
    }
    crate::observability::otel_signal::record_signal_runtime_diagnostic(
        "otel.metric_mark_invalid",
        metric_diagnostic_field.map(str::to_owned),
        format!(
            "OpenTelemetry metric mark {:?} was dropped atomically: {error}",
            event.name()
        ),
        1,
    );
}

fn build_opentelemetry_subscribers(
    endpoints: Vec<OpenTelemetryEndpointConfig>,
) -> PluginResult<Vec<IndexedOpenTelemetryResource<Arc<OpenTelemetrySubscriber>>>> {
    let mut subscribers = Vec::with_capacity(endpoints.len());
    for (index, endpoint) in endpoints.into_iter().enumerate() {
        let subscriber = build_otel_config(index, endpoint).and_then(|config| {
            OpenTelemetrySubscriber::new_for_plugin(config, index)
                .map_err(observability_registration_error)
        });
        match subscriber {
            Ok(value) => subscribers.push(IndexedOpenTelemetryResource {
                index,
                value: OpenTelemetryResource::Active(Arc::new(value)),
            }),
            Err(error) => {
                log::warn!(
                    target: "nemo_relay.plugin",
                    event = "opentelemetry_endpoint_skipped",
                    plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                    resource_kind = "otlp_endpoint",
                    resource_index = index;
                    "OpenTelemetry endpoint was skipped during activation; delivery continues to valid endpoints: {error}"
                );
                subscribers.push(IndexedOpenTelemetryResource {
                    index,
                    value: OpenTelemetryResource::Skipped(error.to_string()),
                });
            }
        }
    }
    Ok(subscribers)
}

fn resolve_signal_endpoints(
    signal: &'static str,
    explicit: Option<&Vec<OpenTelemetrySignalEndpointConfig>>,
    traces: &[OpenTelemetryEndpointConfig],
) -> PluginResult<ResolvedSignalEndpoints> {
    let endpoints = match explicit {
        Some(endpoints) if endpoints.is_empty() => {
            return Err(PluginError::InvalidConfig(format!(
                "enabled OpenTelemetry {signal} section requires at least one explicit endpoint"
            )));
        }
        Some(endpoints) => {
            for (index, endpoint) in endpoints.iter().enumerate() {
                validate_explicit_signal_endpoint(signal, index, endpoint)?;
            }
            endpoints
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, value)| IndexedOpenTelemetryResource {
                    index,
                    value: OpenTelemetryResource::Active(value),
                })
                .collect()
        }
        None if traces.is_empty() => {
            return Err(PluginError::InvalidConfig(format!(
                "enabled OpenTelemetry {signal} section has no explicit or derivable trace endpoints"
            )));
        }
        None => {
            let mut endpoints = Vec::with_capacity(traces.len());
            for (index, trace) in traces.iter().enumerate() {
                let value = match derive_signal_endpoint(signal, index, trace) {
                    Ok(value) => OpenTelemetryResource::Active(value),
                    Err(error) => OpenTelemetryResource::Skipped(error.to_string()),
                };
                endpoints.push(IndexedOpenTelemetryResource { index, value });
            }
            endpoints
        }
    };
    validate_distinct_signal_destinations(signal, &endpoints)?;
    Ok(ResolvedSignalEndpoints { endpoints })
}

fn warn_skipped_signal_endpoints(
    signal: &str,
    endpoints: &[IndexedOpenTelemetryResource<OpenTelemetrySignalEndpointConfig>],
) {
    for endpoint in endpoints {
        let OpenTelemetryResource::Skipped(message) = &endpoint.value else {
            continue;
        };
        log::warn!(
            target: "nemo_relay.plugin",
            event = "opentelemetry_endpoint_skipped",
            plugin_kind = OBSERVABILITY_PLUGIN_KIND,
            resource_kind = "otlp_endpoint",
            signal,
            resource_index = endpoint.index;
            "OpenTelemetry {signal} endpoint was skipped during activation; delivery continues to valid endpoints: {}",
            message
        );
    }
}

fn validate_explicit_signal_endpoint(
    signal: &str,
    index: usize,
    endpoint: &OpenTelemetrySignalEndpointConfig,
) -> PluginResult<()> {
    if endpoint.endpoint.trim().is_empty() {
        return Err(PluginError::InvalidConfig(format!(
            "OpenTelemetry {signal}.endpoints[{index}].endpoint must be nonblank"
        )));
    }
    if endpoint.transport == "http_binary" {
        let path = reqwest::Url::parse(endpoint.endpoint.trim())
            .ok()
            .map(|url| url.path().trim_end_matches('/').to_string());
        if let Some(path) = path {
            for other_signal in ["traces", "logs", "metrics"] {
                if other_signal != signal && path.ends_with(&format!("/v1/{other_signal}")) {
                    return Err(PluginError::InvalidConfig(format!(
                        "OpenTelemetry {signal}.endpoints[{index}] ends in /v1/{other_signal}; configure a {signal} endpoint or a bare authority"
                    )));
                }
            }
        }
    }
    parse_signal_transport(signal, index, &endpoint.transport)?;
    Ok(())
}

fn derive_signal_endpoint(
    signal: &str,
    index: usize,
    trace: &OpenTelemetryEndpointConfig,
) -> PluginResult<OpenTelemetrySignalEndpointConfig> {
    let transport = parse_signal_transport("traces", index, &trace.transport)?;
    let endpoint = match transport {
        OtlpTransport::Grpc => trace.endpoint.clone(),
        OtlpTransport::HttpBinary => derive_http_signal_endpoint(signal, index, &trace.endpoint)?,
    };
    Ok(OpenTelemetrySignalEndpointConfig {
        endpoint,
        transport: trace.transport.clone(),
        headers: trace.headers.clone(),
        header_env: trace.header_env.clone(),
        resource_attributes: trace.resource_attributes.clone(),
        service_name: trace.service_name.clone(),
        service_namespace: trace.service_namespace.clone(),
        service_version: trace.service_version.clone(),
        instrumentation_scope: trace.instrumentation_scope.clone(),
        timeout_millis: trace.timeout_millis,
    })
}

fn derive_http_signal_endpoint(signal: &str, index: usize, endpoint: &str) -> PluginResult<String> {
    let trimmed = endpoint.trim();
    let parsed = reqwest::Url::parse(trimmed).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "OpenTelemetry endpoints[{index}] cannot derive {signal} endpoint from {trimmed:?}: {error}"
        ))
    })?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(PluginError::InvalidConfig(format!(
            "OpenTelemetry endpoints[{index}] cannot derive {signal} endpoint from non-HTTP endpoint {trimmed:?}"
        )));
    }
    let path = parsed.path();
    let is_bare_authority = path == "/";
    if is_bare_authority || path.ends_with("/v1/traces") {
        let derived = match signal {
            "logs" => resolve_http_log_endpoint(trimmed),
            "metrics" => resolve_http_metric_endpoint(trimmed),
            _ => Cow::Borrowed(trimmed),
        };
        return Ok(derived.into_owned());
    }
    Err(PluginError::InvalidConfig(format!(
        "OpenTelemetry endpoints[{index}] path {path:?} cannot derive /v1/{signal}; configure opentelemetry.{signal}.endpoints explicitly"
    )))
}

fn parse_signal_transport(
    signal: &str,
    index: usize,
    transport: &str,
) -> PluginResult<OtlpTransport> {
    match transport {
        "http_binary" => Ok(OtlpTransport::HttpBinary),
        "grpc" => Ok(OtlpTransport::Grpc),
        other => Err(PluginError::InvalidConfig(format!(
            "OpenTelemetry {signal}.endpoints[{index}].transport must be 'http_binary' or 'grpc', got {other:?}"
        ))),
    }
}

fn validate_distinct_signal_destinations(
    signal: &str,
    endpoints: &[IndexedOpenTelemetryResource<OpenTelemetrySignalEndpointConfig>],
) -> PluginResult<()> {
    for (position, endpoint) in endpoints.iter().enumerate() {
        let Some(value) = endpoint.value.as_active() else {
            continue;
        };
        let effective = signal_destination(signal, value);
        for other in &endpoints[..position] {
            let Some(other_value) = other.value.as_active() else {
                continue;
            };
            if value.transport == other_value.transport
                && effective.key == signal_destination(signal, other_value).key
            {
                return Err(PluginError::InvalidConfig(format!(
                    "OpenTelemetry {signal}.endpoints[{}] and {signal}.endpoints[{}] use the same {} destination {:?}",
                    other.index, endpoint.index, value.transport, effective.display
                )));
            }
        }
    }
    Ok(())
}

fn signal_destination(
    signal: &str,
    endpoint: &OpenTelemetrySignalEndpointConfig,
) -> OpenTelemetryDestination {
    let configured = endpoint.endpoint.trim();
    let effective = if endpoint.transport == "http_binary" {
        match signal {
            "logs" => resolve_http_log_endpoint(configured),
            "metrics" => resolve_http_metric_endpoint(configured),
            _ => Cow::Borrowed(configured),
        }
    } else {
        Cow::Borrowed(configured)
    };
    canonicalize_opentelemetry_destination(&effective)
}

fn build_opentelemetry_log_subscribers(
    section: OpenTelemetryLogSectionConfig,
    endpoints: Vec<IndexedOpenTelemetryResource<OpenTelemetrySignalEndpointConfig>>,
) -> PluginResult<Vec<IndexedOpenTelemetryResource<Arc<OpenTelemetryLogSubscriber>>>> {
    let minimum_severity = section
        .minimum_severity
        .parse::<LogSeverity>()
        .map_err(|error| {
            PluginError::InvalidConfig(format!("OpenTelemetry logs.minimum_severity {error}"))
        })?;
    if section.max_queue_size == 0
        || section.max_export_batch_size == 0
        || section.scheduled_delay_millis == 0
        || section.completed_span_context_ttl_millis == 0
        || section.max_export_batch_size > section.max_queue_size
    {
        return Err(PluginError::InvalidConfig(
            "OpenTelemetry logs batch settings and completed_span_context_ttl_millis must be positive and max_export_batch_size must not exceed max_queue_size".to_string(),
        ));
    }
    let mut subscribers = Vec::with_capacity(endpoints.len());
    for endpoint in endpoints {
        let index = endpoint.index;
        let value = match endpoint.value {
            OpenTelemetryResource::Skipped(message) => OpenTelemetryResource::Skipped(message),
            OpenTelemetryResource::Active(endpoint) => {
                let result = build_log_config(index, endpoint, &section, minimum_severity)
                    .and_then(|config| {
                        OpenTelemetryLogSubscriber::new_for_plugin(config, index)
                            .map_err(observability_registration_error)
                    });
                match result {
                    Ok(value) => OpenTelemetryResource::Active(Arc::new(value)),
                    Err(error) => {
                        log::warn!(
                            target: "nemo_relay.plugin",
                            event = "opentelemetry_endpoint_skipped",
                            plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                            resource_kind = "otlp_endpoint",
                            signal = "logs",
                            resource_index = index;
                            "OpenTelemetry log endpoint was skipped during activation; delivery continues to valid endpoints: {error}"
                        );
                        OpenTelemetryResource::Skipped(error.to_string())
                    }
                }
            }
        };
        subscribers.push(IndexedOpenTelemetryResource { index, value });
    }
    Ok(subscribers)
}

fn build_opentelemetry_metric_subscribers(
    section: OpenTelemetryMetricSectionConfig,
    endpoints: Vec<IndexedOpenTelemetryResource<OpenTelemetrySignalEndpointConfig>>,
) -> PluginResult<Vec<IndexedOpenTelemetryResource<Arc<OpenTelemetryMetricSubscriber>>>> {
    let temporality = section
        .temporality
        .parse::<MetricTemporality>()
        .map_err(|error| {
            PluginError::InvalidConfig(format!("OpenTelemetry metrics.temporality {error}"))
        })?;
    if section.export_interval_millis == 0
        || section.max_instruments == 0
        || section.cardinality_limit == 0
        || section.cardinality_limit == usize::MAX
    {
        return Err(PluginError::InvalidConfig(
            "OpenTelemetry metrics export_interval_millis and max_instruments must be greater than 0, and cardinality_limit must be greater than 0 and less than usize::MAX".to_string(),
        ));
    }
    let mut subscribers = Vec::with_capacity(endpoints.len());
    for endpoint in endpoints {
        let index = endpoint.index;
        let value = match endpoint.value {
            OpenTelemetryResource::Skipped(message) => OpenTelemetryResource::Skipped(message),
            OpenTelemetryResource::Active(endpoint) => {
                let result = build_metric_config(index, endpoint, &section, temporality).and_then(
                    |config| {
                        OpenTelemetryMetricSubscriber::new_for_plugin(config, index)
                            .map_err(observability_registration_error)
                    },
                );
                match result {
                    Ok(value) => OpenTelemetryResource::Active(Arc::new(value)),
                    Err(error) => {
                        log::warn!(
                            target: "nemo_relay.plugin",
                            event = "opentelemetry_endpoint_skipped",
                            plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                            resource_kind = "otlp_endpoint",
                            signal = "metrics",
                            resource_index = index;
                            "OpenTelemetry metric endpoint was skipped during activation; delivery continues to valid endpoints: {error}"
                        );
                        OpenTelemetryResource::Skipped(error.to_string())
                    }
                }
            }
        };
        subscribers.push(IndexedOpenTelemetryResource { index, value });
    }
    Ok(subscribers)
}

fn build_log_config(
    index: usize,
    endpoint: OpenTelemetrySignalEndpointConfig,
    section: &OpenTelemetryLogSectionConfig,
    minimum_severity: LogSeverity,
) -> PluginResult<CoreOpenTelemetryLogConfig> {
    let transport = parse_signal_transport("logs", index, &endpoint.transport)?;
    let headers = resolve_signal_headers("logs", index, &endpoint)?;
    let mut config = CoreOpenTelemetryLogConfig::new(endpoint.endpoint.clone())
        .with_transport(transport)
        .with_service_name(endpoint.service_name.clone())
        .with_instrumentation_scope(endpoint.instrumentation_scope.clone())
        .with_timeout(Duration::from_millis(endpoint.timeout_millis))
        .with_minimum_severity(minimum_severity)
        .with_max_queue_size(section.max_queue_size)
        .with_max_export_batch_size(section.max_export_batch_size)
        .with_scheduled_delay(Duration::from_millis(section.scheduled_delay_millis))
        .with_completed_span_context_ttl(Duration::from_millis(
            section.completed_span_context_ttl_millis,
        ));
    config = apply_signal_common(config, &endpoint, headers);
    Ok(config)
}

fn build_metric_config(
    index: usize,
    endpoint: OpenTelemetrySignalEndpointConfig,
    section: &OpenTelemetryMetricSectionConfig,
    temporality: MetricTemporality,
) -> PluginResult<CoreOpenTelemetryMetricConfig> {
    let transport = parse_signal_transport("metrics", index, &endpoint.transport)?;
    let headers = resolve_signal_headers("metrics", index, &endpoint)?;
    let mut config = CoreOpenTelemetryMetricConfig::new(endpoint.endpoint)
        .with_transport(transport)
        .with_service_name(endpoint.service_name)
        .with_instrumentation_scope(endpoint.instrumentation_scope)
        .with_timeout(Duration::from_millis(endpoint.timeout_millis))
        .with_export_interval(Duration::from_millis(section.export_interval_millis))
        .with_temporality(temporality)
        .with_max_instruments(section.max_instruments)
        .with_cardinality_limit(section.cardinality_limit);
    if let Some(namespace) = endpoint.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = endpoint.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in headers {
        config = config.with_header(key, value);
    }
    for (key, value) in endpoint.resource_attributes {
        config = config.with_resource_attribute(key, value);
    }
    Ok(config)
}

fn apply_signal_common(
    mut config: CoreOpenTelemetryLogConfig,
    endpoint: &OpenTelemetrySignalEndpointConfig,
    headers: HashMap<String, String>,
) -> CoreOpenTelemetryLogConfig {
    if let Some(namespace) = &endpoint.service_namespace {
        config = config.with_service_namespace(namespace.as_str());
    }
    if let Some(version) = &endpoint.service_version {
        config = config.with_service_version(version.as_str());
    }
    for (key, value) in headers {
        config = config.with_header(key, value);
    }
    for (key, value) in &endpoint.resource_attributes {
        config = config.with_resource_attribute(key.as_str(), value.as_str());
    }
    config
}

fn resolve_signal_headers(
    signal: &str,
    index: usize,
    endpoint: &OpenTelemetrySignalEndpointConfig,
) -> PluginResult<HashMap<String, String>> {
    let mut headers = endpoint.headers.clone();
    for (key, variable) in &endpoint.header_env {
        if variable.trim().is_empty() || variable.trim() != variable {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry {signal}.endpoints[{index}].header_env.{key} must name a nonblank environment variable without surrounding whitespace"
            )));
        }
        if headers
            .keys()
            .any(|configured| configured.eq_ignore_ascii_case(key))
        {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry {signal}.endpoints[{index}] header {key:?} cannot appear in both headers and header_env"
            )));
        }
        let value = std::env::var(variable).map_err(|_| {
            PluginError::InvalidConfig(format!(
                "OpenTelemetry {signal}.endpoints[{index}].header_env.{key} could not read environment variable {variable:?}"
            ))
        })?;
        if value.trim().is_empty() || value.trim() != value {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry {signal}.endpoints[{index}].header_env.{key} references a blank or padded environment variable {variable:?}"
            )));
        }
        headers.insert(key.clone(), value);
    }
    validate_signal_headers(&headers).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "OpenTelemetry {signal}.endpoints[{index}] has invalid headers: {error}"
        ))
    })?;
    Ok(headers)
}

enum OpenTelemetryShutdownFailure {
    Delivery(PluginError),
    Other(PluginError),
}

struct OpenTelemetryShutdownIssue {
    message: String,
    delivery_failure: bool,
}

impl OpenTelemetryShutdownIssue {
    fn trace(error: crate::observability::otel::OpenTelemetryError) -> Self {
        let message = error.to_string();
        let delivery_failure = message.contains(OTEL_RUNTIME_DELIVERY_FAILURE_MARKER);
        Self {
            message,
            delivery_failure,
        }
    }

    fn other(error: crate::observability::otel::OpenTelemetryError) -> Self {
        Self {
            message: error.to_string(),
            delivery_failure: false,
        }
    }
}

fn signal_shutdown_issue(
    result: crate::observability::otel::Result<()>,
    delivery_summary: Option<String>,
) -> Option<OpenTelemetryShutdownIssue> {
    match (result.err(), delivery_summary) {
        (None, None) => None,
        (Some(error), None) => Some(OpenTelemetryShutdownIssue::other(error)),
        (error, Some(summary)) => {
            let message = error.map_or(summary.clone(), |error| format!("{summary}; {error}"));
            Some(OpenTelemetryShutdownIssue {
                message,
                delivery_failure: true,
            })
        }
    }
}

#[cfg(test)]
fn shutdown_opentelemetry_subscribers(
    subscribers: &[Arc<OpenTelemetrySubscriber>],
) -> Option<OpenTelemetryShutdownFailure> {
    let mut errors = Vec::new();
    if let Err(error) = flush_subscribers() {
        errors.push(OpenTelemetryShutdownIssue::other(
            crate::observability::otel::OpenTelemetryError::Core(error),
        ));
    }
    errors.extend(shutdown_opentelemetry_providers(subscribers));
    shutdown_failure_from_errors(errors)
}

fn shutdown_failure_from_errors(
    errors: Vec<OpenTelemetryShutdownIssue>,
) -> Option<OpenTelemetryShutdownFailure> {
    if errors.is_empty() {
        return None;
    }

    let all_delivery_failures = errors.iter().all(|error| error.delivery_failure);
    let summary = errors
        .into_iter()
        .map(|error| error.message)
        .collect::<Vec<_>>()
        .join("; ");
    let message = if all_delivery_failures {
        format!("{OTEL_RUNTIME_DELIVERY_FAILURE_MARKER}: {summary}")
    } else {
        format!("OpenTelemetry shutdown failures: {summary}")
    };
    let error = PluginError::RegistrationFailed(message);
    Some(if all_delivery_failures {
        OpenTelemetryShutdownFailure::Delivery(error)
    } else {
        OpenTelemetryShutdownFailure::Other(error)
    })
}

#[cfg(test)]
fn shutdown_opentelemetry_providers(
    subscribers: &[Arc<OpenTelemetrySubscriber>],
) -> Vec<OpenTelemetryShutdownIssue> {
    let mut errors = Vec::new();
    for subscriber in subscribers {
        if let Err(error) = subscriber.shutdown_provider() {
            errors.push(OpenTelemetryShutdownIssue::trace(error));
        }
    }
    errors
}

fn shutdown_indexed_opentelemetry_providers(
    subscribers: &[IndexedOpenTelemetryResource<Arc<OpenTelemetrySubscriber>>],
) -> Vec<OpenTelemetryShutdownIssue> {
    let mut errors = Vec::new();
    for subscriber in subscribers {
        let Some(value) = subscriber.value.as_active() else {
            continue;
        };
        if let Err(error) = value.shutdown_provider() {
            errors.push(OpenTelemetryShutdownIssue::trace(error));
        }
    }
    errors
}

struct AtifDispatcher {
    config: AtifSectionConfig,
    agents: HashMap<Uuid, ManagedAtifExporter>,
    scope_owners: HashMap<Uuid, Uuid>,
    scope_subscribers: HashMap<Uuid, String>,
    /// Fatal dispatcher errors (subscriber registration, payload serialization)
    /// that cannot be isolated to a single sink. Once set, the dispatcher stops
    /// observing further events.
    fatal_error: Option<String>,
    runtime_failures: Vec<RuntimeDiagnostic>,
    validated_sinks: HashSet<SinkLabel>,
}

struct ManagedAtifExporter {
    exporter: AtifExporter,
    filename: String,
    local_root: Option<PathBuf>,
    local_path: Option<PathBuf>,
    correlation: AtifCorrelation,
    observed_events: Vec<Event>,
    observed_event_keys: HashSet<String>,
    written: bool,
}

struct PendingAtifWrite {
    agent_uuid: Uuid,
    #[cfg_attr(not(feature = "object-store"), allow(dead_code))]
    session_id: String,
    // `filename` is consumed by the remote upload path, which is gated on the
    // object-store feature; without it, only the local sink reads `local_path`.
    #[cfg_attr(not(feature = "object-store"), allow(dead_code))]
    filename: String,
    local_root: Option<PathBuf>,
    local_path: Option<PathBuf>,
    payload: Vec<u8>,
}

struct AtifFlushWork {
    exports: Vec<PendingAtifExport>,
    scope_subscribers: Vec<(Uuid, String)>,
}

struct PendingAtifExport {
    agent_uuid: Uuid,
    exporter: AtifExporter,
    filename: String,
    local_root: Option<PathBuf>,
    local_path: Option<PathBuf>,
    correlation: AtifCorrelation,
}

#[derive(Clone)]
struct AtifCorrelation {
    session_id: Option<String>,
    session_instance_id: Option<String>,
    user_id: Option<String>,
}

impl AtifCorrelation {
    fn from_event(event: &Event) -> Self {
        let metadata = event.metadata();
        Self {
            session_id: metadata
                .and_then(|value| value.get("session_id"))
                .and_then(Json::as_str)
                .map(ToString::to_string),
            session_instance_id: event.propagation_root_uuid().map(|uuid| uuid.to_string()),
            user_id: metadata
                .and_then(|value| value.get("user_id"))
                .and_then(Json::as_str)
                .map(ToString::to_string),
        }
    }

    fn to_json(&self) -> Json {
        let mut fields = Map::new();
        if let Some(session_id) = &self.session_id {
            fields.insert("session_id".to_string(), Json::String(session_id.clone()));
        }
        if let Some(session_instance_id) = &self.session_instance_id {
            fields.insert(
                "session_instance_id".to_string(),
                Json::String(session_instance_id.clone()),
            );
        }
        if let Some(user_id) = &self.user_id {
            fields.insert("user_id".to_string(), Json::String(user_id.clone()));
        }
        Json::Object(fields)
    }
}

/// Identifier for a single output sink. `Local` is used when `storage` is empty
/// (the legacy local-file path); `Remote(i)` indexes into the configured
/// storage backends.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SinkLabel {
    Local,
    Remote(usize),
}

impl AtifDispatcher {
    fn new(config: AtifSectionConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            scope_owners: HashMap::new(),
            scope_subscribers: HashMap::new(),
            fatal_error: None,
            runtime_failures: Vec::new(),
            validated_sinks: HashSet::new(),
        }
    }

    fn observe_global(
        &mut self,
        event: &Event,
        subscriber_prefix: &str,
        state: Arc<Mutex<Self>>,
        storage: AtifStorageList,
    ) -> Option<(PendingAtifWrite, Vec<SinkLabel>)> {
        if self.fatal_error.is_some() {
            return None;
        }

        if !is_top_level_trajectory_start(event) {
            return self.observe_descendant_from_global(event);
        }

        if self.agents.contains_key(&event.uuid()) {
            return None;
        }

        // The top-level trajectory scope UUID remains the artifact identity. The global
        // dispatcher records the start event itself because the scope-local
        // subscriber is attached after that start event has already been
        // emitted.
        let trajectory_id = event.uuid().to_string();
        let session_id = event
            .propagation_root_uuid()
            .unwrap_or(event.uuid())
            .to_string();
        let (filename, local_root, local_path) =
            match self.prepare_destination(&trajectory_id, event.metadata()) {
                Ok(destination) => destination,
                Err(error) => {
                    self.record_runtime_failure(
                        "atif.destination_render_failed",
                        Some("filename_template".into()),
                        error.clone(),
                        Some(trajectory_id.clone()),
                    );
                    log::warn!(
                        target: "nemo_relay.observability",
                        event = "atif_destination_render_failed",
                        plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                        exporter = "atif",
                        trajectory_id = trajectory_id.as_str();
                        "ATIF destination rendering failed: {error}"
                    );
                    return None;
                }
            };
        let exporter =
            AtifExporter::new_for_trajectory(session_id, trajectory_id, self.agent_info());
        (exporter.subscriber())(event);
        let correlation = AtifCorrelation::from_event(event);
        self.scope_owners.insert(event.uuid(), event.uuid());
        self.agents.insert(
            event.uuid(),
            ManagedAtifExporter {
                exporter,
                filename,
                local_root,
                local_path,
                correlation,
                observed_events: vec![event.clone()],
                observed_event_keys: HashSet::from([event_observation_key(event)]),
                written: false,
            },
        );

        let agent_uuid = event.uuid();
        let name = format!("{subscriber_prefix}{agent_uuid}");
        let callback = atif_scope_subscriber(state, agent_uuid, storage);
        // Attach the scoped subscriber to the trajectory root rather than the
        // global registry so sibling top-level trajectories never share events.
        // With async subscriber delivery, the root scope may already be closed
        // when the dispatcher observes this start event; global routing still
        // handles descendant events by parent UUID in that case.
        if try_scope_register_subscriber(&agent_uuid, &name, callback).is_ok() {
            self.scope_subscribers.insert(agent_uuid, name);
        }
        None
    }

    fn observe_descendant_from_global(
        &mut self,
        event: &Event,
    ) -> Option<(PendingAtifWrite, Vec<SinkLabel>)> {
        let owner = self.scope_owners.get(&event.uuid()).copied().or_else(|| {
            event
                .parent_uuid()
                .and_then(|uuid| self.scope_owners.get(&uuid).copied())
        })?;

        if event.scope_category() == Some(ScopeCategory::Start) {
            self.scope_owners.insert(event.uuid(), owner);
        }

        let pending_write = self.observe_scope(event, owner);

        if event.scope_category() == Some(ScopeCategory::End) && event.uuid() != owner {
            self.scope_owners.remove(&event.uuid());
        }

        pending_write
    }

    fn observe_scope(
        &mut self,
        event: &Event,
        agent_uuid: Uuid,
    ) -> Option<(PendingAtifWrite, Vec<SinkLabel>)> {
        if self.fatal_error.is_some() {
            return None;
        }
        let should_finalize =
            event.uuid() == agent_uuid && event.scope_category() == Some(ScopeCategory::End);
        let agent = self.agents.get_mut(&agent_uuid)?;
        if !agent
            .observed_event_keys
            .insert(event_observation_key(event))
        {
            return None;
        }
        (agent.exporter.subscriber())(event);
        agent.observed_events.push(event.clone());
        if !should_finalize || agent.written {
            return None;
        }
        let write = match prepare_atif_file(agent_uuid, agent) {
            Ok(write) => write,
            Err(err) => {
                self.fatal_error = Some(err.to_string());
                return None;
            }
        };
        let targets = self.sink_targets();
        Some((write, targets))
    }

    fn complete_scope_write(
        &mut self,
        agent_uuid: Uuid,
        results: Vec<(SinkLabel, std::io::Result<()>)>,
    ) -> Option<(Uuid, String)> {
        let is_remote_fallback = results
            .iter()
            .any(|(label, _)| matches!(label, SinkLabel::Remote(_)));
        for (label, result) in results {
            if result.is_ok()
                && label == SinkLabel::Local
                && self.validated_sinks.insert(label.clone())
            {
                log::info!(
                    target: "nemo_relay.observability",
                    event = "storage_access_validated",
                    plugin_kind = "observability",
                    exporter = "atif",
                    resource_kind = "local_file",
                    permission = "write";
                    "ATIF storage access validated"
                );
            } else if let Err(err) = result {
                let (field, message) = match &label {
                    SinkLabel::Local => ("output_directory".to_string(), err.to_string()),
                    SinkLabel::Remote(index) => (format!("storage[{index}]"), err.to_string()),
                };
                match &label {
                    SinkLabel::Local => log::warn!(
                        target: "nemo_relay.observability",
                        event = "storage_access_failed",
                        plugin_kind = "observability",
                        exporter = "atif",
                        resource_kind = "local_file",
                        permission = "write",
                        reason = "write_failed";
                        "ATIF storage access failed"
                    ),
                    SinkLabel::Remote(index) => log::warn!(
                        target: "nemo_relay.observability",
                        event = "atif_remote_delivery_failed",
                        plugin_kind = OBSERVABILITY_PLUGIN_KIND,
                        exporter = "atif",
                        storage_index = *index;
                        "ATIF remote storage upload failed"
                    ),
                }
                self.record_runtime_failure(
                    match &label {
                        SinkLabel::Local if is_remote_fallback => "atif.local_fallback_failed",
                        SinkLabel::Local => "atif.local_write_failed",
                        SinkLabel::Remote(_) => "atif.remote_delivery_failed",
                    },
                    Some(field),
                    message,
                    Some(agent_uuid.to_string()),
                );
            }
        }
        if let Some(agent) = self.agents.get_mut(&agent_uuid) {
            agent.observed_events.clear();
        }
        self.agents.remove(&agent_uuid);
        self.scope_owners.retain(|_, owner| *owner != agent_uuid);
        self.scope_subscribers
            .remove(&agent_uuid)
            .map(|name| (agent_uuid, name))
    }

    fn flush_open_agents(&mut self) -> AtifFlushWork {
        // Plugin teardown may run before an agent scope closes. Remove dynamic
        // scope-local subscribers first so the later scope end event cannot
        // trigger a second write after the dispatcher has flushed.
        let scope_subscribers = std::mem::take(&mut self.scope_subscribers)
            .into_iter()
            .collect();
        let agent_uuids = self
            .agents
            .iter()
            .filter_map(|(agent_uuid, agent)| (!agent.written).then_some(*agent_uuid))
            .collect::<Vec<_>>();
        let mut exports = Vec::with_capacity(agent_uuids.len());
        for agent_uuid in agent_uuids {
            if let Some(agent) = self.agents.get_mut(&agent_uuid) {
                agent.written = true;
                exports.push(PendingAtifExport {
                    agent_uuid,
                    exporter: agent.exporter.clone(),
                    filename: agent.filename.clone(),
                    local_root: agent.local_root.clone(),
                    local_path: agent.local_path.clone(),
                    correlation: agent.correlation.clone(),
                });
            }
        }
        AtifFlushWork {
            exports,
            scope_subscribers,
        }
    }

    fn observed_events(&self, agent_uuid: Uuid) -> Vec<Event> {
        self.agents
            .get(&agent_uuid)
            .map(|agent| agent.observed_events.clone())
            .unwrap_or_default()
    }

    fn last_error_result(&self) -> std::io::Result<()> {
        if let Some(message) = &self.fatal_error {
            return Err(std::io::Error::other(message.clone()));
        }
        if !self.runtime_failures.is_empty() {
            return Err(std::io::Error::other(format!(
                "{ATIF_RUNTIME_DELIVERY_FAILURE_MARKER}: {}",
                self.runtime_failures
                    .iter()
                    .map(|diagnostic| format!("{} ({})", diagnostic.code, diagnostic.count))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(())
    }

    fn agent_info(&self) -> AtifAgentInfo {
        AtifAgentInfo {
            name: self.config.agent_name.clone(),
            version: self.config.agent_version.clone(),
            model_name: Some(self.config.model_name.clone()),
            tool_definitions: self.config.tool_definitions.clone(),
            extra: self.config.extra.clone(),
        }
    }

    fn prepare_destination(
        &self,
        session_id: &str,
        metadata: Option<&Json>,
    ) -> Result<(String, Option<PathBuf>, Option<PathBuf>), String> {
        validate_atif_filename_template(&self.config.filename_template)?;
        if !is_safe_atif_session_id(session_id) {
            return Err("ATIF session_id must be a non-empty path-safe filename component".into());
        }
        let filename = render_atif_filename(&self.config.filename_template, session_id, metadata)?;
        let rendered_path = Path::new(&filename);
        if rendered_path.is_absolute()
            || rendered_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir
                        | Component::CurDir
                        | Component::RootDir
                        | Component::Prefix(_)
                )
            })
        {
            return Err("rendered ATIF filename must remain a path-safe relative path".into());
        }
        let directory = self
            .config
            .output_directory
            .clone()
            .unwrap_or_else(default_output_directory);
        let path = directory.join(&filename);
        Ok((filename, Some(directory), Some(path)))
    }

    fn sink_targets(&self) -> Vec<SinkLabel> {
        if self.config.storage.is_empty() {
            vec![SinkLabel::Local]
        } else {
            (0..self.config.storage.len())
                .map(SinkLabel::Remote)
                .collect()
        }
    }

    fn record_runtime_failure(
        &mut self,
        code: &str,
        field: Option<String>,
        message: String,
        session_id: Option<String>,
    ) {
        let diagnostic = RuntimeDiagnostic {
            code: code.to_string(),
            component: OBSERVABILITY_PLUGIN_KIND.to_string(),
            field,
            message,
            session_id,
            count: 1,
        };
        if let Some(existing) = self
            .runtime_failures
            .iter_mut()
            .find(|existing| existing.code == diagnostic.code && existing.field == diagnostic.field)
        {
            existing.message = diagnostic.message.clone();
            existing.session_id = diagnostic.session_id.clone();
            existing.count += 1;
        } else {
            self.runtime_failures.push(diagnostic.clone());
        }
        record_active_plugin_runtime_diagnostic(diagnostic);
    }
}

fn is_safe_atif_session_id(value: &str) -> bool {
    !matches!(value, "" | "." | "..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~'))
}

fn is_valid_atif_metadata_selector(selector: &str) -> bool {
    !selector.is_empty()
        && selector.split('.').all(|segment| {
            !segment.is_empty()
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        })
}

fn parse_atif_metadata_expression(expression: &str) -> Result<(&str, Option<&str>), String> {
    let (selector, fallback) = expression
        .split_once(":-")
        .map_or((expression, None), |(key, value)| (key, Some(value)));
    if !is_valid_atif_metadata_selector(selector) {
        return Err(format!(
            "ATIF filename_template metadata placeholder '{{metadata.{selector}}}' must contain a dot-separated path of ASCII letters, digits, '-' or '_'"
        ));
    }
    Ok((selector, fallback))
}

fn validate_atif_filename_template(template: &str) -> Result<(), String> {
    const PREFIX: &str = "{metadata.";

    if !template.contains("{session_id}") {
        return Err("ATIF filename_template must contain '{session_id}'".to_string());
    }

    let literal_path = template
        .replace("{session_id}", "session")
        .replace("{metadata.", "metadata.");
    if Path::new(&literal_path).is_absolute()
        || Path::new(&literal_path).components().any(|component| {
            matches!(
                component,
                Component::ParentDir
                    | Component::CurDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
    {
        return Err("ATIF filename_template must be a path-safe relative path".to_string());
    }

    let mut cursor = 0;
    while let Some(relative_start) = template[cursor..].find(PREFIX) {
        let selector_start = cursor + relative_start + PREFIX.len();
        let end = template[selector_start..]
            .find('}')
            .map(|relative_end| selector_start + relative_end)
            .ok_or_else(|| {
                "ATIF filename_template contains an unclosed metadata placeholder".to_string()
            })?;
        let (_, fallback) = parse_atif_metadata_expression(&template[selector_start..end])?;
        if let Some(fallback) = fallback
            && !is_safe_atif_metadata_path(fallback)
        {
            return Err(format!(
                "ATIF filename_template fallback '{fallback}' must be a path-safe relative fragment"
            ));
        }
        cursor = end + 1;
    }
    Ok(())
}

fn render_atif_filename(
    template: &str,
    session_id: &str,
    metadata: Option<&Json>,
) -> Result<String, String> {
    const PREFIX: &str = "{metadata.";

    let mut rendered = template.replace("{session_id}", session_id);
    let mut cursor = 0;
    while let Some(relative_start) = rendered[cursor..].find(PREFIX) {
        let start = cursor + relative_start;
        let selector_start = start + PREFIX.len();
        let end = rendered[selector_start..]
            .find('}')
            .map(|relative_end| selector_start + relative_end)
            .ok_or_else(|| {
                "ATIF filename_template contains an unclosed metadata placeholder".to_string()
            })?;
        let expression = rendered[selector_start..end].to_string();
        let (selector, fallback) = parse_atif_metadata_expression(&expression)?;
        let mut resolved = metadata;
        for segment in selector.split('.') {
            resolved = match resolved {
                Some(Json::Object(object)) => object.get(segment),
                None | Some(Json::Null) => break,
                Some(_) => {
                    return Err(format!(
                        "filename_template placeholder '{{metadata.{selector}}}' traversed a non-object value"
                    ));
                }
            };
        }
        let value = match resolved {
            Some(Json::String(value)) => sanitize_atif_metadata_fragment(value),
            None | Some(Json::Null) => fallback.map(str::to_string).ok_or_else(|| {
                format!(
                    "filename_template placeholder '{{metadata.{selector}}}' must resolve to a string"
                )
            })?,
            Some(_) => {
                return Err(format!(
                    "filename_template placeholder '{{metadata.{selector}}}' resolved to a non-string value"
                ));
            }
        };
        if !is_safe_atif_metadata_path(&value) {
            return Err(format!(
                "metadata path '{selector}' must be a path-safe relative fragment"
            ));
        }
        rendered.replace_range(start..=end, &value);
        cursor = start + value.len();
    }
    Ok(rendered)
}

fn sanitize_atif_metadata_fragment(value: &str) -> String {
    let mut characters = value.chars().peekable();
    let mut sanitized = String::with_capacity(value.len());
    let mut replacing = false;

    while let Some(character) = characters.next() {
        let safe = if character == '.' {
            let mut count = 1;
            while characters.next_if_eq(&'.').is_some() {
                count += 1;
            }
            count == 1 && !(sanitized.is_empty() && characters.peek().is_none())
        } else {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '~')
        };

        if safe {
            sanitized.push(character);
            replacing = false;
        } else if !replacing {
            sanitized.push('-');
            replacing = true;
        }
    }

    sanitized
}

fn is_safe_atif_metadata_path(value: &str) -> bool {
    !value.is_empty()
        && value.split('/').all(|segment| {
            !matches!(segment, "" | "." | "..")
                && segment.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~')
                })
        })
}

fn atif_dispatcher_subscriber(
    manager: Arc<Mutex<AtifDispatcher>>,
    subscriber_prefix: String,
    storage: AtifStorageList,
) -> EventSubscriberFn {
    Arc::new(move |event: &Event| {
        let pending = {
            let Ok(mut guard) = manager.lock() else {
                return;
            };
            guard.observe_global(
                event,
                &subscriber_prefix,
                Arc::clone(&manager),
                Arc::clone(&storage),
            )
        };
        let Some((write, targets)) = pending else {
            return;
        };
        let results = write_atif(&write, storage.as_slice(), &targets);
        let scope_subscriber = {
            let Ok(mut guard) = manager.lock() else {
                return;
            };
            guard.complete_scope_write(write.agent_uuid, results)
        };
        if let Some((scope_uuid, name)) = scope_subscriber {
            let _ = try_scope_deregister_subscriber(&scope_uuid, &name);
        }
    })
}

fn atif_scope_subscriber(
    manager: Arc<Mutex<AtifDispatcher>>,
    agent_uuid: Uuid,
    storage: AtifStorageList,
) -> EventSubscriberFn {
    Arc::new(move |event: &Event| {
        let pending = {
            let Ok(mut guard) = manager.lock() else {
                return;
            };
            guard.observe_scope(event, agent_uuid)
        };
        let Some((write, targets)) = pending else {
            return;
        };
        let results = write_atif(&write, storage.as_slice(), &targets);
        let scope_subscriber = {
            let Ok(mut guard) = manager.lock() else {
                return;
            };
            guard.complete_scope_write(write.agent_uuid, results)
        };
        if let Some((scope_uuid, name)) = scope_subscriber {
            let _ = try_scope_deregister_subscriber(&scope_uuid, &name);
        }
    })
}

fn prepare_atif_file(
    agent_uuid: Uuid,
    agent: &mut ManagedAtifExporter,
) -> std::io::Result<PendingAtifWrite> {
    let trajectory = agent
        .exporter
        .try_export()
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let observed_events = agent.observed_events.clone();
    agent.written = true;
    prepare_atif_payload(
        agent_uuid,
        agent.filename.clone(),
        agent.local_root.clone(),
        agent.local_path.clone(),
        trajectory,
        observed_events,
        agent.correlation.clone(),
    )
}

fn prepare_atif_shutdown_file(
    export: &PendingAtifExport,
    manager: Arc<Mutex<AtifDispatcher>>,
) -> std::io::Result<PendingAtifWrite> {
    let trajectory = export
        .exporter
        .try_export()
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let observed_events = {
        let guard = manager.lock().map_err(|err| {
            std::io::Error::other(format!("ATIF dispatcher lock poisoned: {err}"))
        })?;
        guard.observed_events(export.agent_uuid)
    };
    prepare_atif_payload(
        export.agent_uuid,
        export.filename.clone(),
        export.local_root.clone(),
        export.local_path.clone(),
        trajectory,
        observed_events,
        export.correlation.clone(),
    )
}

fn prepare_atif_payload(
    agent_uuid: Uuid,
    filename: String,
    local_root: Option<PathBuf>,
    local_path: Option<PathBuf>,
    trajectory: crate::observability::atif::AtifTrajectory,
    observed_events: Vec<Event>,
    correlation: AtifCorrelation,
) -> std::io::Result<PendingAtifWrite> {
    let session_id = trajectory.session_id.clone();
    let mut value = serde_json::to_value(trajectory)?;
    if let Some(object) = value.as_object_mut() {
        let existing_extra = object.remove("extra");
        let mut extra = match existing_extra {
            Some(Json::Object(fields)) => fields,
            Some(value) => Map::from_iter([("trajectory_extra".to_string(), value)]),
            None => Map::new(),
        };
        extra.insert(
            "observed_events".to_string(),
            serde_json::to_value(observed_events)?,
        );
        let mut nemo_relay = match extra.remove("nemo_relay") {
            Some(Json::Object(fields)) => fields,
            Some(value) => Map::from_iter([("trajectory_extra".to_string(), value)]),
            None => Map::new(),
        };
        if let Json::Object(correlation_fields) = correlation.to_json() {
            nemo_relay.extend(correlation_fields);
        }
        extra.insert("nemo_relay".to_string(), Json::Object(nemo_relay));
        object.insert("extra".to_string(), Json::Object(extra));
    }
    let payload = serde_json::to_vec_pretty(&value)?;
    Ok(PendingAtifWrite {
        agent_uuid,
        session_id,
        filename,
        local_root,
        local_path,
        payload,
    })
}

fn write_atif(
    write: &PendingAtifWrite,
    storage: &[Arc<AtifRemoteStorage>],
    targets: &[SinkLabel],
) -> Vec<(SinkLabel, std::io::Result<()>)> {
    let mut results = targets
        .iter()
        .map(|label| {
            let result = match label {
                SinkLabel::Local => match (&write.local_root, &write.local_path) {
                    (Some(root), Some(path)) => write_atif_local(root, path, &write.payload),
                    _ => Err(std::io::Error::other(
                        "ATIF local destination has no output path",
                    )),
                },
                SinkLabel::Remote(index) => write_atif_remote(storage, *index, write),
            };
            (label.clone(), result)
        })
        .collect::<Vec<_>>();
    if !targets.is_empty()
        && targets
            .iter()
            .all(|label| matches!(label, SinkLabel::Remote(_)))
        && results.iter().all(|(_, result)| result.is_err())
    {
        let fallback = match (&write.local_root, &write.local_path) {
            (Some(root), Some(path)) => write_atif_local(root, path, &write.payload),
            _ => Err(std::io::Error::other(
                "ATIF local fallback has no output path",
            )),
        };
        results.push((SinkLabel::Local, fallback));
    }
    results
}

fn write_atif_local(root: &Path, path: &Path, payload: &[u8]) -> std::io::Result<()> {
    atomic_private_write(root, path, payload)
}

#[cfg(feature = "object-store")]
fn write_atif_remote(
    storage: &[Arc<AtifRemoteStorage>],
    index: usize,
    write: &PendingAtifWrite,
) -> std::io::Result<()> {
    let sink = storage
        .get(index)
        .ok_or_else(|| std::io::Error::other(format!("ATIF storage[{index}] is not registered")))?;
    sink.put(&write.filename, &write.session_id, &write.payload)
}

#[cfg(not(feature = "object-store"))]
fn write_atif_remote(
    _storage: &[Arc<AtifRemoteStorage>],
    _index: usize,
    _write: &PendingAtifWrite,
) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "ATIF storage support is not enabled in this build",
    ))
}

fn event_observation_key(event: &Event) -> String {
    format!(
        "{}:{}:{:?}",
        event.kind(),
        event.uuid(),
        event.scope_category()
    )
}

fn is_top_level_trajectory_start(event: &Event) -> bool {
    if event.scope_category() != Some(ScopeCategory::Start) {
        return false;
    }
    let is_agent_scope = event.scope_type() == Some(ScopeType::Agent);
    let is_turn_scope = event.scope_type() == Some(ScopeType::Custom)
        && event
            .metadata()
            .and_then(|metadata| metadata.get("nemo_relay_scope_role"))
            .and_then(Json::as_str)
            == Some("turn");
    if !is_agent_scope && !is_turn_scope {
        return false;
    }
    let Some(parent_uuid) = event.parent_uuid() else {
        return false;
    };
    current_scope_stack()
        .read()
        .map(|stack| stack.root_uuid() == parent_uuid || stack.is_propagated_parent(parent_uuid))
        .unwrap_or(false)
}

fn build_otel_config(
    index: usize,
    section: OpenTelemetryEndpointConfig,
) -> PluginResult<CoreOpenTelemetryConfig> {
    if section.endpoint.trim().is_empty() {
        return Err(PluginError::InvalidConfig(
            "OpenTelemetry endpoint must be a nonblank string".to_string(),
        ));
    }
    let transport = match section.transport.as_str() {
        "http_binary" => OtlpTransport::HttpBinary,
        "grpc" => OtlpTransport::Grpc,
        other => {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry transport must be 'http_binary' or 'grpc', got {other:?}"
            )));
        }
    };
    validate_otel_header_env(index, &section)?;
    validate_otel_batch_config(index, &section)?;
    let mut config = CoreOpenTelemetryConfig::new(section.otel_type, section.endpoint)
        .with_transport(transport)
        .with_service_name(section.service_name)
        .with_timeout(Duration::from_millis(section.timeout_millis))
        .with_instrumentation_scope(section.instrumentation_scope)
        .with_mark_projection(section.mark_projection)
        .with_mark_exclude_names(section.mark_exclude_names)
        .with_attribute_mappings(section.attribute_mappings)
        .with_promote_metadata_prefixes(section.promote_metadata_prefixes)
        .with_promote_resource_metadata_prefixes(section.promote_resource_metadata_prefixes);
    if let Some(max_queue_size) = section.max_queue_size {
        config = config.with_max_queue_size(max_queue_size);
    }
    if let Some(max_export_batch_size) = section.max_export_batch_size {
        config = config.with_max_export_batch_size(max_export_batch_size);
    }
    if let Some(scheduled_delay_millis) = section.scheduled_delay_millis {
        config = config.with_scheduled_delay(Duration::from_millis(scheduled_delay_millis));
    }
    if let Some(completed_span_context_ttl_millis) = section.completed_span_context_ttl_millis {
        config = config.with_completed_span_context_ttl(Duration::from_millis(
            completed_span_context_ttl_millis,
        ));
    }
    if let Some(namespace) = section.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = section.service_version {
        config = config.with_service_version(version);
    }
    for (key, value) in section.headers {
        config = config.with_header(key, value);
    }
    config = apply_otel_environment_headers(config, index, section.header_env)?;
    for (key, value) in section.resource_attributes {
        config = config.with_resource_attribute(key, value);
    }
    Ok(config)
}

fn validate_otel_batch_config(
    index: usize,
    section: &OpenTelemetryEndpointConfig,
) -> PluginResult<()> {
    for (field, value) in [
        ("max_queue_size", section.max_queue_size),
        ("max_export_batch_size", section.max_export_batch_size),
    ] {
        if value == Some(0) {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry endpoints[{index}].{field} must be greater than 0"
            )));
        }
    }
    if section.scheduled_delay_millis == Some(0) {
        return Err(PluginError::InvalidConfig(format!(
            "OpenTelemetry endpoints[{index}].scheduled_delay_millis must be greater than 0"
        )));
    }
    if section.completed_span_context_ttl_millis == Some(0) {
        return Err(PluginError::InvalidConfig(format!(
            "OpenTelemetry endpoints[{index}].completed_span_context_ttl_millis must be greater than 0"
        )));
    }
    if matches!(
        (section.max_export_batch_size, section.max_queue_size),
        (Some(batch), Some(queue)) if batch > queue
    ) {
        return Err(PluginError::InvalidConfig(format!(
            "OpenTelemetry endpoints[{index}].max_export_batch_size must be less than or equal to max_queue_size"
        )));
    }
    Ok(())
}

fn validate_otel_header_env(
    index: usize,
    section: &OpenTelemetryEndpointConfig,
) -> PluginResult<()> {
    for (header, variable) in &section.header_env {
        if variable.trim().is_empty() || variable.trim() != variable {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry endpoints[{index}].header_env.{header} must name a nonblank environment variable without surrounding whitespace"
            )));
        }
        if section
            .headers
            .keys()
            .any(|configured| configured.eq_ignore_ascii_case(header))
        {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry endpoints[{index}] header {header:?} cannot appear in both headers and header_env"
            )));
        }
    }
    Ok(())
}

fn apply_otel_environment_headers(
    mut config: CoreOpenTelemetryConfig,
    index: usize,
    header_env: HashMap<String, String>,
) -> PluginResult<CoreOpenTelemetryConfig> {
    for (key, variable) in header_env {
        let value = std::env::var(&variable).map_err(|_| {
            PluginError::InvalidConfig(format!(
                "OpenTelemetry endpoints[{index}].header_env.{key} could not read environment variable {variable:?}"
            ))
        })?;
        if value.trim().is_empty() || value.trim() != value {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry endpoints[{index}].header_env.{key} references a blank or padded environment variable {variable:?}"
            )));
        }
        config = config.with_header(key, value);
    }
    Ok(config)
}

fn parse_observability_config(
    plugin_config: &Map<String, Json>,
) -> PluginResult<ObservabilityConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid observability plugin config: {err}"))
    })
}

fn validate_observability_plugin_config(
    plugin_config: &Map<String, Json>,
) -> Vec<ConfigDiagnostic> {
    validate_observability_plugin_config_with_policy(plugin_config, None)
}

fn validate_observability_plugin_config_with_policy(
    plugin_config: &Map<String, Json>,
    policy: Option<&ConfigPolicy>,
) -> Vec<ConfigDiagnostic> {
    let mut config = match parse_observability_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "observability.invalid_plugin_config".to_string(),
                component: Some(OBSERVABILITY_PLUGIN_KIND.to_string()),
                field: None,
                message: err.to_string(),
            }];
        }
    };
    if let Some(policy) = policy {
        config.policy = apply_global_config_policy(config.policy, policy);
    }

    let mut diagnostics = vec![];
    validate_top_level_observability_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_observability_section_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_observability_section_values(&mut diagnostics, &config);

    diagnostics
}

fn validate_top_level_observability_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    validate_unknown_fields(
        diagnostics,
        policy,
        Some(OBSERVABILITY_PLUGIN_KIND.to_string()),
        plugin_config,
        &[
            "version",
            "atof",
            "atif",
            "opentelemetry",
            "openinference",
            "policy",
            "enable_full_payloads",
        ],
    );
    if plugin_config.contains_key("openinference") {
        push_policy_diag(
            diagnostics,
            UnsupportedBehavior::Error,
            "observability.legacy_openinference_section",
            Some(OBSERVABILITY_PLUGIN_KIND.to_string()),
            Some("openinference".to_string()),
            "the standalone OpenInference section was removed in observability config version 3; configure an opentelemetry endpoint with type = \"openinference\"".to_string(),
        );
    }
}

fn validate_observability_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    validate_section_fields(
        diagnostics,
        policy,
        plugin_config,
        "atof",
        &["enabled", "sinks"],
    );
    if let Some(atof) = plugin_config.get("atof").and_then(Json::as_object) {
        for legacy_field in ["output_directory", "filename", "mode", "endpoints"] {
            if atof.contains_key(legacy_field) {
                push_policy_diag(
                    diagnostics,
                    UnsupportedBehavior::Error,
                    "observability.legacy_atof_field",
                    Some("atof".to_string()),
                    Some(legacy_field.to_string()),
                    format!(
                        "ATOF {legacy_field} was removed in observability config version 2; configure typed ATOF sinks instead"
                    ),
                );
            }
        }
    }
    validate_section_fields(
        diagnostics,
        policy,
        plugin_config,
        "atif",
        &[
            "enabled",
            "agent_name",
            "agent_version",
            "model_name",
            "tool_definitions",
            "extra",
            "output_directory",
            "filename_template",
            "storage",
        ],
    );
    validate_section_fields(
        diagnostics,
        policy,
        plugin_config,
        "opentelemetry",
        &[
            "enabled",
            "traces",
            "endpoints",
            "logs",
            "metrics",
            "mark_projection",
            "mark_exclude_names",
            "attribute_mappings",
            "promote_metadata_prefixes",
            "promote_resource_metadata_prefixes",
            "transport",
            "endpoint",
            "headers",
            "resource_attributes",
            "service_name",
            "service_namespace",
            "service_version",
            "instrumentation_scope",
            "timeout_millis",
        ],
    );
    if let Some(opentelemetry) = plugin_config.get("opentelemetry").and_then(Json::as_object) {
        validate_opentelemetry_endpoint_fields(diagnostics, policy, opentelemetry);
        validate_opentelemetry_signal_fields(diagnostics, policy, opentelemetry, "logs");
        validate_opentelemetry_signal_fields(diagnostics, policy, opentelemetry, "metrics");
        for legacy_field in [
            "mark_projection",
            "mark_exclude_names",
            "attribute_mappings",
            "promote_metadata_prefixes",
            "promote_resource_metadata_prefixes",
            "transport",
            "endpoint",
            "headers",
            "resource_attributes",
            "service_name",
            "service_namespace",
            "service_version",
            "instrumentation_scope",
            "timeout_millis",
        ] {
            if opentelemetry.contains_key(legacy_field) {
                push_policy_diag(
                    diagnostics,
                    UnsupportedBehavior::Error,
                    "observability.legacy_opentelemetry_field",
                    Some("opentelemetry".to_string()),
                    Some(legacy_field.to_string()),
                    format!(
                        "OpenTelemetry {legacy_field} moved into each typed endpoint in observability config version 3"
                    ),
                );
            }
        }
    }
}

fn validate_opentelemetry_signal_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    opentelemetry: &Map<String, Json>,
    signal: &str,
) {
    const COMMON_ENDPOINT_FIELDS: &[&str] = &[
        "endpoint",
        "transport",
        "headers",
        "header_env",
        "resource_attributes",
        "service_name",
        "service_namespace",
        "service_version",
        "instrumentation_scope",
        "timeout_millis",
    ];
    let section_fields = match signal {
        "logs" => &[
            "enabled",
            "endpoints",
            "minimum_severity",
            "max_queue_size",
            "max_export_batch_size",
            "scheduled_delay_millis",
        ][..],
        "metrics" => &[
            "enabled",
            "endpoints",
            "export_interval_millis",
            "temporality",
            "max_instruments",
            "cardinality_limit",
        ][..],
        _ => return,
    };
    let Some(section) = opentelemetry.get(signal).and_then(Json::as_object) else {
        return;
    };
    validate_unknown_fields(
        diagnostics,
        policy,
        Some(format!("opentelemetry.{signal}")),
        section,
        section_fields,
    );
    let Some(endpoints) = section.get("endpoints").and_then(Json::as_array) else {
        return;
    };
    for (index, endpoint) in endpoints.iter().enumerate() {
        let Some(endpoint) = endpoint.as_object() else {
            continue;
        };
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(format!("opentelemetry.{signal}")),
            endpoint,
            COMMON_ENDPOINT_FIELDS,
        );
        if endpoint.contains_key("type") {
            push_policy_diag(
                diagnostics,
                UnsupportedBehavior::Error,
                "observability.unsupported_value",
                Some(format!("opentelemetry.{signal}")),
                Some(format!("endpoints[{index}].type")),
                format!("OpenTelemetry {signal} endpoints do not use trace projection types"),
            );
        }
    }
}

fn validate_opentelemetry_endpoint_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    opentelemetry: &Map<String, Json>,
) {
    const ALLOWED: &[&str] = &[
        "type",
        "endpoint",
        "mark_projection",
        "mark_exclude_names",
        "attribute_mappings",
        "promote_metadata_prefixes",
        "promote_resource_metadata_prefixes",
        "transport",
        "headers",
        "header_env",
        "resource_attributes",
        "service_name",
        "service_namespace",
        "service_version",
        "instrumentation_scope",
        "timeout_millis",
        "max_queue_size",
        "max_export_batch_size",
        "scheduled_delay_millis",
        "completed_span_context_ttl_millis",
    ];
    const REMOVED: &[&str] = &["semantic_selector", "capture_content"];
    let Some(endpoints) = opentelemetry
        .get("traces")
        .or_else(|| opentelemetry.get("endpoints"))
        .and_then(Json::as_array)
    else {
        return;
    };
    for (index, endpoint) in endpoints.iter().enumerate() {
        let Some(endpoint) = endpoint.as_object() else {
            continue;
        };
        for field in endpoint
            .keys()
            .filter(|field| !ALLOWED.contains(&field.as_str()))
        {
            let behavior = if REMOVED.contains(&field.as_str()) {
                UnsupportedBehavior::Error
            } else {
                policy.unknown_field
            };
            push_policy_diag(
                diagnostics,
                behavior,
                if REMOVED.contains(&field.as_str()) {
                    "observability.legacy_opentelemetry_field"
                } else {
                    "observability.unknown_field"
                },
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].{field}")),
                format!("unknown OpenTelemetry endpoint field {field:?}"),
            );
        }
    }
}

fn validate_observability_section_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    config: &ObservabilityConfig,
) {
    if let Some(section) = &config.atof {
        validate_atof_section(diagnostics, &config.policy, section);
    }
    if let Some(section) = &config.atif {
        validate_atif_section(diagnostics, &config.policy, section);
    }
    if let Some(section) = &config.opentelemetry {
        validate_opentelemetry_section(diagnostics, &config.policy, section);
        let signal_field = if section.logs.is_some() {
            Some("logs")
        } else if section.metrics.is_some() {
            Some("metrics")
        } else {
            None
        };
        if config.version == 3
            && let Some(signal_field) = signal_field
        {
            push_policy_diag(
                diagnostics,
                UnsupportedBehavior::Error,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(signal_field.to_string()),
                "observability config version 3 is trace-only; use version 4 for OpenTelemetry logs or metrics"
                    .to_string(),
            );
        }
    }
}

fn validate_atof_section(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtofSectionConfig,
) {
    validate_atof_values(diagnostics, policy, section);
    validate_atof_feature_support(diagnostics, policy, section);
}

#[cfg(not(feature = "atof-streaming"))]
fn validate_atof_feature_support(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtofSectionConfig,
) {
    if section.enabled
        && section
            .sinks
            .iter()
            .any(|sink| matches!(sink, AtofSinkSectionConfig::Stream(_)))
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some("sinks".to_string()),
            "ATOF stream sinks are not enabled in this build".to_string(),
        );
    }
}

#[cfg(feature = "atof-streaming")]
fn validate_atof_feature_support(
    _diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    _section: &AtofSectionConfig,
) {
}

fn validate_atif_section(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtifSectionConfig,
) {
    validate_atif_values(diagnostics, policy, section);
    validate_atif_file_export_support(diagnostics, policy, section);
    validate_atif_storage_support(diagnostics, policy, section);
}

fn validate_atif_file_export_support(
    _diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    _section: &AtifSectionConfig,
) {
}

#[cfg(not(feature = "object-store"))]
fn validate_atif_storage_support(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtifSectionConfig,
) {
    if section.enabled && !section.storage.is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.feature_disabled",
            Some("atif".to_string()),
            Some("storage".to_string()),
            "ATIF storage support is not enabled in this build".to_string(),
        );
    }
}

#[cfg(feature = "object-store")]
fn validate_atif_storage_support(
    _diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    _section: &AtifSectionConfig,
) {
}

fn validate_opentelemetry_section(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &OpenTelemetrySectionConfig,
) {
    let has_enabled_signal = section.logs.as_ref().is_some_and(|signal| signal.enabled)
        || section
            .metrics
            .as_ref()
            .is_some_and(|signal| signal.enabled);
    if section.enabled && section.endpoints.is_empty() && !has_enabled_signal {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("opentelemetry".to_string()),
            Some("endpoints".to_string()),
            "enabled OpenTelemetry section requires at least one endpoint or an enabled log/metric signal"
                .to_string(),
        );
    }
    for (index, endpoint) in section.endpoints.iter().enumerate() {
        if endpoint.endpoint.trim().is_empty() {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].endpoint")),
                "OpenTelemetry endpoint must be a nonblank string".to_string(),
            );
        }
        if !matches!(endpoint.transport.as_str(), "http_binary" | "grpc") {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].transport")),
                "OpenTelemetry endpoint transport must be 'http_binary' or 'grpc'".to_string(),
            );
        }
        if let Err(error) = validate_attribute_mappings(&endpoint.attribute_mappings) {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].attribute_mappings")),
                error,
            );
        }
        if let Err(error) =
            validate_metadata_promotion_prefixes(&endpoint.promote_metadata_prefixes)
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].promote_metadata_prefixes")),
                error,
            );
        }
        if let Err(error) =
            validate_metadata_promotion_prefixes(&endpoint.promote_resource_metadata_prefixes)
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(format!(
                    "endpoints[{index}].promote_resource_metadata_prefixes"
                )),
                error,
            );
        }
        validate_opentelemetry_batch_config(diagnostics, policy, index, endpoint);
        validate_opentelemetry_headers(diagnostics, policy, index, endpoint);
    }
    for error in opentelemetry_destination_collision_errors(&section.endpoints) {
        diagnostics.push(ConfigDiagnostic {
            level: DiagnosticLevel::Error,
            code: "observability.unsafe_otel_destination_collision".to_string(),
            component: Some("opentelemetry".to_string()),
            field: Some(format!("endpoints[{}].endpoint", error.index)),
            message: error.message,
        });
    }
    if let Some(logs) = &section.logs {
        validate_opentelemetry_log_section(diagnostics, policy, logs, &section.endpoints);
    }
    if let Some(metrics) = &section.metrics {
        validate_opentelemetry_metric_section(diagnostics, policy, metrics, &section.endpoints);
    }
    validate_opentelemetry_feature_support(diagnostics, policy, section);
}

fn validate_opentelemetry_log_section(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &OpenTelemetryLogSectionConfig,
    traces: &[OpenTelemetryEndpointConfig],
) {
    if section.minimum_severity.parse::<LogSeverity>().is_err() {
        push_otel_signal_diagnostic(
            diagnostics,
            policy,
            "logs",
            "minimum_severity",
            "must be trace, debug, info, warn, warning, or error",
        );
    }
    for (field, is_invalid) in [
        ("max_queue_size", section.max_queue_size == 0),
        (
            "max_export_batch_size",
            section.max_export_batch_size == 0
                || section.max_export_batch_size > section.max_queue_size,
        ),
        (
            "scheduled_delay_millis",
            section.scheduled_delay_millis == 0,
        ),
    ] {
        if is_invalid {
            push_otel_signal_diagnostic(
                diagnostics,
                policy,
                "logs",
                field,
                "must be greater than 0, and the batch size must not exceed the queue size",
            );
        }
    }
    validate_opentelemetry_signal_endpoints(
        diagnostics,
        policy,
        "logs",
        section.enabled,
        section.endpoints.as_ref(),
        traces,
    );
}

fn validate_opentelemetry_metric_section(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &OpenTelemetryMetricSectionConfig,
    traces: &[OpenTelemetryEndpointConfig],
) {
    if section.temporality.parse::<MetricTemporality>().is_err() {
        push_otel_signal_diagnostic(
            diagnostics,
            policy,
            "metrics",
            "temporality",
            "must be cumulative, delta, or low_memory",
        );
    }
    for (field, is_invalid) in [
        (
            "export_interval_millis",
            section.export_interval_millis == 0,
        ),
        ("max_instruments", section.max_instruments == 0),
        (
            "cardinality_limit",
            section.cardinality_limit == 0 || section.cardinality_limit == usize::MAX,
        ),
    ] {
        if is_invalid {
            push_otel_signal_diagnostic(
                diagnostics,
                policy,
                "metrics",
                field,
                if field == "cardinality_limit" {
                    "must be greater than 0 and less than usize::MAX"
                } else {
                    "must be greater than 0"
                },
            );
        }
    }
    validate_opentelemetry_signal_endpoints(
        diagnostics,
        policy,
        "metrics",
        section.enabled,
        section.endpoints.as_ref(),
        traces,
    );
}

fn validate_opentelemetry_signal_endpoints(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    signal: &'static str,
    enabled: bool,
    explicit: Option<&Vec<OpenTelemetrySignalEndpointConfig>>,
    traces: &[OpenTelemetryEndpointConfig],
) {
    if let Some(endpoints) = explicit {
        for (index, endpoint) in endpoints.iter().enumerate() {
            validate_opentelemetry_signal_endpoint_values(
                diagnostics,
                policy,
                signal,
                index,
                endpoint,
            );
        }
    }
    // Disabled sections do not require an endpoint, but any explicit nonempty
    // endpoint list must still satisfy the same path and collision contracts as
    // an active section.
    let validate_resolution = enabled || explicit.is_some_and(|endpoints| !endpoints.is_empty());
    if validate_resolution {
        match resolve_signal_endpoints(signal, explicit, traces) {
            Ok(resolution) => {
                for endpoint in resolution.endpoints {
                    let OpenTelemetryResource::Skipped(message) = endpoint.value else {
                        continue;
                    };
                    push_skipped_opentelemetry_endpoint_diagnostic(
                        diagnostics,
                        Some(format!("opentelemetry.{signal}")),
                        Some(format!("endpoints[{}]", endpoint.index)),
                        message,
                    );
                }
            }
            Err(error) => {
                push_otel_signal_diagnostic(
                    diagnostics,
                    policy,
                    signal,
                    "endpoints",
                    &error.to_string(),
                );
            }
        }
    }
}

fn validate_opentelemetry_signal_endpoint_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    signal: &str,
    index: usize,
    endpoint: &OpenTelemetrySignalEndpointConfig,
) {
    if endpoint.endpoint.trim().is_empty() {
        push_otel_signal_diagnostic(
            diagnostics,
            policy,
            signal,
            &format!("endpoints[{index}].endpoint"),
            "must be a nonblank string",
        );
    }
    if !matches!(endpoint.transport.as_str(), "http_binary" | "grpc") {
        push_otel_signal_diagnostic(
            diagnostics,
            policy,
            signal,
            &format!("endpoints[{index}].transport"),
            "must be 'http_binary' or 'grpc'",
        );
    }
    validate_case_insensitive_signal_header_duplicates(
        diagnostics,
        policy,
        signal,
        index,
        "headers",
        endpoint.headers.keys(),
    );
    validate_case_insensitive_signal_header_duplicates(
        diagnostics,
        policy,
        signal,
        index,
        "header_env",
        endpoint.header_env.keys(),
    );
    for (header, value) in &endpoint.headers {
        let field = format!("endpoints[{index}].headers.{header}");
        validate_opentelemetry_header_name(diagnostics, policy, &field, header);
        validate_opentelemetry_header_value(diagnostics, policy, &field, header, value);
    }
    for (header, variable) in &endpoint.header_env {
        let field = format!("endpoints[{index}].header_env.{header}");
        validate_opentelemetry_header_name(diagnostics, policy, &field, header);
        if endpoint
            .headers
            .keys()
            .any(|configured| configured.eq_ignore_ascii_case(header))
        {
            push_otel_signal_diagnostic(
                diagnostics,
                policy,
                signal,
                &field,
                "cannot also appear in headers",
            );
        }
        validate_opentelemetry_header_env(diagnostics, policy, &field, variable);
    }
}

fn validate_case_insensitive_signal_header_duplicates<'a>(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    signal: &str,
    index: usize,
    map_name: &str,
    headers: impl Iterator<Item = &'a String>,
) {
    let mut normalized = HashSet::new();
    for header in headers {
        if !normalized.insert(header.to_ascii_lowercase()) {
            push_otel_signal_diagnostic(
                diagnostics,
                policy,
                signal,
                &format!("endpoints[{index}].{map_name}.{header}"),
                "contains a duplicate header ignoring ASCII case",
            );
        }
    }
}

fn push_otel_signal_diagnostic(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    signal: &str,
    field: &str,
    message: &str,
) {
    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "observability.unsupported_value",
        Some(format!("opentelemetry.{signal}")),
        Some(field.to_string()),
        format!("OpenTelemetry {signal}.{field} {message}"),
    );
}

fn validate_opentelemetry_batch_config(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    index: usize,
    endpoint: &OpenTelemetryEndpointConfig,
) {
    for (field, is_zero) in [
        ("max_queue_size", endpoint.max_queue_size == Some(0)),
        (
            "max_export_batch_size",
            endpoint.max_export_batch_size == Some(0),
        ),
        (
            "scheduled_delay_millis",
            endpoint.scheduled_delay_millis == Some(0),
        ),
        (
            "completed_span_context_ttl_millis",
            endpoint.completed_span_context_ttl_millis == Some(0),
        ),
    ] {
        if is_zero {
            push_skipped_opentelemetry_endpoint_diagnostic(
                diagnostics,
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].{field}")),
                format!("OpenTelemetry endpoints[{index}].{field} must be greater than 0"),
            );
        }
    }
    if matches!(
        (endpoint.max_export_batch_size, endpoint.max_queue_size),
        (Some(batch), Some(queue)) if batch > queue
    ) {
        push_skipped_opentelemetry_endpoint_diagnostic(
            diagnostics,
            Some("opentelemetry".to_string()),
            Some(format!("endpoints[{index}].max_export_batch_size")),
            format!(
                "OpenTelemetry endpoints[{index}].max_export_batch_size must be less than or equal to max_queue_size"
            ),
        );
    }
}

fn push_skipped_opentelemetry_endpoint_diagnostic(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    diagnostics.push(ConfigDiagnostic {
        level: DiagnosticLevel::Warning,
        code: "observability.invalid_otel_endpoint".to_string(),
        component,
        field,
        message,
    });
}

struct OpenTelemetryDestinationCollision {
    index: usize,
    message: String,
}

#[derive(Debug, PartialEq, Eq)]
enum OpenTelemetryDestinationKey {
    Url {
        scheme: String,
        host: String,
        port: Option<u16>,
        path: String,
        query: Option<String>,
    },
    Raw(String),
}

struct OpenTelemetryDestination {
    key: OpenTelemetryDestinationKey,
    display: String,
}

fn validate_distinct_opentelemetry_destinations(
    endpoints: &[OpenTelemetryEndpointConfig],
) -> PluginResult<()> {
    if let Some(error) = opentelemetry_destination_collision_errors(endpoints)
        .into_iter()
        .next()
    {
        return Err(PluginError::InvalidConfig(error.message));
    }
    Ok(())
}

fn opentelemetry_destination_collision_errors(
    endpoints: &[OpenTelemetryEndpointConfig],
) -> Vec<OpenTelemetryDestinationCollision> {
    let mut errors = Vec::new();
    for (index, endpoint) in endpoints.iter().enumerate() {
        for (other_index, other) in endpoints[..index].iter().enumerate() {
            let endpoint_destination = opentelemetry_destination(endpoint);
            let other_destination = opentelemetry_destination(other);
            if endpoint.transport == other.transport
                && endpoint_destination.key == other_destination.key
                && endpoint.otel_type != other.otel_type
            {
                errors.push(OpenTelemetryDestinationCollision {
                    index,
                    message: format!(
                        "OpenTelemetry endpoints[{other_index}] ({}) and endpoints[{index}] ({}) use the same {} destination {:?}; different projection types must use independent destinations",
                        opentelemetry_type_name(other.otel_type),
                        opentelemetry_type_name(endpoint.otel_type),
                        endpoint.transport,
                        endpoint_destination.display,
                    ),
                });
            }
        }
    }
    errors
}

fn opentelemetry_destination(endpoint: &OpenTelemetryEndpointConfig) -> OpenTelemetryDestination {
    let configured_endpoint = endpoint.endpoint.trim();
    let effective_endpoint = if endpoint.transport == "http_binary" {
        resolve_http_trace_endpoint(configured_endpoint)
    } else {
        Cow::Borrowed(configured_endpoint)
    };
    canonicalize_opentelemetry_destination(&effective_endpoint)
}

fn canonicalize_opentelemetry_destination(endpoint: &str) -> OpenTelemetryDestination {
    let Ok(url) = reqwest::Url::parse(endpoint) else {
        return raw_opentelemetry_destination(endpoint);
    };
    if !matches!(url.scheme(), "http" | "https") {
        return raw_opentelemetry_destination(endpoint);
    }
    let Some(url_host) = url.host_str() else {
        return raw_opentelemetry_destination(endpoint);
    };

    let scheme = url.scheme().to_string();
    let host = canonical_opentelemetry_host(url_host);
    let port = url.port_or_known_default();
    let path = normalize_opentelemetry_path(url.path());
    let query = url.query().map(str::to_string);
    let display = format!(
        "{scheme}://{host}{}{path}{}",
        port.map(|port| format!(":{port}")).unwrap_or_default(),
        query
            .as_deref()
            .map(|query| format!("?{query}"))
            .unwrap_or_default(),
    );
    OpenTelemetryDestination {
        key: OpenTelemetryDestinationKey::Url {
            scheme,
            host,
            port,
            path,
            query,
        },
        display,
    }
}

fn raw_opentelemetry_destination(endpoint: &str) -> OpenTelemetryDestination {
    OpenTelemetryDestination {
        key: OpenTelemetryDestinationKey::Raw(endpoint.to_string()),
        display: endpoint.to_string(),
    }
}

fn canonical_opentelemetry_host(host: &str) -> String {
    let domain = host.strip_suffix('.').unwrap_or(host);
    let unbracketed = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    let is_loopback_domain = domain == "localhost" || domain.ends_with(".localhost");
    let is_loopback_address = unbracketed
        .parse::<IpAddr>()
        .is_ok_and(|address| address.is_loopback());
    if is_loopback_domain || is_loopback_address {
        "<loopback>".to_string()
    } else {
        host.to_string()
    }
}

fn normalize_opentelemetry_path(path: &str) -> String {
    let mut normalized = String::with_capacity(path.len());
    let mut previous_was_slash = false;
    for character in path.chars() {
        if character == '/' {
            if !previous_was_slash {
                normalized.push(character);
            }
            previous_was_slash = true;
        } else {
            normalized.push(character);
            previous_was_slash = false;
        }
    }
    while normalized.len() > 1 && normalized.ends_with('/') {
        normalized.pop();
    }
    normalized
}

const fn opentelemetry_type_name(otel_type: OpenTelemetryType) -> &'static str {
    match otel_type {
        OpenTelemetryType::Full => "full",
        OpenTelemetryType::GenAi => "gen_ai",
        OpenTelemetryType::OpenInference => "openinference",
    }
}

fn validate_opentelemetry_headers(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    endpoint: &OpenTelemetryEndpointConfig,
) {
    validate_case_insensitive_header_duplicates(
        diagnostics,
        policy,
        index,
        "headers",
        endpoint.headers.keys(),
    );
    validate_case_insensitive_header_duplicates(
        diagnostics,
        policy,
        index,
        "header_env",
        endpoint.header_env.keys(),
    );
    for (header, value) in &endpoint.headers {
        let field = format!("endpoints[{index}].headers.{header}");
        validate_opentelemetry_header_name(diagnostics, policy, &field, header);
        validate_opentelemetry_header_value(diagnostics, policy, &field, header, value);
    }
    for (header, variable) in &endpoint.header_env {
        let field = format!("endpoints[{index}].header_env.{header}");
        validate_opentelemetry_header_name(diagnostics, policy, &field, header);
        if endpoint
            .headers
            .keys()
            .any(|configured| configured.eq_ignore_ascii_case(header))
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(field.clone()),
                format!(
                    "OpenTelemetry endpoints[{index}] header {header:?} cannot appear in both headers and header_env"
                ),
            );
        }
        validate_opentelemetry_header_env(diagnostics, policy, &field, variable);
    }
}

fn validate_case_insensitive_header_duplicates<'a>(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    map_name: &str,
    headers: impl Iterator<Item = &'a String>,
) {
    let mut normalized = HashSet::new();
    for header in headers {
        if !normalized.insert(header.to_ascii_lowercase()) {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("opentelemetry".to_string()),
                Some(format!("endpoints[{index}].{map_name}.{header}")),
                format!(
                    "OpenTelemetry endpoints[{index}].{map_name} contains duplicate header {header:?} ignoring ASCII case"
                ),
            );
        }
    }
}

fn validate_opentelemetry_header_name(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    header: &str,
) {
    if header.trim().is_empty()
        || header.trim() != header
        || reqwest::header::HeaderName::from_bytes(header.as_bytes()).is_err()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("opentelemetry".to_string()),
            Some(field.to_string()),
            format!("OpenTelemetry {field} header name {header:?} is invalid"),
        );
    }
}

fn validate_opentelemetry_header_value(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    header: &str,
    value: &str,
) {
    if reqwest::header::HeaderValue::from_str(value).is_err() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("opentelemetry".to_string()),
            Some(field.to_string()),
            format!("OpenTelemetry header {header:?} has an invalid value"),
        );
    }
}

fn validate_opentelemetry_header_env(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    variable: &str,
) {
    let trimmed = variable.trim();
    let error = if trimmed.is_empty() {
        Some("must name a non-empty environment variable".to_string())
    } else if trimmed != variable {
        Some(format!(
            "must not have surrounding whitespace; got {variable:?}"
        ))
    } else {
        None
    };
    if let Some(error) = error {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("opentelemetry".to_string()),
            Some(field.to_string()),
            format!("OpenTelemetry {field} {error}"),
        );
    }
}

fn validate_opentelemetry_feature_support(
    _diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    _section: &OpenTelemetrySectionConfig,
) {
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if !matches!(version, 3 | 4) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_config_version",
            Some(OBSERVABILITY_PLUGIN_KIND.to_string()),
            Some("version".to_string()),
            format!(
                "observability config version {version} is unsupported; use version 4 (or version 3 for trace-only compatibility)"
            ),
        );
    }
}

fn validate_policy_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    if let Some(policy_json) = plugin_config.get("policy").and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some("policy".to_string()),
            policy_json,
            &["unknown_component", "unknown_field", "unsupported_value"],
        );
    }
}

fn validate_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section: &str,
    known_fields: &[&str],
) {
    if let Some(section_json) = plugin_config.get(section).and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(section.to_string()),
            section_json,
            known_fields,
        );
    }
}

fn validate_atof_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtofSectionConfig,
) {
    if section.enabled && section.sinks.is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some("sinks".to_string()),
            "ATOF requires at least one configured sink when enabled".to_string(),
        );
    }
    let mut stream_sink_names = HashSet::new();
    for (index, sink) in section.sinks.iter().enumerate() {
        validate_atof_sink(diagnostics, policy, index, sink, &mut stream_sink_names);
    }
}

fn validate_atof_sink<'a>(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    sink: &'a AtofSinkSectionConfig,
    stream_sink_names: &mut HashSet<&'a str>,
) {
    match sink {
        AtofSinkSectionConfig::File(file) => {
            if AtofExporterMode::parse(&file.mode).is_none() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "observability.unsupported_value",
                    Some("atof".to_string()),
                    Some(format!("sinks[{index}].mode")),
                    format!("ATOF sinks[{index}].mode must be 'append' or 'overwrite'"),
                );
            }
        }
        AtofSinkSectionConfig::Stream(stream) => {
            validate_atof_stream_sink_name(diagnostics, policy, index, stream, stream_sink_names);
            validate_atof_stream_sink_values(diagnostics, policy, index, stream);
        }
    }
}

fn validate_atof_stream_sink_name<'a>(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    stream: &'a AtofStreamSinkSectionConfig,
    stream_sink_names: &mut HashSet<&'a str>,
) {
    let Some(name) = stream.name.as_deref() else {
        return;
    };
    let trimmed = name.trim();
    let message = if trimmed.is_empty() {
        Some(format!("ATOF sinks[{index}].name must be non-empty"))
    } else if name != trimmed {
        Some(format!(
            "ATOF sinks[{index}].name must not have leading or trailing whitespace"
        ))
    } else if !stream_sink_names.insert(name) {
        Some(format!("ATOF stream sink name {name:?} must be unique"))
    } else {
        None
    };
    if let Some(message) = message {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("sinks[{index}].name")),
            message,
        );
    }
}

fn validate_atof_stream_sink_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    endpoint: &AtofStreamSinkSectionConfig,
) {
    let transport = AtofEndpointTransport::parse(&endpoint.transport);
    if endpoint.url.trim().is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("sinks[{index}].url")),
            format!("ATOF sinks[{index}].url must be non-empty"),
        );
    } else if transport.is_some_and(|transport| !is_valid_atof_stream_url(&endpoint.url, transport))
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("sinks[{index}].url")),
            format!(
                "ATOF sinks[{index}].url must be a valid URL for transport {:?}",
                endpoint.transport
            ),
        );
    }
    if transport.is_none() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("sinks[{index}].transport")),
            format!("ATOF sinks[{index}].transport must be 'http_post', 'websocket', or 'ndjson'"),
        );
    }
    if endpoint.timeout_millis == 0 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("sinks[{index}].timeout_millis")),
            format!("ATOF sinks[{index}].timeout_millis must be greater than 0"),
        );
    }
    if AtofEndpointFieldNamePolicy::parse(&endpoint.field_name_policy).is_none() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("sinks[{index}].field_name_policy")),
            format!("ATOF sinks[{index}].field_name_policy must be 'preserve' or 'replace_dots'"),
        );
    }
    for (header, value) in &endpoint.headers {
        validate_atof_stream_header(
            diagnostics,
            policy,
            &format!("sinks[{index}].headers.{header}"),
            header,
            value,
        );
    }
    for (header, variable) in &endpoint.header_env {
        let field = format!("sinks[{index}].header_env.{header}");
        validate_atof_stream_header_name(diagnostics, policy, &field, header);
        if endpoint
            .headers
            .keys()
            .any(|configured| configured.eq_ignore_ascii_case(header))
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("atof".to_string()),
                Some(field.clone()),
                format!(
                    "ATOF sinks[{index}] header {header:?} cannot appear in both headers and header_env"
                ),
            );
        }
        validate_atof_stream_header_env(diagnostics, policy, &field, variable);
    }
}

#[cfg(feature = "atof-streaming")]
fn is_valid_atof_stream_url(url: &str, transport: AtofEndpointTransport) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else {
        return false;
    };
    url.host_str().is_some()
        && match transport {
            AtofEndpointTransport::HttpPost | AtofEndpointTransport::Ndjson => {
                matches!(url.scheme(), "http" | "https")
            }
            AtofEndpointTransport::Websocket => matches!(url.scheme(), "ws" | "wss"),
        }
}

#[cfg(not(feature = "atof-streaming"))]
fn is_valid_atof_stream_url(url: &str, transport: AtofEndpointTransport) -> bool {
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    !rest.is_empty()
        && !rest.starts_with('/')
        && match transport {
            AtofEndpointTransport::HttpPost | AtofEndpointTransport::Ndjson => {
                matches!(scheme, "http" | "https")
            }
            AtofEndpointTransport::Websocket => matches!(scheme, "ws" | "wss"),
        }
}

fn validate_atof_stream_header(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    header: &str,
    value: &str,
) {
    validate_atof_stream_header_name(diagnostics, policy, field, header);
    #[cfg(not(feature = "atof-streaming"))]
    let _ = value;
    #[cfg(feature = "atof-streaming")]
    if let Err(error) = reqwest::header::HeaderValue::from_str(value) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(field.to_string()),
            format!("ATOF {field} value is invalid: {error}"),
        );
    }
}

fn validate_atof_stream_header_name(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    header: &str,
) {
    #[cfg(feature = "atof-streaming")]
    let is_valid = reqwest::header::HeaderName::from_bytes(header.as_bytes()).is_ok();
    #[cfg(not(feature = "atof-streaming"))]
    let is_valid = !header.trim().is_empty() && header.trim() == header;
    if !is_valid {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(field.to_string()),
            format!("ATOF {field} header name '{header}' is invalid"),
        );
    }
}

fn validate_atof_stream_header_env(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    variable: &str,
) {
    let trimmed = variable.trim();
    if trimmed.is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(field.to_string()),
            format!("ATOF {field} must name a non-empty environment variable"),
        );
    } else if trimmed != variable {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(field.to_string()),
            format!("ATOF {field} must not have surrounding whitespace; got '{variable}'"),
        );
    } else {
        match std::env::var(variable) {
            Ok(value) if value.trim().is_empty() => push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("atof".to_string()),
                Some(field.to_string()),
                format!("ATOF {field} references an environment variable that is blank"),
            ),
            Ok(_) => {}
            Err(error) => push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("atof".to_string()),
                Some(field.to_string()),
                format!("ATOF {field} references an environment variable that is not set: {error}"),
            ),
        }
    }
}

fn validate_atif_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtifSectionConfig,
) {
    if let Err(message) = validate_atif_filename_template(&section.filename_template) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some("filename_template".to_string()),
            message,
        );
    }
    for (index, storage) in section.storage.iter().enumerate() {
        validate_atif_storage_values(diagnostics, policy, index, storage);
    }
}

fn validate_atif_storage_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    storage: &AtifStorageConfig,
) {
    match storage {
        AtifStorageConfig::Http(http) => {
            validate_atif_http_endpoint(
                diagnostics,
                policy,
                &format!("storage[{index}].endpoint"),
                &http.endpoint,
            );
            if http.timeout_millis == 0 {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "observability.unsupported_value",
                    Some("atif".to_string()),
                    Some(format!("storage[{index}].timeout_millis")),
                    format!("ATIF storage[{index}].timeout_millis must be positive"),
                );
            }
            for (header, value) in &http.headers {
                validate_atif_http_header(
                    diagnostics,
                    policy,
                    &format!("storage[{index}].headers.{header}"),
                    header,
                    value,
                );
            }
            for (header, var_name) in &http.header_env {
                validate_atif_http_header_name(
                    diagnostics,
                    policy,
                    &format!("storage[{index}].header_env.{header}"),
                    header,
                );
                validate_atif_storage_env_var(
                    diagnostics,
                    policy,
                    &format!("storage[{index}].header_env.{header}"),
                    Some(var_name.as_str()),
                );
            }
        }
        AtifStorageConfig::S3(s3) => {
            if s3.bucket.trim().is_empty() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "observability.unsupported_value",
                    Some("atif".to_string()),
                    Some(format!("storage[{index}].bucket")),
                    format!("ATIF storage[{index}].bucket must be non-empty"),
                );
            }
            validate_atif_storage_env_var(
                diagnostics,
                policy,
                &format!("storage[{index}].secret_access_key_var"),
                s3.secret_access_key_var.as_deref(),
            );
            validate_atif_storage_env_var(
                diagnostics,
                policy,
                &format!("storage[{index}].session_token_var"),
                s3.session_token_var.as_deref(),
            );
        }
    }
}

fn validate_atif_http_header(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    header: &str,
    _value: &str,
) {
    validate_atif_http_header_name(diagnostics, policy, field, header);
    #[cfg(feature = "object-store")]
    if let Err(err) = reqwest::header::HeaderValue::from_str(_value) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some(field.to_string()),
            format!("ATIF {field} value is invalid: {err}"),
        );
    }
}

fn validate_atif_http_header_name(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    header: &str,
) {
    #[cfg(feature = "object-store")]
    let is_valid = reqwest::header::HeaderName::from_bytes(header.as_bytes()).is_ok();
    #[cfg(not(feature = "object-store"))]
    let is_valid = !header.trim().is_empty() && header.trim() == header;
    if !is_valid {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some(field.to_string()),
            format!("ATIF {field} header name '{header}' is invalid"),
        );
    }
}

fn validate_atif_http_endpoint(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    endpoint: &str,
) {
    let trimmed = endpoint.trim();
    let mut is_valid = !trimmed.is_empty() && trimmed == endpoint;
    #[cfg(feature = "object-store")]
    {
        is_valid = is_valid
            && reqwest::Url::parse(endpoint)
                .map(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
                .unwrap_or(false);
    }
    #[cfg(not(feature = "object-store"))]
    {
        let valid_scheme = trimmed.starts_with("http://") || trimmed.starts_with("https://");
        let has_host = trimmed
            .split_once("://")
            .map(|(_, rest)| !rest.is_empty() && !rest.starts_with('/'))
            .unwrap_or(false);
        is_valid = is_valid && valid_scheme && has_host;
    }
    if !is_valid {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some(field.to_string()),
            format!("ATIF {field} must be a valid http:// or https:// URL"),
        );
    }
}

fn validate_atif_storage_env_var(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    field: &str,
    var_name: Option<&str>,
) {
    let Some(var_name) = var_name else {
        return;
    };
    let trimmed = var_name.trim();
    if trimmed.is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some(field.to_string()),
            format!("ATIF {field} must be the name of an environment variable, not empty"),
        );
        return;
    }
    if trimmed != var_name {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some(field.to_string()),
            format!("ATIF {field} must not have surrounding whitespace; got '{var_name}'"),
        );
        return;
    }
    match std::env::var(var_name) {
        Ok(value) if !value.is_empty() => {}
        Ok(_) => {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("atif".to_string()),
                Some(field.to_string()),
                format!(
                    "ATIF {field}='{var_name}' references an environment variable that is set but empty"
                ),
            );
        }
        Err(_) => {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "observability.unsupported_value",
                Some("atif".to_string()),
                Some(field.to_string()),
                format!(
                    "ATIF {field}='{var_name}' references an environment variable that is not set"
                ),
            );
        }
    }
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    config: &Map<String, Json>,
    known_fields: &[&str],
) {
    for field in config.keys() {
        if !known_fields.contains(&field.as_str()) {
            push_policy_diag(
                diagnostics,
                policy.unknown_field,
                "observability.unknown_field",
                component.clone(),
                Some(field.clone()),
                format!(
                    "field '{}' is not recognized for '{}'",
                    field,
                    component.as_deref().unwrap_or("unknown")
                ),
            );
        }
    }
}

fn push_policy_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };
    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component,
        field,
        message,
    });
}

fn observability_registration_error(error: impl std::fmt::Display) -> PluginError {
    PluginError::RegistrationFailed(error.to_string())
}

fn default_observability_config_version() -> u32 {
    4
}

fn default_atof_mode() -> String {
    "append".to_string()
}

fn default_atof_endpoint_transport() -> String {
    AtofEndpointTransport::default().as_str().to_string()
}

fn default_atof_endpoint_field_name_policy() -> String {
    AtofEndpointFieldNamePolicy::default().as_str().to_string()
}

fn default_agent_name() -> String {
    "NeMo Relay".to_string()
}

fn default_agent_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn default_model_name() -> String {
    "unknown".to_string()
}

fn default_atif_filename_template() -> String {
    "nemo-relay-atif-{session_id}.json".to_string()
}

fn default_otlp_transport() -> String {
    "http_binary".to_string()
}

fn default_otel_service_name() -> String {
    "unknown_service".to_string()
}

fn default_otel_instrumentation_scope() -> String {
    "opentelemetry".to_string()
}

fn default_timeout_millis() -> u64 {
    3_000
}

fn default_otel_log_minimum_severity() -> String {
    "info".to_string()
}

fn default_otel_log_max_queue_size() -> usize {
    2_048
}

fn default_otel_log_max_export_batch_size() -> usize {
    512
}

fn default_otel_log_scheduled_delay_millis() -> u64 {
    1_000
}

fn default_otel_metric_export_interval_millis() -> u64 {
    60_000
}

fn default_otel_metric_temporality() -> String {
    "cumulative".to_string()
}

fn default_otel_metric_max_instruments() -> usize {
    256
}

fn default_otel_metric_cardinality_limit() -> usize {
    2_000
}

fn default_output_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(not(feature = "object-store"))]
struct AtifRemoteStorage;

/// Remote storage handle for ATIF trajectory uploads.
///
/// The handle owns a dedicated OS thread that runs a single-threaded tokio
/// runtime. Subscriber callbacks (which run on the runtime that emitted the
/// event) submit uploads over a synchronous channel and block on the reply, so
/// the handle stays safe to drive from any thread regardless of whether the
/// caller is already inside another tokio runtime.
#[cfg(feature = "object-store")]
struct AtifRemoteStorage {
    sender: std::sync::mpsc::Sender<AtifUploadRequest>,
    key_prefix: String,
    index: usize,
    resource_kind: &'static str,
    access_state: AtomicU8,
}

#[cfg(feature = "object-store")]
struct AtifUploadRequest {
    key: String,
    filename: String,
    session_id: String,
    payload: Vec<u8>,
    reply: std::sync::mpsc::Sender<std::io::Result<()>>,
}

#[cfg(feature = "object-store")]
#[derive(Clone)]
struct HttpUploadConfig {
    endpoint: String,
    headers: HashMap<String, String>,
    timeout: Duration,
}

#[cfg(feature = "object-store")]
#[derive(Default)]
struct S3BuilderOverrides {
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    region: Option<String>,
    endpoint_url: Option<String>,
    allow_http: Option<bool>,
}

#[cfg(feature = "object-store")]
impl S3BuilderOverrides {
    fn resolve(index: usize, s3: &S3StorageConfig) -> std::io::Result<Self> {
        Ok(Self {
            access_key_id: s3.access_key_id.clone(),
            secret_access_key: resolve_env_var_field(
                &format!("storage[{index}].secret_access_key_var"),
                s3.secret_access_key_var.as_deref(),
            )?,
            session_token: resolve_env_var_field(
                &format!("storage[{index}].session_token_var"),
                s3.session_token_var.as_deref(),
            )?,
            region: s3.region.clone(),
            endpoint_url: s3.endpoint_url.clone(),
            allow_http: s3.allow_http,
        })
    }

    fn apply(
        self,
        mut builder: object_store::aws::AmazonS3Builder,
    ) -> object_store::aws::AmazonS3Builder {
        if let Some(value) = self.access_key_id {
            builder = builder.with_access_key_id(value);
        }
        if let Some(value) = self.secret_access_key {
            builder = builder.with_secret_access_key(value);
        }
        if let Some(value) = self.session_token {
            builder = builder.with_token(value);
        }
        if let Some(value) = self.region {
            builder = builder.with_region(value);
        }
        if let Some(value) = self.endpoint_url {
            builder = builder.with_endpoint(value);
        }
        if let Some(value) = self.allow_http {
            builder = builder.with_allow_http(value);
        }
        builder
    }
}

#[cfg(feature = "object-store")]
fn resolve_env_var_field(field: &str, var_name: Option<&str>) -> std::io::Result<Option<String>> {
    let Some(var_name) = var_name else {
        return Ok(None);
    };
    if var_name.trim().is_empty() || var_name.trim() != var_name {
        return Err(std::io::Error::other(format!(
            "ATIF {field} must be the name of an environment variable, not '{var_name}'"
        )));
    }
    match std::env::var(var_name) {
        Ok(value) if !value.is_empty() => Ok(Some(value)),
        Ok(_) => Err(std::io::Error::other(format!(
            "ATIF {field}='{var_name}' references an environment variable that is set but empty"
        ))),
        Err(_) => Err(std::io::Error::other(format!(
            "ATIF {field}='{var_name}' references an environment variable that is not set"
        ))),
    }
}

#[cfg(feature = "object-store")]
impl AtifRemoteStorage {
    fn from_config(index: usize, config: &AtifStorageConfig) -> std::io::Result<Self> {
        match config {
            AtifStorageConfig::Http(http) => Self::build_http(index, http),
            AtifStorageConfig::S3(s3) => Self::build_s3(index, s3),
        }
    }

    fn build_http(index: usize, http: &HttpStorageConfig) -> std::io::Result<Self> {
        let upload_config = HttpUploadConfig::resolve(index, http)?;
        let (req_tx, req_rx) = std::sync::mpsc::channel::<AtifUploadRequest>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<std::io::Result<()>>();

        std::thread::Builder::new()
            .name("nemo-relay-atif-storage".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(err) => {
                        let _ = ready_tx.send(Err(std::io::Error::other(format!(
                            "failed to build ATIF storage runtime: {err}"
                        ))));
                        return;
                    }
                };
                let client = match reqwest::Client::builder()
                    .timeout(upload_config.timeout)
                    .build()
                {
                    Ok(client) => client,
                    Err(err) => {
                        let _ = ready_tx.send(Err(std::io::Error::other(format!(
                            "failed to build HTTP client for ATIF storage[{}]: {err}",
                            index
                        ))));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                drop(ready_tx);

                while let Ok(request) = req_rx.recv() {
                    let result = runtime.block_on(post_atif_http(
                        &client,
                        &upload_config,
                        request.filename,
                        request.session_id,
                        request.payload,
                    ));
                    let _ = request.reply.send(result);
                }
            })
            .map_err(|err| {
                std::io::Error::other(format!("failed to spawn ATIF storage thread: {err}"))
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: req_tx,
                key_prefix: String::new(),
                index,
                resource_kind: "http_endpoint",
                access_state: AtomicU8::new(0),
            }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err(std::io::Error::other(
                "ATIF storage thread exited before signalling readiness",
            )),
        }
    }

    fn build_s3(index: usize, s3: &S3StorageConfig) -> std::io::Result<Self> {
        let bucket = s3.bucket.clone();
        let key_prefix = normalize_storage_key_prefix(s3.key_prefix.as_deref());
        let overrides = S3BuilderOverrides::resolve(index, s3)?;

        let (req_tx, req_rx) = std::sync::mpsc::channel::<AtifUploadRequest>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<std::io::Result<()>>();

        std::thread::Builder::new()
            .name("nemo-relay-atif-storage".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(err) => {
                        let _ = ready_tx.send(Err(std::io::Error::other(format!(
                            "failed to build ATIF storage runtime: {err}"
                        ))));
                        return;
                    }
                };
                let store = match overrides
                    .apply(object_store::aws::AmazonS3Builder::from_env())
                    .with_bucket_name(&bucket)
                    .build()
                {
                    Ok(store) => Arc::new(store) as Arc<dyn object_store::ObjectStore>,
                    Err(err) => {
                        let _ = ready_tx.send(Err(std::io::Error::other(format!(
                            "failed to build S3 client for bucket '{bucket}': {err}"
                        ))));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                drop(ready_tx);

                while let Ok(request) = req_rx.recv() {
                    let result = runtime.block_on(async {
                        use object_store::ObjectStoreExt as _;
                        store
                            .put(
                                &object_store::path::Path::from(request.key.clone()),
                                object_store::PutPayload::from(request.payload),
                            )
                            .await
                            .map(|_| ())
                            .map_err(|err| {
                                std::io::Error::other(format!(
                                    "S3 upload to '{}' failed: {err}",
                                    request.key
                                ))
                            })
                    });
                    let _ = request.reply.send(result);
                }
            })
            .map_err(|err| {
                std::io::Error::other(format!("failed to spawn ATIF storage thread: {err}"))
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: req_tx,
                key_prefix,
                index,
                resource_kind: "s3_bucket",
                access_state: AtomicU8::new(0),
            }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err(std::io::Error::other(
                "ATIF storage thread exited before signalling readiness",
            )),
        }
    }

    fn put(&self, filename: &str, session_id: &str, payload: &[u8]) -> std::io::Result<()> {
        let key = format!("{}{}", self.key_prefix, filename);
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.sender
            .send(AtifUploadRequest {
                key,
                filename: filename.to_string(),
                session_id: session_id.to_string(),
                payload: payload.to_vec(),
                reply: reply_tx,
            })
            .map_err(|_| std::io::Error::other("ATIF storage thread is not running"))?;
        let (result, failure_reason) = match reply_rx.recv() {
            Ok(result) => (result, "upload_failed"),
            Err(_) => (
                Err(std::io::Error::other(
                    "ATIF storage thread dropped the upload reply",
                )),
                "reply_channel_closed",
            ),
        };
        match &result {
            Ok(()) => {
                if self.access_state.swap(2, Ordering::AcqRel) != 2 {
                    log::info!(
                        target: "nemo_relay.observability",
                        event = "storage_access_validated",
                        plugin_kind = "observability",
                        exporter = "atif",
                        resource_index = self.index,
                        resource_kind = self.resource_kind,
                        permission = "write";
                        "ATIF storage access validated"
                    );
                }
            }
            Err(_) => {
                if self.access_state.swap(1, Ordering::AcqRel) != 1 {
                    log::warn!(
                        target: "nemo_relay.observability",
                        event = "storage_access_failed",
                        plugin_kind = "observability",
                        exporter = "atif",
                        resource_index = self.index,
                        resource_kind = self.resource_kind,
                        permission = "write",
                        reason = failure_reason;
                        "ATIF storage access failed"
                    );
                }
            }
        }
        result
    }
}

#[cfg(feature = "object-store")]
impl HttpUploadConfig {
    fn resolve(index: usize, http: &HttpStorageConfig) -> std::io::Result<Self> {
        let endpoint = http.endpoint.trim();
        if endpoint.is_empty() || endpoint != http.endpoint {
            return Err(std::io::Error::other(format!(
                "ATIF storage[{index}].endpoint must be non-empty and must not have surrounding whitespace"
            )));
        }
        let parsed = reqwest::Url::parse(endpoint).map_err(|err| {
            std::io::Error::other(format!(
                "ATIF storage[{index}].endpoint must be a valid URL: {err}"
            ))
        })?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(std::io::Error::other(format!(
                "ATIF storage[{index}].endpoint must be a valid http:// or https:// URL"
            )));
        }
        if http.timeout_millis == 0 {
            return Err(std::io::Error::other(format!(
                "ATIF storage[{index}].timeout_millis must be positive"
            )));
        }

        let mut headers = http.headers.clone();
        for (header, var_name) in &http.header_env {
            let value = resolve_env_var_field(
                &format!("storage[{index}].header_env.{header}"),
                Some(var_name.as_str()),
            )?
            .expect("resolve_env_var_field returns Some when var_name is Some");
            headers.insert(header.clone(), value);
        }
        validate_http_headers(index, &headers)?;

        Ok(Self {
            endpoint: parsed.to_string(),
            headers,
            timeout: Duration::from_millis(http.timeout_millis),
        })
    }
}

#[cfg(feature = "object-store")]
fn validate_http_headers(index: usize, headers: &HashMap<String, String>) -> std::io::Result<()> {
    for (header, value) in headers {
        reqwest::header::HeaderName::from_bytes(header.as_bytes()).map_err(|err| {
            std::io::Error::other(format!(
                "ATIF storage[{index}] header name '{header}' is invalid: {err}"
            ))
        })?;
        reqwest::header::HeaderValue::from_str(value).map_err(|err| {
            std::io::Error::other(format!(
                "ATIF storage[{index}] value for header '{header}' is invalid: {err}"
            ))
        })?;
    }
    Ok(())
}

#[cfg(feature = "object-store")]
async fn post_atif_http(
    client: &reqwest::Client,
    config: &HttpUploadConfig,
    filename: String,
    session_id: String,
    payload: Vec<u8>,
) -> std::io::Result<()> {
    let mut request = client.post(&config.endpoint);
    for (header, value) in &config.headers {
        request = request.header(header.as_str(), value.as_str());
    }
    let response = request
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header("x-nemo-relay-atif-filename", filename.clone())
        .header("x-nemo-relay-atif-session-id", session_id)
        .body(payload)
        .send()
        .await
        .map_err(|err| {
            std::io::Error::other(format!(
                "HTTP ATIF upload to '{}' failed: {err}",
                config.endpoint
            ))
        })?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "HTTP ATIF upload to '{}' for '{}' failed with status {}",
            config.endpoint,
            filename,
            response.status()
        )))
    }
}

#[cfg(feature = "object-store")]
fn normalize_storage_key_prefix(raw: Option<&str>) -> String {
    let trimmed = raw.unwrap_or("").trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.ends_with('/') {
        trimmed.to_string()
    } else {
        format!("{trimmed}/")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/plugin_component_tests.rs"]
mod tests;
