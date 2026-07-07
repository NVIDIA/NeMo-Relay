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
//! Observability Format (ATOF), OpenTelemetry, and OpenInference each register
//! one global subscriber when enabled. Agent Trajectory Interchange Format
//! (ATIF) uses a global dispatcher that detects top-level agent or turn scopes
//! and creates one scope-local exporter for each trajectory run. Coding-agent
//! turns that need bounded traces carry role metadata; their declared scope
//! type is preserved in the exported event stream.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
#[cfg(any(feature = "otel", feature = "openinference", feature = "object-store"))]
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use uuid::Uuid;

use crate::api::event::{Event, ScopeCategory};
use crate::api::runtime::{EventSubscriberFn, current_scope_stack};
use crate::api::scope::ScopeType;
use crate::api::subscriber::{
    scope_deregister_subscriber, try_scope_deregister_subscriber, try_scope_register_subscriber,
};
use crate::error::FlowError;
use crate::observability::atif::{AtifAgentInfo, AtifExporter};
use crate::observability::atof::{
    AtofEndpointConfig as CoreAtofEndpointConfig, AtofEndpointFieldNamePolicy,
    AtofEndpointTransport, AtofExporter, AtofExporterConfig as CoreAtofExporterConfig,
    AtofExporterMode,
};
#[cfg(feature = "openinference")]
use crate::observability::openinference::{
    OpenInferenceConfig as CoreOpenInferenceConfig, OpenInferenceSubscriber,
    OtlpTransport as OpenInferenceTransport,
};
#[cfg(feature = "otel")]
use crate::observability::otel::{
    OpenTelemetryConfig as CoreOpenTelemetryConfig, OpenTelemetrySubscriber,
};
use crate::observability::{MarkProjection, default_mark_exclude_names};
use crate::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistration, PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior,
    deregister_plugin, register_plugin,
};

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
    pub opentelemetry: Option<OtlpSectionConfig>,
    /// OpenInference trace export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openinference: Option<OtlpSectionConfig>,
    /// Observability-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            version: default_observability_config_version(),
            atof: None,
            atif: None,
            opentelemetry: None,
            openinference: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// Filesystem-backed ATOF JSONL exporter config.
///
/// When enabled, this section wraps
/// [`crate::observability::atof::AtofExporter`] and writes the raw ATOF event
/// stream as JSONL. The exporter uses the current working directory and a
/// timestamped filename when no explicit path settings are supplied.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AtofSectionConfig {
    /// Whether ATOF JSONL export is active.
    #[serde(default)]
    pub enabled: bool,
    /// Directory containing the JSONL output file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    /// Output filename. Defaults to the underlying ATOF exporter timestamped filename.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// File open mode: `append` or `overwrite`.
    #[serde(default = "default_atof_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "atof_mode_schema"))]
    pub mode: String,
    /// Optional streaming endpoints that receive every raw ATOF event.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub endpoints: Vec<AtofEndpointSectionConfig>,
}

impl Default for AtofSectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_directory: None,
            filename: None,
            mode: default_atof_mode(),
            endpoints: Vec::new(),
        }
    }
}

/// Streaming destination for raw ATOF events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AtofEndpointSectionConfig {
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
    /// Per-endpoint timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
    /// Field name policy applied before sending events: `preserve` or `replace_dots`.
    #[serde(default = "default_atof_endpoint_field_name_policy")]
    pub field_name_policy: String,
}

/// Per-trajectory ATIF exporter config.
///
/// When enabled, this section creates a dispatcher that opens a separate
/// [`crate::observability::atif::AtifExporter`] for each top-level agent or turn scope. The
/// `{session_id}` placeholder in [`AtifSectionConfig::filename_template`] is required so
/// concurrent sibling trajectories cannot overwrite each other's files.
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
    /// Representation used for mark events: `event` or `tool`.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mark_projection_schema"))]
    pub mark_projection: MarkProjection,
    /// Mark names excluded from tool projection. Defaults to `llm.chunk`.
    #[serde(default = "default_mark_exclude_names")]
    pub mark_exclude_names: Vec<String>,
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
    /// Filename template. `{session_id}` is replaced with the top-level trajectory scope UUID.
    /// When [`storage`] is non-empty, the rendered filename is appended to each backend's key prefix.
    ///
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
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
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

/// Shared OTLP exporter config for OpenTelemetry and OpenInference.
///
/// The `opentelemetry` and `openinference` sections share the same shape but
/// construct different subscriber implementations. Both sections are disabled
/// by default and use `http_binary` transport unless configured otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OtlpSectionConfig {
    /// Whether the subscriber is active.
    #[serde(default)]
    pub enabled: bool,
    /// Representation used for mark events: `event` or `tool`.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mark_projection_schema"))]
    pub mark_projection: MarkProjection,
    /// Mark names excluded from tool projection. Defaults to `llm.chunk`.
    #[serde(default = "default_mark_exclude_names")]
    pub mark_exclude_names: Vec<String>,
    /// OTLP transport: `http_binary` or `grpc`.
    #[serde(default = "default_otlp_transport")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "otlp_transport_schema"))]
    pub transport: String,
    /// OTLP endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Extra exporter headers or metadata.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Extra resource attributes.
    #[serde(default)]
    pub resource_attributes: HashMap<String, String>,
    /// `service.name` resource attribute.
    #[serde(default = "default_service_name")]
    pub service_name: String,
    /// Optional `service.namespace` resource attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_namespace: Option<String>,
    /// Optional `service.version` resource attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_version: Option<String>,
    /// Instrumentation scope name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instrumentation_scope: Option<String>,
    /// Export timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
}

impl Default for OtlpSectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mark_projection: MarkProjection::default(),
            mark_exclude_names: default_mark_exclude_names(),
            transport: default_otlp_transport(),
            endpoint: None,
            headers: HashMap::new(),
            resource_attributes: HashMap::new(),
            service_name: default_service_name(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: None,
            timeout_millis: default_timeout_millis(),
        }
    }
}

crate::editor_config! {
    impl ObservabilityConfig {
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
            nested: OtlpSectionConfig,
            default: OtlpSectionConfig,
        },
        openinference => {
            label: "OpenInference",
            kind: Section,
            optional: true,
            nested: OtlpSectionConfig,
            default: OtlpSectionConfig,
        },
        policy => {
            label: "policy",
            kind: Section,
            nested: ConfigPolicy,
            default: ConfigPolicy,
        },
    }
}

crate::editor_config! {
    impl AtofSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        output_directory => { label: "output_directory", kind: String, optional: true },
        filename => { label: "filename", kind: String, optional: true },
        mode => { label: "mode", kind: Enum, values: ["append", "overwrite"] },
        endpoints => { label: "endpoints", kind: Json, optional: true },
    }
}

crate::editor_config! {
    impl AtifSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        agent_name => { label: "agent_name", kind: String },
        agent_version => { label: "agent_version", kind: String },
        model_name => { label: "model_name", kind: String },
        mark_projection => { label: "mark_projection", kind: Enum, values: ["event", "tool"] },
        mark_exclude_names => { label: "mark_exclude_names", kind: Json },
        tool_definitions => { label: "tool_definitions", kind: Json, optional: true },
        extra => { label: "extra", kind: Json, optional: true },
        output_directory => { label: "output_directory", kind: String, optional: true },
        filename_template => { label: "filename_template", kind: String },
        storage => { label: "storage", kind: Json, optional: true },
    }
}

crate::editor_config! {
    impl OtlpSectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        mark_projection => { label: "mark_projection", kind: Enum, values: ["event", "tool"] },
        mark_exclude_names => { label: "mark_exclude_names", kind: Json },
        transport => { label: "transport", kind: Enum, values: ["http_binary", "grpc"] },
        endpoint => { label: "endpoint", kind: String, optional: true },
        headers => { label: "headers", kind: StringMap },
        resource_attributes => { label: "resource_attributes", kind: StringMap },
        service_name => { label: "service_name", kind: String },
        service_namespace => { label: "service_namespace", kind: String, optional: true },
        service_version => { label: "service_version", kind: String, optional: true },
        instrumentation_scope => { label: "instrumentation_scope", kind: String, optional: true },
        timeout_millis => { label: "timeout_millis", kind: Integer },
    }
}

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
    match register_plugin(Arc::new(ObservabilityPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message)) if message.contains("already registered") => {
            Ok(())
        }
        Err(err) => Err(err),
    }
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
    string_enum_schema(generator, &["event", "tool"], Some("event"))
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
    if let Some(atof) = config.atof.filter(|section| section.enabled) {
        register_atof_exporter(atof, ctx)?;
    }
    if let Some(atif) = config.atif.filter(|section| section.enabled) {
        register_atif_dispatcher(atif, ctx)?;
    }
    if let Some(otel) = config.opentelemetry.filter(|section| section.enabled) {
        register_opentelemetry(otel, ctx)?;
    }
    if let Some(openinference) = config.openinference.filter(|section| section.enabled) {
        register_openinference(openinference, ctx)?;
    }
    Ok(())
}

fn register_atof_exporter(
    section: AtofSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let mode = AtofExporterMode::parse(&section.mode).ok_or_else(|| {
        PluginError::InvalidConfig("ATOF mode must be 'append' or 'overwrite'".to_string())
    })?;
    let mut config = CoreAtofExporterConfig::new().with_mode(mode);
    if let Some(output_directory) = section.output_directory {
        config = config.with_output_directory(output_directory);
    }
    if let Some(filename) = section.filename {
        config = config.with_filename(filename);
    }
    let endpoints = section
        .endpoints
        .into_iter()
        .enumerate()
        .map(|(index, endpoint)| build_atof_endpoint_config(index, endpoint))
        .collect::<PluginResult<Vec<_>>>()?;
    config = config.with_endpoints(endpoints);

    let exporter = Arc::new(AtofExporter::new(config).map_err(observability_registration_error)?);
    ctx.register_subscriber("atof", exporter.subscriber())?;
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("atof.shutdown"),
        Box::new(move || {
            exporter
                .shutdown()
                .map_err(observability_registration_error)
        }),
    ));
    Ok(())
}

fn build_atof_endpoint_config(
    index: usize,
    endpoint: AtofEndpointSectionConfig,
) -> PluginResult<CoreAtofEndpointConfig> {
    let transport = AtofEndpointTransport::parse(&endpoint.transport).ok_or_else(|| {
        PluginError::InvalidConfig(format!(
            "ATOF endpoints[{index}].transport must be 'http_post', 'websocket', or 'ndjson'"
        ))
    })?;
    let field_name_policy = AtofEndpointFieldNamePolicy::parse(&endpoint.field_name_policy)
        .ok_or_else(|| {
            PluginError::InvalidConfig(format!(
                "ATOF endpoints[{index}].field_name_policy must be 'preserve' or 'replace_dots'"
            ))
        })?;
    let mut config = CoreAtofEndpointConfig::new(endpoint.url, transport)
        .with_timeout_millis(endpoint.timeout_millis)
        .with_field_name_policy(field_name_policy);
    for (key, value) in endpoint.headers {
        config = config.with_header(key, value);
    }
    Ok(config)
}

type AtifStorageList = Arc<Vec<Arc<AtifRemoteStorage>>>;

fn register_atif_dispatcher(
    section: AtifSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    if !section.filename_template.contains("{session_id}") {
        return Err(PluginError::InvalidConfig(
            "ATIF filename_template must contain '{session_id}'".to_string(),
        ));
    }

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
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("atif.shutdown"),
        Box::new(move || {
            let work = {
                let mut guard = manager.lock().map_err(|err| {
                    PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
                })?;
                guard.flush_open_agents()
            };
            for (scope_uuid, name) in work.scope_subscribers {
                deregister_atif_shutdown_subscriber(&scope_uuid, &name)?;
            }
            for export in work.exports {
                let write = prepare_atif_shutdown_file(&export, Arc::clone(&manager))
                    .map_err(observability_registration_error)?;
                let agent_uuid = write.agent_uuid;
                let targets = {
                    let guard = manager.lock().map_err(|err| {
                        PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
                    })?;
                    guard.sink_targets()
                };
                let results = write_atif(&write, shutdown_storage.as_slice(), &targets);
                let mut guard = manager.lock().map_err(|err| {
                    PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
                })?;
                let _ = guard.complete_scope_write(agent_uuid, results);
            }
            let guard = manager.lock().map_err(|err| {
                PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
            })?;
            guard
                .last_error_result()
                .map_err(observability_registration_error)
        }),
    ));
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
    AtifRemoteStorage::from_config(index, config)
        .map(Arc::new)
        .map_err(observability_registration_error)
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

#[cfg(feature = "otel")]
fn register_opentelemetry(
    section: OtlpSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let subscriber = Arc::new(
        OpenTelemetrySubscriber::new(build_otel_config(section)?)
            .map_err(observability_registration_error)?,
    );
    ctx.register_subscriber("opentelemetry", subscriber.subscriber())?;
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("opentelemetry.shutdown"),
        Box::new(move || {
            subscriber
                .shutdown()
                .map_err(observability_registration_error)
        }),
    ));
    Ok(())
}

#[cfg(not(feature = "otel"))]
fn register_opentelemetry(
    _section: OtlpSectionConfig,
    _ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Err(PluginError::InvalidConfig(
        "OpenTelemetry support is not enabled in this build".to_string(),
    ))
}

#[cfg(feature = "openinference")]
fn register_openinference(
    section: OtlpSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let subscriber = Arc::new(
        OpenInferenceSubscriber::new(build_openinference_config(section)?)
            .map_err(observability_registration_error)?,
    );
    ctx.register_subscriber("openinference", subscriber.subscriber())?;
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("openinference.shutdown"),
        Box::new(move || {
            subscriber
                .shutdown()
                .map_err(observability_registration_error)
        }),
    ));
    Ok(())
}

#[cfg(not(feature = "openinference"))]
fn register_openinference(
    _section: OtlpSectionConfig,
    _ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Err(PluginError::InvalidConfig(
        "OpenInference support is not enabled in this build".to_string(),
    ))
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
    /// Per-sink last error. A sink that recorded an error is skipped on
    /// subsequent trajectories; other sinks continue to receive writes.
    sink_errors: HashMap<SinkLabel, String>,
}

struct ManagedAtifExporter {
    exporter: AtifExporter,
    filename: String,
    local_path: Option<PathBuf>,
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
    local_path: Option<PathBuf>,
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
            sink_errors: HashMap::new(),
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

        // The top-level trajectory scope UUID is the ATIF session ID. The global
        // dispatcher records the start event itself because the scope-local
        // subscriber is attached after that start event has already been
        // emitted.
        let session_id = event.uuid().to_string();
        let exporter = AtifExporter::new(session_id.clone(), self.agent_info())
            .with_mark_projection(self.config.mark_projection)
            .with_mark_exclude_names(self.config.mark_exclude_names.clone());
        (exporter.subscriber())(event);
        let (filename, local_path) = self.prepare_destination(&session_id);
        self.scope_owners.insert(event.uuid(), event.uuid());
        self.agents.insert(
            event.uuid(),
            ManagedAtifExporter {
                exporter,
                filename,
                local_path,
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
        for (label, result) in results {
            if let Err(err) = result {
                self.sink_errors.insert(label, err.to_string());
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
                    local_path: agent.local_path.clone(),
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

    fn prepare_destination(&self, session_id: &str) -> (String, Option<PathBuf>) {
        let filename = self
            .config
            .filename_template
            .replace("{session_id}", session_id);
        if !self.config.storage.is_empty() {
            return (filename, None);
        }
        let directory = self
            .config
            .output_directory
            .clone()
            .unwrap_or_else(default_output_directory);
        let path = directory.join(&filename);
        (filename, Some(path))
    }

    fn sink_targets(&self) -> Vec<SinkLabel> {
        if self.config.storage.is_empty() {
            if self.sink_errors.contains_key(&SinkLabel::Local) {
                Vec::new()
            } else {
                vec![SinkLabel::Local]
            }
        } else {
            (0..self.config.storage.len())
                .map(SinkLabel::Remote)
                .filter(|label| !self.sink_errors.contains_key(label))
                .collect()
        }
    }
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
        agent.local_path.clone(),
        trajectory,
        observed_events,
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
        export.local_path.clone(),
        trajectory,
        observed_events,
    )
}

fn prepare_atif_payload(
    agent_uuid: Uuid,
    filename: String,
    local_path: Option<PathBuf>,
    trajectory: crate::observability::atif::AtifTrajectory,
    observed_events: Vec<Event>,
) -> std::io::Result<PendingAtifWrite> {
    let mut value = serde_json::to_value(trajectory)?;
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "extra".to_string(),
            serde_json::json!({
                "observed_events": observed_events,
            }),
        );
    }
    let payload = serde_json::to_vec_pretty(&value)?;
    Ok(PendingAtifWrite {
        agent_uuid,
        session_id: agent_uuid.to_string(),
        filename,
        local_path,
        payload,
    })
}

fn write_atif(
    write: &PendingAtifWrite,
    storage: &[Arc<AtifRemoteStorage>],
    targets: &[SinkLabel],
) -> Vec<(SinkLabel, std::io::Result<()>)> {
    targets
        .iter()
        .map(|label| {
            let result = match label {
                SinkLabel::Local => match &write.local_path {
                    Some(path) => write_atif_local(path, &write.payload),
                    None => Err(std::io::Error::other(
                        "ATIF local destination has no output path",
                    )),
                },
                SinkLabel::Remote(index) => write_atif_remote(storage, *index, write),
            };
            (label.clone(), result)
        })
        .collect()
}

fn write_atif_local(path: &PathBuf, payload: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, payload)
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
        .map(|stack| stack.root_uuid() == parent_uuid)
        .unwrap_or(false)
}

#[cfg(feature = "otel")]
fn build_otel_config(section: OtlpSectionConfig) -> PluginResult<CoreOpenTelemetryConfig> {
    let mut config = match section.transport.as_str() {
        "http_binary" => CoreOpenTelemetryConfig::http_binary(section.service_name),
        "grpc" => CoreOpenTelemetryConfig::grpc(section.service_name),
        other => {
            return Err(PluginError::InvalidConfig(format!(
                "OpenTelemetry transport must be 'http_binary' or 'grpc', got {other:?}"
            )));
        }
    }
    .with_timeout(Duration::from_millis(section.timeout_millis));
    config = config
        .with_mark_projection(section.mark_projection)
        .with_mark_exclude_names(section.mark_exclude_names);

    if let Some(endpoint) = section.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(namespace) = section.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = section.service_version {
        config = config.with_service_version(version);
    }
    if let Some(scope) = section.instrumentation_scope {
        config = config.with_instrumentation_scope(scope);
    }
    for (key, value) in section.headers {
        config = config.with_header(key, value);
    }
    for (key, value) in section.resource_attributes {
        config = config.with_resource_attribute(key, value);
    }
    Ok(config)
}

#[cfg(feature = "openinference")]
fn build_openinference_config(section: OtlpSectionConfig) -> PluginResult<CoreOpenInferenceConfig> {
    let transport = match section.transport.as_str() {
        "http_binary" => OpenInferenceTransport::HttpBinary,
        "grpc" => OpenInferenceTransport::Grpc,
        other => {
            return Err(PluginError::InvalidConfig(format!(
                "OpenInference transport must be 'http_binary' or 'grpc', got {other:?}"
            )));
        }
    };
    let mut config = CoreOpenInferenceConfig::new()
        .with_transport(transport)
        .with_service_name(section.service_name)
        .with_timeout(Duration::from_millis(section.timeout_millis))
        .with_mark_projection(section.mark_projection)
        .with_mark_exclude_names(section.mark_exclude_names);

    if let Some(endpoint) = section.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(namespace) = section.service_namespace {
        config = config.with_service_namespace(namespace);
    }
    if let Some(version) = section.service_version {
        config = config.with_service_version(version);
    }
    if let Some(scope) = section.instrumentation_scope {
        config = config.with_instrumentation_scope(scope);
    }
    for (key, value) in section.headers {
        config = config.with_header(key, value);
    }
    for (key, value) in section.resource_attributes {
        config = config.with_resource_attribute(key, value);
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
    let config = match parse_observability_config(plugin_config) {
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
        ],
    );
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
        &[
            "enabled",
            "output_directory",
            "filename",
            "mode",
            "endpoints",
        ],
    );
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
            "mark_projection",
            "mark_exclude_names",
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
            "mark_projection",
            "mark_exclude_names",
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
    validate_section_fields(
        diagnostics,
        policy,
        plugin_config,
        "openinference",
        &[
            "enabled",
            "mark_projection",
            "mark_exclude_names",
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
    }
    if let Some(section) = &config.openinference {
        validate_openinference_section(diagnostics, &config.policy, section);
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
    if section.enabled && !section.endpoints.is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some("endpoints".to_string()),
            "ATOF streaming endpoints are not enabled in this build".to_string(),
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
    section: &OtlpSectionConfig,
) {
    validate_otlp_values(diagnostics, policy, "opentelemetry", section);
    validate_opentelemetry_feature_support(diagnostics, policy, section);
}

#[cfg(not(feature = "otel"))]
fn validate_opentelemetry_feature_support(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &OtlpSectionConfig,
) {
    if section.enabled {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.feature_disabled",
            Some("opentelemetry".to_string()),
            Some("enabled".to_string()),
            "OpenTelemetry support is not enabled in this build".to_string(),
        );
    }
}

#[cfg(feature = "otel")]
fn validate_opentelemetry_feature_support(
    _diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    _section: &OtlpSectionConfig,
) {
}

fn validate_openinference_section(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &OtlpSectionConfig,
) {
    validate_otlp_values(diagnostics, policy, "openinference", section);
    validate_openinference_feature_support(diagnostics, policy, section);
}

#[cfg(not(feature = "openinference"))]
fn validate_openinference_feature_support(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &OtlpSectionConfig,
) {
    if section.enabled {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.feature_disabled",
            Some("openinference".to_string()),
            Some("enabled".to_string()),
            "OpenInference support is not enabled in this build".to_string(),
        );
    }
}

#[cfg(feature = "openinference")]
fn validate_openinference_feature_support(
    _diagnostics: &mut Vec<ConfigDiagnostic>,
    _policy: &ConfigPolicy,
    _section: &OtlpSectionConfig,
) {
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if version != 1 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_config_version",
            Some(OBSERVABILITY_PLUGIN_KIND.to_string()),
            Some("version".to_string()),
            format!("observability config version {version} is unsupported"),
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
    if AtofExporterMode::parse(&section.mode).is_none() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some("mode".to_string()),
            "ATOF mode must be 'append' or 'overwrite'".to_string(),
        );
    }
    for (index, endpoint) in section.endpoints.iter().enumerate() {
        validate_atof_endpoint_values(diagnostics, policy, index, endpoint);
    }
}

fn validate_atof_endpoint_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    index: usize,
    endpoint: &AtofEndpointSectionConfig,
) {
    if endpoint.url.trim().is_empty() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("endpoints[{index}].url")),
            format!("ATOF endpoints[{index}].url must be non-empty"),
        );
    } else if !is_valid_atof_endpoint_url(&endpoint.url) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("endpoints[{index}].url")),
            format!("ATOF endpoints[{index}].url must be a valid URL"),
        );
    }
    if AtofEndpointTransport::parse(&endpoint.transport).is_none() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("endpoints[{index}].transport")),
            format!(
                "ATOF endpoints[{index}].transport must be 'http_post', 'websocket', or 'ndjson'"
            ),
        );
    }
    if endpoint.timeout_millis == 0 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("endpoints[{index}].timeout_millis")),
            format!("ATOF endpoints[{index}].timeout_millis must be greater than 0"),
        );
    }
    if AtofEndpointFieldNamePolicy::parse(&endpoint.field_name_policy).is_none() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atof".to_string()),
            Some(format!("endpoints[{index}].field_name_policy")),
            format!(
                "ATOF endpoints[{index}].field_name_policy must be 'preserve' or 'replace_dots'"
            ),
        );
    }
}

#[cfg(feature = "atof-streaming")]
fn is_valid_atof_endpoint_url(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok()
}

#[cfg(not(feature = "atof-streaming"))]
fn is_valid_atof_endpoint_url(_url: &str) -> bool {
    true
}

fn validate_atif_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section: &AtifSectionConfig,
) {
    if !section.filename_template.contains("{session_id}") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some("atif".to_string()),
            Some("filename_template".to_string()),
            "ATIF filename_template must contain '{session_id}'".to_string(),
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

fn validate_otlp_values(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    section_name: &str,
    section: &OtlpSectionConfig,
) {
    if !matches!(section.transport.as_str(), "http_binary" | "grpc") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "observability.unsupported_value",
            Some(section_name.to_string()),
            Some("transport".to_string()),
            format!("{section_name} transport must be 'http_binary' or 'grpc'"),
        );
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
    1
}

fn default_atof_mode() -> String {
    "append".to_string()
}

fn default_atof_endpoint_transport() -> String {
    "http_post".to_string()
}

fn default_atof_endpoint_field_name_policy() -> String {
    "preserve".to_string()
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

fn default_service_name() -> String {
    "nemo-relay".to_string()
}

fn default_timeout_millis() -> u64 {
    3_000
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
        reply_rx
            .recv()
            .map_err(|_| std::io::Error::other("ATIF storage thread dropped the upload reply"))?
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
