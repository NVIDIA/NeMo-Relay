// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in observability plugin component.
//!
//! This module packages NeMo Flow's first-party observability exporters behind
//! the shared plugin configuration system. Each exporter section is opt-in:
//! omitted sections and sections with `enabled = false` validate but do not
//! register subscribers or construct exporters.
//!
//! The plugin intentionally infers subscriber names from the component namespace
//! so configuration remains portable across bindings. ATOF, OpenTelemetry, and
//! OpenInference each register one global subscriber when enabled. ATIF uses a
//! global dispatcher that detects direct child agent scopes and creates one
//! scope-local exporter for each top-level agent run.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use uuid::Uuid;

use crate::api::event::{Event, ScopeCategory};
use crate::api::runtime::{EventSubscriberFn, current_scope_stack};
use crate::api::scope::ScopeType;
use crate::api::subscriber::{scope_deregister_subscriber, scope_register_subscriber};
use crate::observability::atif::{AtifAgentInfo, AtifExporter};
use crate::observability::atof::{
    AtofExporter, AtofExporterConfig as CoreAtofExporterConfig, AtofExporterMode,
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
    pub mode: String,
}

impl Default for AtofSectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_directory: None,
            filename: None,
            mode: default_atof_mode(),
        }
    }
}

/// Per-agent ATIF trajectory exporter config.
///
/// When enabled, this section creates a dispatcher that opens a separate
/// [`crate::observability::atif::AtifExporter`] for each top-level agent scope. The `{session_id}`
/// placeholder in [`AtifSectionConfig::filename_template`] is required so
/// concurrent sibling agents cannot overwrite each other's trajectory files.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Directory containing trajectory JSON files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    /// Filename template. `{session_id}` is replaced with the top-level agent scope UUID.
    #[serde(default = "default_atif_filename_template")]
    pub filename_template: String,
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
        }
    }
}

/// Shared OTLP exporter config for OpenTelemetry and OpenInference.
///
/// The `opentelemetry` and `openinference` sections share the same shape but
/// construct different subscriber implementations. Both sections are disabled
/// by default and use `http_binary` transport unless configured otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpSectionConfig {
    /// Whether the subscriber is active.
    #[serde(default)]
    pub enabled: bool,
    /// OTLP transport: `http_binary` or `grpc`.
    #[serde(default = "default_otlp_transport")]
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

fn register_atif_dispatcher(
    section: AtifSectionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    if !section.filename_template.contains("{session_id}") {
        return Err(PluginError::InvalidConfig(
            "ATIF filename_template must contain '{session_id}'".to_string(),
        ));
    }

    let manager = Arc::new(Mutex::new(AtifDispatcher::new(section)));
    let dispatcher = atif_dispatcher_subscriber(Arc::clone(&manager), ctx.qualify_name("atif-"));
    ctx.register_subscriber("atif", dispatcher)?;
    ctx.add_registration(PluginRegistration::new(
        "observability",
        ctx.qualify_name("atif.shutdown"),
        Box::new(move || {
            let work = {
                let mut guard = manager.lock().map_err(|err| {
                    PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
                })?;
                guard
                    .flush_open_agents()
                    .map_err(observability_registration_error)?
            };
            for (scope_uuid, name) in work.scope_subscribers {
                let _ = scope_deregister_subscriber(&scope_uuid, &name);
            }
            for write in work.writes {
                let agent_uuid = write.agent_uuid;
                let result = write_atif_file(&write);
                let mut guard = manager.lock().map_err(|err| {
                    PluginError::Internal(format!("ATIF dispatcher lock poisoned: {err}"))
                })?;
                guard
                    .finish_agent_write(agent_uuid, result)
                    .map_err(observability_registration_error)?;
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
    scope_subscribers: HashMap<Uuid, String>,
    last_error: Option<String>,
}

struct ManagedAtifExporter {
    exporter: AtifExporter,
    path: PathBuf,
    observed_events: Vec<Event>,
    written: bool,
}

struct PendingAtifWrite {
    agent_uuid: Uuid,
    path: PathBuf,
    payload: Vec<u8>,
}

struct AtifFlushWork {
    writes: Vec<PendingAtifWrite>,
    scope_subscribers: Vec<(Uuid, String)>,
}

impl AtifDispatcher {
    fn new(config: AtifSectionConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            scope_subscribers: HashMap::new(),
            last_error: None,
        }
    }

    fn observe_global(&mut self, event: &Event, subscriber_prefix: &str, state: Arc<Mutex<Self>>) {
        if self.last_error.is_some() || !is_top_level_agent_start(event) {
            return;
        }

        // The top-level agent scope UUID is the ATIF session ID. The global
        // dispatcher records the start event itself because the scope-local
        // subscriber is attached after that start event has already been
        // emitted.
        let session_id = event.uuid().to_string();
        let exporter = AtifExporter::new(session_id.clone(), self.agent_info());
        (exporter.subscriber())(event);
        let path = self.output_path(&session_id);
        self.agents.insert(
            event.uuid(),
            ManagedAtifExporter {
                exporter,
                path,
                observed_events: vec![event.clone()],
                written: false,
            },
        );

        let agent_uuid = event.uuid();
        let name = format!("{subscriber_prefix}{agent_uuid}");
        let callback = atif_scope_subscriber(state, agent_uuid);
        // Attach the per-agent subscriber to the agent scope rather than the
        // global registry so sibling top-level agents never share events.
        if let Err(err) = scope_register_subscriber(&agent_uuid, &name, callback) {
            self.last_error = Some(format!("failed to register ATIF scope subscriber: {err}"));
        } else {
            self.scope_subscribers.insert(agent_uuid, name);
        }
    }

    fn observe_scope(&mut self, event: &Event, agent_uuid: Uuid) -> Option<PendingAtifWrite> {
        if self.last_error.is_some() {
            return None;
        }
        let should_finalize =
            event.uuid() == agent_uuid && event.scope_category() == Some(ScopeCategory::End);
        let agent = self.agents.get_mut(&agent_uuid)?;
        (agent.exporter.subscriber())(event);
        agent.observed_events.push(event.clone());
        if !should_finalize || agent.written {
            return None;
        }
        match prepare_atif_file(agent_uuid, agent) {
            Ok(write) => Some(write),
            Err(err) => {
                self.last_error = Some(err.to_string());
                None
            }
        }
    }

    fn complete_scope_write(
        &mut self,
        agent_uuid: Uuid,
        result: std::io::Result<()>,
    ) -> Option<(Uuid, String)> {
        if self.finish_agent_write(agent_uuid, result).is_err() {
            return None;
        }
        self.agents.remove(&agent_uuid);
        self.scope_subscribers
            .remove(&agent_uuid)
            .map(|name| (agent_uuid, name))
    }

    fn flush_open_agents(&mut self) -> std::io::Result<AtifFlushWork> {
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
        let mut writes = Vec::with_capacity(agent_uuids.len());
        for agent_uuid in agent_uuids {
            if let Some(agent) = self.agents.get_mut(&agent_uuid) {
                writes.push(prepare_atif_file(agent_uuid, agent)?);
            }
        }
        Ok(AtifFlushWork {
            writes,
            scope_subscribers,
        })
    }

    fn finish_agent_write(
        &mut self,
        agent_uuid: Uuid,
        result: std::io::Result<()>,
    ) -> std::io::Result<()> {
        match result {
            Ok(()) => {
                if let Some(agent) = self.agents.get_mut(&agent_uuid) {
                    agent.observed_events.clear();
                }
                Ok(())
            }
            Err(err) => {
                if let Some(agent) = self.agents.get_mut(&agent_uuid) {
                    agent.written = false;
                }
                self.last_error = Some(err.to_string());
                Err(err)
            }
        }
    }

    fn last_error_result(&self) -> std::io::Result<()> {
        if let Some(message) = &self.last_error {
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

    fn output_path(&self, session_id: &str) -> PathBuf {
        let directory = self
            .config
            .output_directory
            .clone()
            .unwrap_or_else(default_output_directory);
        let filename = self
            .config
            .filename_template
            .replace("{session_id}", session_id);
        directory.join(filename)
    }
}

fn atif_dispatcher_subscriber(
    manager: Arc<Mutex<AtifDispatcher>>,
    subscriber_prefix: String,
) -> EventSubscriberFn {
    Arc::new(move |event: &Event| {
        let Ok(mut guard) = manager.lock() else {
            return;
        };
        guard.observe_global(event, &subscriber_prefix, Arc::clone(&manager));
    })
}

fn atif_scope_subscriber(
    manager: Arc<Mutex<AtifDispatcher>>,
    agent_uuid: Uuid,
) -> EventSubscriberFn {
    Arc::new(move |event: &Event| {
        let pending_write = {
            let Ok(mut guard) = manager.lock() else {
                return;
            };
            guard.observe_scope(event, agent_uuid)
        };
        let Some(write) = pending_write else {
            return;
        };
        let result = write_atif_file(&write);
        let scope_subscriber = {
            let Ok(mut guard) = manager.lock() else {
                return;
            };
            guard.complete_scope_write(write.agent_uuid, result)
        };
        if let Some((scope_uuid, name)) = scope_subscriber {
            let _ = scope_deregister_subscriber(&scope_uuid, &name);
        }
    })
}

fn prepare_atif_file(
    agent_uuid: Uuid,
    agent: &mut ManagedAtifExporter,
) -> std::io::Result<PendingAtifWrite> {
    let trajectory = agent.exporter.export();
    let mut value = serde_json::to_value(trajectory)?;
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "extra".to_string(),
            serde_json::json!({
                "observed_events": agent.observed_events,
            }),
        );
    }
    let payload = serde_json::to_vec_pretty(&value)?;
    agent.written = true;
    Ok(PendingAtifWrite {
        agent_uuid,
        path: agent.path.clone(),
        payload,
    })
}

fn write_atif_file(write: &PendingAtifWrite) -> std::io::Result<()> {
    if let Some(parent) = write.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&write.path, &write.payload)?;
    Ok(())
}

fn is_top_level_agent_start(event: &Event) -> bool {
    if event.scope_category() != Some(ScopeCategory::Start)
        || event.scope_type() != Some(ScopeType::Agent)
    {
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
        .with_timeout(Duration::from_millis(section.timeout_millis));

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
    validate_unknown_fields(
        &mut diagnostics,
        &config.policy,
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

    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "atof",
        &["enabled", "output_directory", "filename", "mode"],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
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
        ],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "opentelemetry",
        &[
            "enabled",
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
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "openinference",
        &[
            "enabled",
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

    if let Some(section) = &config.atof {
        validate_atof_values(&mut diagnostics, &config.policy, section);
        #[cfg(target_arch = "wasm32")]
        if section.enabled {
            push_policy_diag(
                &mut diagnostics,
                config.policy.unsupported_value,
                "observability.unsupported_value",
                Some("atof".to_string()),
                Some("enabled".to_string()),
                "ATOF file export is not supported on WebAssembly".to_string(),
            );
        }
    }
    if let Some(section) = &config.atif {
        validate_atif_values(&mut diagnostics, &config.policy, section);
        #[cfg(target_arch = "wasm32")]
        if section.enabled {
            push_policy_diag(
                &mut diagnostics,
                config.policy.unsupported_value,
                "observability.unsupported_value",
                Some("atif".to_string()),
                Some("enabled".to_string()),
                "ATIF file export is not supported on WebAssembly".to_string(),
            );
        }
    }
    if let Some(section) = &config.opentelemetry {
        validate_otlp_values(&mut diagnostics, &config.policy, "opentelemetry", section);
        #[cfg(not(feature = "otel"))]
        if section.enabled {
            push_policy_diag(
                &mut diagnostics,
                config.policy.unsupported_value,
                "observability.feature_disabled",
                Some("opentelemetry".to_string()),
                Some("enabled".to_string()),
                "OpenTelemetry support is not enabled in this build".to_string(),
            );
        }
    }
    if let Some(section) = &config.openinference {
        validate_otlp_values(&mut diagnostics, &config.policy, "openinference", section);
        #[cfg(not(feature = "openinference"))]
        if section.enabled {
            push_policy_diag(
                &mut diagnostics,
                config.policy.unsupported_value,
                "observability.feature_disabled",
                Some("openinference".to_string()),
                Some("enabled".to_string()),
                "OpenInference support is not enabled in this build".to_string(),
            );
        }
    }

    diagnostics
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

fn default_agent_name() -> String {
    "NeMo Flow".to_string()
}

fn default_agent_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn default_model_name() -> String {
    "unknown".to_string()
}

fn default_atif_filename_template() -> String {
    "nemo-flow-atif-{session_id}.json".to_string()
}

fn default_otlp_transport() -> String {
    "http_binary".to_string()
}

fn default_service_name() -> String {
    "nemo-flow".to_string()
}

fn default_timeout_millis() -> u64 {
    3_000
}

fn default_output_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
#[path = "../../tests/unit/observability/plugin_component_tests.rs"]
mod tests;
