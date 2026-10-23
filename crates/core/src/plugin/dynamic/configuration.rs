// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core-owned `plugins.toml` discovery and dynamic-plugin selection.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value as Json};

use super::read_bounded_regular_file;
use super::trust::DynamicPluginLoadTrust;
use super::{
    DynamicPluginCheckState, DynamicPluginFailure, DynamicPluginHostPolicy, DynamicPluginKind,
    DynamicPluginManifest, DynamicPluginRecord, DynamicPluginStartupClass,
    DynamicPluginValidationStatus, EvaluatedDynamicPluginHostPolicy, FileDynamicPluginHostPolicy,
    evaluate_dynamic_plugin_host_policy, validate_dynamic_plugin_config_schema,
};
use crate::diagnostics::Diagnostics;
use crate::plugin::{
    PluginConfig, PluginError, Result, plugin_config_paths, resolve_plugin_config_documents,
};

const DYNAMIC_PLUGIN_STATE_FILENAME: &str = ".dynamic-plugins.json";
const DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION: u32 = 1;

/// One validation result produced by core before dynamic code is loaded.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginValidationReport {
    /// Canonical plugin identifier.
    pub plugin_id: String,
    /// Canonical manifest path.
    pub manifest_ref: String,
    /// Runtime lane.
    pub kind: DynamicPluginKind,
    /// Effective validation status.
    pub status: DynamicPluginValidationStatus,
    /// Failure observed during policy, trust, or schema validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<DynamicPluginFailure>,
    /// Whether this plugin will be activated by the current host request.
    pub selected: bool,
}

/// Combined static and dynamic host report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PluginHostReport {
    /// Static plugin initialization report.
    pub config: crate::plugin::ConfigReport,
    /// Dynamic validation reports in discovery order.
    #[serde(default)]
    pub dynamic_plugins: Vec<DynamicPluginValidationReport>,
    /// Existing `plugins.toml` files that contributed to the resolved configuration.
    #[serde(default)]
    pub config_paths: Vec<String>,
    /// Fully merged plugin configuration with sensitive values redacted.
    #[serde(default = "empty_resolved_config")]
    pub resolved_config: Json,
}

impl Default for PluginHostReport {
    fn default() -> Self {
        Self {
            config: crate::plugin::ConfigReport::default(),
            dynamic_plugins: Vec::new(),
            config_paths: Vec::new(),
            resolved_config: empty_resolved_config(),
        }
    }
}

fn empty_resolved_config() -> Json {
    Json::Object(Map::new())
}

/// Selects the scope of a standalone dynamic-plugin validation request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case", tag = "target", content = "value")]
pub(crate) enum PluginHostValidationTarget {
    /// Validate every enabled declaration in the effective host configuration.
    #[default]
    All,
    /// Validate one declaration by its plugin identifier, including disabled entries.
    PluginId(String),
    /// Validate one authored manifest outside lifecycle selection.
    ManifestPath(PathBuf),
}

/// Typed request for standalone, non-activating dynamic-plugin validation.
#[cfg(test)]
#[derive(Debug, Clone, serde::Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct PluginHostValidationRequest {
    /// Programmatic static-plugin configuration applied after file layers.
    #[serde(default)]
    pub config: PluginConfig,
    /// Optional explicit `plugins.toml` layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_plugins_toml: Option<PathBuf>,
    /// Validation target.
    #[serde(default, flatten)]
    pub target: PluginHostValidationTarget,
}

#[cfg(test)]
impl<'de> serde::Deserialize<'de> for PluginHostValidationRequest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireRequest {
            #[serde(default)]
            config: PluginConfig,
            #[serde(default)]
            additional_plugins_toml: Option<PathBuf>,
            #[serde(default)]
            target: Option<String>,
            #[serde(default)]
            value: Option<String>,
        }

        let request = WireRequest::deserialize(deserializer)?;
        let target = match request.target.as_deref().unwrap_or("all") {
            "all" => PluginHostValidationTarget::All,
            "plugin_id" => PluginHostValidationTarget::PluginId(
                request
                    .value
                    .ok_or_else(|| serde::de::Error::missing_field("value"))?,
            ),
            "manifest_path" => PluginHostValidationTarget::ManifestPath(PathBuf::from(
                request
                    .value
                    .ok_or_else(|| serde::de::Error::missing_field("value"))?,
            )),
            target => {
                return Err(serde::de::Error::unknown_variant(
                    target,
                    &["all", "plugin_id", "manifest_path"],
                ));
            }
        };
        Ok(Self {
            config: request.config,
            additional_plugins_toml: request.additional_plugins_toml,
            target,
        })
    }
}

pub(crate) struct ResolvedPluginHostConfig {
    pub(crate) config: PluginConfig,
    pub(crate) policy: DynamicPluginHostPolicy,
    pub(crate) dynamic_plugins: Vec<super::VerifiedDynamicPluginSpec>,
    pub(super) load_trust: HashMap<String, DynamicPluginLoadTrust>,
    pub(super) native_artifacts: Vec<super::native::PreparedNativePlugin>,
    pub(super) dynamic_reports: Vec<ResolvedDynamicPluginReport>,
    pub(crate) diagnostics: Vec<crate::plugin::ConfigDiagnostic>,
    pub(crate) config_paths: Vec<String>,
    pub(crate) resolved_config: Json,
}

/// A validation report with the lifecycle selection used to request it.
///
/// `DynamicPluginValidationReport::selected` describes whether the host will
/// activate the plugin. Keep the requested selection separately so preflight
/// validation can report a required plugin that failed validation.
pub(super) struct ResolvedDynamicPluginReport {
    pub(super) requested: bool,
    pub(super) report: DynamicPluginValidationReport,
}

#[derive(Debug, Deserialize)]
struct PluginFile {
    #[serde(default)]
    plugins: PluginFilePlugins,
}

struct PluginFileDocument {
    source: PathBuf,
    value: Json,
    file: PluginFile,
}

#[derive(Debug, Default, Deserialize)]
struct PluginFilePlugins {
    #[serde(default)]
    dynamic: Vec<FileDynamicPlugin>,
    #[serde(default)]
    policy: Option<FileDynamicPluginHostPolicy>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDynamicPlugin {
    manifest: String,
    #[serde(default)]
    config: Map<String, Json>,
}

struct ResolvedDynamicPluginDeclaration {
    source: PathBuf,
    declared: FileDynamicPlugin,
    manifest: DynamicPluginManifest,
    manifest_ref: String,
}

#[derive(Debug, Deserialize)]
struct PersistedDynamicPluginRegistry {
    #[serde(default = "default_state_schema_version")]
    schema_version: u32,
    #[serde(default)]
    records: Vec<DynamicPluginRecord>,
}

struct StateSelection {
    selected: bool,
    environment_ref: Option<String>,
}

const fn default_state_schema_version() -> u32 {
    DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION
}

/// Resolves and validates all active dynamic plugins before any runtime code is loaded.
pub(crate) fn resolve_plugin_host_config(
    programmatic: PluginConfig,
    explicit_path: Option<&Path>,
    diagnostic_context: Option<&Diagnostics>,
) -> Result<ResolvedPluginHostConfig> {
    resolve_plugin_host_config_inner(programmatic, explicit_path, true, diagnostic_context)
}

/// Validates the process-wide static and dynamic plugin host.
///
/// This resolves the same configuration layers as [`super::initialize`] but
/// does not acquire the activation lease or load any dynamic code.
pub fn validate(
    config: impl Into<super::PluginHostConfig>,
    additional_plugins_toml: Option<PathBuf>,
) -> Result<PluginHostReport> {
    let config = config.into();
    let prepared = PreparedPluginHostConfig::load(additional_plugins_toml.as_deref())?;
    prepared.diagnostic_settings(config.diagnostics)?;
    let resolved = prepared.resolve(
        config.plugins,
        additional_plugins_toml.as_deref(),
        false,
        None,
    )?;
    validation_report(resolved, PluginHostValidationTarget::All)
}

/// Validates only the supplied static plugin configuration.
///
/// Unlike [`validate`], this does not discover or merge any `plugins.toml`
/// layers. It is intended for component-specific validators whose callers
/// supplied the complete configuration to validate.
pub fn validate_exact(config: PluginConfig) -> PluginHostReport {
    PluginHostReport {
        config: validate_static_preflight_configuration(&config),
        dynamic_plugins: Vec::new(),
        config_paths: Vec::new(),
        resolved_config: sanitized_plugin_config(&config),
    }
}

fn validate_static_preflight_configuration(
    configuration: &PluginConfig,
) -> crate::plugin::ConfigReport {
    let mut report = crate::plugin::validate_static_plugin_config(configuration);
    for component in &configuration.components {
        if component.enabled
            && component.kind == crate::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND
        {
            report.diagnostics.extend(
                crate::observability::plugin_component::validate_observability_plaintext_destinations(
                    &component.config,
                ),
            );
        }
    }
    report
}

/// Validates a targeted dynamic-plugin request for internal diagnostics.
#[cfg(test)]
pub(crate) fn validate_request(request: PluginHostValidationRequest) -> Result<PluginHostReport> {
    let resolved = resolve_plugin_host_config_inner(
        request.config,
        request.additional_plugins_toml.as_deref(),
        false,
        None,
    )?;
    validation_report(resolved, request.target)
}

fn validation_report(
    resolved: ResolvedPluginHostConfig,
    target: PluginHostValidationTarget,
) -> Result<PluginHostReport> {
    let mut config_report = validate_static_preflight_configuration(&resolved.config);
    config_report
        .diagnostics
        .splice(0..0, resolved.diagnostics.iter().cloned());
    let dynamic_plugins = match target {
        PluginHostValidationTarget::All => resolved
            .dynamic_reports
            .into_iter()
            .filter(|entry| entry.requested)
            .map(|entry| entry.report)
            .collect(),
        PluginHostValidationTarget::PluginId(plugin_id) => {
            let reports = resolved
                .dynamic_reports
                .into_iter()
                .filter(|entry| entry.report.plugin_id == plugin_id)
                .map(|entry| entry.report)
                .collect::<Vec<_>>();
            if reports.is_empty() {
                return Err(PluginError::NotFound(format!(
                    "dynamic plugin '{plugin_id}' was not declared in the effective plugins.toml configuration"
                )));
            }
            reports
        }
        PluginHostValidationTarget::ManifestPath(manifest_path) => {
            let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)?;
            let evaluated_policy = evaluate_dynamic_plugin_host_policy(&resolved.policy, &manifest);
            vec![validate_declaration(
                &manifest,
                manifest_ref,
                true,
                &serde_json::Map::new(),
                &evaluated_policy,
            )]
        }
    };
    Ok(PluginHostReport {
        config: config_report,
        dynamic_plugins,
        config_paths: resolved.config_paths,
        resolved_config: resolved.resolved_config,
    })
}

fn resolve_plugin_host_config_inner(
    programmatic: PluginConfig,
    explicit_path: Option<&Path>,
    reject_required_failures: bool,
    diagnostic_context: Option<&Diagnostics>,
) -> Result<ResolvedPluginHostConfig> {
    PreparedPluginHostConfig::load(explicit_path)?.resolve(
        programmatic,
        explicit_path,
        reject_required_failures,
        diagnostic_context,
    )
}

pub(super) struct PreparedPluginHostConfig {
    files: Vec<PluginFileDocument>,
    diagnostics: Vec<crate::plugin::ConfigDiagnostic>,
}
impl PreparedPluginHostConfig {
    pub(super) fn load(explicit_path: Option<&Path>) -> Result<Self> {
        let paths = plugin_config_paths(explicit_path, crate::plugin::user_config_dir());
        let (files, diagnostics) = read_plugin_files(&paths, explicit_path)?;
        Ok(Self { files, diagnostics })
    }
    pub(super) fn diagnostic_settings(
        &self,
        programmatic: Option<crate::diagnostics::DiagnosticsSettings>,
    ) -> Result<crate::diagnostics::DiagnosticsSettings> {
        // The existing resolver owns file order. Merge service fields independently
        // so they never enter plugin config reports, fingerprints, or callbacks.
        let settings = if let Some(settings) = programmatic {
            settings
        } else {
            let mut merged = Json::Object(Map::new());
            for file in &self.files {
                if let Some(value) = file.value.get("diagnostics") {
                    crate::plugin::merge_json_value(&mut merged, value.clone());
                }
            }
            serde_json::from_value(merged)
                .map_err(|_| PluginError::InvalidConfig("diagnostics: invalid_settings".into()))?
        };
        settings
            .limits
            .validate()
            .map_err(|_| PluginError::InvalidConfig("diagnostics: invalid_limits".into()))?;
        Ok(settings)
    }
    pub(super) fn resolve(
        self,
        programmatic: PluginConfig,
        explicit_path: Option<&Path>,
        reject_required_failures: bool,
        diagnostic_context: Option<&Diagnostics>,
    ) -> Result<ResolvedPluginHostConfig> {
        let Self {
            files,
            mut diagnostics,
        } = self;
        let resolved = resolve_plugin_config_documents(
            programmatic,
            explicit_path,
            files
                .iter()
                .map(|document| {
                    let mut value = document.value.clone();
                    if let Some(root) = value.as_object_mut() {
                        root.remove("diagnostics");
                    }
                    (document.source.clone(), value)
                })
                .collect(),
            diagnostic_context,
        )?;
        diagnostics.extend(resolved.diagnostics.iter().cloned());
        let (mut policy, declarations) = resolve_dynamic_plugin_declarations(files)?;
        let mut active = Vec::new();
        let mut native_artifacts = Vec::new();
        let mut load_trust = HashMap::new();
        let mut reports = Vec::new();

        policy.apply_secure_defaults();
        for ResolvedDynamicPluginDeclaration {
            source,
            declared,
            manifest,
            manifest_ref,
        } in declarations
        {
            let plugin_id = manifest.plugin.id.trim().to_owned();
            let state = state_for_plugin(&source, &manifest_ref, &plugin_id)?;
            let selected = state.as_ref().map(|state| state.selected).unwrap_or(true);
            let evaluated_policy = evaluate_dynamic_plugin_host_policy(&policy, &manifest);
            let (report, artifact_bytes) = validate_declaration_with_artifact(
                &manifest,
                manifest_ref.clone(),
                selected,
                &declared.config,
                &evaluated_policy,
                diagnostic_context,
            );
            let valid = report.failure.is_none();
            let effective_selected = report.selected;
            let failure = report.failure.clone();
            reports.push(ResolvedDynamicPluginReport {
                requested: selected,
                report,
            });
            if selected
                && !valid
                && reject_required_failures
                && evaluated_policy.startup_class == DynamicPluginStartupClass::Required
            {
                return Err(PluginError::InvalidConfig(
                    failure.map(|failure| failure.message).unwrap_or_else(|| {
                        format!("dynamic plugin '{plugin_id}' failed validation")
                    }),
                ));
            }
            if effective_selected {
                if reject_required_failures
                    && manifest.plugin.kind == DynamicPluginKind::RustDynamic
                {
                    native_artifacts.push(super::native::PreparedNativePlugin::new(
                        manifest.clone(),
                        manifest_ref.clone(),
                        artifact_bytes,
                    )?);
                }
                load_trust.insert(
                    plugin_id.clone(),
                    DynamicPluginLoadTrust {
                        manifest: manifest.clone(),
                        policy: evaluated_policy,
                    },
                );
                active.push(super::VerifiedDynamicPluginSpec {
                    plugin_id,
                    kind: manifest.plugin.kind,
                    manifest_ref,
                    environment_ref: state.and_then(|state| state.environment_ref),
                    config: declared.config,
                });
            }
        }
        Ok(ResolvedPluginHostConfig {
            config_paths: resolved.config_paths,
            resolved_config: sanitize_resolved_config(resolved.resolved_config),
            config: resolved.config,
            policy,
            dynamic_plugins: active,
            native_artifacts,
            load_trust,
            dynamic_reports: reports,
            diagnostics,
        })
    }
}

fn resolve_dynamic_plugin_declarations(
    files: Vec<PluginFileDocument>,
) -> Result<(
    DynamicPluginHostPolicy,
    Vec<ResolvedDynamicPluginDeclaration>,
)> {
    let mut policy = DynamicPluginHostPolicy::default();
    let mut declarations = Vec::new();
    let mut declaration_indices = HashMap::new();

    for PluginFileDocument { source, file, .. } in files {
        if let Some(file_policy) = file.plugins.policy {
            policy.merge_from(file_policy.into());
        }
        let mut seen_ids = HashSet::new();
        for declared in file.plugins.dynamic {
            let manifest_path = resolve_manifest_path(&source, &declared.manifest);
            let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)?;
            let plugin_id = manifest.plugin.id.trim().to_owned();
            if !seen_ids.insert(plugin_id.clone()) {
                return Err(PluginError::InvalidConfig(format!(
                    "duplicate dynamic plugin id '{plugin_id}' in {}",
                    source.display()
                )));
            }
            let declaration = ResolvedDynamicPluginDeclaration {
                source: source.clone(),
                declared,
                manifest,
                manifest_ref,
            };
            if let Some(index) = declaration_indices.get(&plugin_id) {
                // Dynamic plugin declarations layer by manifest ID. The later source owns the
                // effective manifest, lifecycle state, and opaque configuration.
                declarations[*index] = declaration;
            } else {
                declaration_indices.insert(plugin_id, declarations.len());
                declarations.push(declaration);
            }
        }
    }
    Ok((policy, declarations))
}

pub(crate) fn sanitized_plugin_config(config: &PluginConfig) -> Json {
    let config = serde_json::to_value(config)
        .expect("PluginConfig must serialize to a JSON value for diagnostics");
    sanitize_resolved_config(config)
}

fn sanitize_resolved_config(mut config: Json) -> Json {
    redact_config_value(&mut config, None);
    redact_opaque_dynamic_plugin_configs(&mut config);
    config
}

fn redact_opaque_dynamic_plugin_configs(config: &mut Json) {
    let Some(dynamic_plugins) = config
        .pointer_mut("/plugins/dynamic")
        .and_then(Json::as_array_mut)
    else {
        return;
    };
    for plugin in dynamic_plugins {
        if let Some(plugin_config) = plugin.get_mut("config") {
            redact_sensitive_value(plugin_config);
        }
    }
}

fn redact_config_value(value: &mut Json, field_name: Option<&str>) {
    if field_name.is_some_and(is_sensitive_field_name) {
        redact_sensitive_value(value);
        return;
    }
    if field_name.is_some_and(is_url_field_name) {
        if let Some(endpoint) = value.as_str() {
            *value = Json::String(diagnostic_endpoint(endpoint));
        }
        return;
    }
    match value {
        Json::Object(values) => {
            if field_name == Some("headers") {
                for value in values.values_mut() {
                    redact_sensitive_value(value);
                }
            } else {
                for (name, value) in values {
                    redact_config_value(value, Some(name));
                }
            }
        }
        Json::Array(values) => {
            for value in values {
                redact_config_value(value, None);
            }
        }
        _ => {}
    }
}

fn redact_sensitive_value(value: &mut Json) {
    match value {
        Json::Object(values) => {
            for value in values.values_mut() {
                redact_sensitive_value(value);
            }
        }
        Json::Array(values) => {
            for value in values {
                redact_sensitive_value(value);
            }
        }
        _ => *value = Json::String("[REDACTED]".to_string()),
    }
}

fn is_sensitive_field_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    matches!(
        name.as_str(),
        "authorization" | "api_key" | "apikey" | "password" | "secret" | "token"
    ) || name.contains("credential")
        || name.contains("secret")
        || name.contains("token")
        || name.contains("password")
}

fn is_url_field_name(name: &str) -> bool {
    let normalized = name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    normalized == "url"
        || normalized == "endpoint"
        || normalized.ends_with("url")
        || normalized.ends_with("endpoint")
}

fn diagnostic_endpoint(endpoint: &str) -> String {
    let Ok(mut endpoint) = reqwest::Url::parse(endpoint) else {
        return endpoint.to_string();
    };
    let _ = endpoint.set_username("");
    let _ = endpoint.set_password(None);
    endpoint.set_query(None);
    endpoint.set_fragment(None);
    endpoint.into()
}

fn validate_declaration(
    manifest: &DynamicPluginManifest,
    manifest_ref: String,
    selected: bool,
    config: &Map<String, Json>,
    evaluated_policy: &EvaluatedDynamicPluginHostPolicy,
) -> DynamicPluginValidationReport {
    validate_declaration_with_artifact(
        manifest,
        manifest_ref,
        selected,
        config,
        evaluated_policy,
        None,
    )
    .0
}

fn validate_declaration_with_artifact(
    manifest: &DynamicPluginManifest,
    manifest_ref: String,
    selected: bool,
    config: &Map<String, Json>,
    evaluated_policy: &EvaluatedDynamicPluginHostPolicy,
    diagnostic_context: Option<&Diagnostics>,
) -> (DynamicPluginValidationReport, Option<Vec<u8>>) {
    let plugin_id = manifest.plugin.id.trim().to_owned();
    let (trust, artifact_bytes) = super::trust::evaluate_dynamic_plugin_trust_with_artifact(
        manifest,
        &manifest_ref,
        evaluated_policy,
    );
    let schema_failure = validate_dynamic_plugin_config_schema(
        manifest,
        &manifest_ref,
        &Json::Object(config.clone()),
    )
    .err();
    let has_diagnostic_failure =
        evaluated_policy.failure().is_some() || trust.failure.is_some() || schema_failure.is_some();
    if let Some(context) = diagnostic_context.filter(|_| has_diagnostic_failure) {
        use crate::diagnostics::{
            DiagnosticClassifier, FailureClass, PluginOperationFailureCode as Operation,
            PluginPhase,
        };
        let observer = super::diagnostics::PluginDiagnostics::new(
            context,
            &plugin_id,
            super::diagnostics::execution(manifest.plugin.kind),
            uuid::Uuid::now_v7(),
        );
        if let Some(failure) = evaluated_policy.failure() {
            let classified = super::diagnostics::PolicyFailureClassifier.classify(failure);
            observer.classified_failure(classified, PluginPhase::Policy);
        }
        if let Some(failure) = trust.failure.as_ref() {
            let classified = super::diagnostics::TrustFailureClassifier.classify(failure);
            observer.classified_failure(classified, PluginPhase::Validation);
        }
        if schema_failure.is_some() {
            // The schema helper includes reading and validation errors; retain
            // Unknown until its source exposes a typed cause.
            observer.failure(
                Operation::SchemaValidation,
                FailureClass::Unknown,
                PluginPhase::Validation,
            );
        }
    }
    let mut status = DynamicPluginValidationStatus {
        manifest: DynamicPluginCheckState::Valid,
        compatibility: DynamicPluginCheckState::Valid,
        integrity: trust.integrity,
        environment: DynamicPluginCheckState::Unknown,
        authenticity: trust.authenticity,
        policy_satisfied: evaluated_policy.check_state(),
        checked_at: None,
        message: Some("validated by core".into()),
    };
    let failure = evaluated_policy
        .last_error(&plugin_id)
        .or_else(|| trust.last_error(&plugin_id))
        .or_else(|| {
            schema_failure
                .as_ref()
                .map(|error| super::DynamicPluginFailure {
                    phase: super::DynamicPluginFailurePhase::Validation,
                    code: "config_schema_failed".into(),
                    message: error.to_string(),
                })
        });
    if schema_failure.is_some() {
        status.manifest = DynamicPluginCheckState::Invalid;
    }
    let effective_selected = selected
        && evaluated_policy.policy_satisfied
        && schema_failure.is_none()
        && trust.is_satisfied();
    (
        DynamicPluginValidationReport {
            plugin_id,
            manifest_ref,
            kind: manifest.plugin.kind,
            status,
            failure,
            selected: effective_selected,
        },
        artifact_bytes,
    )
}

fn read_plugin_files(
    paths: &[PathBuf],
    explicit_path: Option<&Path>,
) -> Result<(
    Vec<PluginFileDocument>,
    Vec<crate::plugin::ConfigDiagnostic>,
)> {
    let mut files = Vec::new();
    let mut diagnostics = Vec::new();
    for source in paths {
        if !plugin_file_is_present(source, std::fs::symlink_metadata(source))? {
            if explicit_path == Some(source.as_path()) {
                diagnostics.push(crate::plugin::ConfigDiagnostic {
                    level: crate::plugin::DiagnosticLevel::Warning,
                    code: "plugin.configuration_file_missing".to_string(),
                    component: None,
                    field: Some("additional_plugins_toml".to_string()),
                    message: format!(
                        "explicit plugin configuration file does not exist: {}",
                        source.display()
                    ),
                });
            }
            continue;
        }
        let raw = read_utf8_plugin_file(source)?;
        let table = raw.parse::<toml::Table>().map_err(|error| {
            PluginError::InvalidConfig(format!(
                "invalid plugin TOML in {}: {error}",
                source.display()
            ))
        })?;
        let file = table.clone().try_into().map_err(|error| {
            PluginError::InvalidConfig(format!(
                "invalid plugin TOML in {}: {error}",
                source.display()
            ))
        })?;
        files.push(PluginFileDocument {
            source: source.clone(),
            value: serde_json::to_value(table)?,
            file,
        });
    }
    Ok((files, diagnostics))
}

fn plugin_file_is_present(
    source: &Path,
    inspection: std::io::Result<std::fs::Metadata>,
) -> Result<bool> {
    match inspection {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(PluginError::InvalidConfig(format!(
            "failed to inspect plugin configuration {}: {error}",
            source.display()
        ))),
    }
}

fn read_utf8_plugin_file(path: &Path) -> Result<String> {
    let bytes = read_bounded_regular_file(path, "plugin configuration").map_err(|error| {
        PluginError::InvalidConfig(format!("failed to read {}: {error}", path.display()))
    })?;
    String::from_utf8(bytes).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "plugin TOML {} is not UTF-8: {error}",
            path.display()
        ))
    })
}

fn resolve_manifest_path(source: &Path, reference: &str) -> PathBuf {
    let path = PathBuf::from(reference);
    if path.is_absolute() {
        path
    } else {
        source
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

fn state_for_plugin(
    plugins_toml: &Path,
    manifest_ref: &str,
    plugin_id: &str,
) -> Result<Option<StateSelection>> {
    let state_path = plugins_toml
        .parent()
        .map(|parent| parent.join(DYNAMIC_PLUGIN_STATE_FILENAME))
        .unwrap_or_else(|| PathBuf::from(DYNAMIC_PLUGIN_STATE_FILENAME));
    if !state_path.exists() {
        return Ok(None);
    }
    let raw = read_bounded_regular_file(&state_path, "dynamic plugin lifecycle state").map_err(
        |error| {
            PluginError::InvalidConfig(format!(
                "failed to read dynamic plugin lifecycle state {}: {error}",
                state_path.display()
            ))
        },
    )?;
    let raw = String::from_utf8(raw).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "invalid dynamic plugin lifecycle state {}: {error}",
            state_path.display()
        ))
    })?;
    let state: PersistedDynamicPluginRegistry = serde_json::from_str(&raw).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "invalid dynamic plugin lifecycle state {}: {error}",
            state_path.display()
        ))
    })?;
    if state.schema_version != DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION {
        return Err(PluginError::InvalidConfig(format!(
            "unsupported dynamic plugin lifecycle schema_version {} in {}; expected {}",
            state.schema_version,
            state_path.display(),
            DYNAMIC_PLUGIN_STATE_SCHEMA_VERSION
        )));
    }
    let mut matches = state
        .records
        .into_iter()
        .filter(|record| record.metadata.id == plugin_id)
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin lifecycle state {} contains duplicate record '{plugin_id}'",
            state_path.display()
        )));
    }
    let Some(record) = matches.pop() else {
        return Ok(Some(StateSelection {
            selected: false,
            environment_ref: None,
        }));
    };
    if record.source.manifest_ref.as_deref() != Some(manifest_ref) {
        return Err(PluginError::InvalidConfig(format!(
            "dynamic plugin lifecycle state {} has manifest identity mismatch for '{plugin_id}'",
            state_path.display()
        )));
    }
    Ok(Some(StateSelection {
        selected: record.spec.present && record.spec.enabled,
        environment_ref: record.source.environment_ref,
    }))
}

#[cfg(test)]
#[path = "../../../tests/unit/plugin_dynamic_configuration_tests.rs"]
mod tests;
