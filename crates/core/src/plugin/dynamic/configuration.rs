// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core-owned `plugins.toml` discovery and dynamic-plugin selection.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value as Json};

use super::read_bounded_regular_file;
use super::{
    DynamicPluginCheckState, DynamicPluginFailure, DynamicPluginHostPolicy, DynamicPluginKind,
    DynamicPluginManifest, DynamicPluginRecord, DynamicPluginStartupClass,
    DynamicPluginValidationStatus, EvaluatedDynamicPluginHostPolicy, FileDynamicPluginHostPolicy,
    evaluate_dynamic_plugin_host_policy, evaluate_dynamic_plugin_trust,
    validate_dynamic_plugin_config_schema,
};
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PluginHostReport {
    /// Static plugin initialization report.
    pub config: crate::plugin::ConfigReport,
    /// Dynamic validation reports in discovery order.
    #[serde(default)]
    pub dynamic_plugins: Vec<DynamicPluginValidationReport>,
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
#[derive(Debug, Clone, serde::Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct PluginHostValidationRequest {
    /// Programmatic static-plugin configuration at the lowest precedence.
    #[serde(default)]
    pub config: PluginConfig,
    /// Optional explicit `plugins.toml` layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_plugins_toml: Option<PathBuf>,
    /// Validation target.
    #[serde(default, flatten)]
    pub target: PluginHostValidationTarget,
}

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
    pub(crate) dynamic_reports: Vec<DynamicPluginValidationReport>,
    pub(crate) diagnostics: Vec<crate::plugin::ConfigDiagnostic>,
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
) -> Result<ResolvedPluginHostConfig> {
    resolve_plugin_host_config_inner(programmatic, explicit_path, true)
}

/// Validates the process-wide static and dynamic plugin host.
///
/// This resolves the same configuration layers as [`super::initialize`] but
/// does not acquire the activation lease or load any dynamic code.
pub fn validate(
    config: PluginConfig,
    additional_plugins_toml: Option<PathBuf>,
) -> Result<PluginHostReport> {
    validate_request(PluginHostValidationRequest {
        config,
        additional_plugins_toml,
        target: PluginHostValidationTarget::All,
    })
}

/// Validates only the supplied static plugin configuration.
///
/// Unlike [`validate`], this does not discover or merge any `plugins.toml`
/// layers. It is intended for component-specific validators whose callers
/// supplied the complete configuration to validate.
pub fn validate_exact(config: PluginConfig) -> PluginHostReport {
    PluginHostReport {
        config: crate::plugin::validate_static_plugin_config(&config),
        dynamic_plugins: Vec::new(),
    }
}

/// Validates a targeted dynamic-plugin request for internal diagnostics.
pub(crate) fn validate_request(request: PluginHostValidationRequest) -> Result<PluginHostReport> {
    let resolved = resolve_plugin_host_config_inner(
        request.config,
        request.additional_plugins_toml.as_deref(),
        false,
    )?;
    let mut config_report = crate::plugin::validate_static_plugin_config(&resolved.config);
    config_report
        .diagnostics
        .splice(0..0, resolved.diagnostics.iter().cloned());
    let dynamic_plugins = match request.target {
        PluginHostValidationTarget::All => resolved
            .dynamic_reports
            .into_iter()
            .filter(|report| report.selected)
            .collect(),
        PluginHostValidationTarget::PluginId(plugin_id) => {
            let reports = resolved
                .dynamic_reports
                .into_iter()
                .filter(|report| report.plugin_id == plugin_id)
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
    })
}

fn resolve_plugin_host_config_inner(
    programmatic: PluginConfig,
    explicit_path: Option<&Path>,
    reject_required_failures: bool,
) -> Result<ResolvedPluginHostConfig> {
    let paths = plugin_config_paths(explicit_path, crate::plugin::user_config_dir());
    let files = read_plugin_files(&paths)?;
    let resolved = resolve_plugin_config_documents(
        programmatic,
        explicit_path,
        files
            .iter()
            .map(|document| (document.source.clone(), document.value.clone()))
            .collect(),
    )?;
    let mut policy = DynamicPluginHostPolicy::default();
    let mut active = Vec::new();
    let mut reports = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut declarations = Vec::new();

    for PluginFileDocument { source, file, .. } in files {
        if let Some(file_policy) = file.plugins.policy {
            policy.merge_from(file_policy.into());
        }
        declarations.extend(
            file.plugins
                .dynamic
                .into_iter()
                .map(|declared| (source.clone(), declared)),
        );
    }
    policy.apply_secure_defaults();
    for (source, declared) in declarations {
        let manifest_path = resolve_manifest_path(&source, &declared.manifest);
        let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)?;
        let plugin_id = manifest.plugin.id.trim().to_owned();
        if !seen_ids.insert(plugin_id.clone()) {
            return Err(PluginError::InvalidConfig(format!(
                "duplicate dynamic plugin id '{plugin_id}' in resolved plugins.toml layers"
            )));
        }
        let state = state_for_plugin(&source, &manifest_ref, &plugin_id)?;
        let selected = state.as_ref().map(|state| state.selected).unwrap_or(true);
        let evaluated_policy = evaluate_dynamic_plugin_host_policy(&policy, &manifest);
        let report = validate_declaration(
            &manifest,
            manifest_ref.clone(),
            selected,
            &declared.config,
            &evaluated_policy,
        );
        let valid = report.failure.is_none();
        let effective_selected = report.selected;
        let failure = report.failure.clone();
        reports.push(report);
        if selected
            && !valid
            && reject_required_failures
            && evaluated_policy.startup_class == DynamicPluginStartupClass::Required
        {
            return Err(PluginError::InvalidConfig(
                failure
                    .map(|failure| failure.message)
                    .unwrap_or_else(|| format!("dynamic plugin '{plugin_id}' failed validation")),
            ));
        }
        if effective_selected {
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
        config: resolved.config,
        policy,
        dynamic_plugins: active,
        dynamic_reports: reports,
        diagnostics: resolved.diagnostics,
    })
}

fn validate_declaration(
    manifest: &DynamicPluginManifest,
    manifest_ref: String,
    selected: bool,
    config: &Map<String, Json>,
    evaluated_policy: &EvaluatedDynamicPluginHostPolicy,
) -> DynamicPluginValidationReport {
    let plugin_id = manifest.plugin.id.trim().to_owned();
    let trust = evaluate_dynamic_plugin_trust(manifest, &manifest_ref, evaluated_policy);
    let schema_failure = validate_dynamic_plugin_config_schema(
        manifest,
        &manifest_ref,
        &Json::Object(config.clone()),
    )
    .err();
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
    let is_valid = failure.is_none();
    DynamicPluginValidationReport {
        plugin_id,
        manifest_ref,
        kind: manifest.plugin.kind,
        status,
        failure,
        selected: selected && is_valid,
    }
}

fn read_plugin_files(paths: &[PathBuf]) -> Result<Vec<PluginFileDocument>> {
    let mut files = Vec::new();
    for source in paths {
        match std::fs::symlink_metadata(source) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(PluginError::InvalidConfig(format!(
                    "failed to inspect plugin configuration {}: {error}",
                    source.display()
                )));
            }
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
    Ok(files)
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
