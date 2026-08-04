// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, PluginConfig, default_plugin_config_paths,
    merge_plugin_config_documents, user_config_dir,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::{PluginHostConfigError, Result};
use crate::io::{load_bounded_dynamic_plugin_manifest, read_bounded_utf8_regular_file};
use crate::policy::{DynamicPluginHostPolicy, FileDynamicPluginHostPolicy};
use crate::state::pin_plugin_config_path;

/// Source selection used by file-backed plugin-host initialization.
#[derive(Debug, Clone)]
pub struct PluginFileResolveOptions {
    /// Optional file that replaces the ambient user-level `plugins.toml` layer.
    pub plugin_config_path: Option<PathBuf>,
    /// Directory used for nearest-project discovery. `None` suppresses project discovery.
    pub current_dir: Option<PathBuf>,
    /// Ambient user configuration directory.
    pub user_config_dir: Option<PathBuf>,
    /// System-level plugin configuration path.
    pub system_config_path: PathBuf,
}

impl PluginFileResolveOptions {
    /// Builds source selection from the current process environment.
    pub fn from_environment(plugin_config_path: Option<PathBuf>) -> Self {
        let user_only = std::env::var("NEMO_RELAY_CONFIG_SCOPE").ok().as_deref() == Some("user");
        Self {
            plugin_config_path,
            current_dir: (!user_only)
                .then(std::env::current_dir)
                .transpose()
                .ok()
                .flatten(),
            user_config_dir: user_config_dir(),
            system_config_path: PathBuf::from("/etc/nemo-relay/plugins.toml"),
        }
    }

    /// Returns selected source paths in increasing precedence order.
    pub fn selected_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(selected) = &self.plugin_config_path {
            paths.push(selected.clone());
        } else if let Some(user_dir) = &self.user_config_dir {
            paths.push(user_dir.join("plugins.toml"));
        }
        if let Some(current_dir) = self.current_dir.as_deref() {
            let mut implicit = default_plugin_config_paths(Some(current_dir), None);
            implicit.pop();
            paths.extend(implicit);
        }
        paths.push(self.system_config_path.clone());
        paths
    }
}

/// One dynamic declaration resolved from a physical `plugins.toml` source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDynamicPluginConfig {
    /// Canonical plugin ID derived from the referenced manifest.
    pub plugin_id: String,
    /// Canonical absolute manifest reference.
    pub manifest_ref: String,
    /// Component configuration supplied by the declaration.
    pub config: Map<String, Value>,
    /// Whether the declaration explicitly included `config`.
    pub has_explicit_config: bool,
    /// Physical `plugins.toml` containing the declaration.
    pub source: PathBuf,
}

/// Fully resolved static and dynamic file-backed plugin configuration.
#[derive(Debug, Clone)]
pub struct ResolvedPluginFileConfiguration {
    /// Effective static Relay plugin configuration, including the caller overlay.
    pub config: PluginConfig,
    /// Effective JSON document before conversion to [`PluginConfig`].
    pub runtime_value: Option<Value>,
    /// Dynamic declarations in source and declaration order.
    pub dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    /// Fully layered dynamic-plugin host policy.
    pub dynamic_plugin_policy: DynamicPluginHostPolicy,
    /// Redacted inherited-source diagnostics.
    pub diagnostics: Vec<ConfigDiagnostic>,
    /// Existing physical sources that contributed static, dynamic, or policy configuration.
    pub contributing_sources: Vec<PathBuf>,
    /// Selected source spellings corresponding to [`Self::contributing_sources`].
    ///
    /// These are captured during the same physical-path resolution pass so presentation adapters
    /// do not need to re-resolve aliases after configuration has been consumed.
    #[doc(hidden)]
    pub contributing_selected_sources: Vec<PathBuf>,
    /// Selected source paths, including absent files whose sibling lifecycle state may remain.
    #[doc(hidden)]
    pub selected_sources: Vec<PathBuf>,
    /// Whether any physical source or caller-supplied configuration participated.
    pub had_input: bool,
}

/// Resolves standard plugin file discovery and an optional caller static overlay.
pub fn resolve_plugin_files(
    caller_config: Option<PluginConfig>,
    options: PluginFileResolveOptions,
) -> Result<ResolvedPluginFileConfiguration> {
    resolve_plugin_files_from_paths(options.selected_paths(), caller_config)
}

/// Resolves a supplied low-to-high precedence list of physical plugin files.
#[doc(hidden)]
pub fn resolve_plugin_files_from_paths<I>(
    paths: I,
    caller_config: Option<PluginConfig>,
) -> Result<ResolvedPluginFileConfiguration>
where
    I: IntoIterator<Item = PathBuf>,
{
    let paths = paths.into_iter().collect::<Vec<_>>();
    let mut seen_selected_sources = HashSet::new();
    let mut selected_source_mappings = Vec::with_capacity(paths.len());
    for selected_path in paths.into_iter().rev() {
        let physical_path = pin_plugin_config_path(&selected_path)?;
        if seen_selected_sources.insert(physical_path.clone()) {
            selected_source_mappings.push((selected_path, physical_path));
        }
    }
    selected_source_mappings.reverse();
    let selected_sources = selected_source_mappings
        .iter()
        .map(|(_, physical_path)| physical_path.clone())
        .collect::<Vec<_>>();
    let mut dynamic_plugins = Vec::new();
    let mut dynamic_plugin_policy = DynamicPluginHostPolicy::default();
    let mut seen_plugin_ids = HashSet::new();
    let mut contributing_sources = Vec::new();
    let mut contributing_selected_sources = Vec::new();
    let mut runtime_documents = Vec::new();
    let mut enabled_sources = HashMap::new();
    let mut seen_physical_sources = HashSet::new();

    for (selected_path, physical_path) in &selected_source_mappings {
        if !physical_path.try_exists().map_err(|error| {
            PluginHostConfigError::InvalidConfig(format!(
                "failed to inspect plugin configuration file {}: {error}",
                physical_path.display()
            ))
        })? {
            continue;
        }
        let path = physical_path.clone();
        if !seen_physical_sources.insert(path.clone()) {
            continue;
        }
        let raw = read_bounded_utf8_regular_file(&path, "plugin configuration file")?;
        let mut parsed = raw
            .parse::<toml::Table>()
            .map(toml::Value::Table)
            .map_err(|error| PluginHostConfigError::toml_parse("plugin TOML", &path, &error))?;
        contributing_sources.push(path.clone());
        contributing_selected_sources.push(selected_path.clone());
        let resolved = resolve_dynamic_plugin_refs(&path, &mut parsed, &mut seen_plugin_ids)?;
        dynamic_plugins.extend(resolved.dynamic_plugins);
        dynamic_plugin_policy.merge_from(resolved.dynamic_plugin_policy);
        let runtime_value = serde_json::to_value(remove_dynamic_plugin_sections(parsed))?;
        record_enabled_sources(&path, &runtime_value, &mut enabled_sources);
        runtime_documents.push((path, runtime_value));
    }

    let (mut runtime_value, _) = merge_plugin_config_documents(runtime_documents)
        .map_err(|error| {
            PluginHostConfigError::InvalidConfig(format!(
                "failed to merge static plugin configuration: {}",
                crate::error::sanitize_parser_reason(&error.to_string())
            ))
        })?
        .unwrap_or_else(|| (Value::Object(Map::new()), Vec::new()));
    let had_caller_config = caller_config.is_some();
    if let Some(caller_config) = caller_config.as_ref() {
        let diagnostics = programmatic_enable_override_diagnostics(
            &runtime_value,
            &enabled_sources,
            caller_config,
        );
        layer_config(
            &mut runtime_value,
            plugin_config_overlay_value(caller_config)?,
        );
        let mut inherited = inherited_source_diagnostics(&contributing_sources);
        inherited.extend(diagnostics);
        return finish_resolution(
            runtime_value,
            dynamic_plugins,
            dynamic_plugin_policy,
            inherited,
            ResolvedPluginFileSourcePaths {
                contributing_sources,
                contributing_selected_sources,
                selected_sources,
            },
            had_caller_config,
        );
    }

    let diagnostics = inherited_source_diagnostics(&contributing_sources);
    finish_resolution(
        runtime_value,
        dynamic_plugins,
        dynamic_plugin_policy,
        diagnostics,
        ResolvedPluginFileSourcePaths {
            contributing_sources,
            contributing_selected_sources,
            selected_sources,
        },
        had_caller_config,
    )
}

struct ResolvedPluginFileSourcePaths {
    contributing_sources: Vec<PathBuf>,
    contributing_selected_sources: Vec<PathBuf>,
    selected_sources: Vec<PathBuf>,
}

fn finish_resolution(
    runtime_value: Value,
    dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    dynamic_plugin_policy: DynamicPluginHostPolicy,
    diagnostics: Vec<ConfigDiagnostic>,
    source_paths: ResolvedPluginFileSourcePaths,
    had_caller_config: bool,
) -> Result<ResolvedPluginFileConfiguration> {
    let ResolvedPluginFileSourcePaths {
        contributing_sources,
        contributing_selected_sources,
        selected_sources,
    } = source_paths;
    let had_input = had_caller_config || !contributing_sources.is_empty();
    let serialized_value = match &runtime_value {
        Value::Object(object) if object.is_empty() => None,
        _ => Some(runtime_value.clone()),
    };
    let config = serde_json::from_value(runtime_value).map_err(|error| {
        PluginHostConfigError::InvalidConfig(format!(
            "resolved static plugin configuration is invalid: {}",
            crate::error::sanitize_parser_reason(&error.to_string())
        ))
    })?;
    Ok(ResolvedPluginFileConfiguration {
        config,
        runtime_value: serialized_value,
        dynamic_plugins,
        dynamic_plugin_policy,
        diagnostics,
        contributing_sources,
        contributing_selected_sources,
        selected_sources,
        had_input,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PluginTomlPluginsSection {
    #[serde(default)]
    dynamic: Vec<FileDynamicPluginConfig>,
    #[serde(default)]
    policy: Option<FileDynamicPluginHostPolicy>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDynamicPluginConfig {
    manifest: String,
    #[serde(default)]
    config: Option<Map<String, Value>>,
}

struct ResolvedDynamicPluginRefs {
    dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    dynamic_plugin_policy: DynamicPluginHostPolicy,
}

fn resolve_dynamic_plugin_refs(
    source: &Path,
    value: &mut toml::Value,
    seen_plugin_ids: &mut HashSet<String>,
) -> Result<ResolvedDynamicPluginRefs> {
    let Some(root) = value.as_table_mut() else {
        return Ok(ResolvedDynamicPluginRefs {
            dynamic_plugins: Vec::new(),
            dynamic_plugin_policy: DynamicPluginHostPolicy::default(),
        });
    };
    let Some(plugins_value) = root.get("plugins").cloned() else {
        return Ok(ResolvedDynamicPluginRefs {
            dynamic_plugins: Vec::new(),
            dynamic_plugin_policy: DynamicPluginHostPolicy::default(),
        });
    };
    let plugins: PluginTomlPluginsSection = plugins_value.try_into().map_err(|error| {
        PluginHostConfigError::toml_parse("dynamic plugin config", source, &error)
    })?;
    let mut resolved = Vec::with_capacity(plugins.dynamic.len());
    for dynamic in plugins.dynamic {
        let manifest_path = resolve_dynamic_manifest_path(source, &dynamic.manifest);
        let (manifest, manifest_ref) = load_bounded_dynamic_plugin_manifest(&manifest_path)
            .map_err(|error| contextualize_manifest_error(source, error))?;
        let plugin_id = manifest.plugin.id.trim().to_owned();
        if !seen_plugin_ids.insert(plugin_id.clone()) {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "duplicate dynamic plugin id '{}' in {} across plugins.toml sources",
                plugin_id,
                source.display()
            )));
        }
        resolved.push(ResolvedDynamicPluginConfig {
            plugin_id,
            manifest_ref,
            has_explicit_config: dynamic.config.is_some(),
            config: dynamic.config.unwrap_or_default(),
            source: source.to_path_buf(),
        });
    }
    Ok(ResolvedDynamicPluginRefs {
        dynamic_plugins: resolved,
        dynamic_plugin_policy: plugins.policy.map(Into::into).unwrap_or_default(),
    })
}

fn contextualize_manifest_error(
    source: &Path,
    error: PluginHostConfigError,
) -> PluginHostConfigError {
    match error {
        PluginHostConfigError::NotFound { .. } => error,
        other => PluginHostConfigError::InvalidConfig(format!(
            "invalid dynamic plugin manifest referenced by {}: {other}",
            source.display()
        )),
    }
}

fn resolve_dynamic_manifest_path(source: &Path, manifest: &str) -> PathBuf {
    let manifest = PathBuf::from(manifest);
    if manifest.is_absolute() {
        manifest
    } else {
        source
            .parent()
            .map(|parent| parent.join(&manifest))
            .unwrap_or(manifest)
    }
}

fn remove_dynamic_plugin_sections(mut value: toml::Value) -> toml::Value {
    if let Some(root) = value.as_table_mut()
        && let Some(toml::Value::Table(plugins)) = root.get_mut("plugins")
    {
        plugins.remove("dynamic");
        plugins.remove("policy");
        if plugins.is_empty() {
            root.remove("plugins");
        }
    }
    value
}

fn inherited_source_diagnostics(sources: &[PathBuf]) -> Vec<ConfigDiagnostic> {
    sources
        .iter()
        .map(|source| {
            let source = source.display().to_string();
            log::warn!(
                target: "nemo_relay.plugin",
                event = "plugin_configuration_inherited",
                config_path = source.as_str();
                "Inherited plugin configuration from discovered file"
            );
            ConfigDiagnostic {
                level: DiagnosticLevel::Warning,
                code: "plugin.configuration_inherited".to_owned(),
                component: None,
                field: None,
                message: format!("inherited plugin configuration from discovered file: {source}"),
            }
        })
        .collect()
}

#[derive(Clone)]
struct ComponentEnabledSource {
    enabled: bool,
    path: PathBuf,
}

fn record_enabled_sources(
    path: &Path,
    document: &Value,
    sources: &mut HashMap<String, ComponentEnabledSource>,
) {
    let Some(components) = document.get("components").and_then(Value::as_array) else {
        return;
    };
    for component in components {
        let Some(kind) = component_kind(component) else {
            continue;
        };
        if let Some(enabled) = component.get("enabled").and_then(Value::as_bool) {
            sources.insert(
                kind.to_owned(),
                ComponentEnabledSource {
                    enabled,
                    path: path.to_path_buf(),
                },
            );
        }
    }
}

fn programmatic_enable_override_diagnostics(
    discovered: &Value,
    enabled_sources: &HashMap<String, ComponentEnabledSource>,
    programmatic: &PluginConfig,
) -> Vec<ConfigDiagnostic> {
    let Some(discovered_components) = discovered.get("components").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut consumed = HashMap::new();
    let mut diagnostics = Vec::new();
    for component in &programmatic.components {
        let nth = consumed.entry(component.kind.as_str()).or_insert(0usize);
        let discovered_component =
            nth_component_by_kind(discovered_components, &component.kind, *nth)
                .and_then(|index| discovered_components.get(index));
        *nth += 1;
        let discovered_enabled = discovered_component
            .and_then(|component| component.get("enabled"))
            .and_then(Value::as_bool);
        let file_disabled = discovered_enabled == Some(false)
            || (discovered_enabled.is_none()
                && enabled_sources
                    .get(&component.kind)
                    .is_some_and(|source| !source.enabled));
        if !component.enabled || !file_disabled {
            continue;
        }
        let source = enabled_sources
            .get(&component.kind)
            .map(|source| format!(" from {}", source.path.display()))
            .unwrap_or_default();
        diagnostics.push(ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "plugin.component_reenabled".to_owned(),
            component: Some(component.kind.clone()),
            field: Some("enabled".to_owned()),
            message: format!(
                "programmatic configuration enabled plugin component '{}' and overrode enabled = false{source}",
                component.kind
            ),
        });
    }
    diagnostics
}

fn plugin_config_overlay_value(config: &PluginConfig) -> Result<Value> {
    let mut overlay = serde_json::to_value(config)?;
    let Value::Object(root) = &mut overlay else {
        return Ok(overlay);
    };
    if config.version == PluginConfig::default().version {
        root.remove("version");
    }
    remove_default_policy_overlay(root, &config.policy);
    Ok(overlay)
}

fn remove_default_policy_overlay(root: &mut Map<String, Value>, config: &ConfigPolicy) {
    let Some(Value::Object(policy)) = root.get_mut("policy") else {
        return;
    };
    let defaults = ConfigPolicy::default();
    for (field, is_default) in [
        (
            "unknown_component",
            config.unknown_component == defaults.unknown_component,
        ),
        (
            "unknown_field",
            config.unknown_field == defaults.unknown_field,
        ),
        (
            "unsupported_value",
            config.unsupported_value == defaults.unsupported_value,
        ),
    ] {
        if is_default {
            policy.remove(field);
        }
    }
    if policy.is_empty() {
        root.remove("policy");
    }
}

fn layer_config(left: &mut Value, right: Value) {
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            for (key, value) in right {
                match (key.as_str(), left.get_mut(&key)) {
                    ("components", Some(existing)) => merge_plugin_components(existing, value),
                    (_, Some(existing)) => merge_json_value(existing, value),
                    (_, None) => {
                        left.insert(key, value);
                    }
                }
            }
        }
        (left, right) => *left = right,
    }
}

fn merge_plugin_components(left: &mut Value, right: Value) {
    let Value::Array(left_components) = left else {
        *left = right;
        return;
    };
    let Value::Array(right_components) = right else {
        *left = right;
        return;
    };
    let base_component_count = left_components.len();
    let mut consumed = HashMap::new();
    for component in right_components {
        let Some(kind) = component_kind(&component).map(str::to_owned) else {
            left_components.push(component);
            continue;
        };
        let nth = consumed.entry(kind.clone()).or_insert(0usize);
        let slot = nth_component_by_kind(&left_components[..base_component_count], &kind, *nth);
        *nth += 1;
        match slot {
            Some(index) => merge_plugin_component(&mut left_components[index], component),
            None => left_components.push(component),
        }
    }
}

fn merge_plugin_component(existing: &mut Value, higher_priority: Value) {
    let is_observability = component_kind(&higher_priority).or_else(|| component_kind(existing))
        == Some("observability");
    match (existing, higher_priority) {
        (Value::Object(existing), Value::Object(higher_priority)) => {
            for (key, value) in higher_priority {
                match (key.as_str(), existing.get_mut(&key)) {
                    ("config", Some(existing_config)) => merge_plugin_config_value(
                        existing_config,
                        value,
                        &mut Vec::new(),
                        is_observability,
                    ),
                    (_, Some(existing_value)) => merge_json_value(existing_value, value),
                    (_, None) => {
                        existing.insert(key, value);
                    }
                }
            }
        }
        (existing, higher_priority) => *existing = higher_priority,
    }
}

fn merge_plugin_config_value(
    lower_priority: &mut Value,
    higher_priority: Value,
    path: &mut Vec<String>,
    is_observability: bool,
) {
    match (lower_priority, higher_priority) {
        (Value::Object(lower_priority), Value::Object(higher_priority)) => {
            for (key, value) in higher_priority {
                path.push(key.clone());
                match lower_priority.get_mut(&key) {
                    Some(existing) => {
                        merge_plugin_config_value(existing, value, path, is_observability)
                    }
                    None => {
                        lower_priority.insert(key, value);
                    }
                }
                path.pop();
            }
        }
        (Value::Array(lower_priority), Value::Array(mut higher_priority))
            if plugin_config_list_concatenates(path, is_observability) =>
        {
            higher_priority.append(lower_priority);
            *lower_priority = higher_priority;
        }
        (lower_priority, higher_priority) => *lower_priority = higher_priority,
    }
}

fn plugin_config_list_concatenates(path: &[String], is_observability: bool) -> bool {
    path.len() == 1
        || (is_observability
            && matches!(
                path,
                [section, field]
                    if (section == "atof" && field == "sinks")
                        || (section == "opentelemetry" && field == "endpoints")
                        || (section == "atif" && field == "storage")
            ))
}

fn merge_json_value(left: &mut Value, right: Value) {
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            for (key, value) in right {
                match left.get_mut(&key) {
                    Some(existing) => merge_json_value(existing, value),
                    None => {
                        left.insert(key, value);
                    }
                }
            }
        }
        (left, right) => *left = right,
    }
}

fn component_kind(value: &Value) -> Option<&str> {
    value.get("kind").and_then(Value::as_str)
}

fn nth_component_by_kind(components: &[Value], kind: &str, nth: usize) -> Option<usize> {
    components
        .iter()
        .enumerate()
        .filter(|(_, component)| component_kind(component) == Some(kind))
        .nth(nth)
        .map(|(index, _)| index)
}

#[cfg(test)]
#[path = "../tests/unit/resolver.rs"]
mod tests;
