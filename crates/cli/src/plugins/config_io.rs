// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable plugin configuration file and validation helpers.

use std::path::{Path, PathBuf};

use console::style;
use nemo_relay::plugin::{ConfigPolicy, PluginConfig, validate_plugin_config};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::configuration::{
    global_plugin_config_path, project_plugin_config_path, user_plugin_config_path,
};
use crate::error::CliError;
use crate::plugins::ConfigurationScope;
use crate::server::register_and_validate_plugin_components;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetScope {
    User,
    Project,
    Global,
}

/// A physical `plugins.toml` document together with its typed runtime plugin config.
///
/// The raw TOML remains the source of truth for host-only sections such as
/// `[[plugins.dynamic]]`. Rendering patches the typed runtime fields back into that raw document
/// instead of reconstructing the whole file from [`PluginConfig`].
#[derive(Debug, Clone)]
pub(crate) struct PluginConfigDocument {
    path: PathBuf,
    root: toml::Value,
    config: PluginConfig,
}

/// One dynamic plugin reference declared in the physical document.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DynamicPluginConfigEntry {
    /// Original index in `plugins.dynamic`; suitable for updating this document clone.
    pub(crate) index: usize,
    /// Manifest reference exactly as declared in TOML.
    pub(crate) manifest: String,
    /// Manifest path resolved relative to the physical `plugins.toml` file.
    pub(crate) manifest_path: PathBuf,
    /// `None` distinguishes an omitted config from an explicitly empty object.
    pub(crate) config: Option<Map<String, Value>>,
}

impl PluginConfigDocument {
    pub(crate) fn read(path: &Path) -> Result<Self, CliError> {
        let root = read_plugin_toml_root(path)?;
        let config = plugin_config_from_toml(&root)?;
        Ok(Self {
            path: path.to_path_buf(),
            root,
            config,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn config(&self) -> &PluginConfig {
        &self.config
    }

    pub(crate) fn config_mut(&mut self) -> &mut PluginConfig {
        &mut self.config
    }

    pub(crate) fn set_config(&mut self, config: PluginConfig) {
        self.config = config;
    }

    pub(crate) fn dynamic_entries(&self) -> Result<Vec<DynamicPluginConfigEntry>, CliError> {
        let Some(dynamic) = dynamic_plugin_entries(&self.root, &self.path)? else {
            return Ok(Vec::new());
        };

        dynamic
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let entry = entry.as_table().ok_or_else(|| {
                    invalid_dynamic_entry_error(&self.path, index, "must be a table")
                })?;
                let manifest = entry
                    .get("manifest")
                    .and_then(toml::Value::as_str)
                    .ok_or_else(|| {
                        invalid_dynamic_entry_error(&self.path, index, "manifest must be a string")
                    })?
                    .to_owned();
                let config = entry
                    .get("config")
                    .map(|config| dynamic_config_from_toml(config, &self.path, index))
                    .transpose()?;
                Ok(DynamicPluginConfigEntry {
                    index,
                    manifest_path: resolve_manifest_ref(&self.path, &manifest),
                    manifest,
                    config,
                })
            })
            .collect()
    }

    /// Set a dynamic plugin config, including an explicitly empty config object.
    pub(crate) fn set_dynamic_config(
        &mut self,
        index: usize,
        config: Map<String, Value>,
    ) -> Result<(), CliError> {
        let entry = self.dynamic_entry_mut(index)?;
        let config = toml::Value::try_from(Value::Object(config)).map_err(|error| {
            CliError::Config(format!(
                "could not convert dynamic plugin config to TOML: {error}"
            ))
        })?;
        entry.insert("config".to_owned(), config);
        Ok(())
    }

    /// Remove a dynamic plugin config while retaining its manifest reference.
    pub(crate) fn remove_dynamic_config(&mut self, index: usize) -> Result<(), CliError> {
        self.dynamic_entry_mut(index)?.remove("config");
        Ok(())
    }

    /// Patches a dynamic config relative to the JSON view that was originally loaded.
    ///
    /// Unchanged TOML values remain in the raw document. This matters for host extensions that
    /// use TOML-native values, such as datetimes, which do not have an equivalent JSON type.
    pub(crate) fn patch_dynamic_config(
        &mut self,
        index: usize,
        original: Option<&Map<String, Value>>,
        updated: Option<Map<String, Value>>,
    ) -> Result<(), CliError> {
        match (original, updated) {
            (_, None) => self.remove_dynamic_config(index),
            (None, Some(updated)) => self.set_dynamic_config(index, updated),
            (Some(original), Some(updated)) => {
                let entry = self.dynamic_entry_mut(index)?;
                let Some(raw) = entry.get_mut("config") else {
                    let updated = json_to_toml(Value::Object(updated))?;
                    entry.insert("config".to_owned(), updated);
                    return Ok(());
                };
                patch_json_value(
                    raw,
                    &Value::Object(original.clone()),
                    &Value::Object(updated),
                )
            }
        }
    }

    /// Render a full document after overlaying the typed runtime config onto the raw TOML.
    pub(crate) fn render(&self) -> Result<String, CliError> {
        let mut root = self.root.clone();
        patch_plugin_config(&mut root, &self.config)?;
        toml::to_string_pretty(&root)
            .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))
    }

    pub(crate) fn write(&self) -> Result<(), CliError> {
        let rendered = self.render()?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, rendered)?;
        Ok(())
    }

    fn dynamic_entry_mut(
        &mut self,
        index: usize,
    ) -> Result<&mut toml::map::Map<String, toml::Value>, CliError> {
        let dynamic = dynamic_plugin_entries_mut(&mut self.root, &self.path)?.ok_or_else(|| {
            CliError::Config(format!(
                "dynamic plugin index {index} is out of range in {}",
                self.path.display()
            ))
        })?;
        let entry = dynamic.get_mut(index).ok_or_else(|| {
            CliError::Config(format!(
                "dynamic plugin index {index} is out of range in {}",
                self.path.display()
            ))
        })?;
        entry
            .as_table_mut()
            .ok_or_else(|| invalid_dynamic_entry_error(&self.path, index, "must be a table"))
    }
}

fn patch_json_value(
    raw: &mut toml::Value,
    original: &Value,
    updated: &Value,
) -> Result<(), CliError> {
    if original == updated {
        return Ok(());
    }
    match (raw, original, updated) {
        (toml::Value::Table(raw), Value::Object(original), Value::Object(updated)) => {
            patch_json_object(raw, original, updated)
        }
        (toml::Value::Array(raw), Value::Array(original), Value::Array(updated))
            if raw.len() == original.len() =>
        {
            patch_json_array(raw, original, updated)
        }
        (raw, _, updated) => {
            *raw = json_to_toml(updated.clone())?;
            Ok(())
        }
    }
}

fn patch_json_object(
    raw: &mut toml::map::Map<String, toml::Value>,
    original: &Map<String, Value>,
    updated: &Map<String, Value>,
) -> Result<(), CliError> {
    for key in original.keys() {
        if !updated.contains_key(key) {
            raw.remove(key);
        }
    }
    for (key, updated) in updated {
        match (raw.get_mut(key), original.get(key)) {
            (Some(raw), Some(original)) => patch_json_value(raw, original, updated)?,
            _ => {
                raw.insert(key.clone(), json_to_toml(updated.clone())?);
            }
        }
    }
    Ok(())
}

fn patch_json_array(
    raw: &mut Vec<toml::Value>,
    original: &[Value],
    updated: &[Value],
) -> Result<(), CliError> {
    let shared_len = original.len().min(updated.len());
    for index in 0..shared_len {
        patch_json_value(&mut raw[index], &original[index], &updated[index])?;
    }
    raw.truncate(updated.len());
    for value in &updated[shared_len..] {
        raw.push(json_to_toml(value.clone())?);
    }
    Ok(())
}

fn json_to_toml(value: Value) -> Result<toml::Value, CliError> {
    toml::Value::try_from(value).map_err(|error| {
        CliError::Config(format!(
            "could not convert dynamic plugin config value to TOML: {error}"
        ))
    })
}

pub(crate) fn target_scope(command: &ConfigurationScope) -> Result<TargetScope, CliError> {
    match command {
        ConfigurationScope::Default | ConfigurationScope::User => Ok(TargetScope::User),
        ConfigurationScope::Project => Ok(TargetScope::Project),
        ConfigurationScope::Global => Ok(TargetScope::Global),
        ConfigurationScope::Invalid => Err(CliError::Config(
            "choose only one of --user, --project, or --global".into(),
        )),
    }
}

#[derive(Debug, Clone, Serialize)]
struct DynamicPluginReferenceEntry {
    manifest: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    config: Map<String, Value>,
}

pub(crate) fn target_path(scope: TargetScope) -> Result<PathBuf, CliError> {
    match scope {
        TargetScope::User => user_plugin_config_path().ok_or_else(|| {
            CliError::Config(
                "cannot determine user config directory; set HOME or XDG_CONFIG_HOME".into(),
            )
        }),
        TargetScope::Project => {
            let cwd = std::env::current_dir()?;
            Ok(project_plugin_config_path(&cwd))
        }
        TargetScope::Global => Ok(global_plugin_config_path()),
    }
}

#[cfg(test)]
pub(crate) fn read_plugin_config(path: &Path) -> Result<PluginConfig, CliError> {
    Ok(PluginConfigDocument::read(path)?.config)
}

fn plugin_config_from_toml(parsed: &toml::Value) -> Result<PluginConfig, CliError> {
    serde_json::from_value(
        serde_json::to_value(parsed)
            .map_err(|error| CliError::Config(format!("invalid plugin TOML shape: {error}")))?,
    )
    .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))
}

#[cfg(test)]
pub(crate) fn write_plugin_config(path: &Path, config: &PluginConfig) -> Result<(), CliError> {
    let mut document = PluginConfigDocument::read(path)?;
    document.set_config(config.clone());
    document.write()
}

pub(crate) fn append_dynamic_plugin_reference(
    path: &Path,
    manifest_ref: &str,
) -> Result<(), CliError> {
    let mut root = read_plugin_toml_root(path)?;

    let root_table = root
        .as_table_mut()
        .expect("root plugin TOML is always a table");
    let plugins = root_table
        .entry("plugins")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()))
        .as_table_mut()
        .ok_or_else(|| {
            CliError::Config(format!(
                "invalid plugin TOML in {}: [plugins] must be a table",
                path.display()
            ))
        })?;
    let dynamic = plugins
        .entry("dynamic")
        .or_insert_with(|| toml::Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| {
            CliError::Config(format!(
                "invalid plugin TOML in {}: plugins.dynamic must be an array of tables",
                path.display()
            ))
        })?;
    dynamic.push(
        toml::Value::try_from(DynamicPluginReferenceEntry {
            manifest: manifest_ref.to_owned(),
            config: Map::new(),
        })
        .map_err(|error| {
            CliError::Config(format!(
                "could not serialize dynamic plugin reference for {}: {error}",
                path.display()
            ))
        })?,
    );

    write_plugin_toml_root(path, &root)?;
    Ok(())
}

pub(crate) fn remove_dynamic_plugin_reference(
    path: &Path,
    plugin_id: &str,
    target_manifest_ref: Option<&str>,
) -> Result<bool, CliError> {
    if !path.exists() {
        return Ok(false);
    }

    let mut root = read_plugin_toml_root(path)?;
    let Some(root_table) = root.as_table_mut() else {
        return Ok(false);
    };
    let Some(plugins_value) = root_table.get_mut("plugins") else {
        return Ok(false);
    };
    let plugins = plugins_value.as_table_mut().ok_or_else(|| {
        CliError::Config(format!(
            "invalid plugin TOML in {}: [plugins] must be a table",
            path.display()
        ))
    })?;
    let Some(dynamic_value) = plugins.get_mut("dynamic") else {
        return Ok(false);
    };
    let dynamic_entries = dynamic_value.as_array_mut().ok_or_else(|| {
        CliError::Config(format!(
            "invalid plugin TOML in {}: plugins.dynamic must be an array of tables",
            path.display()
        ))
    })?;

    let original_len = dynamic_entries.len();
    let mut retained = Vec::with_capacity(original_len);
    let target_manifest_ref =
        target_manifest_ref.map(|manifest_ref| resolve_manifest_ref(path, manifest_ref));
    for entry in dynamic_entries.drain(..) {
        let manifest_ref = entry
            .as_table()
            .and_then(|entry| entry.get("manifest"))
            .and_then(toml::Value::as_str)
            .map(|manifest| resolve_manifest_ref(path, manifest));

        let remove = manifest_ref.as_ref().is_some_and(|manifest_ref| {
            target_manifest_ref
                .as_ref()
                .is_some_and(|target_manifest_ref| manifest_ref == target_manifest_ref)
                || crate::configuration::load_bounded_dynamic_plugin_manifest(manifest_ref)
                    .map(|(manifest, _)| manifest.plugin.id.trim() == plugin_id)
                    .unwrap_or(false)
        });

        if !remove {
            retained.push(entry);
        }
    }

    let removed = retained.len() != original_len;
    *dynamic_entries = retained;
    if dynamic_entries.is_empty() {
        plugins.remove("dynamic");
    }
    if plugins.is_empty() {
        root_table.remove("plugins");
    }
    if removed {
        write_plugin_toml_root(path, &root)?;
    }
    Ok(removed)
}

fn read_plugin_toml_root(path: &Path) -> Result<toml::Value, CliError> {
    if path.exists() {
        let raw = std::fs::read_to_string(path)?;
        raw.parse::<toml::Table>()
            .map(toml::Value::Table)
            .map_err(|error| {
                CliError::Config(format!(
                    "invalid plugin TOML in {}: {error}",
                    path.display()
                ))
            })
    } else {
        Ok(toml::Value::Table(toml::map::Map::new()))
    }
}

fn write_plugin_toml_root(path: &Path, root: &toml::Value) -> Result<(), CliError> {
    let rendered = toml::to_string_pretty(root)
        .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, rendered)?;
    Ok(())
}

fn dynamic_plugin_entries<'a>(
    root: &'a toml::Value,
    path: &Path,
) -> Result<Option<&'a Vec<toml::Value>>, CliError> {
    let Some(plugins) = root.as_table().and_then(|root| root.get("plugins")) else {
        return Ok(None);
    };
    let plugins = plugins.as_table().ok_or_else(|| {
        CliError::Config(format!(
            "invalid plugin TOML in {}: [plugins] must be a table",
            path.display()
        ))
    })?;
    let Some(dynamic) = plugins.get("dynamic") else {
        return Ok(None);
    };
    dynamic.as_array().map(Some).ok_or_else(|| {
        CliError::Config(format!(
            "invalid plugin TOML in {}: plugins.dynamic must be an array of tables",
            path.display()
        ))
    })
}

fn dynamic_plugin_entries_mut<'a>(
    root: &'a mut toml::Value,
    path: &Path,
) -> Result<Option<&'a mut Vec<toml::Value>>, CliError> {
    let Some(plugins) = root.as_table_mut().and_then(|root| root.get_mut("plugins")) else {
        return Ok(None);
    };
    let plugins = plugins.as_table_mut().ok_or_else(|| {
        CliError::Config(format!(
            "invalid plugin TOML in {}: [plugins] must be a table",
            path.display()
        ))
    })?;
    let Some(dynamic) = plugins.get_mut("dynamic") else {
        return Ok(None);
    };
    dynamic.as_array_mut().map(Some).ok_or_else(|| {
        CliError::Config(format!(
            "invalid plugin TOML in {}: plugins.dynamic must be an array of tables",
            path.display()
        ))
    })
}

fn dynamic_config_from_toml(
    config: &toml::Value,
    path: &Path,
    index: usize,
) -> Result<Map<String, Value>, CliError> {
    if !config.is_table() {
        return Err(invalid_dynamic_entry_error(
            path,
            index,
            "config must be a table",
        ));
    }
    let config = serde_json::to_value(config).map_err(|error| {
        invalid_dynamic_entry_error(path, index, &format!("config is invalid: {error}"))
    })?;
    config
        .as_object()
        .cloned()
        .ok_or_else(|| invalid_dynamic_entry_error(path, index, "config must be a table"))
}

fn invalid_dynamic_entry_error(path: &Path, index: usize, message: &str) -> CliError {
    CliError::Config(format!(
        "invalid plugin TOML in {}: plugins.dynamic[{index}] {message}",
        path.display()
    ))
}

fn patch_plugin_config(root: &mut toml::Value, config: &PluginConfig) -> Result<(), CliError> {
    let desired = pruned_plugin_config_toml(config)?;
    let desired = desired
        .as_table()
        .expect("serialized plugin config is always a table");
    let root = root
        .as_table_mut()
        .expect("root plugin TOML is always a table");

    patch_required_field(root, desired, "version");
    patch_policy(root, desired);
    patch_components(root, desired);
    Ok(())
}

fn pruned_plugin_config_toml(config: &PluginConfig) -> Result<toml::Value, CliError> {
    let mut value = serde_json::to_value(config)
        .map_err(|error| CliError::Config(format!("could not serialize plugin config: {error}")))?;
    prune_plugin_defaults(&mut value);
    serde_json::from_value(value).map_err(|error| {
        CliError::Config(format!("could not convert plugin config to TOML: {error}"))
    })
}

fn patch_required_field(
    root: &mut toml::map::Map<String, toml::Value>,
    desired: &toml::map::Map<String, toml::Value>,
    key: &str,
) {
    root.insert(
        key.to_owned(),
        desired
            .get(key)
            .expect("serialized plugin config contains required fields")
            .clone(),
    );
}

fn patch_policy(
    root: &mut toml::map::Map<String, toml::Value>,
    desired: &toml::map::Map<String, toml::Value>,
) {
    const POLICY_FIELDS: [&str; 3] = ["unknown_component", "unknown_field", "unsupported_value"];
    let desired_policy = desired.get("policy").and_then(toml::Value::as_table);
    let existing_policy = root.get_mut("policy").and_then(toml::Value::as_table_mut);

    match (existing_policy, desired_policy) {
        (Some(existing), desired) => {
            for key in POLICY_FIELDS {
                match desired.and_then(|policy| policy.get(key)) {
                    Some(value) => {
                        existing.insert(key.to_owned(), value.clone());
                    }
                    None => {
                        existing.remove(key);
                    }
                }
            }
            if existing.is_empty() {
                root.remove("policy");
            }
        }
        (None, Some(desired)) => {
            root.insert("policy".to_owned(), toml::Value::Table(desired.clone()));
        }
        (None, None) => {}
    }
}

fn patch_components(
    root: &mut toml::map::Map<String, toml::Value>,
    desired: &toml::map::Map<String, toml::Value>,
) {
    let desired = desired
        .get("components")
        .and_then(toml::Value::as_array)
        .expect("serialized plugin config components are an array");
    let existing = root
        .get("components")
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut consumed = vec![false; existing.len()];
    let mut patched = Vec::with_capacity(desired.len());

    for desired_component in desired {
        let kind = desired_component
            .as_table()
            .and_then(|component| component.get("kind"))
            .and_then(toml::Value::as_str);
        let matching_index = existing.iter().enumerate().position(|(index, component)| {
            !consumed[index]
                && component
                    .as_table()
                    .and_then(|component| component.get("kind"))
                    .and_then(toml::Value::as_str)
                    == kind
        });
        let Some(index) = matching_index else {
            patched.push(desired_component.clone());
            continue;
        };
        consumed[index] = true;
        let mut component = existing[index].clone();
        if let (Some(component), Some(desired_component)) =
            (component.as_table_mut(), desired_component.as_table())
        {
            for key in ["kind", "enabled", "config"] {
                match desired_component.get(key) {
                    Some(value) => {
                        component.insert(key.to_owned(), value.clone());
                    }
                    None => {
                        component.remove(key);
                    }
                }
            }
        }
        patched.push(component);
    }

    root.insert("components".to_owned(), toml::Value::Array(patched));
}

pub(crate) fn resolve_manifest_ref(source: &Path, manifest: &str) -> PathBuf {
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

#[cfg(test)]
pub(super) fn print_preview(config: &PluginConfig) -> Result<(), CliError> {
    print_rendered_preview(
        &toml::to_string_pretty(&pruned_plugin_config_toml(config)?)
            .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))?,
    )
}

pub(super) fn print_document_preview(document: &PluginConfigDocument) -> Result<(), CliError> {
    print_rendered_preview(&document.render()?)
}

fn print_rendered_preview(rendered: &str) -> Result<(), CliError> {
    println!();
    println!(
        "{} {}",
        style("❯").green(),
        style("plugins.toml preview").bold()
    );
    println!("{}", style("─".repeat(58)).black().bright());
    print!("{rendered}");
    println!("{}", style("─".repeat(58)).black().bright());
    Ok(())
}

pub(crate) fn validate_config(config: &PluginConfig) -> Result<(), CliError> {
    register_and_validate_plugin_components(config)
        .map_err(|error| CliError::Config(error.to_string()))?;
    let report = validate_plugin_config(config);
    if report.has_errors() {
        let messages = report
            .diagnostics
            .into_iter()
            .filter(|diagnostic| diagnostic.level == nemo_relay::plugin::DiagnosticLevel::Error)
            .map(|diagnostic| diagnostic.message)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(CliError::Config(format!(
            "plugin validation failed: {messages}"
        )));
    }
    Ok(())
}

pub(super) fn prune_plugin_defaults(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    remove_default_field(
        object,
        "policy",
        serde_json::to_value(ConfigPolicy::default()).expect("policy default serializes"),
    );
    if let Some(components) = object.get_mut("components").and_then(Value::as_array_mut) {
        for component in components {
            if let Some(component) = component.as_object_mut()
                && component.get("enabled") == Some(&Value::Bool(true))
            {
                component.remove("enabled");
            }
        }
    }
}

pub(super) fn remove_default_field(object: &mut Map<String, Value>, key: &str, default: Value) {
    let Some(value) = object.get_mut(key) else {
        return;
    };
    remove_matching_defaults(value, &default);
    if value == &default || value.as_object().is_some_and(|value| value.is_empty()) {
        object.remove(key);
    }
}

pub(super) fn remove_matching_defaults(value: &mut Value, default: &Value) {
    let (Some(value), Some(default)) = (value.as_object_mut(), default.as_object()) else {
        return;
    };
    let default_keys = default.keys().cloned().collect::<Vec<_>>();
    for key in default_keys {
        if value.get(&key) == default.get(&key) {
            value.remove(&key);
        }
    }
}
