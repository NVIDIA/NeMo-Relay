// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable plugin configuration file and validation helpers.

use std::path::{Path, PathBuf};

use console::style;
use nemo_relay::plugin::dynamic::DynamicPluginManifest;
use nemo_relay::plugin::{ConfigPolicy, PluginConfig, validate_plugin_config};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_pii_redaction::component::register_pii_redaction_component;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::{
    PluginsScopeArgs, global_plugin_config_path, project_plugin_config_path,
    user_plugin_config_path,
};
use crate::error::CliError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetScope {
    User,
    Project,
    Global,
}

pub(crate) fn target_scope(command: &PluginsScopeArgs) -> Result<TargetScope, CliError> {
    let selected = [command.user, command.project, command.global]
        .into_iter()
        .filter(|selected| *selected)
        .count();
    if selected > 1 {
        return Err(CliError::Config(
            "choose only one of --user, --project, or --global".into(),
        ));
    }
    if command.project {
        Ok(TargetScope::Project)
    } else if command.global {
        Ok(TargetScope::Global)
    } else {
        Ok(TargetScope::User)
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

pub(crate) fn read_plugin_config(path: &Path) -> Result<PluginConfig, CliError> {
    if !path.exists() {
        return Ok(PluginConfig::default());
    }
    let raw = std::fs::read_to_string(path)?;
    let parsed = raw
        .parse::<toml::Table>()
        .map(toml::Value::Table)
        .map_err(|error| {
            CliError::Config(format!(
                "invalid plugin TOML in {}: {error}",
                path.display()
            ))
        })?;
    serde_json::from_value(
        serde_json::to_value(parsed)
            .map_err(|error| CliError::Config(format!("invalid plugin TOML shape: {error}")))?,
    )
    .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))
}

pub(crate) fn write_plugin_config(path: &Path, config: &PluginConfig) -> Result<(), CliError> {
    let mut value = serde_json::to_value(config)
        .map_err(|error| CliError::Config(format!("could not serialize plugin config: {error}")))?;
    prune_plugin_defaults(&mut value);
    let toml_value: toml::Value = serde_json::from_value(value).map_err(|error| {
        CliError::Config(format!("could not convert plugin config to TOML: {error}"))
    })?;
    let rendered = toml::to_string_pretty(&toml_value)
        .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, rendered)?;
    Ok(())
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
    let target_manifest_ref = target_manifest_ref.map(PathBuf::from);
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
                || DynamicPluginManifest::load_from_path(manifest_ref)
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

fn resolve_manifest_ref(source: &Path, manifest: &str) -> PathBuf {
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

pub(super) fn print_preview(config: &PluginConfig) -> Result<(), CliError> {
    println!();
    println!(
        "{} {}",
        style("❯").green(),
        style("plugins.toml preview").bold()
    );
    println!("{}", style("─".repeat(58)).black().bright());
    let mut value = serde_json::to_value(config)
        .map_err(|error| CliError::Config(format!("could not serialize plugin config: {error}")))?;
    prune_plugin_defaults(&mut value);
    let toml_value: toml::Value = serde_json::from_value(value).map_err(|error| {
        CliError::Config(format!("could not convert plugin config to TOML: {error}"))
    })?;
    let rendered = toml::to_string_pretty(&toml_value)
        .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))?;
    print!("{rendered}");
    println!("{}", style("─".repeat(58)).black().bright());
    Ok(())
}

pub(crate) fn validate_config(config: &PluginConfig) -> Result<(), CliError> {
    register_adaptive_component().map_err(|error| {
        CliError::Config(format!("adaptive plugin registration failed: {error}"))
    })?;
    register_pii_redaction_component().map_err(|error| {
        CliError::Config(format!("PII redaction plugin registration failed: {error}"))
    })?;
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
