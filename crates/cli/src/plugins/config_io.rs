// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable plugin configuration file and validation helpers.

use std::path::{Path, PathBuf};

use console::style;
use nemo_relay::plugin::{ConfigPolicy, PluginConfig, validate_plugin_config};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_pii_redaction::component::register_pii_redaction_component;
use serde_json::{Map, Value};

use crate::config::{
    PluginsEditCommand, global_plugin_config_path, project_plugin_config_path,
    user_plugin_config_path,
};
use crate::error::CliError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetScope {
    User,
    Project,
    Global,
}

pub(crate) fn target_scope(command: &PluginsEditCommand) -> Result<TargetScope, CliError> {
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
