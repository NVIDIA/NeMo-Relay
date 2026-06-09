// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code-specific provider routing setup.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{Value, json};

use super::command::PluginShimProviderAction;
use super::shared::{
    backup, backup_path, home_dir, read_json_object, remove_backup, restore_file_snapshot,
    snapshot_optional_file, write_json,
};

pub(super) fn claude_provider(
    action: PluginShimProviderAction,
    gateway_url: &str,
) -> Result<ExitCode, String> {
    match action {
        PluginShimProviderAction::Enable => {
            let path = claude_settings_path()?;
            let mut settings = read_json_object(&path)?;
            if settings.get("env").is_some_and(|env| !env.is_object()) {
                return Err(format!("{} has a non-object env field", path.display()));
            }
            let backup_snapshot = snapshot_optional_file(&backup_path(&path))?;
            if json_env_string(&settings, "ANTHROPIC_BASE_URL") != Some(gateway_url) {
                backup_claude_settings(&path)?;
            }
            let env = settings
                .as_object_mut()
                .expect("read_json_object returns an object")
                .entry("env")
                .or_insert_with(|| json!({}));
            let env = env.as_object_mut().expect("env was validated as an object");
            env.insert("ANTHROPIC_BASE_URL".into(), json!(gateway_url));
            if let Err(error) = write_json(&path, &settings) {
                restore_file_snapshot(&backup_snapshot)?;
                return Err(error);
            }
            println!("set ANTHROPIC_BASE_URL={gateway_url} in {}", path.display());
            Ok(ExitCode::SUCCESS)
        }
        PluginShimProviderAction::Restore => {
            let path = claude_settings_path()?;
            let backup = backup_path(&path);
            if !backup.exists() {
                println!(
                    "no backup found at {}; no managed Claude provider routing to restore",
                    backup.display()
                );
                return Ok(ExitCode::SUCCESS);
            }
            let mut settings = read_json_object(&path)?;
            if json_env_string(&settings, "ANTHROPIC_BASE_URL") == Some(gateway_url) {
                let backup_settings = read_json_object(&backup)?;
                restore_json_env_value(&mut settings, &backup_settings, "ANTHROPIC_BASE_URL")?;
                write_json(&path, &settings)?;
                remove_backup(&path)?;
                println!(
                    "restored managed ANTHROPIC_BASE_URL in {} from {}",
                    path.display(),
                    backup.display()
                );
            } else {
                println!(
                    "current Claude provider routing is not managed by Relay; left {} unchanged",
                    path.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        PluginShimProviderAction::Status => {
            println!(
                "{}",
                claude_settings_base_url().unwrap_or_else(|| {
                    "ANTHROPIC_BASE_URL is not configured in Claude settings".into()
                })
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

pub(super) fn json_env_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get("env")
        .and_then(Value::as_object)
        .and_then(|env| env.get(key))
        .and_then(Value::as_str)
}

pub(super) fn remove_json_env_string(value: &mut Value, key: &str) -> Result<bool, String> {
    let Some(object) = value.as_object_mut() else {
        return Err("Claude settings must be a JSON object".into());
    };
    let Some(env) = object.get_mut("env") else {
        return Ok(false);
    };
    let Some(env) = env.as_object_mut() else {
        return Err("Claude settings env field must be a JSON object".into());
    };
    let removed = env.remove(key).is_some();
    if env.is_empty() {
        object.remove("env");
    }
    Ok(removed)
}

pub(super) fn restore_json_env_value(
    value: &mut Value,
    backup: &Value,
    key: &str,
) -> Result<(), String> {
    let backup_value = backup
        .get("env")
        .and_then(Value::as_object)
        .and_then(|env| env.get(key))
        .cloned();
    if let Some(backup_value) = backup_value {
        let Some(object) = value.as_object_mut() else {
            return Err("Claude settings must be a JSON object".into());
        };
        let env = object.entry("env").or_insert_with(|| json!({}));
        let Some(env) = env.as_object_mut() else {
            return Err("Claude settings env field must be a JSON object".into());
        };
        env.insert(key.into(), backup_value);
    } else {
        remove_json_env_string(value, key)?;
    }
    Ok(())
}

pub(super) fn backup_claude_settings(path: &Path) -> Result<(), String> {
    let backup_file = backup_path(path);
    if backup_file.exists() {
        return Ok(());
    }
    if path.exists() {
        backup(path)
    } else {
        if let Some(parent) = backup_file.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(&backup_file, b"{}\n")
            .map_err(|error| format!("failed to write {}: {error}", backup_file.display()))
    }
}

pub(super) fn claude_settings_path() -> Result<PathBuf, String> {
    Ok(home_dir()?.join(".claude").join("settings.json"))
}

pub(super) fn claude_settings_base_url() -> Option<String> {
    let path = claude_settings_path().ok()?;
    let value = read_json_object(&path).ok()?;
    value
        .get("env")
        .and_then(Value::as_object)
        .and_then(|env| env.get("ANTHROPIC_BASE_URL"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}
