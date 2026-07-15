// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code-specific provider routing setup.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::agents::shared::host::{home_dir, read_json_object, write_json};
use crate::filesystem::{
    FileSnapshot, backup, backup_path, remove_backup, restore_file_snapshot, snapshot_optional_file,
};

const ABSENT_SETTINGS_BACKUP_KEY: &str = "__nemo_relay_original_settings_absent";
const MANAGED_PROVIDER_BACKUP_KEY: &str = "__nemo_relay_managed_anthropic_base_url";

pub(crate) struct ClaudeSetupSnapshot {
    files: Vec<FileSnapshot>,
}

pub(crate) fn snapshot_claude_setup() -> Result<ClaudeSetupSnapshot, String> {
    let settings = claude_settings_path()?;
    let files = [settings.clone(), backup_path(&settings)]
        .iter()
        .map(|path| snapshot_optional_file(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ClaudeSetupSnapshot { files })
}

pub(crate) fn restore_claude_setup(snapshot: &ClaudeSetupSnapshot) -> Result<(), String> {
    let errors = snapshot
        .files
        .iter()
        .filter_map(|file| restore_file_snapshot(file).err())
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

pub(crate) fn enable_claude_provider(gateway_url: &str) -> Result<(), String> {
    let path = claude_settings_path()?;
    let mut settings = read_json_object(&path)?;
    if settings.get("env").is_some_and(|env| !env.is_object()) {
        return Err(format!("{} has a non-object env field", path.display()));
    }
    let backup_snapshot = snapshot_optional_file(&backup_path(&path))?;
    let current_provider = json_env_string(&settings, "ANTHROPIC_BASE_URL");
    let backup_file = backup_path(&path);
    let previous_managed_provider = read_json_object(&backup_file).ok().and_then(|backup| {
        backup
            .get(MANAGED_PROVIDER_BACKUP_KEY)
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    let managed_provider = current_provider == Some(gateway_url)
        || previous_managed_provider
            .as_deref()
            .is_some_and(|previous| current_provider == Some(previous))
        || (backup_file.exists() && current_provider == Some(crate::bootstrap::DEFAULT_URL));
    if !managed_provider && let Err(error) = backup_claude_settings(&path, true) {
        restore_file_snapshot(&backup_snapshot)?;
        return Err(error);
    }
    if backup_file.exists()
        && let Err(error) = record_managed_provider(&backup_file, gateway_url)
    {
        restore_file_snapshot(&backup_snapshot)?;
        return Err(error);
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
    Ok(())
}

fn record_managed_provider(backup: &Path, gateway_url: &str) -> Result<(), String> {
    let mut value = read_json_object(backup)?;
    value
        .as_object_mut()
        .expect("read_json_object returns an object")
        .insert(MANAGED_PROVIDER_BACKUP_KEY.into(), json!(gateway_url));
    write_json(backup, &value)
}

pub(crate) fn restore_claude_provider(gateway_url: &str) -> Result<(), String> {
    let path = claude_settings_path()?;
    let backup = backup_path(&path);
    if !backup.exists() {
        println!(
            "no backup found at {}; no managed Claude provider routing to restore",
            backup.display()
        );
        return Ok(());
    }
    let mut settings = read_json_object(&path)?;
    if json_env_string(&settings, "ANTHROPIC_BASE_URL") == Some(gateway_url) {
        let backup_settings = read_json_object(&backup)?;
        restore_json_env_value(&mut settings, &backup_settings, "ANTHROPIC_BASE_URL")?;
        if backup_settings.get(ABSENT_SETTINGS_BACKUP_KEY) == Some(&Value::Bool(true))
            && settings.as_object().is_some_and(serde_json::Map::is_empty)
        {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!("failed to remove {}: {error}", path.display()));
                }
            }
        } else {
            write_json(&path, &settings)?;
        }
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
    Ok(())
}

pub(crate) fn json_env_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get("env")
        .and_then(Value::as_object)
        .and_then(|env| env.get(key))
        .and_then(Value::as_str)
}

pub(crate) fn remove_json_env_string(value: &mut Value, key: &str) -> Result<bool, String> {
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

pub(crate) fn restore_json_env_value(
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

pub(crate) fn backup_claude_settings(path: &Path, replace_existing: bool) -> Result<(), String> {
    let backup_file = backup_path(path);
    if backup_file.exists() && !replace_existing {
        return Ok(());
    }
    if path.exists() {
        if replace_existing && backup_file.exists() {
            fs::remove_file(&backup_file).map_err(|error| {
                format!(
                    "failed to remove stale backup {}: {error}",
                    backup_file.display()
                )
            })?;
        }
        backup(path)
    } else {
        if let Some(parent) = backup_file.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(
            &backup_file,
            format!("{{\"{ABSENT_SETTINGS_BACKUP_KEY}\":true}}\n"),
        )
        .map_err(|error| format!("failed to write {}: {error}", backup_file.display()))
    }
}

pub(crate) fn claude_settings_path() -> Result<PathBuf, String> {
    Ok(home_dir()?.join(".claude").join("settings.json"))
}

pub(crate) fn claude_settings_base_url() -> Option<String> {
    let path = claude_settings_path().ok()?;
    let value = read_json_object(&path).ok()?;
    value
        .get("env")
        .and_then(Value::as_object)
        .and_then(|env| env.get("ANTHROPIC_BASE_URL"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}
