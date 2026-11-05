// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code-specific provider routing setup.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::agents::shared::host::{
    home_dir, read_json_object, write_json, write_json_preserving_symlink,
};
use crate::filesystem::{
    FileSnapshot, backup, backup_path, remove_backup, remove_file_preserving_symlink,
    restore_file_snapshot, snapshot_optional_file,
};

const ABSENT_SETTINGS_BACKUP_KEY: &str = "__nemo_relay_original_settings_absent";
const DANGLING_SETTINGS_SYMLINK_BACKUP_KEY: &str =
    "__nemo_relay_original_settings_dangling_symlink";
const MANAGED_PROVIDER_BACKUP_KEY: &str = "__nemo_relay_managed_anthropic_base_url";
const GATEWAY_HINT_HEADERS_ENV: &str = "CLAUDE_CODE_GATEWAY_HINT_HEADERS";
const MANAGED_GATEWAY_HINT_HEADERS_BACKUP_KEY: &str = "__nemo_relay_added_gateway_hint_headers";

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
    let previous_backup = read_json_object(&backup_file).ok();
    let previous_managed_provider = previous_backup.as_ref().and_then(|backup| {
        backup
            .get(MANAGED_PROVIDER_BACKUP_KEY)
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    let add_gateway_hints = settings
        .get("env")
        .and_then(|env| env.get(GATEWAY_HINT_HEADERS_ENV))
        .is_none();
    let owns_gateway_hints = previous_backup.as_ref().is_some_and(|backup| {
        backup.get(MANAGED_GATEWAY_HINT_HEADERS_BACKUP_KEY) == Some(&Value::Bool(true))
    }) && json_env_string(&settings, GATEWAY_HINT_HEADERS_ENV)
        == Some("1");
    let managed_provider = current_provider == Some(gateway_url)
        || previous_managed_provider
            .as_deref()
            .is_some_and(|previous| current_provider == Some(previous))
        || (backup_file.exists() && current_provider == Some(crate::bootstrap::DEFAULT_URL));
    if (!managed_provider || (add_gateway_hints && !backup_file.exists()))
        && let Err(error) = backup_claude_settings(&path, true)
    {
        restore_file_snapshot(&backup_snapshot)?;
        return Err(error);
    }
    if backup_file.exists()
        && let Err(error) = record_managed_provider(
            &backup_file,
            gateway_url,
            owns_gateway_hints || add_gateway_hints,
        )
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
    if add_gateway_hints {
        env.insert(GATEWAY_HINT_HEADERS_ENV.into(), json!("1"));
    }
    if let Err(error) = write_json_preserving_symlink(&path, &settings) {
        restore_file_snapshot(&backup_snapshot)?;
        return Err(error);
    }
    println!("set ANTHROPIC_BASE_URL={gateway_url} in {}", path.display());
    Ok(())
}

fn record_managed_provider(
    backup: &Path,
    gateway_url: &str,
    owns_gateway_hints: bool,
) -> Result<(), String> {
    let mut value = read_json_object(backup)?;
    let object = value
        .as_object_mut()
        .expect("read_json_object returns an object");
    object.insert(MANAGED_PROVIDER_BACKUP_KEY.into(), json!(gateway_url));
    if owns_gateway_hints {
        object.insert(MANAGED_GATEWAY_HINT_HEADERS_BACKUP_KEY.into(), json!(true));
    } else {
        object.remove(MANAGED_GATEWAY_HINT_HEADERS_BACKUP_KEY);
    }
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
    let restore_provider = json_env_string(&settings, "ANTHROPIC_BASE_URL") == Some(gateway_url);
    let mut backup_settings = match read_json_object(&backup) {
        Ok(settings) => settings,
        Err(error) if !restore_provider => {
            log::warn!(
                target: "nemo_relay.cli",
                event = "claude_setup_restore_skipped",
                reason = "backup_ownership_unknown";
                "Cannot establish Relay ownership from Claude setup backup {}; leaving user settings unchanged: {error}",
                backup.display()
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let managed_gateway_hints =
        backup_settings.get(MANAGED_GATEWAY_HINT_HEADERS_BACKUP_KEY) == Some(&Value::Bool(true));
    if !restore_provider && !managed_gateway_hints {
        println!(
            "current Claude provider routing is not managed by Relay; left {} unchanged",
            path.display()
        );
        return Ok(());
    }
    let snapshot = snapshot_claude_setup()?;
    let update = (|| {
        if restore_provider {
            restore_json_env_value(&mut settings, &backup_settings, "ANTHROPIC_BASE_URL")?;
        }
        // This default was inserted only when absent. An older routing backup may contain a
        // historical hint value, which is not the value to restore for this separate setting.
        let remove_gateway_hints = managed_gateway_hints
            && json_env_string(&settings, GATEWAY_HINT_HEADERS_ENV) == Some("1");
        if remove_gateway_hints {
            remove_json_env_string(&mut settings, GATEWAY_HINT_HEADERS_ENV)?;
        }
        if restore_provider || remove_gateway_hints {
            write_restored_claude_settings(&path, &settings, &backup_settings)?;
        }
        if restore_provider {
            remove_backup(&path)?;
        } else {
            // Routing now belongs to the user. Keep its backup but retire our independent
            // hint ownership so a later user-supplied "1" cannot be removed by another uninstall.
            backup_settings
                .as_object_mut()
                .expect("read_json_object returns an object")
                .remove(MANAGED_GATEWAY_HINT_HEADERS_BACKUP_KEY);
            write_json(&backup, &backup_settings)?;
        }
        Ok(())
    })();
    if let Err(error) = update {
        return match restore_claude_setup(&snapshot) {
            Ok(()) => Err(error),
            Err(rollback_error) => Err(format!(
                "{error}; failed to restore Claude setup after cleanup failure: {rollback_error}"
            )),
        };
    }
    if restore_provider {
        println!(
            "restored managed ANTHROPIC_BASE_URL in {} from {}",
            path.display(),
            backup.display()
        );
    } else {
        println!(
            "left user-managed Claude provider routing unchanged in {}",
            path.display()
        );
    }
    Ok(())
}

fn write_restored_claude_settings(
    path: &Path,
    settings: &Value,
    backup_settings: &Value,
) -> Result<(), String> {
    if backup_settings.get(ABSENT_SETTINGS_BACKUP_KEY) == Some(&Value::Bool(true))
        && settings.as_object().is_some_and(serde_json::Map::is_empty)
    {
        if backup_settings.get(DANGLING_SETTINGS_SYMLINK_BACKUP_KEY) == Some(&Value::Bool(true)) {
            remove_file_preserving_symlink(path)?;
        } else {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!("failed to remove {}: {error}", path.display()));
                }
            }
        }
        Ok(())
    } else {
        write_json_preserving_symlink(path, settings)
    }
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
        let mut backup = json!({
            ABSENT_SETTINGS_BACKUP_KEY: true,
        });
        if path_is_dangling_symlink(path)? {
            backup[DANGLING_SETTINGS_SYMLINK_BACKUP_KEY] = Value::Bool(true);
        }
        write_json(&backup_file, &backup)
    }
}

fn path_is_dangling_symlink(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Ok(!path.exists()),
        Ok(_) => Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "failed to inspect {} for symlink metadata: {error}",
            path.display()
        )),
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
