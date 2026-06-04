// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex-specific plugin setup, provider routing, and hook configuration.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use serde_json::{Value, json};
use toml_edit::{DocumentMut, Item, Table, value};

use crate::config::CodingAgent;
use crate::installer::{generated_hooks, merge_hooks};

use super::shared::{
    FileSnapshot, atomic_write, backup, backup_path, current_exe, ensure_table, home_dir,
    read_json_object, remove_backup, restore_file_snapshot, snapshot_optional_file, write_json,
};

pub(super) fn install_codex(gateway_url: &str) -> Result<ExitCode, String> {
    let codex_dir = home_dir()?.join(".codex");
    fs::create_dir_all(&codex_dir)
        .map_err(|error| format!("failed to create {}: {error}", codex_dir.display()))?;
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    prepare_codex_config(&config_path)?;
    let hooks_snapshot = snapshot_optional_file(&hooks_path)?;
    let hooks_backup_snapshot = snapshot_optional_file(&backup_path(&hooks_path))?;
    if let Err(error) = install_codex_hooks(&hooks_path, gateway_url) {
        if let Err(rollback_error) =
            restore_codex_hooks_snapshot(&hooks_snapshot, &hooks_backup_snapshot)
        {
            return Err(format!(
                "{error}; additionally failed to roll back Codex hooks at {}: {rollback_error}",
                hooks_path.display()
            ));
        }
        return Err(error);
    }
    if let Err(error) = install_codex_config(&config_path, gateway_url) {
        if let Err(rollback_error) =
            restore_codex_hooks_snapshot(&hooks_snapshot, &hooks_backup_snapshot)
        {
            return Err(format!(
                "{error}; additionally failed to roll back Codex hooks at {}: {rollback_error}",
                hooks_path.display()
            ));
        }
        return Err(error);
    }
    println!("updated {}", config_path.display());
    println!("updated {}", hooks_path.display());
    println!("Codex Relay sidecar startup is hook-supervised; no daemon was installed.");
    Ok(ExitCode::SUCCESS)
}

pub(super) fn uninstall_codex(installed_gateway_url: &str) -> Result<ExitCode, String> {
    let codex_dir = home_dir()?.join(".codex");
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let hook_gateway_url =
        codex_provider_gateway_url(&config_path).unwrap_or_else(|| installed_gateway_url.into());
    let hooks_snapshot = snapshot_optional_file(&hooks_path)?;
    let hooks_backup_snapshot = snapshot_optional_file(&backup_path(&hooks_path))?;
    let has_remaining_hooks = uninstall_codex_hooks(&hooks_path, &hook_gateway_url)?;
    if let Err(error) =
        uninstall_codex_config(&config_path, installed_gateway_url, has_remaining_hooks)
    {
        if let Err(rollback_error) =
            restore_codex_hooks_snapshot(&hooks_snapshot, &hooks_backup_snapshot)
        {
            return Err(format!(
                "{error}; additionally failed to roll back Codex hooks at {}: {rollback_error}",
                hooks_path.display()
            ));
        }
        return Err(error);
    }
    println!("updated {}", config_path.display());
    println!("updated {}", hooks_path.display());
    println!("removed Codex Relay hook-supervised sidecar setup.");
    Ok(ExitCode::SUCCESS)
}

pub(super) fn prepare_codex_config(path: &Path) -> Result<(), String> {
    let raw = read_optional_text(path)?;
    raw.parse::<DocumentMut>()
        .map(|_| ())
        .map_err(|error| format!("invalid TOML in {}: {error}", path.display()))
}

pub(super) fn install_codex_config(path: &Path, gateway_url: &str) -> Result<(), String> {
    let raw = read_optional_text(path)?;
    let mut doc = raw
        .parse::<DocumentMut>()
        .map_err(|error| format!("invalid TOML in {}: {error}", path.display()))?;
    let backup_snapshot = snapshot_optional_file(&backup_path(path))?;
    if !codex_config_doc_has_managed_install(&doc, gateway_url) {
        backup(path)?;
    }
    doc["model_provider"] = value("nemo-relay-openai");
    ensure_table(&mut doc, "features")["hooks"] = value(true);

    let providers = ensure_table(&mut doc, "model_providers");
    let mut provider = Table::new();
    provider["name"] = value("NeMo Relay");
    provider["base_url"] = value(gateway_url);
    provider["wire_api"] = value("responses");
    provider["requires_openai_auth"] = value(true);
    provider["supports_websockets"] = value(false);
    providers["nemo-relay-openai"] = Item::Table(provider);

    if let Err(error) = atomic_write(path, doc.to_string().as_bytes()) {
        restore_file_snapshot(&backup_snapshot)?;
        return Err(error);
    }
    Ok(())
}

pub(super) fn read_optional_text(path: &Path) -> Result<String, String> {
    match fs::read_to_string(path) {
        Ok(raw) => Ok(raw),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

pub(super) fn uninstall_codex_config(
    path: &Path,
    gateway_url: &str,
    preserve_hooks: bool,
) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut doc = raw
        .parse::<DocumentMut>()
        .map_err(|error| format!("invalid TOML in {}: {error}", path.display()))?;
    let backup = backup_path(path);
    let backup_doc = if backup.exists() {
        let raw = fs::read_to_string(&backup)
            .map_err(|error| format!("failed to read {}: {error}", backup.display()))?;
        Some(
            raw.parse::<DocumentMut>()
                .map_err(|error| format!("invalid TOML in {}: {error}", backup.display()))?,
        )
    } else {
        None
    };

    let provider_is_managed = codex_provider_item_is_managed(&doc, gateway_url);
    if let Some(backup_doc) = backup_doc.as_ref() {
        if provider_is_managed {
            restore_top_level_item_if_str(
                &mut doc,
                backup_doc,
                "model_provider",
                "nemo-relay-openai",
            );
            restore_table_item(&mut doc, backup_doc, "model_providers", "nemo-relay-openai");
        }
        if !preserve_hooks || feature_hooks_enabled(&doc) != Some(true) {
            restore_table_item_if_bool(&mut doc, backup_doc, "features", "hooks", true);
        }
    } else {
        if provider_is_managed
            && doc
                .get("model_provider")
                .and_then(Item::as_value)
                .and_then(|value| value.as_str())
                == Some("nemo-relay-openai")
        {
            doc.as_table_mut().remove("model_provider");
        }
        if provider_is_managed
            && let Some(providers) = doc.get_mut("model_providers").and_then(Item::as_table_mut)
        {
            providers.remove("nemo-relay-openai");
        }
        if !preserve_hooks {
            remove_table_item_if_bool(&mut doc, "features", "hooks", true);
        }
    }

    remove_empty_table(&mut doc, "model_providers");
    remove_empty_table(&mut doc, "features");
    atomic_write(path, doc.to_string().as_bytes())?;
    remove_backup(path)
}

pub(super) fn install_codex_hooks(path: &Path, gateway_url: &str) -> Result<(), String> {
    let relay = current_exe()?;
    let command = codex_hook_command(gateway_url);
    let generated = generated_hooks(CodingAgent::Codex, &command);
    let mut existing = if path.exists() {
        let raw = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let existing = serde_json::from_str::<Value>(&raw)
            .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))?;
        if !hook_config_contains_generated_groups(&existing, &generated) {
            backup(path)?;
        }
        existing
    } else {
        json!({})
    };
    remove_managed_codex_hook_groups(&mut existing, &relay, Some(gateway_url));
    let merged = merge_hooks(existing, generated).map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(&merged).map_err(|error| error.to_string())?;
    let mut output = bytes;
    output.push(b'\n');
    atomic_write(path, &output)
}

pub(super) fn uninstall_codex_hooks(path: &Path, _gateway_url: &str) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let mut value = read_json_object(path)?;
    let relay = current_exe()?;
    remove_managed_codex_hook_groups(&mut value, &relay, None);
    let has_remaining_hooks = hook_config_has_hook_groups(&value);
    write_json(path, &value)?;
    Ok(has_remaining_hooks)
}

pub(super) fn remove_managed_codex_hook_groups(
    value: &mut Value,
    relay: &Path,
    keep_gateway_url: Option<&str>,
) {
    let Some(hooks) = value.get_mut("hooks").and_then(Value::as_object_mut) else {
        return;
    };
    let events: Vec<String> = hooks.keys().cloned().collect();
    for event in events {
        let should_remove_event = hooks
            .get_mut(&event)
            .and_then(Value::as_array_mut)
            .map(|groups| {
                groups.retain(|group| {
                    !managed_codex_hook_group_for_relay(group, relay, keep_gateway_url)
                });
                groups.is_empty()
            })
            .unwrap_or(false);
        if should_remove_event {
            hooks.remove(&event);
        }
    }
}

pub(super) fn managed_codex_hook_group_for_relay(
    group: &Value,
    relay: &Path,
    keep_gateway_url: Option<&str>,
) -> bool {
    let Some(hooks) = group.get("hooks").and_then(Value::as_array) else {
        return false;
    };
    let [hook] = hooks.as_slice() else {
        return false;
    };
    if hook.get("type").and_then(Value::as_str) != Some("command")
        || hook.get("timeout").and_then(Value::as_u64) != Some(30)
    {
        return false;
    }
    let Some(command) = hook.get("command").and_then(Value::as_str) else {
        return false;
    };
    if keep_gateway_url.is_some_and(|gateway_url| command == codex_hook_command(gateway_url)) {
        return false;
    }
    command == legacy_codex_hook_command(relay)
        || command == legacy_named_codex_hook_command()
        || command.starts_with("nemo-relay plugin-shim hook codex --gateway-url ")
        || command.starts_with(&format!(
            "{} plugin-shim hook codex --gateway-url ",
            shell_quote(relay)
        ))
}

pub(super) fn hook_config_contains_generated_groups(existing: &Value, generated: &Value) -> bool {
    let Some(generated_hooks) = generated.get("hooks").and_then(Value::as_object) else {
        return false;
    };
    generated_hooks.iter().all(|(event, groups)| {
        groups.as_array().is_some_and(|groups| {
            groups
                .iter()
                .all(|group| generated_event_contains_group(existing, event, group))
        })
    })
}

pub(super) fn generated_event_contains_group(config: &Value, event: &str, group: &Value) -> bool {
    config
        .get("hooks")
        .and_then(Value::as_object)
        .and_then(|hooks| hooks.get(event))
        .and_then(Value::as_array)
        .is_some_and(|groups| groups.iter().any(|existing| existing == group))
}

pub(super) fn hook_config_has_hook_groups(config: &Value) -> bool {
    config
        .get("hooks")
        .and_then(Value::as_object)
        .is_some_and(|hooks| {
            hooks
                .values()
                .any(|groups| groups.as_array().is_some_and(|groups| !groups.is_empty()))
        })
}

pub(super) fn codex_config_doc_has_managed_install(doc: &DocumentMut, gateway_url: &str) -> bool {
    doc.get("model_provider")
        .and_then(Item::as_value)
        .and_then(|value| value.as_str())
        == Some("nemo-relay-openai")
        && codex_provider_item_is_managed(doc, gateway_url)
        && feature_hooks_enabled(doc) == Some(true)
}

pub(super) fn codex_provider_gateway_url(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let doc = raw.parse::<DocumentMut>().ok()?;
    doc.get("model_providers")
        .and_then(Item::as_table)
        .and_then(|providers| providers.get("nemo-relay-openai"))
        .and_then(Item::as_table)
        .and_then(|provider| provider.get("base_url"))
        .and_then(Item::as_value)
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
}

pub(super) fn restore_top_level_item(doc: &mut DocumentMut, backup: &DocumentMut, key: &str) {
    if let Some(item) = backup.as_table().get(key).cloned() {
        doc.as_table_mut().insert(key, item);
    } else {
        doc.as_table_mut().remove(key);
    }
}

pub(super) fn restore_top_level_item_if_str(
    doc: &mut DocumentMut,
    backup: &DocumentMut,
    key: &str,
    expected: &str,
) {
    let current = doc
        .get(key)
        .and_then(Item::as_value)
        .and_then(|value| value.as_str());
    if current == Some(expected) {
        restore_top_level_item(doc, backup, key);
    }
}

pub(super) fn restore_table_item(
    doc: &mut DocumentMut,
    backup: &DocumentMut,
    table: &str,
    key: &str,
) {
    if let Some(item) = backup
        .get(table)
        .and_then(Item::as_table)
        .and_then(|table| table.get(key))
        .cloned()
    {
        ensure_table(doc, table).insert(key, item);
    } else if let Some(table) = doc.get_mut(table).and_then(Item::as_table_mut) {
        table.remove(key);
    }
}

pub(super) fn restore_table_item_if_bool(
    doc: &mut DocumentMut,
    backup: &DocumentMut,
    table: &str,
    key: &str,
    expected: bool,
) {
    let current = doc
        .get(table)
        .and_then(Item::as_table)
        .and_then(|table| table.get(key))
        .and_then(Item::as_value)
        .and_then(|value| value.as_bool());
    if current == Some(expected) {
        restore_table_item(doc, backup, table, key);
    }
}

pub(super) fn codex_provider_item_is_managed(doc: &DocumentMut, gateway_url: &str) -> bool {
    doc.get("model_providers")
        .and_then(Item::as_table)
        .and_then(|providers| providers.get("nemo-relay-openai"))
        .and_then(Item::as_table)
        .is_some_and(|provider| codex_provider_table_is_managed_for_gateway(provider, gateway_url))
}

pub(super) fn codex_provider_table_is_managed_for_gateway(
    provider: &Table,
    gateway_url: &str,
) -> bool {
    provider
        .get("name")
        .and_then(Item::as_value)
        .and_then(|value| value.as_str())
        == Some("NeMo Relay")
        && provider
            .get("base_url")
            .and_then(Item::as_value)
            .and_then(|value| value.as_str())
            == Some(gateway_url)
        && provider
            .get("wire_api")
            .and_then(Item::as_value)
            .and_then(|value| value.as_str())
            == Some("responses")
        && provider
            .get("requires_openai_auth")
            .and_then(Item::as_value)
            .and_then(|value| value.as_bool())
            == Some(true)
        && provider
            .get("supports_websockets")
            .and_then(Item::as_value)
            .and_then(|value| value.as_bool())
            == Some(false)
}

pub(super) fn feature_hooks_enabled(doc: &DocumentMut) -> Option<bool> {
    doc.get("features")
        .and_then(Item::as_table)
        .and_then(|table| table.get("hooks"))
        .and_then(Item::as_value)
        .and_then(|value| value.as_bool())
}

pub(super) fn remove_empty_table(doc: &mut DocumentMut, key: &str) {
    let is_empty = doc
        .get(key)
        .and_then(Item::as_table)
        .is_some_and(Table::is_empty);
    if is_empty {
        doc.as_table_mut().remove(key);
    }
}

pub(super) fn remove_table_item_if_bool(
    doc: &mut DocumentMut,
    table: &str,
    key: &str,
    expected: bool,
) {
    let should_remove = doc
        .get(table)
        .and_then(Item::as_table)
        .and_then(|table| table.get(key))
        .and_then(Item::as_value)
        .and_then(|value| value.as_bool())
        == Some(expected);
    if should_remove && let Some(table) = doc.get_mut(table).and_then(Item::as_table_mut) {
        table.remove(key);
    }
}

pub(super) fn codex_provider_installed(gateway_url: &str) -> bool {
    let Ok(path) = home_dir().map(|home| home.join(".codex").join("config.toml")) else {
        return false;
    };
    let Ok(raw) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(doc) = raw.parse::<DocumentMut>() else {
        return false;
    };
    codex_config_doc_has_managed_install(&doc, gateway_url)
}

pub(super) fn codex_hooks_installed(gateway_url: &str) -> Result<bool, String> {
    let path = home_dir()?.join(".codex").join("hooks.json");
    let value = read_json_object(&path)?;
    let generated = generated_hooks(CodingAgent::Codex, &codex_hook_command(gateway_url));
    Ok(hook_config_contains_generated_groups(&value, &generated))
}

pub(super) fn restore_codex_hooks_snapshot(
    hooks: &FileSnapshot,
    hooks_backup: &FileSnapshot,
) -> Result<(), String> {
    restore_file_snapshot(hooks)?;
    restore_file_snapshot(hooks_backup)
}

pub(super) fn shell_quote(path: &Path) -> String {
    shell_quote_for_platform(path, cfg!(windows))
}

pub(super) fn shell_quote_for_platform(path: &Path, windows: bool) -> String {
    shell_quote_arg_for_platform(&path.display().to_string(), windows)
}

pub(super) fn shell_quote_arg_for_platform(raw: &str, windows: bool) -> String {
    if windows {
        return cmd_quote_arg(raw);
    }
    posix_quote_arg(raw)
}

pub(super) fn posix_quote_arg(raw: &str) -> String {
    if raw.is_empty() {
        "''".into()
    } else if raw
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | ':' | '.' | '_' | '-'))
    {
        raw.to_string()
    } else {
        format!("'{}'", raw.replace('\'', "'\\''"))
    }
}

pub(super) fn cmd_quote_arg(raw: &str) -> String {
    if raw.chars().all(|ch| {
        ch.is_ascii_alphanumeric()
            || matches!(ch, '/' | '\\' | ':' | '.' | '_' | '-' | '=' | '@' | '+')
    }) {
        raw.to_string()
    } else {
        let mut escaped = String::new();
        for ch in raw.chars() {
            match ch {
                '%' => escaped.push_str("%%"),
                '"' | '^' | '&' | '|' | '<' | '>' => {
                    escaped.push('^');
                    escaped.push(ch);
                }
                _ => escaped.push(ch),
            }
        }
        format!("\"{escaped}\"")
    }
}

pub(super) fn codex_hook_command(gateway_url: &str) -> String {
    format!(
        "nemo-relay plugin-shim hook codex --gateway-url {}",
        shell_quote_arg_for_platform(gateway_url, cfg!(windows))
    )
}

#[cfg(test)]
pub(super) fn codex_hook_command_for_platform(
    relay: &Path,
    gateway_url: &str,
    windows: bool,
) -> String {
    format!(
        "{} plugin-shim hook codex --gateway-url {}",
        shell_quote_for_platform(relay, windows),
        shell_quote_arg_for_platform(gateway_url, windows)
    )
}

pub(super) fn legacy_codex_hook_command(relay: &Path) -> String {
    format!("{} plugin-shim hook codex", shell_quote(relay))
}

pub(super) fn legacy_named_codex_hook_command() -> &'static str {
    "nemo-relay plugin-shim hook codex"
}
