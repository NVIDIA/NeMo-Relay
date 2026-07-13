// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex-specific plugin setup, provider routing, and hook configuration.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{Value, json};
use toml_edit::{DocumentMut, InlineTable, Item, Table, Value as TomlValue, value};

use crate::agents::CodingAgent;
use crate::configuration::{BOOTSTRAP_CLIENT_TOKEN_HEADER, BootstrapChallengeKey, RELAY_PLUGIN_ID};
use crate::hooks::generated_hooks;
#[cfg(test)]
use crate::hooks::merge_hooks;

use super::app_server::{CodexAppServerClient, CodexHookMetadata, CodexHooksClient};
use crate::agents::shared::host::{
    atomic_write, atomic_write_private, current_exe, ensure_table, home_dir, read_json_object,
    shell_quote, write_json,
};
use crate::filesystem::{
    FileSnapshot, backup, backup_path, remove_backup, restore_file_snapshot, snapshot_optional_file,
};
use crate::process::{portable_executable_path, shell_quote_arg_for_platform};

pub(crate) const CODEX_PLUGIN_ID: &str = RELAY_PLUGIN_ID;
pub(crate) const CODEX_PLUGIN_HOOK_KEY_PREFIX: &str =
    "nemo-relay-plugin@nemo-relay-local:hooks/hooks.json:";

pub(crate) struct CodexSetupSnapshot {
    files: Vec<FileSnapshot>,
    hooks: Vec<CodexHookMetadata>,
    trust_state: Vec<(String, Option<Value>)>,
}

pub(crate) fn snapshot_codex_setup() -> Result<CodexSetupSnapshot, String> {
    let home = home_dir()?;
    let codex_dir = codex_home_dir()?;
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let mut client = CodexAppServerClient::start()?;
    let hooks = relay_codex_plugin_hooks(&mut client, &home)?;
    let trust_state = snapshot_relay_owned_hook_trust_state(&config_path, &hooks)?;
    let files = codex_install_snapshots(&config_path, &hooks_path)?;
    Ok(CodexSetupSnapshot {
        files,
        hooks,
        trust_state,
    })
}

pub(crate) fn restore_codex_setup(snapshot: &CodexSetupSnapshot) -> Result<(), String> {
    let mut errors = Vec::new();
    if let Err(error) = restore_codex_install_snapshots(&snapshot.files) {
        errors.push(format!("failed to restore Codex files: {error}"));
    }
    match (home_dir(), CodexAppServerClient::start()) {
        (Ok(home), Ok(mut client)) => {
            if let Err(error) = client.restore_hook_trust(&snapshot.trust_state) {
                errors.push(format!("failed to restore Codex hook trust: {error}"));
            } else if let Err(error) =
                verify_restored_hook_trust(&mut client, &home, &snapshot.hooks)
            {
                errors.push(error);
            }
        }
        (Err(error), _) | (_, Err(error)) => errors.push(error),
    }
    if let Err(error) = restore_codex_install_snapshots(&snapshot.files) {
        errors.push(format!("failed to restore exact Codex files: {error}"));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

#[cfg(test)]
pub(crate) fn install_codex(
    gateway_url: &str,
    plugin_hooks_path: &Path,
) -> Result<ExitCode, String> {
    install_codex_with_generation(gateway_url, plugin_hooks_path, None)
}

pub(crate) fn install_codex_with_generation(
    gateway_url: &str,
    plugin_hooks_path: &Path,
    generation_token: Option<&str>,
) -> Result<ExitCode, String> {
    let expected_command =
        expected_plugin_hook_command_with_token(plugin_hooks_path, generation_token)?;
    validate_plugin_hooks(plugin_hooks_path, &expected_command)?;
    install_codex_with_trust(
        gateway_url,
        &expected_command,
        |home, config_path, command| {
            let mut client = CodexAppServerClient::start()?;
            auto_trust_codex_hooks(&mut client, home, config_path, command)
        },
    )
}

pub(crate) fn install_codex_with_trust<F>(
    gateway_url: &str,
    expected_command: &str,
    trust_hooks: F,
) -> Result<ExitCode, String>
where
    F: FnOnce(&Path, &Path, &str) -> Result<(), String>,
{
    let home = home_dir()?;
    let codex_dir = codex_home_dir()?;
    fs::create_dir_all(&codex_dir)
        .map_err(|error| format!("failed to create {}: {error}", codex_dir.display()))?;
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    prepare_codex_config(&config_path)?;
    let snapshots = codex_install_snapshots(&config_path, &hooks_path)?;
    let install_result = remove_legacy_codex_hooks(&hooks_path)
        .and_then(|()| install_codex_config(&config_path, gateway_url))
        .and_then(|()| trust_hooks(&home, &config_path, expected_command));
    if let Err(error) = install_result {
        return match restore_codex_install_snapshots(&snapshots) {
            Ok(()) => Err(error),
            Err(rollback_error) => Err(format!(
                "{error}; additionally failed to roll back Codex configuration: {rollback_error}"
            )),
        };
    }
    println!("updated {}", config_path.display());
    if hooks_path.exists() {
        println!("updated {}", hooks_path.display());
    }
    println!("configured Codex Relay provider and plugin hooks; no daemon was installed.");
    Ok(ExitCode::SUCCESS)
}

pub(crate) fn uninstall_codex(
    installed_gateway_url: &str,
    _plugin_hooks_path: &Path,
) -> Result<ExitCode, String> {
    let mut client = CodexAppServerClient::start()?;
    uninstall_codex_with_client(installed_gateway_url, Some(&mut client))
}

pub(crate) fn uninstall_codex_with_client(
    installed_gateway_url: &str,
    client: Option<&mut dyn CodexHooksClient>,
) -> Result<ExitCode, String> {
    let home = home_dir()?;
    let codex_dir = codex_home_dir()?;
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let client = client
        .ok_or_else(|| "Codex app-server is required to clear plugin hook trust".to_string())?;
    let hooks = relay_codex_plugin_hooks(client, &home)?;
    let trust_state = snapshot_relay_owned_hook_trust_state(&config_path, &hooks)?;
    let trust_keys = trust_state
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    let snapshots = codex_install_snapshots(&config_path, &hooks_path)?;
    let uninstall_result =
        clear_and_verify_hook_trust(client, &home, &config_path, &hooks, &trust_keys)
            .and_then(|()| uninstall_codex_hooks(&hooks_path, installed_gateway_url))
            .and_then(|has_remaining_hooks| {
                uninstall_codex_config(&config_path, installed_gateway_url, has_remaining_hooks)
            });
    if let Err(error) = uninstall_result {
        return rollback_codex_uninstall(client, &home, &hooks, &trust_state, &snapshots, error);
    }
    println!("updated {}", config_path.display());
    println!("updated {}", hooks_path.display());
    println!("removed Codex Relay provider and plugin hook trust.");
    Ok(ExitCode::SUCCESS)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CodexHookTrustReport {
    trusted: Vec<String>,
    untrusted: Vec<String>,
    modified: Vec<String>,
    disabled: Vec<String>,
    missing_required: Vec<String>,
    duplicate_required: Vec<String>,
}

impl CodexHookTrustReport {
    pub(crate) fn ready(&self) -> bool {
        self.untrusted.is_empty()
            && self.modified.is_empty()
            && self.disabled.is_empty()
            && self.missing_required.is_empty()
            && self.duplicate_required.is_empty()
            && !self.trusted.is_empty()
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "trusted": self.trusted,
            "untrusted": self.untrusted,
            "modified": self.modified,
            "disabled": self.disabled,
            "missing_required": self.missing_required,
            "duplicate_required": self.duplicate_required,
        })
    }

    pub(crate) fn summary(&self) -> String {
        format!(
            "untrusted={}, modified={}, disabled={}, missing required={}, duplicate required={}",
            self.untrusted.len(),
            self.modified.len(),
            self.disabled.len(),
            self.missing_required.join(", "),
            self.duplicate_required.join(", ")
        )
    }
}

pub(crate) fn empty_codex_hook_trust_report() -> CodexHookTrustReport {
    CodexHookTrustReport {
        missing_required: CodingAgent::Codex
            .hook_events()
            .iter()
            .map(|event| (*event).to_string())
            .collect(),
        ..CodexHookTrustReport::default()
    }
}

pub(crate) fn codex_hook_trust_report(
    plugin_hooks_path: &Path,
) -> Result<CodexHookTrustReport, String> {
    codex_hook_trust_report_with_generation(plugin_hooks_path, None)
}

pub(crate) fn codex_hook_trust_report_with_generation(
    plugin_hooks_path: &Path,
    generation_token: Option<&str>,
) -> Result<CodexHookTrustReport, String> {
    let home = home_dir()?;
    let expected_command =
        expected_plugin_hook_command_with_token(plugin_hooks_path, generation_token)?;
    validate_plugin_hooks(plugin_hooks_path, &expected_command)?;
    let mut client = CodexAppServerClient::start()?;
    codex_hook_trust_report_with_client(&mut client, &home, &expected_command)
}

pub(crate) fn codex_hook_trust_report_with_client(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    expected_command: &str,
) -> Result<CodexHookTrustReport, String> {
    let hooks = relay_codex_hooks(client, cwd, expected_command)?;
    Ok(codex_hook_trust_report_for(&hooks))
}

pub(crate) fn auto_trust_codex_hooks(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    config_path: &Path,
    expected_command: &str,
) -> Result<(), String> {
    let hooks = relay_codex_hooks(client, cwd, expected_command)?;
    let before = codex_hook_trust_report_for(&hooks);
    if !before.missing_required.is_empty() || !before.duplicate_required.is_empty() {
        return Err(format!(
            "Codex must discover exactly one Relay handler per required event (missing: {}; duplicate: {})",
            before.missing_required.join(", "),
            before.duplicate_required.join(", ")
        ));
    }
    let state = snapshot_hook_trust_state(config_path, &hooks)?;
    let trust_result = client.trust_hooks(&hooks).and_then(|()| {
        let verified_hooks = relay_codex_hooks(client, cwd, expected_command)?;
        let verified = codex_hook_trust_report_for(&verified_hooks);
        let unverified_targets = hooks
            .iter()
            .filter(|target| {
                !verified_hooks.iter().any(|actual| {
                    actual.key == target.key
                        && actual.current_hash == target.current_hash
                        && actual.trust_status == "trusted"
                        && actual.enabled
                })
            })
            .map(|hook| hook.key.as_str())
            .collect::<Vec<_>>();
        if verified.ready() && unverified_targets.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Codex did not enable and trust all generated Relay hooks: {}; unverified targeted hooks={}",
                verified.summary(),
                unverified_targets.join(", ")
            ))
        }
    });
    if let Err(error) = trust_result {
        return restore_hook_trust_after_failure(
            client,
            cwd,
            expected_command,
            &hooks,
            &state,
            error,
        );
    }
    Ok(())
}

fn relay_codex_hooks(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    expected_command: &str,
) -> Result<Vec<CodexHookMetadata>, String> {
    let hooks = relay_codex_plugin_hooks(client, cwd)?
        .into_iter()
        .filter(|hook| hook.command.as_deref() == Some(expected_command))
        .collect::<Vec<_>>();
    validate_loaded_hook_sources(&hooks, expected_command)?;
    Ok(hooks)
}

fn validate_loaded_hook_sources(
    hooks: &[CodexHookMetadata],
    expected_command: &str,
) -> Result<(), String> {
    let expected = generated_hooks(CodingAgent::Codex, expected_command);
    let sources = hooks
        .iter()
        .map(|hook| hook.source_path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for source in sources {
        let path = Path::new(source);
        let actual = read_json_object(path)?;
        if actual != expected {
            return Err(format!(
                "Codex loaded modified Relay hooks from {}; run `nemo-relay install codex --force`",
                path.display()
            ));
        }
    }
    Ok(())
}

fn relay_codex_plugin_hooks(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
) -> Result<Vec<CodexHookMetadata>, String> {
    Ok(client
        .list_hooks(cwd)?
        .into_iter()
        .filter(|hook| {
            hook.source == "plugin"
                && hook.plugin_id.as_deref() == Some(CODEX_PLUGIN_ID)
                && hook.handler_type == "command"
                && is_generated_codex_hook_event(&hook.event_name)
        })
        .collect())
}

fn clear_and_verify_hook_trust(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    config_path: &Path,
    hooks: &[CodexHookMetadata],
    keys: &[String],
) -> Result<(), String> {
    if keys.is_empty() {
        return Ok(());
    }
    client.clear_hook_trust(keys)?;
    let mut uncleared = Vec::new();
    if !hooks.is_empty() {
        let cleared = relay_codex_plugin_hooks(client, cwd)?;
        uncleared.extend(
            hooks
                .iter()
                .filter(|expected| {
                    !cleared.iter().any(|actual| {
                        actual.key == expected.key && actual.trust_status.as_str() == "untrusted"
                    })
                })
                .map(|hook| hook.key.clone()),
        );
    }
    let persisted = configured_hook_trust_keys(config_path)?;
    uncleared.extend(
        keys.iter()
            .filter(|key| persisted.contains(key.as_str()))
            .cloned(),
    );
    uncleared.sort();
    uncleared.dedup();
    if uncleared.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Codex did not clear trust for Relay plugin hooks: {}",
            uncleared.join(", ")
        ))
    }
}

pub(crate) fn configured_hook_trust_keys(config_path: &Path) -> Result<BTreeSet<String>, String> {
    let raw = read_optional_text(config_path)?;
    let config = toml::from_str::<toml::Value>(&raw)
        .map_err(|error| format!("invalid TOML in {}: {error}", config_path.display()))?;
    Ok(config
        .get("hooks")
        .and_then(|hooks| hooks.get("state"))
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(|state| state.keys())
        .cloned()
        .collect())
}

fn relay_owned_hook_trust_keys(
    config_path: &Path,
    hooks: &[CodexHookMetadata],
) -> Result<Vec<String>, String> {
    let mut keys = Vec::new();
    for hook in hooks {
        if !keys.contains(&hook.key) {
            keys.push(hook.key.clone());
        }
    }
    for key in configured_hook_trust_keys(config_path)? {
        if key.starts_with(CODEX_PLUGIN_HOOK_KEY_PREFIX) && !keys.contains(&key) {
            keys.push(key);
        }
    }
    Ok(keys)
}

fn snapshot_relay_owned_hook_trust_state(
    config_path: &Path,
    hooks: &[CodexHookMetadata],
) -> Result<Vec<(String, Option<Value>)>, String> {
    let keys = relay_owned_hook_trust_keys(config_path, hooks)?;
    snapshot_hook_trust_keys(config_path, &keys)
}

fn snapshot_hook_trust_keys(
    config_path: &Path,
    keys: &[String],
) -> Result<Vec<(String, Option<Value>)>, String> {
    let raw = read_optional_text(config_path)?;
    let config = toml::from_str::<toml::Value>(&raw)
        .map_err(|error| format!("invalid TOML in {}: {error}", config_path.display()))?;
    let state = config
        .get("hooks")
        .and_then(|hooks| hooks.get("state"))
        .and_then(toml::Value::as_table);
    keys.iter()
        .map(|key| {
            let value = state
                .and_then(|state| state.get(key))
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| {
                    format!("failed to snapshot Codex hook trust for {key}: {error}")
                })?;
            Ok((key.clone(), value))
        })
        .collect()
}

fn snapshot_hook_trust_state(
    config_path: &Path,
    hooks: &[CodexHookMetadata],
) -> Result<Vec<(String, Option<Value>)>, String> {
    let keys = hooks
        .iter()
        .map(|hook| hook.key.clone())
        .collect::<Vec<_>>();
    snapshot_hook_trust_keys(config_path, &keys)
}

fn rollback_codex_uninstall(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    hooks: &[CodexHookMetadata],
    trust_state: &[(String, Option<Value>)],
    snapshots: &[FileSnapshot],
    original_error: String,
) -> Result<ExitCode, String> {
    let mut rollback_errors = Vec::new();
    if let Err(error) = client.restore_hook_trust(trust_state) {
        rollback_errors.push(format!("failed to restore Codex hook trust: {error}"));
    } else if let Err(error) = verify_restored_hook_trust(client, cwd, hooks) {
        rollback_errors.push(error);
    }
    if let Err(error) = restore_codex_install_snapshots(snapshots) {
        rollback_errors.push(format!("failed to restore Codex files: {error}"));
    }
    if rollback_errors.is_empty() {
        Err(original_error)
    } else {
        Err(format!(
            "{original_error}; additionally failed to roll back Codex uninstall: {}",
            rollback_errors.join("; ")
        ))
    }
}

fn verify_restored_hook_trust(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    expected: &[CodexHookMetadata],
) -> Result<(), String> {
    if expected.is_empty() {
        return Ok(());
    }
    let restored = relay_codex_plugin_hooks(client, cwd)?;
    let matches = expected.iter().all(|expected| {
        restored.iter().any(|actual| {
            actual.key == expected.key
                && actual.trust_status == expected.trust_status
                && actual.enabled == expected.enabled
        })
    });
    matches
        .then_some(())
        .ok_or_else(|| "failed to verify restored Codex hook trust after uninstall rollback".into())
}

fn restore_hook_trust_after_failure(
    client: &mut dyn CodexHooksClient,
    cwd: &Path,
    expected_command: &str,
    before: &[CodexHookMetadata],
    state: &[(String, Option<Value>)],
    original_error: String,
) -> Result<(), String> {
    if let Err(rollback_error) = client.restore_hook_trust(state) {
        return Err(format!(
            "{original_error}; additionally failed to restore Codex hook trust: {rollback_error}"
        ));
    }
    let restored = relay_codex_hooks(client, cwd, expected_command).map_err(|rollback_error| {
        format!(
            "{original_error}; additionally failed to verify restored Codex hook trust: {rollback_error}"
        )
    })?;
    let restored_matches = before.iter().all(|expected| {
        restored.iter().any(|actual| {
            actual.key == expected.key
                && actual.trust_status == expected.trust_status
                && actual.enabled == expected.enabled
        })
    });
    if !restored_matches {
        return Err(format!(
            "{original_error}; additionally failed to verify restored Codex hook trust state"
        ));
    }
    Err(original_error)
}

#[cfg(test)]
pub(crate) fn expected_plugin_hook_command(plugin_hooks_path: &Path) -> Result<String, String> {
    expected_plugin_hook_command_with_token(plugin_hooks_path, None)
}

fn expected_plugin_hook_command_with_token(
    plugin_hooks_path: &Path,
    generation_token: Option<&str>,
) -> Result<String, String> {
    let relay = current_exe()?;
    let relay = relay.canonicalize().unwrap_or(relay);
    let relay = portable_executable_path(relay);
    let generation_path = plugin_generation_file(plugin_hooks_path)?;
    let captured;
    let generation_token = match generation_token {
        Some(token) => token,
        None => {
            captured = crate::installation::generation::InstallGeneration::capture(
                generation_path.clone(),
            )?;
            captured.token()
        }
    };
    codex_plugin_hook_command(&relay, &generation_path, generation_token)
}

fn plugin_generation_file(plugin_hooks_path: &Path) -> Result<PathBuf, String> {
    let generation = plugin_hooks_path
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join(crate::installation::generation::GENERATION_FILE_NAME))
        .ok_or_else(|| {
            format!(
                "Codex plugin hooks path {} is not inside a plugin hooks directory",
                plugin_hooks_path.display()
            )
        })?;
    if generation.is_absolute() {
        Ok(generation)
    } else {
        env::current_dir()
            .map(|current| current.join(generation))
            .map_err(|error| format!("failed to resolve the Codex plugin generation path: {error}"))
    }
}

fn validate_plugin_hooks(path: &Path, expected_command: &str) -> Result<(), String> {
    let actual = read_json_object(path)?;
    let expected = generated_hooks(CodingAgent::Codex, expected_command);
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{} does not match the generated NeMo Relay plugin hooks; run `nemo-relay install codex --force`",
            path.display()
        ))
    }
}

pub(crate) fn codex_hook_trust_report_for(hooks: &[CodexHookMetadata]) -> CodexHookTrustReport {
    let mut report = CodexHookTrustReport::default();
    for hook in hooks {
        match hook.trust_status.as_str() {
            "trusted" => report.trusted.push(hook.key.clone()),
            "modified" => report.modified.push(hook.key.clone()),
            _ => report.untrusted.push(hook.key.clone()),
        }
        if !hook.enabled {
            report.disabled.push(hook.key.clone());
        }
    }
    report.missing_required = CodingAgent::Codex
        .hook_events()
        .iter()
        .filter(|event| {
            let normalized = normalize_hook_event(event);
            !hooks
                .iter()
                .any(|hook| normalize_hook_event(&hook.event_name) == normalized)
        })
        .map(|event| (*event).to_string())
        .collect();
    report.duplicate_required = CodingAgent::Codex
        .hook_events()
        .iter()
        .filter(|event| {
            let normalized = normalize_hook_event(event);
            hooks
                .iter()
                .filter(|hook| normalize_hook_event(&hook.event_name) == normalized)
                .count()
                > 1
        })
        .map(|event| (*event).to_string())
        .collect();
    report
}

fn is_generated_codex_hook_event(event: &str) -> bool {
    let normalized = normalize_hook_event(event);
    CodingAgent::Codex
        .hook_events()
        .iter()
        .any(|expected| normalize_hook_event(expected) == normalized)
}

fn normalize_hook_event(event: &str) -> String {
    event
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn codex_install_snapshots(
    config_path: &Path,
    hooks_path: &Path,
) -> Result<Vec<FileSnapshot>, String> {
    [
        config_path.to_path_buf(),
        backup_path(config_path),
        hooks_path.to_path_buf(),
        backup_path(hooks_path),
    ]
    .iter()
    .map(|path| snapshot_optional_file(path))
    .collect()
}

fn restore_codex_install_snapshots(snapshots: &[FileSnapshot]) -> Result<(), String> {
    let errors = snapshots
        .iter()
        .filter_map(|snapshot| restore_file_snapshot(snapshot).err())
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

pub(crate) fn prepare_codex_config(path: &Path) -> Result<(), String> {
    let raw = read_optional_text(path)?;
    raw.parse::<DocumentMut>()
        .map(|_| ())
        .map_err(|error| format!("invalid TOML in {}: {error}", path.display()))
}

pub(crate) fn install_codex_config(path: &Path, gateway_url: &str) -> Result<(), String> {
    let challenge = BootstrapChallengeKey::load().map_err(|error| error.to_string())?;
    let client_token = challenge.client_token();
    let raw = read_optional_text(path)?;
    let mut doc = raw
        .parse::<DocumentMut>()
        .map_err(|error| format!("invalid TOML in {}: {error}", path.display()))?;
    let backup_snapshot = snapshot_optional_file(&backup_path(path))?;
    let has_managed_proof =
        codex_provider_client_token(&doc).is_some_and(|token| challenge.verify_client_token(token));
    let provider_extensions = codex_provider_user_extensions(&doc, gateway_url);
    let unmodified_managed_install = codex_config_doc_has_managed_install(&doc, gateway_url)
        && has_managed_proof
        && codex_provider_has_only_generated_fields(&doc);
    if !unmodified_managed_install
        && let Err(error) = refresh_codex_config_backup(
            path,
            &raw,
            &doc,
            gateway_url,
            has_managed_proof,
            &challenge,
        )
    {
        return match restore_file_snapshot(&backup_snapshot) {
            Ok(()) => Err(error),
            Err(restore_error) => Err(format!(
                "{error}; additionally failed to restore the Codex backup: {restore_error}"
            )),
        };
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
    let mut headers = InlineTable::new();
    headers.insert(BOOTSTRAP_CLIENT_TOKEN_HEADER, TomlValue::from(client_token));
    provider["http_headers"] = Item::Value(TomlValue::InlineTable(headers));
    providers["nemo-relay-openai"] = Item::Table(provider);
    if let Some(extensions) = provider_extensions.as_ref() {
        merge_codex_provider_extensions(&mut doc, extensions);
    }

    if let Err(error) = atomic_write_private(path, doc.to_string().as_bytes()) {
        restore_file_snapshot(&backup_snapshot)?;
        return Err(error);
    }
    Ok(())
}

/// Refresh the uninstall baseline without carrying installer-owned fields forward.
///
/// A user can edit one field of an installed config before a forced reinstall. The current file
/// then contains both that user change and Relay's provider, hook flag, and client proof. Reusing
/// the whole file as the new backup would make those generated fields survive uninstall. Apply
/// the same ownership rules as uninstall to reconstruct the user baseline first.
fn refresh_codex_config_backup(
    path: &Path,
    raw: &str,
    current: &DocumentMut,
    gateway_url: &str,
    has_managed_proof: bool,
    challenge: &BootstrapChallengeKey,
) -> Result<(), String> {
    let previous = read_codex_backup_doc_for_refresh(path)?
        .map(|backup| sanitize_codex_backup_doc(backup, gateway_url, Some(challenge)));
    if previous.is_none() && !has_managed_proof {
        if !path.exists() {
            return Ok(());
        }
        return atomic_write_private(&backup_path(path), raw.as_bytes());
    }

    let empty = DocumentMut::new();
    let previous = previous.as_ref().unwrap_or(&empty);
    let mut baseline = current.clone();
    let preserved_provider = codex_extended_provider_without_proof(&baseline, gateway_url);
    let provider_is_managed = codex_provider_item_is_managed(&baseline, gateway_url);
    restore_codex_config_from_backup(&mut baseline, previous, provider_is_managed, false);
    restore_codex_client_proof_from_backup(&mut baseline, previous, Some(challenge));
    if let Some(provider) = preserved_provider {
        ensure_table(&mut baseline, "model_providers")
            .insert("nemo-relay-openai", Item::Table(provider));
    }
    remove_empty_table(&mut baseline, "model_providers");
    remove_empty_table(&mut baseline, "features");
    atomic_write_private(&backup_path(path), baseline.to_string().as_bytes())
}

fn read_codex_backup_doc_for_refresh(path: &Path) -> Result<Option<DocumentMut>, String> {
    let backup = backup_path(path);
    let raw = match fs::read_to_string(&backup) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!("failed to read {}: {error}", backup.display()));
        }
    };
    // A stale backup from an interrupted or older installation is not a trustworthy baseline.
    // The enclosing install transaction still snapshots and restores its bytes on later failure.
    Ok(raw.parse::<DocumentMut>().ok())
}

fn restore_codex_client_proof_from_backup(
    doc: &mut DocumentMut,
    backup: &DocumentMut,
    challenge: Option<&BootstrapChallengeKey>,
) {
    let current_is_managed = codex_provider_header(doc, BOOTSTRAP_CLIENT_TOKEN_HEADER)
        .and_then(TomlValue::as_str)
        .is_some_and(|token| {
            challenge.is_some_and(|challenge| challenge.verify_client_token(token))
        });
    if !current_is_managed {
        return;
    }
    let replacement = codex_provider_header(backup, BOOTSTRAP_CLIENT_TOKEN_HEADER)
        .filter(|value| {
            !value.as_str().is_some_and(|token| {
                challenge.is_some_and(|challenge| challenge.verify_client_token(token))
            })
        })
        .cloned();
    let Some(provider) = doc
        .get_mut("model_providers")
        .and_then(Item::as_table_mut)
        .and_then(|providers| providers.get_mut("nemo-relay-openai"))
        .and_then(Item::as_table_mut)
    else {
        return;
    };
    let Some(headers) = provider.get_mut("http_headers") else {
        return;
    };

    let remove_headers = if let Some(headers) = headers.as_inline_table_mut() {
        if !headers.contains_key(BOOTSTRAP_CLIENT_TOKEN_HEADER) {
            return;
        }
        match replacement {
            Some(value) => {
                headers.insert(BOOTSTRAP_CLIENT_TOKEN_HEADER, value);
            }
            None => {
                headers.remove(BOOTSTRAP_CLIENT_TOKEN_HEADER);
            }
        }
        headers.is_empty()
    } else if let Some(headers) = headers.as_table_mut() {
        if !headers.contains_key(BOOTSTRAP_CLIENT_TOKEN_HEADER) {
            return;
        }
        match replacement {
            Some(value) => {
                headers.insert(BOOTSTRAP_CLIENT_TOKEN_HEADER, Item::Value(value));
            }
            None => {
                headers.remove(BOOTSTRAP_CLIENT_TOKEN_HEADER);
            }
        }
        headers.is_empty()
    } else {
        false
    };
    if remove_headers {
        provider.remove("http_headers");
    }
}

/// Remove installer-owned state from a backup produced by an older partial-reinstall bug.
///
/// The client proof is the ownership signal: only a token authenticated by this user's current
/// bootstrap key permits cleanup. Exact generated providers are removed wholesale. A provider
/// edited by the user is retained, with only Relay's reserved proof and hook feature removed.
fn sanitize_codex_backup_doc(
    mut backup: DocumentMut,
    gateway_url: &str,
    challenge: Option<&BootstrapChallengeKey>,
) -> DocumentMut {
    let reserved_token =
        codex_provider_header(&backup, BOOTSTRAP_CLIENT_TOKEN_HEADER).and_then(TomlValue::as_str);
    let has_managed_proof = reserved_token.is_some_and(|token| {
        challenge.is_some_and(|challenge| challenge.verify_client_token(token))
    });
    let provider_is_managed = codex_provider_item_is_managed(&backup, gateway_url);
    let has_generated_lineage = provider_is_managed && reserved_token.is_some();
    if !has_managed_proof && !has_generated_lineage {
        return backup;
    }

    if provider_is_managed {
        let preserved_provider = codex_extended_provider_without_proof(&backup, gateway_url);
        if top_level_item_is_str(&backup, "model_provider", "nemo-relay-openai") {
            backup.as_table_mut().remove("model_provider");
        }
        if let Some(providers) = backup
            .get_mut("model_providers")
            .and_then(Item::as_table_mut)
        {
            providers.remove("nemo-relay-openai");
        }
        if let Some(provider) = preserved_provider {
            ensure_table(&mut backup, "model_providers")
                .insert("nemo-relay-openai", Item::Table(provider));
        }
    } else {
        let empty = DocumentMut::new();
        restore_codex_client_proof_from_backup(&mut backup, &empty, challenge);
    }
    remove_table_item_if_bool(&mut backup, "features", "hooks", true);
    remove_empty_table(&mut backup, "model_providers");
    remove_empty_table(&mut backup, "features");
    backup
}

/// Extract fields a user added to an otherwise generated provider table.
fn codex_provider_user_extensions(doc: &DocumentMut, gateway_url: &str) -> Option<Table> {
    if !codex_provider_item_is_managed(doc, gateway_url) {
        return None;
    }
    let mut extensions = doc
        .get("model_providers")?
        .as_table()?
        .get("nemo-relay-openai")?
        .as_table()?
        .clone();
    for key in [
        "name",
        "base_url",
        "wire_api",
        "requires_openai_auth",
        "supports_websockets",
    ] {
        extensions.remove(key);
    }
    remove_codex_provider_header(&mut extensions, BOOTSTRAP_CLIENT_TOKEN_HEADER);
    (!extensions.is_empty()).then_some(extensions)
}

fn codex_extended_provider_without_proof(doc: &DocumentMut, gateway_url: &str) -> Option<Table> {
    codex_provider_user_extensions(doc, gateway_url)?;
    let mut provider = doc
        .get("model_providers")?
        .as_table()?
        .get("nemo-relay-openai")?
        .as_table()?
        .clone();
    remove_codex_provider_header(&mut provider, BOOTSTRAP_CLIENT_TOKEN_HEADER);
    Some(provider)
}

fn remove_codex_provider_header(provider: &mut Table, name: &str) {
    let Some(headers) = provider.get_mut("http_headers") else {
        return;
    };
    let remove_headers = if let Some(headers) = headers.as_inline_table_mut() {
        headers.remove(name);
        headers.is_empty()
    } else if let Some(headers) = headers.as_table_mut() {
        headers.remove(name);
        headers.is_empty()
    } else {
        false
    };
    if remove_headers {
        provider.remove("http_headers");
    }
}

fn merge_codex_provider_extensions(doc: &mut DocumentMut, extensions: &Table) {
    let providers = ensure_table(doc, "model_providers");
    if !providers
        .get("nemo-relay-openai")
        .is_some_and(Item::is_table)
    {
        providers["nemo-relay-openai"] = Item::Table(Table::new());
    }
    let provider = providers["nemo-relay-openai"]
        .as_table_mut()
        .expect("provider table was just inserted");
    for (key, item) in extensions.iter() {
        if key == "http_headers" {
            merge_codex_provider_headers(provider, item);
        } else {
            provider.insert(key, item.clone());
        }
    }
}

fn merge_codex_provider_headers(provider: &mut Table, extensions: &Item) {
    let Some(entries) = codex_header_entries(extensions) else {
        provider.insert("http_headers", extensions.clone());
        return;
    };
    let Some(headers) = provider.get_mut("http_headers") else {
        provider.insert("http_headers", extensions.clone());
        return;
    };
    if let Some(headers) = headers.as_inline_table_mut() {
        for (name, value) in entries {
            headers.insert(&name, value);
        }
    } else if let Some(headers) = headers.as_table_mut() {
        for (name, value) in entries {
            headers.insert(&name, Item::Value(value));
        }
    } else {
        *headers = extensions.clone();
    }
}

fn codex_header_entries(headers: &Item) -> Option<Vec<(String, TomlValue)>> {
    if let Some(headers) = headers.as_inline_table() {
        return Some(
            headers
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect(),
        );
    }
    headers.as_table().and_then(|headers| {
        headers
            .iter()
            .map(|(name, item)| {
                item.as_value()
                    .cloned()
                    .map(|value| (name.to_string(), value))
            })
            .collect::<Option<Vec<_>>>()
    })
}

fn codex_provider_has_only_generated_fields(doc: &DocumentMut) -> bool {
    let Some(provider) = doc
        .get("model_providers")
        .and_then(Item::as_table)
        .and_then(|providers| providers.get("nemo-relay-openai"))
        .and_then(Item::as_table)
    else {
        return false;
    };
    let generated_fields = [
        "name",
        "base_url",
        "wire_api",
        "requires_openai_auth",
        "supports_websockets",
        "http_headers",
    ];
    if provider.len() != generated_fields.len()
        || !generated_fields
            .iter()
            .all(|field| provider.contains_key(field))
    {
        return false;
    }
    let Some(headers) = provider.get("http_headers") else {
        return false;
    };
    headers.as_inline_table().is_some_and(|headers| {
        headers.len() == 1 && headers.contains_key(BOOTSTRAP_CLIENT_TOKEN_HEADER)
    }) || headers.as_table().is_some_and(|headers| {
        headers.len() == 1 && headers.contains_key(BOOTSTRAP_CLIENT_TOKEN_HEADER)
    })
}

pub(crate) fn codex_provider_header<'a>(doc: &'a DocumentMut, name: &str) -> Option<&'a TomlValue> {
    let headers = doc
        .get("model_providers")
        .and_then(Item::as_table)
        .and_then(|providers| providers.get("nemo-relay-openai"))
        .and_then(Item::as_table)
        .and_then(|provider| provider.get("http_headers"))?;
    headers
        .as_inline_table()
        .and_then(|headers| {
            headers
                .iter()
                .find_map(|(key, value)| key.eq_ignore_ascii_case(name).then_some(value))
        })
        .or_else(|| {
            headers.as_table().and_then(|headers| {
                headers.iter().find_map(|(key, item)| {
                    key.eq_ignore_ascii_case(name)
                        .then(|| item.as_value())
                        .flatten()
                })
            })
        })
}

pub(crate) fn read_optional_text(path: &Path) -> Result<String, String> {
    match fs::read_to_string(path) {
        Ok(raw) => Ok(raw),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

pub(crate) fn uninstall_codex_config(
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
    let challenge = BootstrapChallengeKey::load_existing().map_err(|error| error.to_string())?;
    let backup_doc = read_codex_backup_doc(path)?
        .map(|backup| sanitize_codex_backup_doc(backup, gateway_url, challenge.as_ref()));
    let preserved_provider = codex_extended_provider_without_proof(&doc, gateway_url);
    let provider_is_managed = codex_provider_item_is_managed(&doc, gateway_url);
    match backup_doc.as_ref() {
        Some(backup_doc) => {
            restore_codex_config_from_backup(
                &mut doc,
                backup_doc,
                provider_is_managed,
                preserve_hooks,
            );
        }
        None => remove_codex_config_without_backup(&mut doc, provider_is_managed, preserve_hooks),
    }
    if let Some(provider) = preserved_provider {
        ensure_table(&mut doc, "model_providers")
            .insert("nemo-relay-openai", Item::Table(provider));
    }
    let empty_backup = DocumentMut::new();
    restore_codex_client_proof_from_backup(
        &mut doc,
        backup_doc.as_ref().unwrap_or(&empty_backup),
        challenge.as_ref(),
    );

    remove_empty_table(&mut doc, "model_providers");
    remove_empty_table(&mut doc, "features");
    atomic_write(path, doc.to_string().as_bytes())?;
    remove_backup(path)
}

fn read_codex_backup_doc(path: &Path) -> Result<Option<DocumentMut>, String> {
    let backup = backup_path(path);
    if !backup.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&backup)
        .map_err(|error| format!("failed to read {}: {error}", backup.display()))?;
    raw.parse::<DocumentMut>()
        .map(Some)
        .map_err(|error| format!("invalid TOML in {}: {error}", backup.display()))
}

fn restore_codex_config_from_backup(
    doc: &mut DocumentMut,
    backup_doc: &DocumentMut,
    provider_is_managed: bool,
    preserve_hooks: bool,
) {
    if provider_is_managed {
        restore_top_level_item_if_str(doc, backup_doc, "model_provider", "nemo-relay-openai");
        restore_table_item(doc, backup_doc, "model_providers", "nemo-relay-openai");
    }
    if !preserve_hooks || feature_hooks_enabled(doc) != Some(true) {
        restore_table_item_if_bool(doc, backup_doc, "features", "hooks", true);
    }
}

fn remove_codex_config_without_backup(
    doc: &mut DocumentMut,
    provider_is_managed: bool,
    preserve_hooks: bool,
) {
    if !provider_is_managed {
        return;
    }
    if top_level_item_is_str(doc, "model_provider", "nemo-relay-openai") {
        doc.as_table_mut().remove("model_provider");
    }
    if let Some(providers) = doc.get_mut("model_providers").and_then(Item::as_table_mut) {
        providers.remove("nemo-relay-openai");
    }
    if !preserve_hooks {
        remove_table_item_if_bool(doc, "features", "hooks", true);
    }
}

pub(crate) fn remove_legacy_codex_hooks(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let original = read_json_object(path)?;
    let mut updated = original.clone();
    let relay = current_exe()?;
    remove_managed_codex_hook_groups(&mut updated, &relay, None);
    if updated == original {
        return Ok(());
    }
    backup(path)?;
    write_json(path, &updated)
}

#[cfg(test)]
pub(crate) fn install_codex_hooks(path: &Path, gateway_url: &str) -> Result<(), String> {
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

pub(crate) fn uninstall_codex_hooks(path: &Path, _gateway_url: &str) -> Result<bool, String> {
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

pub(crate) fn remove_managed_codex_hook_groups(
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
                groups.retain_mut(|group| {
                    let Some(commands) = group.get_mut("hooks").and_then(Value::as_array_mut)
                    else {
                        return true;
                    };
                    let before = commands.len();
                    commands.retain(|hook| {
                        !managed_codex_hook_for_relay(hook, relay, keep_gateway_url)
                    });
                    commands.len() == before || !commands.is_empty()
                });
                groups.is_empty()
            })
            .unwrap_or(false);
        if should_remove_event {
            hooks.remove(&event);
        }
    }
}

fn managed_codex_hook_for_relay(
    hook: &Value,
    relay: &Path,
    keep_gateway_url: Option<&str>,
) -> bool {
    if hook.get("type").and_then(Value::as_str) != Some("command") {
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
        || legacy_relay_hook_command(command)
}

fn legacy_relay_hook_command(command: &str) -> bool {
    let Some((program, arguments)) = [" hook-forward codex", " plugin-shim hook codex"]
        .into_iter()
        .find_map(|separator| command.split_once(separator))
    else {
        return false;
    };
    if !arguments.is_empty() && !arguments.starts_with(" --gateway-url ") {
        return false;
    }
    let executable = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .trim_matches(['\'', '"'])
        .to_ascii_lowercase();
    matches!(executable.as_str(), "nemo-relay" | "nemo-relay.exe")
}

#[cfg(test)]
pub(crate) fn hook_config_contains_generated_groups(existing: &Value, generated: &Value) -> bool {
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

#[cfg(test)]
pub(crate) fn generated_event_contains_group(config: &Value, event: &str, group: &Value) -> bool {
    config
        .get("hooks")
        .and_then(Value::as_object)
        .and_then(|hooks| hooks.get(event))
        .and_then(Value::as_array)
        .is_some_and(|groups| groups.iter().any(|existing| existing == group))
}

pub(crate) fn hook_config_has_hook_groups(config: &Value) -> bool {
    config
        .get("hooks")
        .and_then(Value::as_object)
        .is_some_and(|hooks| {
            hooks
                .values()
                .any(|groups| groups.as_array().is_some_and(|groups| !groups.is_empty()))
        })
}

pub(crate) fn codex_config_doc_has_managed_install(doc: &DocumentMut, gateway_url: &str) -> bool {
    doc.get("model_provider")
        .and_then(Item::as_value)
        .and_then(|value| value.as_str())
        == Some("nemo-relay-openai")
        && codex_provider_item_is_managed(doc, gateway_url)
        && feature_hooks_enabled(doc) == Some(true)
}

#[cfg(test)]
pub(crate) fn codex_provider_gateway_url(path: &Path) -> Option<String> {
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

pub(crate) fn restore_top_level_item(doc: &mut DocumentMut, backup: &DocumentMut, key: &str) {
    if let Some(item) = backup.as_table().get(key).cloned() {
        doc.as_table_mut().insert(key, item);
    } else {
        doc.as_table_mut().remove(key);
    }
}

pub(crate) fn restore_top_level_item_if_str(
    doc: &mut DocumentMut,
    backup: &DocumentMut,
    key: &str,
    expected: &str,
) {
    if top_level_item_is_str(doc, key, expected) {
        restore_top_level_item(doc, backup, key);
    }
}

fn top_level_item_is_str(doc: &DocumentMut, key: &str, expected: &str) -> bool {
    doc.get(key)
        .and_then(Item::as_value)
        .and_then(|value| value.as_str())
        == Some(expected)
}

pub(crate) fn restore_table_item(
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

pub(crate) fn restore_table_item_if_bool(
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

pub(crate) fn codex_provider_item_is_managed(doc: &DocumentMut, gateway_url: &str) -> bool {
    doc.get("model_providers")
        .and_then(Item::as_table)
        .and_then(|providers| providers.get("nemo-relay-openai"))
        .and_then(Item::as_table)
        .is_some_and(|provider| codex_provider_table_is_managed_for_gateway(provider, gateway_url))
}

pub(crate) fn codex_provider_table_is_managed_for_gateway(
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

pub(crate) fn feature_hooks_enabled(doc: &DocumentMut) -> Option<bool> {
    doc.get("features")
        .and_then(Item::as_table)
        .and_then(|table| table.get("hooks"))
        .and_then(Item::as_value)
        .and_then(|value| value.as_bool())
}

pub(crate) fn remove_empty_table(doc: &mut DocumentMut, key: &str) {
    let is_empty = doc
        .get(key)
        .and_then(Item::as_table)
        .is_some_and(Table::is_empty);
    if is_empty {
        doc.as_table_mut().remove(key);
    }
}

pub(crate) fn remove_table_item_if_bool(
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

pub(crate) fn codex_provider_installed(gateway_url: &str) -> bool {
    let Ok(path) = codex_home_dir().map(|home| home.join("config.toml")) else {
        return false;
    };
    let Ok(raw) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(doc) = raw.parse::<DocumentMut>() else {
        return false;
    };
    let Ok(Some(key)) = BootstrapChallengeKey::load_existing() else {
        return false;
    };
    codex_config_doc_has_managed_install(&doc, gateway_url)
        && codex_provider_client_token(&doc).is_some_and(|token| key.verify_client_token(token))
}

pub(crate) fn codex_provider_client_token(doc: &DocumentMut) -> Option<&str> {
    codex_provider_header(doc, BOOTSTRAP_CLIENT_TOKEN_HEADER).and_then(TomlValue::as_str)
}

pub(crate) fn codex_hooks_installed(path: &Path) -> Result<bool, String> {
    codex_hooks_installed_with_generation(path, None)
}

pub(crate) fn codex_hooks_installed_with_generation(
    path: &Path,
    generation_token: Option<&str>,
) -> Result<bool, String> {
    let value = read_json_object(path)?;
    let generated = generated_hooks(
        CodingAgent::Codex,
        &expected_plugin_hook_command_with_token(path, generation_token)?,
    );
    Ok(value == generated)
}

pub(crate) fn codex_home_dir() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("CODEX_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    Ok(home_dir()?.join(".codex"))
}

pub(crate) fn codex_hook_command(gateway_url: &str) -> String {
    format!(
        "nemo-relay hook-forward codex --gateway-url {}",
        shell_quote_arg_for_platform(gateway_url, cfg!(windows))
    )
}

pub(crate) fn codex_plugin_hook_command(
    relay: &Path,
    generation: &Path,
    generation_token: &str,
) -> Result<String, String> {
    crate::hooks::persistent_hook_forward_command(
        relay,
        CodingAgent::Codex,
        generation,
        generation_token,
    )
}

#[cfg(test)]
pub(crate) fn codex_plugin_hook_command_for_platform(
    relay: &Path,
    generation: &Path,
    generation_token: &str,
    windows: bool,
) -> String {
    crate::hooks::persistent_hook_forward_command_for_platform(
        relay,
        CodingAgent::Codex,
        generation,
        generation_token,
        windows,
    )
}

pub(crate) fn legacy_codex_hook_command(relay: &Path) -> String {
    format!("{} plugin-shim hook codex", shell_quote(relay))
}

pub(crate) fn legacy_named_codex_hook_command() -> &'static str {
    "nemo-relay plugin-shim hook codex"
}
