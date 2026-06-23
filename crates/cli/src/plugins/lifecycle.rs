// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginManifest, DynamicPluginRecord,
    DynamicPluginValidationStatus,
};

use crate::config::{
    PluginsAddCommand, PluginsDisableCommand, PluginsEnableCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsRemoveCommand, PluginsValidateCommand, ResolvedConfig,
    ResolvedDynamicPluginConfig, ServerArgs, resolve_plugins_config,
};
use crate::error::{CliError, PluginLifecycleFailureKind};

use super::config_io::{
    append_dynamic_plugin_reference, remove_dynamic_plugin_reference, target_scope,
};

mod json;
mod render;
mod state;
mod target;

use self::json::{
    ValidateJsonContext, failure_envelope, generic_failure_envelope, inspect_success_envelope,
    list_success_envelope, print_json, validate_success_envelope,
};
use self::render::{render_inspect, render_list, render_validation_summary};
use self::state::{
    RegistryScope, ScopedRegistry, collect_records, find_record_by_id, load_scoped_registries,
    scoped_paths_for_add,
};
use self::target::PluginTarget;

pub(crate) fn add(command: PluginsAddCommand, server: &ServerArgs) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let (manifest, manifest_ref) = load_manifest_for_action("add", &command.path)?;
    let plugin_id = manifest.plugin.id.trim().to_owned();
    let revived = match find_record_by_id(&scopes, &plugin_id)? {
        Some(existing) if !existing.record.is_tombstoned() => {
            return Err(CliError::Config(format!(
                "dynamic plugin '{}' is already registered in the {} lifecycle scope",
                plugin_id,
                existing.scope.label()
            )));
        }
        Some(_) => true,
        None => false,
    };

    if server.config.is_some() && scope_flags_selected(&command.scope) {
        return Err(CliError::Config(
            "--config cannot be combined with --user, --project, or --global for `plugins add`"
                .into(),
        ));
    }

    let (plugins_toml_path, state_path, scope) =
        scoped_paths_for_add(target_scope(&command.scope)?, server.config.as_ref())?;
    let scope_index = ensure_scope(&mut scopes, scope, plugins_toml_path.clone(), state_path);
    let record = validated_record_from_manifest(manifest, manifest_ref.clone())?;

    scopes[scope_index]
        .registry
        .add(record)
        .map_err(|error| CliError::Config(error.to_string()))?;
    append_dynamic_plugin_reference(&plugins_toml_path, &manifest_ref)?;
    if let Err(error) = scopes[scope_index].save() {
        let _ = remove_dynamic_plugin_reference(&plugins_toml_path, &plugin_id);
        return Err(error);
    }

    println!("Added dynamic plugin {}", plugin_id);
    println!("scope: {}", scope.label());
    println!("manifest: {}", manifest_ref);
    println!("plugins_toml: {}", plugins_toml_path.display());
    println!("state_path: {}", scopes[scope_index].state_path.display());
    println!("revived: {}", if revived { "true" } else { "false" });
    println!("desired.enabled: false");
    Ok(())
}

pub(crate) fn validate(
    command: PluginsValidateCommand,
    server: &ServerArgs,
) -> Result<(), CliError> {
    match PluginTarget::parse(&command.target) {
        PluginTarget::Path(path) => {
            if !path.exists() {
                return Err(plugin_not_found(
                    "plugins validate",
                    Some(command.target.clone()),
                    format!("dynamic plugin target '{}' does not exist", command.target),
                ));
            }
            let (manifest, manifest_ref) = load_manifest_for_action("validate", &path)?;
            if command.json {
                print_json(&validate_success_envelope(ValidateJsonContext {
                    command: "plugins validate",
                    target: Some(command.target.as_str()),
                    target_kind: "path",
                    resolved_plugin_id: Some(manifest.plugin.id.as_str()),
                    manifest: &manifest,
                    manifest_ref: &manifest_ref,
                    entry: None,
                    host_config: None,
                }))?;
            } else {
                println!(
                    "{}",
                    render_validation_summary(&manifest, &manifest_ref, None, None)
                );
            }
            Ok(())
        }
        PluginTarget::Id(plugin_id) => {
            let resolved = resolve_plugins_config(server.config.as_ref())?;
            let host_config_by_id = host_config_by_id(&resolved);
            let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
            let entry = find_registered_entry(&scopes, "plugins validate", &plugin_id)?;
            let manifest_ref = manifest_ref_from_record(&entry.record)?;
            let (manifest, manifest_ref) = load_manifest_for_action("validate", &manifest_ref)?;
            scopes[entry.scope_index]
                .registry
                .update_validation_status(
                    &plugin_id,
                    DynamicPluginValidationStatus {
                        manifest: DynamicPluginCheckState::Valid,
                        compatibility: DynamicPluginCheckState::Valid,
                        integrity: DynamicPluginCheckState::Unknown,
                        environment: DynamicPluginCheckState::Unknown,
                        authenticity: DynamicPluginCheckState::Unknown,
                        policy_satisfied: DynamicPluginCheckState::Unknown,
                        checked_at: None,
                        message: Some("validated by CLI".into()),
                    },
                )
                .map_err(|error| CliError::Config(error.to_string()))?;
            scopes[entry.scope_index].save()?;
            let refreshed = find_record_by_id(&scopes, &plugin_id)?
                .expect("validated registry record should still exist");
            if command.json {
                print_json(&validate_success_envelope(ValidateJsonContext {
                    command: "plugins validate",
                    target: Some(plugin_id.as_str()),
                    target_kind: "plugin_id",
                    resolved_plugin_id: Some(plugin_id.as_str()),
                    manifest: &manifest,
                    manifest_ref: &manifest_ref,
                    entry: Some(&refreshed),
                    host_config: host_config_by_id.get(&plugin_id),
                }))?;
            } else {
                println!(
                    "{}",
                    render_validation_summary(
                        &manifest,
                        &manifest_ref,
                        Some(&refreshed),
                        host_config_by_id.get(&plugin_id),
                    )
                );
            }
            Ok(())
        }
    }
}

pub(crate) fn list(command: PluginsListCommand, server: &ServerArgs) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let records = collect_records(&scopes, command.all);
    if records.is_empty() {
        if command.json {
            print_json(&list_success_envelope(
                "plugins list",
                None,
                &records,
                &host_config_by_id,
            ))?;
        } else {
            println!("No dynamic plugins registered.");
        }
        return Ok(());
    }
    if command.json {
        print_json(&list_success_envelope(
            "plugins list",
            None,
            &records,
            &host_config_by_id,
        ))?;
    } else {
        println!("{}", render_list(&records, &host_config_by_id));
    }
    Ok(())
}

pub(crate) fn inspect(command: PluginsInspectCommand, server: &ServerArgs) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let entry = find_registered_entry(&scopes, "plugins inspect", &command.id)?;
    let manifest_ref = manifest_ref_from_record(&entry.record)?;
    let (manifest, manifest_ref) = load_manifest_for_action("inspect", &manifest_ref)?;
    if command.json {
        print_json(&inspect_success_envelope(
            "plugins inspect",
            command.id.as_str(),
            &entry,
            &manifest,
            &manifest_ref,
            host_config_by_id.get(&command.id),
        ))?;
    } else {
        println!(
            "{}",
            render_inspect(
                &entry,
                &manifest,
                &manifest_ref,
                host_config_by_id.get(&command.id),
            )
        );
    }
    Ok(())
}

pub(crate) fn enable(command: PluginsEnableCommand, server: &ServerArgs) -> Result<(), CliError> {
    mutate_enabled_state(command.id, server, true)
}

pub(crate) fn disable(command: PluginsDisableCommand, server: &ServerArgs) -> Result<(), CliError> {
    mutate_enabled_state(command.id, server, false)
}

pub(crate) fn remove(command: PluginsRemoveCommand, server: &ServerArgs) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let entry = find_registered_entry(&scopes, "plugins remove", &command.id)?;
    let original_plugins_toml = std::fs::read(&entry.plugins_toml_path).ok();

    scopes[entry.scope_index]
        .registry
        .remove(&command.id)
        .map_err(|error| CliError::Config(error.to_string()))?;
    let removed_reference = remove_dynamic_plugin_reference(&entry.plugins_toml_path, &command.id)?;
    if let Err(error) = scopes[entry.scope_index].save() {
        let _ = restore_plugins_toml(&entry.plugins_toml_path, original_plugins_toml.as_deref());
        return Err(error);
    }

    println!("Removed dynamic plugin {}", command.id);
    println!("scope: {}", entry.scope.label());
    println!(
        "plugins_toml: {}",
        if removed_reference {
            entry.plugins_toml_path.display().to_string()
        } else {
            "<already absent>".into()
        }
    );
    println!("state_path: {}", entry.state_path.display());
    println!("status: tombstoned");
    Ok(())
}

fn mutate_enabled_state(
    plugin_id: String,
    server: &ServerArgs,
    enabled: bool,
) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let command = if enabled {
        "plugins enable"
    } else {
        "plugins disable"
    };
    let entry = find_registered_entry(&scopes, command, &plugin_id)?;
    if entry.record.is_tombstoned() {
        return Err(plugin_refused(
            command,
            Some(plugin_id.clone()),
            format!(
                "dynamic plugin '{}' is tombstoned and cannot be {}d",
                plugin_id,
                if enabled { "enable" } else { "disable" }
            ),
        ));
    }
    if enabled {
        scopes[entry.scope_index]
            .registry
            .enable(&plugin_id)
            .map_err(|error| CliError::Config(error.to_string()))?;
    } else {
        scopes[entry.scope_index]
            .registry
            .disable(&plugin_id)
            .map_err(|error| CliError::Config(error.to_string()))?;
    }
    scopes[entry.scope_index].save()?;

    println!(
        "{} dynamic plugin {}",
        if enabled { "Enabled" } else { "Disabled" },
        plugin_id
    );
    println!("scope: {}", entry.scope.label());
    println!("state_path: {}", entry.state_path.display());
    Ok(())
}

fn load_and_hydrate_scopes(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<Vec<ScopedRegistry>, CliError> {
    let mut scopes = load_scoped_registries(explicit)?;
    for plugin in &resolved.dynamic_plugins {
        if find_record_by_id(&scopes, &plugin.plugin_id)?.is_some() {
            continue;
        }
        let scope_index = scopes
            .iter()
            .position(|scope| scope.plugins_toml_path == plugin.source)
            .ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin '{}' resolved from {} but no matching lifecycle scope exists",
                    plugin.plugin_id,
                    plugin.source.display()
                ))
            })?;
        let (manifest, manifest_ref) = load_manifest_for_action("hydrate", &plugin.manifest_ref)?;
        scopes[scope_index]
            .registry
            .add(validated_record_from_manifest(manifest, manifest_ref)?)
            .map_err(|error| CliError::Config(error.to_string()))?;
    }
    Ok(scopes)
}

fn validated_record_from_manifest(
    manifest: DynamicPluginManifest,
    manifest_ref: String,
) -> Result<DynamicPluginRecord, CliError> {
    let mut record = manifest
        .into_record(Some(manifest_ref))
        .map_err(|error| CliError::Config(error.to_string()))?;
    record.status.validation = DynamicPluginValidationStatus {
        manifest: DynamicPluginCheckState::Valid,
        compatibility: DynamicPluginCheckState::Valid,
        integrity: DynamicPluginCheckState::Unknown,
        environment: DynamicPluginCheckState::Unknown,
        authenticity: DynamicPluginCheckState::Unknown,
        policy_satisfied: DynamicPluginCheckState::Unknown,
        checked_at: None,
        message: Some("validated by CLI".into()),
    };
    Ok(record)
}

fn host_config_by_id(resolved: &ResolvedConfig) -> HashMap<String, ResolvedDynamicPluginConfig> {
    resolved
        .dynamic_plugins
        .iter()
        .cloned()
        .map(|plugin| (plugin.plugin_id.clone(), plugin))
        .collect()
}

fn find_registered_entry(
    scopes: &[ScopedRegistry],
    command: &'static str,
    plugin_id: &str,
) -> Result<self::state::ScopedDynamicPluginRecord, CliError> {
    find_record_by_id(scopes, plugin_id)?.ok_or_else(|| {
        plugin_not_found(
            command,
            Some(plugin_id.to_owned()),
            format!(
                "dynamic plugin '{}' is not registered; run `nemo-relay plugins add <path>`",
                plugin_id
            ),
        )
    })
}

fn load_manifest_for_action(
    action: &str,
    path: impl Into<PathBuf>,
) -> Result<(DynamicPluginManifest, String), CliError> {
    let path = path.into();
    DynamicPluginManifest::load_from_path(&path)
        .map_err(|error| CliError::Config(format!("dynamic plugin {action} failed: {error}")))
}

fn manifest_ref_from_record(record: &DynamicPluginRecord) -> Result<String, CliError> {
    record.source.manifest_ref.clone().ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin '{}' has no manifest_ref in lifecycle state",
            record.metadata.id
        ))
    })
}

fn ensure_scope(
    scopes: &mut Vec<ScopedRegistry>,
    scope: RegistryScope,
    plugins_toml_path: PathBuf,
    state_path: PathBuf,
) -> usize {
    if let Some(index) = scopes.iter().position(|existing| {
        existing.scope == scope
            && existing.plugins_toml_path == plugins_toml_path
            && existing.state_path == state_path
    }) {
        return index;
    }
    scopes.push(ScopedRegistry {
        scope,
        plugins_toml_path,
        state_path,
        registry: nemo_relay::plugin::dynamic::DynamicPluginRegistry::new(),
    });
    scopes.len() - 1
}

fn scope_flags_selected(scope: &crate::config::PluginsScopeArgs) -> bool {
    scope.user || scope.project || scope.global
}

fn restore_plugins_toml(path: &std::path::Path, original: Option<&[u8]>) -> Result<(), CliError> {
    match original {
        Some(bytes) => std::fs::write(path, bytes)?,
        None if path.exists() => std::fs::remove_file(path)?,
        None => {}
    }
    Ok(())
}

pub(crate) fn render_plugin_error(
    error: &CliError,
    json: bool,
) -> Result<Option<ExitCode>, CliError> {
    let Some((command, target, kind, message)) = error.plugin_lifecycle() else {
        return Ok(None);
    };

    let exit_code = match kind {
        PluginLifecycleFailureKind::Failed => ExitCode::from(1),
        PluginLifecycleFailureKind::NotFound => ExitCode::from(2),
        PluginLifecycleFailureKind::Refused => ExitCode::from(3),
    };

    if json {
        print_json(&failure_envelope(command, target, kind, message))?;
    } else {
        eprintln!("{message}");
    }
    Ok(Some(exit_code))
}

pub(crate) fn render_generic_plugin_json_error(
    command: &'static str,
    target: Option<&str>,
    message: &str,
) -> Result<ExitCode, CliError> {
    print_json(&generic_failure_envelope(command, target, message))?;
    Ok(ExitCode::from(1))
}

fn plugin_not_found(
    command: &'static str,
    target: Option<String>,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::NotFound,
        message: message.into(),
    }
}

fn plugin_refused(
    command: &'static str,
    target: Option<String>,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::Refused,
        message: message.into(),
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/plugins_lifecycle_tests.rs"]
mod tests;
