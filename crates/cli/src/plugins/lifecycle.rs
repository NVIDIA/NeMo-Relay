// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginManifest, DynamicPluginRecord,
    DynamicPluginValidationStatus,
};

use crate::config::{
    PluginsAddCommand, PluginsDisableCommand, PluginsEnableCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsRemoveCommand, PluginsValidateCommand, ResolvedConfig,
    ResolvedDynamicPluginConfig, ServerArgs, resolve_plugins_config,
};
use crate::error::CliError;

use super::config_io::{
    append_dynamic_plugin_reference, remove_dynamic_plugin_reference, target_scope,
};

mod render;
mod state;
mod target;

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
    if let Some(existing) = find_record_by_id(&scopes, &plugin_id)?
        && !existing.record.is_tombstoned()
    {
        return Err(CliError::Config(format!(
            "dynamic plugin '{}' is already registered in the {} lifecycle scope",
            plugin_id,
            existing.scope.label()
        )));
    }

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
    println!("desired.enabled: false");
    Ok(())
}

pub(crate) fn validate(
    command: PluginsValidateCommand,
    server: &ServerArgs,
) -> Result<(), CliError> {
    match PluginTarget::parse(&command.target) {
        PluginTarget::Path(path) => {
            let (manifest, manifest_ref) = load_manifest_for_action("validate", &path)?;
            println!(
                "{}",
                render_validation_summary(&manifest, &manifest_ref, None, None)
            );
            Ok(())
        }
        PluginTarget::Id(plugin_id) => {
            let resolved = resolve_plugins_config(server.config.as_ref())?;
            let host_config_by_id = host_config_by_id(&resolved);
            let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
            let entry = find_record_by_id(&scopes, &plugin_id)?.ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin '{}' is not registered; run `nemo-relay plugins add <path>`",
                    plugin_id
                ))
            })?;
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
            println!(
                "{}",
                render_validation_summary(
                    &manifest,
                    &manifest_ref,
                    Some(&refreshed),
                    host_config_by_id.get(&plugin_id),
                )
            );
            Ok(())
        }
    }
}

pub(crate) fn list(command: PluginsListCommand, server: &ServerArgs) -> Result<(), CliError> {
    let _ = command;
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let records = collect_records(&scopes, false);
    if records.is_empty() {
        println!("No dynamic plugins registered.");
        return Ok(());
    }
    println!("{}", render_list(&records, &host_config_by_id));
    Ok(())
}

pub(crate) fn inspect(command: PluginsInspectCommand, server: &ServerArgs) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let entry = find_record_by_id(&scopes, &command.id)?.ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin '{}' is not registered; run `nemo-relay plugins add <path>`",
            command.id
        ))
    })?;
    let manifest_ref = manifest_ref_from_record(&entry.record)?;
    let (manifest, manifest_ref) = load_manifest_for_action("inspect", &manifest_ref)?;
    println!(
        "{}",
        render_inspect(
            &entry,
            &manifest,
            &manifest_ref,
            host_config_by_id.get(&command.id),
        )
    );
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
    let entry = find_record_by_id(&scopes, &command.id)?.ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin '{}' is not registered; run `nemo-relay plugins add <path>`",
            command.id
        ))
    })?;

    scopes[entry.scope_index]
        .registry
        .remove(&command.id)
        .map_err(|error| CliError::Config(error.to_string()))?;
    let removed_reference = remove_dynamic_plugin_reference(&entry.plugins_toml_path, &command.id)?;
    scopes[entry.scope_index].save()?;

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
    let entry = find_record_by_id(&scopes, &plugin_id)?.ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin '{}' is not registered; run `nemo-relay plugins add <path>`",
            plugin_id
        ))
    })?;
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

#[cfg(test)]
#[path = "../../tests/coverage/plugins_lifecycle_tests.rs"]
mod tests;
