// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginCompatibility, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginKind, DynamicPluginLoadContract, DynamicPluginManifest,
    DynamicPluginRecord, DynamicPluginValidationStatus,
};
use serde_json::{Map, Value};

use crate::config::{
    PluginsAddCommand, PluginsDisableCommand, PluginsEnableCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsRemoveCommand, PluginsValidateCommand, ResolvedConfig,
    ResolvedDynamicPluginConfig, ServerArgs, resolve_plugins_config,
};
use crate::error::{CliError, PluginLifecycleFailureKind};
use crate::plugins::policy::{
    EvaluatedDynamicPluginHostPolicy, evaluate_dynamic_plugin_host_policy,
};

use super::config_io::{
    append_dynamic_plugin_reference, remove_dynamic_plugin_reference, target_scope,
};
use super::schema::PluginConfigSchema;

mod environment;
mod responses;
mod state;
mod target;
mod trust;

use self::environment::{
    ProcessPythonEnvironmentCommandRunner, PythonEnvironmentCommandRunner, environment_state,
    provision_python_environment, remove_managed_environment,
};
use self::responses::{
    ValidateResponseInput, failure, generic_failure, inspect_data, inspect_success, list_success,
    print_response_json, validate_success,
};
use self::state::{
    RegistryScope, ScopedDynamicPluginRecord, ScopedRegistry, collect_records, find_record_by_id,
    load_scoped_registries, scoped_paths_for_add,
};
use self::target::PluginTarget;
use self::trust::{EvaluatedDynamicPluginTrust, evaluate_dynamic_plugin_trust};

const VALIDATION_MESSAGE: &str = "validated by CLI";

pub(crate) fn add(command: PluginsAddCommand, server: &ServerArgs) -> Result<(), CliError> {
    add_with_environment_runner(command, server, &ProcessPythonEnvironmentCommandRunner)
}

fn add_with_environment_runner(
    command: PluginsAddCommand,
    server: &ServerArgs,
    environment_runner: &impl PythonEnvironmentCommandRunner,
) -> Result<(), CliError> {
    const COMMAND: &str = "plugins add";

    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let (manifest, manifest_ref) = load_manifest_for_action("add", &command.path)?;
    let plugin_id = manifest.plugin.id.trim().to_owned();
    load_config_schema_for_manifest(&manifest, &manifest_ref)?;
    let revived = match find_record_by_id(&scopes, &plugin_id)? {
        Some(existing) if !existing.record.is_tombstoned() => {
            return Err(CliError::Config(format!(
                "dynamic plugin '{}' is already registered in the {} lifecycle scope",
                plugin_id, existing.scope
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
    let policy = evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    if !policy.policy_satisfied {
        return Err(plugin_refused_with_code(
            COMMAND,
            Some(plugin_id.clone()),
            "policy_blocked",
            policy
                .failure()
                .map(|failure| failure.display(&plugin_id).to_string())
                .unwrap_or_else(|| {
                    format!("dynamic plugin '{}' is blocked by host policy", plugin_id)
                }),
        ));
    }
    if let Some(failure) = trust.failure() {
        return Err(plugin_refused_with_code(
            COMMAND,
            Some(plugin_id.clone()),
            trust_refusal_code(&trust),
            failure.display(&plugin_id).to_string(),
        ));
    }
    let environment_ref = provision_python_environment(
        &manifest,
        &manifest_ref,
        &scopes[scope_index].state_path,
        environment_runner,
    )
    .map_err(|message| {
        plugin_failed_with_code(
            COMMAND,
            Some(plugin_id.clone()),
            "environment_failed",
            message,
        )
    })?;
    let environment_ref_string = environment_ref
        .as_ref()
        .map(|environment| environment.display().to_string());
    let record = match validated_record_from_manifest(
        manifest,
        manifest_ref.clone(),
        environment_ref_string,
        &scopes[scope_index].state_path,
        &policy,
        &trust,
    ) {
        Ok(record) => record,
        Err(error) => {
            cleanup_provisioned_environment(
                &scopes[scope_index].state_path,
                &plugin_id,
                environment_ref.as_deref(),
            );
            return Err(error);
        }
    };
    let original_plugins_toml = std::fs::read(&plugins_toml_path).ok();

    if let Err(error) = scopes[scope_index]
        .registry
        .add(record)
        .map_err(|error| CliError::Config(error.to_string()))
    {
        cleanup_provisioned_environment(
            &scopes[scope_index].state_path,
            &plugin_id,
            environment_ref.as_deref(),
        );
        return Err(error);
    }
    if let Err(error) = append_dynamic_plugin_reference(&plugins_toml_path, &manifest_ref) {
        cleanup_provisioned_environment(
            &scopes[scope_index].state_path,
            &plugin_id,
            environment_ref.as_deref(),
        );
        return Err(error);
    }
    if let Err(error) = scopes[scope_index].save() {
        let _ = restore_plugins_toml(&plugins_toml_path, original_plugins_toml.as_deref());
        cleanup_provisioned_environment(
            &scopes[scope_index].state_path,
            &plugin_id,
            environment_ref.as_deref(),
        );
        return Err(error);
    }

    println!(
        "{} dynamic plugin {}",
        if revived { "Revived" } else { "Added" },
        plugin_id
    );
    Ok(())
}

fn cleanup_provisioned_environment(state_path: &Path, plugin_id: &str, environment: Option<&Path>) {
    if let Some(environment) = environment {
        let _ = remove_managed_environment(
            state_path,
            plugin_id,
            environment.to_string_lossy().as_ref(),
        );
    }
}

pub(crate) fn enforce_required_dynamic_plugin_startup(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<(), CliError> {
    let (scopes, touched_scope_indices) = load_and_hydrate_scopes_with_updates(explicit, resolved)?;
    for scope_index in touched_scope_indices {
        scopes[scope_index].save()?;
    }
    let required_failures = collect_records(&scopes, false)
        .into_iter()
        .filter(|entry| entry.record.spec.enabled)
        .filter_map(|entry| required_startup_failure(&entry, resolved.dynamic_plugins.as_slice()))
        .collect::<Vec<_>>();

    if required_failures.is_empty() {
        return Ok(());
    }

    Err(CliError::Config(format!(
        "required dynamic plugin startup preflight failed:\n{}",
        required_failures.join("\n")
    )))
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
            let resolved = resolve_plugins_config(server.config.as_ref())?;
            let (manifest, manifest_ref) = load_manifest_for_action("validate", &path)?;
            load_config_schema_for_manifest(&manifest, &manifest_ref)?;
            let policy =
                evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
            let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
            if command.json {
                print_response_json(&validate_success(ValidateResponseInput {
                    command: "plugins validate",
                    target: Some(command.target.as_str()),
                    target_kind: "path",
                    resolved_plugin_id: Some(manifest.plugin.id.as_str()),
                    manifest: &manifest,
                    manifest_ref: &manifest_ref,
                    entry: None,
                    host_config: None,
                    policy: &policy,
                    trust: &trust,
                }))?;
            } else {
                println!(
                    "{}",
                    PluginValidationSummaryView {
                        manifest: &manifest,
                        manifest_ref: &manifest_ref,
                        entry: None,
                        host_config: None,
                        policy: &policy,
                        trust: &trust,
                    }
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
            if let Some(schema) = load_config_schema_for_manifest(&manifest, &manifest_ref)? {
                let config = host_config_by_id
                    .get(&plugin_id)
                    .map(|host_config| Value::Object(host_config.config.clone()))
                    .unwrap_or_else(|| Value::Object(Map::new()));
                schema.validate(&config)?;
            }
            let policy =
                evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
            let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
            update_registry_validation_status(
                &mut scopes[entry.scope_index],
                &plugin_id,
                &manifest,
                &policy,
                &trust,
            )?;
            scopes[entry.scope_index].save()?;
            let refreshed = find_record_by_id(&scopes, &plugin_id)?
                .expect("validated registry record should still exist");
            if command.json {
                print_response_json(&validate_success(ValidateResponseInput {
                    command: "plugins validate",
                    target: Some(plugin_id.as_str()),
                    target_kind: "plugin_id",
                    resolved_plugin_id: Some(plugin_id.as_str()),
                    manifest: &manifest,
                    manifest_ref: &manifest_ref,
                    entry: Some(&refreshed),
                    host_config: host_config_by_id.get(&plugin_id),
                    policy: &policy,
                    trust: &trust,
                }))?;
            } else {
                println!(
                    "{}",
                    PluginValidationSummaryView {
                        manifest: &manifest,
                        manifest_ref: &manifest_ref,
                        entry: Some(&refreshed),
                        host_config: host_config_by_id.get(&plugin_id),
                        policy: &policy,
                        trust: &trust,
                    }
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
            print_response_json(&list_success(
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
        print_response_json(&list_success(
            "plugins list",
            None,
            &records,
            &host_config_by_id,
        ))?;
    } else {
        println!(
            "{}",
            PluginListView {
                records: &records,
                host_config_by_id: &host_config_by_id,
            }
        );
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
        print_response_json(&inspect_success(
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
            PluginInspectView {
                entry: &entry,
                manifest: &manifest,
                manifest_ref: &manifest_ref,
                host_config: host_config_by_id.get(&command.id),
            }
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
    let mut scopes = load_scoped_registries(server.config.as_ref())?;
    if find_record_by_id(&scopes, &command.id)?.is_none() {
        let resolved = resolve_plugins_config(server.config.as_ref())?;
        scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    }
    let entry = find_registered_entry(&scopes, "plugins remove", &command.id)?;
    let original_plugins_toml = std::fs::read(&entry.plugins_toml_path).ok();
    let environment_ref = entry.record.source.environment_ref.clone();

    scopes[entry.scope_index]
        .registry
        .remove(&command.id)
        .map_err(|error| CliError::Config(error.to_string()))?;
    remove_dynamic_plugin_reference(
        &entry.plugins_toml_path,
        &command.id,
        entry.record.source.manifest_ref.as_deref(),
    )?;
    if let Err(error) = scopes[entry.scope_index].save() {
        let _ = restore_plugins_toml(&entry.plugins_toml_path, original_plugins_toml.as_deref());
        return Err(error);
    }

    if let Some(environment_ref) = environment_ref {
        remove_managed_environment(&entry.state_path, &command.id, &environment_ref)
            .map_err(CliError::Config)?;
        scopes[entry.scope_index]
            .registry
            .update_environment(&command.id, None, DynamicPluginCheckState::Unknown)
            .map_err(|error| CliError::Config(error.to_string()))?;
        scopes[entry.scope_index].save()?;
    }

    println!("Removed dynamic plugin {}", command.id);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveDynamicPluginComponent {
    pub(crate) plugin_id: String,
    pub(crate) kind: DynamicPluginKind,
    pub(crate) manifest_ref: Option<String>,
    pub(crate) environment_ref: Option<String>,
    pub(crate) config: Map<String, Value>,
}

pub(crate) fn active_dynamic_plugin_components(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<Vec<ActiveDynamicPluginComponent>, CliError> {
    let scopes = load_and_hydrate_scopes(explicit, resolved)?;
    let host_config_by_id = host_config_by_id(resolved);
    let mut components = Vec::new();

    for resolved_plugin in &resolved.dynamic_plugins {
        let Some(entry) = find_record_by_id(&scopes, &resolved_plugin.plugin_id)? else {
            return Err(CliError::Config(format!(
                "dynamic plugin '{}' is present in resolved config but not lifecycle state",
                resolved_plugin.plugin_id
            )));
        };
        if entry.record.is_tombstoned() || !entry.record.spec.enabled {
            continue;
        }
        let host_config = host_config_by_id
            .get(&entry.record.metadata.id)
            .ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin '{}' is enabled but has no resolved host config",
                    entry.record.metadata.id
                ))
            })?;
        components.push(ActiveDynamicPluginComponent {
            plugin_id: entry.record.metadata.id.clone(),
            kind: entry.record.metadata.kind,
            manifest_ref: match entry.record.metadata.kind {
                DynamicPluginKind::RustDynamic => Some(manifest_ref_from_record(&entry.record)?),
                DynamicPluginKind::Worker => entry.record.source.manifest_ref.clone(),
            },
            environment_ref: entry.record.source.environment_ref.clone(),
            config: host_config.config.clone(),
        });
    }

    Ok(components)
}

fn mutate_enabled_state(
    plugin_id: String,
    server: &ServerArgs,
    enabled: bool,
) -> Result<(), CliError> {
    let command = if enabled {
        "plugins enable"
    } else {
        "plugins disable"
    };
    let mut scopes = if enabled {
        let resolved = resolve_plugins_config(server.config.as_ref())?;
        let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
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
        let manifest_ref = manifest_ref_from_record(&entry.record)?;
        let (manifest, manifest_ref) = load_manifest_for_action(command, &manifest_ref)?;
        let policy =
            evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
        let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
        update_registry_validation_status(
            &mut scopes[entry.scope_index],
            &plugin_id,
            &manifest,
            &policy,
            &trust,
        )?;
        if !policy.policy_satisfied {
            scopes[entry.scope_index].save()?;
            return Err(plugin_refused_with_code(
                command,
                Some(plugin_id.clone()),
                "policy_blocked",
                policy
                    .failure()
                    .map(|failure| failure.display(&plugin_id).to_string())
                    .unwrap_or_else(|| {
                        format!("dynamic plugin '{}' is blocked by host policy", plugin_id)
                    }),
            ));
        }
        if let Some(failure) = trust.failure() {
            scopes[entry.scope_index].save()?;
            return Err(plugin_refused_with_code(
                command,
                Some(plugin_id.clone()),
                trust_refusal_code(&trust),
                failure.display(&plugin_id).to_string(),
            ));
        }
        if let Some(environment_error) = scopes[entry.scope_index]
            .registry
            .get(&plugin_id)
            .and_then(|record| record.status.last_error.as_ref())
            .filter(|error| error.code == "environment_failed")
        {
            let message = environment_error.message.clone();
            scopes[entry.scope_index].save()?;
            return Err(plugin_refused_with_code(
                command,
                Some(plugin_id.clone()),
                "environment_failed",
                message,
            ));
        }
        scopes
    } else {
        load_scoped_registries(server.config.as_ref())?
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
    Ok(())
}

fn load_and_hydrate_scopes(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<Vec<ScopedRegistry>, CliError> {
    let (scopes, touched_scope_indices) = load_and_hydrate_scopes_with_updates(explicit, resolved)?;
    for scope_index in touched_scope_indices {
        scopes[scope_index].save()?;
    }
    Ok(scopes)
}

fn load_and_hydrate_scopes_with_updates(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<(Vec<ScopedRegistry>, Vec<usize>), CliError> {
    let mut scopes = load_scoped_registries(explicit)?;
    let mut touched_scope_indices = BTreeSet::new();
    for plugin in &resolved.dynamic_plugins {
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
        touched_scope_indices.insert(scope_index);
        let (manifest, manifest_ref) = load_manifest_for_action("hydrate", &plugin.manifest_ref)?;
        let policy =
            evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
        let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
        if find_record_by_id(&scopes, &plugin.plugin_id)?.is_some() {
            update_registry_validation_status(
                &mut scopes[scope_index],
                &plugin.plugin_id,
                &manifest,
                &policy,
                &trust,
            )?;
        } else {
            let state_path = scopes[scope_index].state_path.clone();
            let record = validated_record_from_manifest(
                manifest,
                manifest_ref,
                None,
                &state_path,
                &policy,
                &trust,
            )?;
            scopes[scope_index]
                .registry
                .add(record)
                .map_err(|error| CliError::Config(error.to_string()))?;
        }
    }
    Ok((scopes, touched_scope_indices.into_iter().collect()))
}

fn validated_record_from_manifest(
    manifest: DynamicPluginManifest,
    manifest_ref: String,
    environment_ref: Option<String>,
    state_path: &Path,
    policy: &EvaluatedDynamicPluginHostPolicy,
    trust: &EvaluatedDynamicPluginTrust,
) -> Result<DynamicPluginRecord, CliError> {
    let environment = environment_state(&manifest, state_path, environment_ref.as_deref());
    let mut record = manifest
        .into_record(Some(manifest_ref))
        .map_err(|error| CliError::Config(error.to_string()))?;
    record.source.environment_ref = environment_ref;
    record.status.validation = DynamicPluginValidationStatus {
        manifest: DynamicPluginCheckState::Valid,
        compatibility: DynamicPluginCheckState::Valid,
        integrity: trust.integrity,
        environment,
        authenticity: trust.authenticity,
        policy_satisfied: policy.check_state(),
        checked_at: None,
        message: Some(VALIDATION_MESSAGE.into()),
    };
    record.status.startup_class = Some(policy.startup_class);
    record.status.attestation_mode = Some(policy.attestation_mode);
    record.status.last_error = policy
        .last_error(&record.metadata.id)
        .or_else(|| trust.last_error(&record.metadata.id))
        .or_else(|| {
            environment_last_error(
                &record.metadata.id,
                environment,
                record.source.environment_ref.as_deref(),
            )
        });
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

fn update_registry_policy_status(
    scope: &mut ScopedRegistry,
    plugin_id: &str,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> Result<(), CliError> {
    scope
        .registry
        .update_policy_status(
            plugin_id,
            policy.check_state(),
            policy.startup_class,
            policy.attestation_mode,
            policy.last_error(plugin_id),
        )
        .map_err(|error| CliError::Config(error.to_string()))
}

fn update_registry_validation_status(
    scope: &mut ScopedRegistry,
    plugin_id: &str,
    manifest: &DynamicPluginManifest,
    policy: &EvaluatedDynamicPluginHostPolicy,
    trust: &EvaluatedDynamicPluginTrust,
) -> Result<(), CliError> {
    let environment_ref = scope
        .registry
        .get(plugin_id)
        .and_then(|record| record.source.environment_ref.as_deref());
    let environment = environment_state(manifest, &scope.state_path, environment_ref);
    let environment_error = environment_last_error(plugin_id, environment, environment_ref);
    scope
        .registry
        .update_validation_status(
            plugin_id,
            DynamicPluginValidationStatus {
                manifest: DynamicPluginCheckState::Valid,
                compatibility: DynamicPluginCheckState::Valid,
                integrity: trust.integrity,
                environment,
                authenticity: trust.authenticity,
                policy_satisfied: policy.check_state(),
                checked_at: None,
                message: Some(VALIDATION_MESSAGE.into()),
            },
        )
        .map_err(|error| CliError::Config(error.to_string()))?;
    update_registry_policy_status(scope, plugin_id, policy)?;
    scope
        .registry
        .update_last_error(
            plugin_id,
            policy
                .last_error(plugin_id)
                .or_else(|| trust.last_error(plugin_id))
                .or(environment_error),
        )
        .map_err(|error| CliError::Config(error.to_string()))
}

fn environment_last_error(
    plugin_id: &str,
    environment: DynamicPluginCheckState,
    environment_ref: Option<&str>,
) -> Option<DynamicPluginFailure> {
    (environment == DynamicPluginCheckState::Invalid).then(|| DynamicPluginFailure {
        phase: DynamicPluginFailurePhase::Validation,
        code: "environment_failed".into(),
        message: environment_ref.map_or_else(
            || {
                format!(
                    "dynamic plugin '{}' has no lifecycle-managed Python environment; run `nemo-relay plugins remove {}` to remove the manual registration, then run `nemo-relay plugins add <path>`",
                    plugin_id, plugin_id
                )
            },
            |environment_ref| {
                format!(
                    "dynamic plugin '{}' configured Python environment {} is unavailable",
                    plugin_id, environment_ref
                )
            },
        ),
    })
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

fn load_config_schema_for_manifest(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
) -> Result<Option<PluginConfigSchema>, CliError> {
    let schema_path = manifest
        .resolve_config_schema_path(manifest_ref)
        .map_err(|error| {
            CliError::Config(format!(
                "dynamic plugin '{}' config schema path could not be resolved from '{}': {error}",
                manifest.plugin.id, manifest_ref
            ))
        })?;
    schema_path
        .map(|path| PluginConfigSchema::load(manifest.plugin.id.trim(), path))
        .transpose()
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

fn required_startup_failure(
    entry: &ScopedDynamicPluginRecord,
    resolved_plugins: &[ResolvedDynamicPluginConfig],
) -> Option<String> {
    if entry.record.status.startup_class
        != Some(nemo_relay::plugin::dynamic::DynamicPluginStartupClass::Required)
    {
        return None;
    }

    if entry.record.status.validation.policy_satisfied == DynamicPluginCheckState::Invalid {
        return Some(format!(
            "- {}: {}",
            entry.record.metadata.id,
            entry
                .record
                .status
                .last_error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or("blocked by host policy")
        ));
    }
    if entry.record.status.validation.integrity == DynamicPluginCheckState::Invalid
        || entry.record.status.validation.authenticity == DynamicPluginCheckState::Invalid
    {
        return Some(format!(
            "- {}: {}",
            entry.record.metadata.id,
            entry
                .record
                .status
                .last_error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or("required dynamic plugin trust verification failed")
        ));
    }
    if entry.record.status.validation.environment == DynamicPluginCheckState::Invalid {
        return Some(format!(
            "- {}: {}",
            entry.record.metadata.id,
            entry
                .record
                .status
                .last_error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or("required dynamic plugin environment is unavailable")
        ));
    }

    let manifest_ref = entry
        .record
        .source
        .manifest_ref
        .as_deref()
        .map(Path::new)
        .map(Path::to_path_buf);
    if manifest_ref.is_none() {
        return Some(format!(
            "- {}: required dynamic plugin has no manifest_ref in lifecycle state",
            entry.record.metadata.id
        ));
    }

    let manifest_ref = manifest_ref.expect("manifest_ref checked above");
    if !resolved_plugins
        .iter()
        .any(|plugin| plugin.plugin_id == entry.record.metadata.id)
    {
        if !manifest_ref.exists() {
            return Some(format!(
                "- {}: required dynamic plugin manifest is no longer available at {}",
                entry.record.metadata.id,
                manifest_ref.display()
            ));
        }

        if let Err(error) = DynamicPluginManifest::load_from_path(&manifest_ref) {
            return Some(format!(
                "- {}: required dynamic plugin manifest at {} is unreadable: {}",
                entry.record.metadata.id,
                manifest_ref.display(),
                error
            ));
        }
    }

    None
}

pub(crate) fn render_plugin_error(
    error: &CliError,
    json: bool,
) -> Result<Option<ExitCode>, CliError> {
    let Some((command, target, kind, code, message)) = error.as_plugin_lifecycle_error_context()
    else {
        return Ok(None);
    };

    let exit_code = match kind {
        PluginLifecycleFailureKind::Failed => ExitCode::from(1),
        PluginLifecycleFailureKind::NotFound => ExitCode::from(2),
        PluginLifecycleFailureKind::Refused => ExitCode::from(3),
    };

    if json {
        print_response_json(&failure(command, target, kind, code, message))?;
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
    print_response_json(&generic_failure(command, target, message))?;
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
        code: None,
        message: message.into(),
    }
}

fn plugin_refused(
    command: &'static str,
    target: Option<String>,
    message: impl Into<String>,
) -> CliError {
    plugin_refused_with_code(command, target, "refused", message)
}

fn plugin_refused_with_code(
    command: &'static str,
    target: Option<String>,
    code: &'static str,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::Refused,
        code: Some(code),
        message: message.into(),
    }
}

fn plugin_failed_with_code(
    command: &'static str,
    target: Option<String>,
    code: &'static str,
    message: impl Into<String>,
) -> CliError {
    CliError::PluginLifecycle {
        command,
        target,
        kind: PluginLifecycleFailureKind::Failed,
        code: Some(code),
        message: message.into(),
    }
}

fn trust_refusal_code(trust: &EvaluatedDynamicPluginTrust) -> &'static str {
    trust.refusal_code().unwrap_or("refused")
}

fn list_validation_state(record: &DynamicPluginRecord) -> DynamicPluginCheckState {
    let validation = &record.status.validation;
    if validation.manifest == DynamicPluginCheckState::Invalid
        || validation.compatibility == DynamicPluginCheckState::Invalid
        || validation.integrity == DynamicPluginCheckState::Invalid
        || validation.environment == DynamicPluginCheckState::Invalid
        || validation.authenticity == DynamicPluginCheckState::Invalid
        || validation.policy_satisfied == DynamicPluginCheckState::Invalid
    {
        DynamicPluginCheckState::Invalid
    } else if validation.manifest == DynamicPluginCheckState::Unknown
        || validation.compatibility == DynamicPluginCheckState::Unknown
    {
        DynamicPluginCheckState::Unknown
    } else {
        DynamicPluginCheckState::Valid
    }
}

struct PluginListView<'a> {
    records: &'a [ScopedDynamicPluginRecord],
    host_config_by_id: &'a HashMap<String, ResolvedDynamicPluginConfig>,
}

impl fmt::Display for PluginListView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths = PluginListWidths::from_records(self.records);

        write!(
            f,
            "{:<id_width$} {:<scope_width$} {:<enabled_width$} {:<state_width$} {:<validation_width$} {:<policy_width$} HOST CONFIG",
            "ID",
            "SCOPE",
            "ENABLED",
            "STATE",
            "VALIDATION",
            "POLICY",
            id_width = widths.id,
            scope_width = widths.scope,
            enabled_width = widths.enabled,
            state_width = widths.state,
            validation_width = widths.validation,
            policy_width = widths.policy,
        )?;
        for entry in self.records {
            let scope: &'static str = entry.scope.into();
            let validation: &'static str = list_validation_state(&entry.record).into();
            let policy: &'static str = entry.record.status.validation.policy_satisfied.into();
            write!(
                f,
                "\n{:<id_width$} {:<scope_width$} {:<enabled_width$} {:<state_width$} {:<validation_width$} {:<policy_width$} {}",
                entry.record.metadata.id,
                scope,
                entry.record.spec.enabled,
                lifecycle_state_label(&entry.record),
                validation,
                policy,
                host_config_label(self.host_config_by_id.get(&entry.record.metadata.id)),
                id_width = widths.id,
                scope_width = widths.scope,
                enabled_width = widths.enabled,
                state_width = widths.state,
                validation_width = widths.validation,
                policy_width = widths.policy,
            )?;
        }
        Ok(())
    }
}

struct PluginListWidths {
    id: usize,
    scope: usize,
    enabled: usize,
    state: usize,
    validation: usize,
    policy: usize,
}

impl PluginListWidths {
    fn from_records(records: &[ScopedDynamicPluginRecord]) -> Self {
        Self {
            id: column_width(
                "ID",
                records
                    .iter()
                    .map(|entry| entry.record.metadata.id.as_str()),
            ),
            scope: column_width(
                "SCOPE",
                records.iter().map(|entry| {
                    let scope: &'static str = entry.scope.into();
                    scope
                }),
            ),
            enabled: column_width(
                "ENABLED",
                records.iter().map(|entry| {
                    if entry.record.spec.enabled {
                        "true"
                    } else {
                        "false"
                    }
                }),
            ),
            state: column_width(
                "STATE",
                records
                    .iter()
                    .map(|entry| lifecycle_state_label(&entry.record)),
            ),
            validation: column_width(
                "VALIDATION",
                records.iter().map(|entry| {
                    let validation: &'static str = list_validation_state(&entry.record).into();
                    validation
                }),
            ),
            policy: column_width(
                "POLICY",
                records.iter().map(|entry| {
                    let policy: &'static str =
                        entry.record.status.validation.policy_satisfied.into();
                    policy
                }),
            ),
        }
    }
}

fn column_width<'a>(header: &'static str, values: impl Iterator<Item = &'a str>) -> usize {
    values
        .map(str::len)
        .chain(std::iter::once(header.len()))
        .max()
        .unwrap_or(header.len())
}

struct PluginInspectView<'a> {
    entry: &'a ScopedDynamicPluginRecord,
    manifest: &'a DynamicPluginManifest,
    manifest_ref: &'a str,
    host_config: Option<&'a ResolvedDynamicPluginConfig>,
}

impl fmt::Display for PluginInspectView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let view = inspect_data(
            self.entry,
            self.manifest,
            self.manifest_ref,
            self.host_config,
        );
        let yaml = serde_yaml::to_string(&view).map_err(|_| fmt::Error)?;
        write!(f, "{}", yaml.trim_end())
    }
}

struct PluginValidationSummaryView<'a> {
    manifest: &'a DynamicPluginManifest,
    manifest_ref: &'a str,
    entry: Option<&'a ScopedDynamicPluginRecord>,
    host_config: Option<&'a ResolvedDynamicPluginConfig>,
    policy: &'a EvaluatedDynamicPluginHostPolicy,
    trust: &'a EvaluatedDynamicPluginTrust,
}

impl fmt::Display for PluginValidationSummaryView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let environment = self
            .entry
            .map(|entry| entry.record.status.validation.environment)
            .unwrap_or(DynamicPluginCheckState::Unknown);
        if self.policy.policy_satisfied
            && self.trust.is_satisfied()
            && environment != DynamicPluginCheckState::Invalid
        {
            writeln!(f, "Dynamic plugin '{}' is valid.", self.manifest.plugin.id)?;
        } else if self.policy.policy_satisfied
            && self.trust.is_satisfied()
            && environment == DynamicPluginCheckState::Invalid
        {
            writeln!(
                f,
                "Dynamic plugin '{}' manifest is valid, but its runtime environment is unavailable.",
                self.manifest.plugin.id
            )?;
        } else if self.policy.policy_satisfied {
            writeln!(
                f,
                "Dynamic plugin '{}' manifest is valid, but trust verification blocks it.",
                self.manifest.plugin.id
            )?;
        } else {
            writeln!(
                f,
                "Dynamic plugin '{}' manifest is valid, but host policy blocks it.",
                self.manifest.plugin.id
            )?;
        }
        writeln!(f, "kind: {}", self.manifest.plugin.kind)?;
        writeln!(
            f,
            "policy_state: {}",
            <&'static str>::from(self.policy.check_state())
        )?;
        writeln!(
            f,
            "integrity_state: {}",
            <&'static str>::from(self.trust.integrity)
        )?;
        writeln!(
            f,
            "environment_state: {}",
            <&'static str>::from(environment)
        )?;
        writeln!(
            f,
            "authenticity_state: {}",
            <&'static str>::from(self.trust.authenticity)
        )?;
        writeln!(f, "startup_class: {}", self.policy.startup_class)?;
        writeln!(f, "attestation_mode: {}", self.policy.attestation_mode)?;
        if let Some(failure) = self.policy.failure() {
            writeln!(
                f,
                "policy_error: {}",
                failure.display(&self.manifest.plugin.id)
            )?;
        }
        if let Some(failure) = self.trust.failure() {
            writeln!(
                f,
                "trust_error: {}",
                failure.display(&self.manifest.plugin.id)
            )?;
        }
        if let Some(entry) = self.entry {
            writeln!(f, "manifest: {}", self.manifest_ref)?;
            writeln!(f, "scope: {}", entry.scope)?;
            writeln!(f, "lifecycle_state_path: {}", entry.state_path.display())?;
            writeln!(f, "desired.enabled: {}", entry.record.spec.enabled)?;
            write!(f, "host_config: {}", host_config_label(self.host_config))?;
        } else {
            write!(f, "manifest: {}", self.manifest_ref)?;
        }
        Ok(())
    }
}

fn lifecycle_state_label(record: &DynamicPluginRecord) -> &'static str {
    if record.is_tombstoned() {
        "tombstoned"
    } else {
        record.status.runtime.state.into()
    }
}

fn host_config_label(host_config: Option<&ResolvedDynamicPluginConfig>) -> &'static str {
    host_config
        .map(|plugin| {
            let status: &'static str = plugin.host_config_status().into();
            status
        })
        .unwrap_or("absent")
}

fn redacted_host_config_json(host_config: &ResolvedDynamicPluginConfig) -> Value {
    if host_config.config.is_empty() && !host_config.has_explicit_config {
        return Value::Null;
    }

    Value::Object(
        host_config
            .config
            .keys()
            .cloned()
            .map(|key| (key, Value::String("<redacted>".into())))
            .collect(),
    )
}

pub(super) fn inspect_load_data(record: &DynamicPluginRecord) -> Value {
    match &record.load {
        DynamicPluginLoadContract::Worker(load) => serde_json::json!({
            "runtime": load.runtime,
            "entrypoint": load.entrypoint,
        }),
        DynamicPluginLoadContract::RustDynamic(load) => serde_json::json!({
            "library": load.library,
            "symbol": load.symbol,
        }),
    }
}

pub(super) fn inspect_compat_data(record: &DynamicPluginRecord) -> Value {
    match &record.compatibility {
        DynamicPluginCompatibility::Worker(compatibility) => serde_json::json!({
            "relay": compatibility.relay,
            "worker_protocol": compatibility.worker_protocol,
        }),
        DynamicPluginCompatibility::RustDynamic(compatibility) => serde_json::json!({
            "relay": compatibility.relay,
            "native_api": compatibility.native_api,
        }),
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/plugins_lifecycle_tests.rs"]
mod tests;
