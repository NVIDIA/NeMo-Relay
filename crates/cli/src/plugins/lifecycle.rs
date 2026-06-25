// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use base64::Engine;
use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginCompatibility,
    DynamicPluginFailure, DynamicPluginFailurePhase, DynamicPluginLoadContract,
    DynamicPluginManifest, DynamicPluginRecord, DynamicPluginValidationStatus,
};
use ring::signature::{ED25519, UnparsedPublicKey};
use serde_json::Value;
use sha2::{Digest, Sha256};

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

mod responses;
mod state;
mod target;

use self::responses::{
    ValidateResponseInput, failure, generic_failure, inspect_data, inspect_success, list_success,
    print_response_json, validate_success,
};
use self::state::{
    RegistryScope, ScopedDynamicPluginRecord, ScopedRegistry, collect_records, find_record_by_id,
    load_scoped_registries, scoped_paths_for_add,
};
use self::target::PluginTarget;

const VALIDATION_MESSAGE: &str = "validated by CLI";

pub(crate) fn add(command: PluginsAddCommand, server: &ServerArgs) -> Result<(), CliError> {
    let resolved = resolve_plugins_config(server.config.as_ref())?;
    let mut scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved)?;
    let (manifest, manifest_ref) = load_manifest_for_action("add", &command.path)?;
    let plugin_id = manifest.plugin.id.trim().to_owned();
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
        return Err(plugin_refused(
            "plugins add",
            Some(plugin_id.clone()),
            policy.refusal_message(&plugin_id),
        ));
    }
    if let Some(message) = trust.message.as_ref() {
        return Err(plugin_refused(
            "plugins add",
            Some(plugin_id.clone()),
            message.clone(),
        ));
    }
    let record = validated_record_from_manifest(manifest, manifest_ref.clone(), &policy, &trust)?;
    let original_plugins_toml = std::fs::read(&plugins_toml_path).ok();

    scopes[scope_index]
        .registry
        .add(record)
        .map_err(|error| CliError::Config(error.to_string()))?;
    append_dynamic_plugin_reference(&plugins_toml_path, &manifest_ref)?;
    if let Err(error) = scopes[scope_index].save() {
        let _ = restore_plugins_toml(&plugins_toml_path, original_plugins_toml.as_deref());
        return Err(error);
    }

    println!(
        "{} dynamic plugin {}",
        if revived { "Revived" } else { "Added" },
        plugin_id
    );
    Ok(())
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
            let policy =
                evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
            let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
            scopes[entry.scope_index]
                .registry
                .update_validation_status(
                    &plugin_id,
                    DynamicPluginValidationStatus {
                        manifest: DynamicPluginCheckState::Valid,
                        compatibility: DynamicPluginCheckState::Valid,
                        integrity: trust.integrity,
                        environment: DynamicPluginCheckState::Unknown,
                        authenticity: trust.authenticity,
                        policy_satisfied: policy.check_state(),
                        checked_at: None,
                        message: Some(VALIDATION_MESSAGE.into()),
                    },
                )
                .map_err(|error| CliError::Config(error.to_string()))?;
            update_registry_policy_status(&mut scopes[entry.scope_index], &plugin_id, &policy)?;
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

    println!("Removed dynamic plugin {}", command.id);
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
    let manifest_ref = manifest_ref_from_record(&entry.record)?;
    let (manifest, manifest_ref) = load_manifest_for_action(command, &manifest_ref)?;
    let policy = evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    update_registry_validation_status(&mut scopes[entry.scope_index], &plugin_id, &policy, &trust)?;
    if enabled && !policy.policy_satisfied {
        scopes[entry.scope_index].save()?;
        return Err(plugin_refused(
            command,
            Some(plugin_id.clone()),
            policy.refusal_message(&plugin_id),
        ));
    }
    if enabled && let Some(message) = trust.message.as_ref() {
        scopes[entry.scope_index].save()?;
        return Err(plugin_refused(
            command,
            Some(plugin_id.clone()),
            message.clone(),
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
    Ok(load_and_hydrate_scopes_with_updates(explicit, resolved)?.0)
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
                &policy,
                &trust,
            )?;
        } else {
            scopes[scope_index]
                .registry
                .add(validated_record_from_manifest(
                    manifest,
                    manifest_ref,
                    &policy,
                    &trust,
                )?)
                .map_err(|error| CliError::Config(error.to_string()))?;
        }
    }
    Ok((scopes, touched_scope_indices.into_iter().collect()))
}

fn validated_record_from_manifest(
    manifest: DynamicPluginManifest,
    manifest_ref: String,
    policy: &EvaluatedDynamicPluginHostPolicy,
    trust: &EvaluatedDynamicPluginTrust,
) -> Result<DynamicPluginRecord, CliError> {
    let mut record = manifest
        .into_record(Some(manifest_ref))
        .map_err(|error| CliError::Config(error.to_string()))?;
    record.status.validation = DynamicPluginValidationStatus {
        manifest: DynamicPluginCheckState::Valid,
        compatibility: DynamicPluginCheckState::Valid,
        integrity: trust.integrity,
        environment: DynamicPluginCheckState::Unknown,
        authenticity: trust.authenticity,
        policy_satisfied: policy.check_state(),
        checked_at: None,
        message: Some(VALIDATION_MESSAGE.into()),
    };
    record.status.startup_class = Some(policy.startup_class);
    record.status.attestation_mode = Some(policy.attestation_mode);
    record.status.last_error = policy
        .last_error()
        .or_else(|| trust.last_error(policy.attestation_mode));
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
            policy.last_error(),
        )
        .map_err(|error| CliError::Config(error.to_string()))
}

fn update_registry_validation_status(
    scope: &mut ScopedRegistry,
    plugin_id: &str,
    policy: &EvaluatedDynamicPluginHostPolicy,
    trust: &EvaluatedDynamicPluginTrust,
) -> Result<(), CliError> {
    scope
        .registry
        .update_validation_status(
            plugin_id,
            DynamicPluginValidationStatus {
                manifest: DynamicPluginCheckState::Valid,
                compatibility: DynamicPluginCheckState::Valid,
                integrity: trust.integrity,
                environment: DynamicPluginCheckState::Unknown,
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
                .last_error()
                .or_else(|| trust.last_error(policy.attestation_mode)),
        )
        .map_err(|error| CliError::Config(error.to_string()))
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

#[derive(Debug)]
struct EvaluatedDynamicPluginTrust {
    integrity: DynamicPluginCheckState,
    authenticity: DynamicPluginCheckState,
    message: Option<String>,
}

impl EvaluatedDynamicPluginTrust {
    fn last_error(
        &self,
        attestation_mode: DynamicPluginAttestationMode,
    ) -> Option<DynamicPluginFailure> {
        self.message.as_ref().map(|message| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Validation,
            code: match attestation_mode {
                DynamicPluginAttestationMode::IntegrityOnly => "integrity_verification_failed",
                DynamicPluginAttestationMode::SignatureIfPresent
                | DynamicPluginAttestationMode::SignatureRequired => {
                    "attestation_verification_failed"
                }
            }
            .into(),
            message: message.clone(),
        })
    }
}

fn evaluate_dynamic_plugin_trust(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> EvaluatedDynamicPluginTrust {
    let Some(artifact) = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
    else {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: Some(format!(
                "dynamic plugin '{}' is missing source.artifact required for integrity verification",
                manifest.plugin.id
            )),
        };
    };

    let Some(expected_digest) = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
    else {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: Some(format!(
                "dynamic plugin '{}' is missing integrity.sha256 required for host trust verification",
                manifest.plugin.id
            )),
        };
    };

    let artifact_path = resolve_artifact_path(manifest_ref, artifact);
    let actual_digest = match file_sha256(&artifact_path) {
        Ok(digest) => digest,
        Err(error) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Invalid,
                authenticity: DynamicPluginCheckState::Unknown,
                message: Some(format!(
                    "dynamic plugin '{}' artifact {} could not be read for integrity verification: {}",
                    manifest.plugin.id,
                    artifact_path.display(),
                    error
                )),
            };
        }
    };

    if actual_digest != expected_digest.trim() {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Invalid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: Some(format!(
                "dynamic plugin '{}' failed integrity verification for {}: expected {}, got {}",
                manifest.plugin.id,
                artifact_path.display(),
                expected_digest.trim(),
                actual_digest
            )),
        };
    }

    evaluate_authenticity(manifest, manifest_ref, artifact_path.as_path(), policy)
}

fn evaluate_authenticity(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    artifact_path: &Path,
    policy: &EvaluatedDynamicPluginHostPolicy,
) -> EvaluatedDynamicPluginTrust {
    let signature_ref = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty());

    match policy.attestation_mode {
        DynamicPluginAttestationMode::IntegrityOnly => EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Valid,
            authenticity: DynamicPluginCheckState::Unknown,
            message: None,
        },
        DynamicPluginAttestationMode::SignatureIfPresent => match signature_ref {
            Some(signature_ref) => verify_signature(
                manifest,
                manifest_ref,
                artifact_path,
                signature_ref,
                &policy.trusted_public_keys,
            ),
            None => EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Unknown,
                message: None,
            },
        },
        DynamicPluginAttestationMode::SignatureRequired => match signature_ref {
            Some(signature_ref) => verify_signature(
                manifest,
                manifest_ref,
                artifact_path,
                signature_ref,
                &policy.trusted_public_keys,
            ),
            None => EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Invalid,
                message: Some(format!(
                    "dynamic plugin '{}' requires integrity.signature under host policy",
                    manifest.plugin.id
                )),
            },
        },
    }
}

fn verify_signature(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
    artifact_path: &Path,
    signature_ref: &str,
    trusted_public_keys: &[String],
) -> EvaluatedDynamicPluginTrust {
    if trusted_public_keys.is_empty() {
        return EvaluatedDynamicPluginTrust {
            integrity: DynamicPluginCheckState::Valid,
            authenticity: DynamicPluginCheckState::Invalid,
            message: Some(format!(
                "dynamic plugin '{}' requires signature verification, but no trusted_public_keys are configured in host policy",
                manifest.plugin.id
            )),
        };
    }

    let signature_path = resolve_artifact_path(manifest_ref, signature_ref);
    let signature_bytes = match read_signature_bytes(&signature_path) {
        Ok(signature_bytes) => signature_bytes,
        Err(error) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Invalid,
                message: Some(format!(
                    "dynamic plugin '{}' signature {} could not be read: {}",
                    manifest.plugin.id,
                    signature_path.display(),
                    error
                )),
            };
        }
    };

    let artifact_bytes = match fs::read(artifact_path) {
        Ok(artifact_bytes) => artifact_bytes,
        Err(error) => {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Invalid,
                message: Some(format!(
                    "dynamic plugin '{}' artifact {} could not be read for signature verification: {}",
                    manifest.plugin.id,
                    artifact_path.display(),
                    error
                )),
            };
        }
    };

    let mut parse_errors = Vec::new();
    for trusted_public_key in trusted_public_keys {
        let public_key_bytes = match parse_ed25519_public_key(trusted_public_key) {
            Ok(public_key_bytes) => public_key_bytes,
            Err(error) => {
                parse_errors.push(error);
                continue;
            }
        };

        let verifier = UnparsedPublicKey::new(&ED25519, public_key_bytes);
        if verifier.verify(&artifact_bytes, &signature_bytes).is_ok() {
            return EvaluatedDynamicPluginTrust {
                integrity: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Valid,
                message: None,
            };
        }
    }

    let parse_error_suffix = if parse_errors.is_empty() {
        String::new()
    } else {
        format!("; key parse errors: {}", parse_errors.join("; "))
    };

    EvaluatedDynamicPluginTrust {
        integrity: DynamicPluginCheckState::Valid,
        authenticity: DynamicPluginCheckState::Invalid,
        message: Some(format!(
            "dynamic plugin '{}' failed signature verification for {} against configured host policy keys{}",
            manifest.plugin.id,
            signature_path.display(),
            parse_error_suffix
        )),
    }
}

fn read_signature_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let raw = fs::read(path).map_err(|error| error.to_string())?;
    let trimmed = String::from_utf8_lossy(&raw).trim().to_owned();
    if trimmed.is_empty() {
        return Err("signature file is empty".into());
    }

    let encoded = trimmed
        .strip_prefix("ed25519:")
        .unwrap_or(trimmed.as_str())
        .trim();
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| format!("invalid base64 signature: {error}"))
}

fn parse_ed25519_public_key(value: &str) -> Result<Vec<u8>, String> {
    let encoded = value
        .trim()
        .strip_prefix("ed25519:")
        .ok_or_else(|| format!("unsupported trusted public key format '{value}'"))?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("invalid ed25519 trusted public key '{value}': {error}"))
}

fn resolve_artifact_path(manifest_ref: &str, artifact_ref: &str) -> PathBuf {
    let artifact_path = PathBuf::from(artifact_ref);
    if artifact_path.is_absolute() {
        artifact_path
    } else {
        Path::new(manifest_ref)
            .parent()
            .map(|parent| parent.join(&artifact_path))
            .unwrap_or(artifact_path)
    }
}

fn file_sha256(path: &Path) -> Result<String, std::io::Error> {
    let bytes = fs::read(path)?;
    let mut digest = Sha256::new();
    digest.update(&bytes);
    Ok(format!(
        "sha256:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
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
        print_response_json(&failure(command, target, kind, message))?;
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
            let validation: &'static str = entry.record.status.validation.manifest.into();
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
                    let validation: &'static str = entry.record.status.validation.manifest.into();
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
        if self.policy.policy_satisfied && self.trust.message.is_none() {
            writeln!(f, "Dynamic plugin '{}' is valid.", self.manifest.plugin.id)?;
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
            "authenticity_state: {}",
            <&'static str>::from(self.trust.authenticity)
        )?;
        writeln!(f, "startup_class: {}", self.policy.startup_class)?;
        writeln!(f, "attestation_mode: {}", self.policy.attestation_mode)?;
        if let Some(message) = &self.policy.message {
            writeln!(f, "policy_error: {message}")?;
        }
        if let Some(message) = &self.trust.message {
            writeln!(f, "trust_error: {message}")?;
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
        .unwrap_or("missing")
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
