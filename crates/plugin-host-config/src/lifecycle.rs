// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::environment::{validate_environment_state, validate_python_entrypoint_artifact};
use crate::error::{PluginHostConfigError, Result};
use crate::io::load_bounded_dynamic_plugin_manifest;
use crate::policy::{EvaluatedDynamicPluginHostPolicy, evaluate_dynamic_plugin_host_policy};
use crate::resolver::{ResolvedDynamicPluginConfig, ResolvedPluginFileConfiguration};
use crate::snapshot::DynamicPluginActivationSnapshot;
use crate::state::{
    DynamicPluginLifecycleState, LifecycleStateLock, lock_lifecycle_state, pin_plugin_config_path,
    read_locked_lifecycle_state, save_locked_lifecycle_state, sibling_lifecycle_state_path,
};
use crate::trust::{EvaluatedDynamicPluginTrust, evaluate_dynamic_plugin_trust};
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationResource, DynamicPluginActivationSpec, DynamicPluginCheckState,
    DynamicPluginFailure, DynamicPluginFailurePhase, DynamicPluginKind, DynamicPluginLoadContract,
    DynamicPluginManifest, DynamicPluginRecord, DynamicPluginValidationStatus,
    PlannedDynamicPluginActivation, PluginHostActivationPlan, WorkerRuntime,
};

const VALIDATION_MESSAGE: &str = "validated by Relay plugin host";

/// One enabled, live lifecycle record ready for activation planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconciledDynamicPlugin {
    /// Canonical plugin ID.
    pub plugin_id: String,
    /// Dynamic execution lane.
    pub kind: DynamicPluginKind,
    /// Desired-state lifecycle generation.
    pub lifecycle_generation: u64,
    /// Canonical authored manifest reference retained by lifecycle state.
    pub manifest_ref: String,
    /// Lifecycle-managed worker environment, when applicable.
    pub environment_ref: Option<String>,
    /// Component-local host configuration.
    pub config: serde_json::Map<String, serde_json::Value>,
    /// Physical `plugins.toml` that owns the declaration and lifecycle record.
    pub source: PathBuf,
}

/// Durable lifecycle reconciliation result for one resolved plugin file configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconciledPluginLifecycle {
    /// Enabled, non-tombstoned dynamic plugins in declaration order.
    pub enabled_plugins: Vec<ReconciledDynamicPlugin>,
}

struct SourceRegistry {
    state_path: PathBuf,
    registry: DynamicPluginLifecycleState,
    _lock: LifecycleStateLock,
}

/// Reconciles declarations with source-local lifecycle state and durably saves validation status.
///
/// Missing records are hydrated disabled. This operation never installs a plugin, provisions an
/// environment, changes enablement, or edits `plugins.toml`.
pub fn reconcile_plugin_lifecycle(
    resolved: &ResolvedPluginFileConfiguration,
) -> Result<ReconciledPluginLifecycle> {
    if resolved.dynamic_plugins.is_empty() {
        return Ok(ReconciledPluginLifecycle::default());
    }

    let source_to_state = resolved
        .dynamic_plugins
        .iter()
        .map(|plugin| {
            (
                plugin.source.clone(),
                sibling_lifecycle_state_path(&plugin.source),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let declared_state_paths = source_to_state.values().cloned().collect::<BTreeSet<_>>();
    let mut state_paths = declared_state_paths.clone();
    for source in &resolved.selected_sources {
        let source = pin_plugin_config_path(source)?;
        let state_path = sibling_lifecycle_state_path(&source);
        if state_path.try_exists().map_err(|error| {
            PluginHostConfigError::io("inspect dynamic plugin lifecycle state", &state_path, error)
        })? {
            state_paths.insert(state_path);
        }
    }
    let mut locks = BTreeMap::new();
    for state_path in state_paths {
        locks.insert(state_path.clone(), lock_lifecycle_state(&state_path)?);
    }
    let mut registries = BTreeMap::new();
    for (state_path, lock) in locks {
        let registry = read_locked_lifecycle_state(&lock)?;
        registries.insert(
            state_path.clone(),
            SourceRegistry {
                state_path,
                registry,
                _lock: lock,
            },
        );
    }

    for declaration in &resolved.dynamic_plugins {
        let state_path = source_to_state.get(&declaration.source).ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin '{}' resolved from {} but no matching lifecycle scope exists",
                declaration.plugin_id,
                declaration.source.display()
            ))
        })?;
        let declaration_ref = lifecycle_declaration_ref(&declaration.source)?;
        let (manifest, manifest_ref) =
            load_bounded_dynamic_plugin_manifest(&declaration.manifest_ref)?;
        let reloaded_plugin_id = manifest.plugin.id.trim();
        if reloaded_plugin_id != declaration.plugin_id {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin manifest {} changed identity during lifecycle reconciliation: resolved '{}' but reloaded '{}'",
                manifest_ref, declaration.plugin_id, reloaded_plugin_id
            )));
        }
        validate_python_entrypoint_artifact(&manifest, &manifest_ref)
            .map_err(PluginHostConfigError::InvalidConfig)?;
        let policy =
            evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
        let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
        let mut conflicting_foreign_states = Vec::new();
        for (candidate_state_path, registry) in &registries {
            if candidate_state_path == state_path {
                continue;
            }
            let Some(record) = registry.registry.get(&declaration.plugin_id) else {
                continue;
            };
            if record.is_tombstoned() {
                continue;
            }
            conflicting_foreign_states.push(candidate_state_path.clone());
        }
        if !conflicting_foreign_states.is_empty() {
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin '{}' has live lifecycle state outside its declaring source {}: {}",
                declaration.plugin_id,
                declaration.source.display(),
                conflicting_foreign_states
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let source_registry = registries.get_mut(state_path).ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin '{}' resolved from {} but no matching lifecycle transaction exists",
                declaration.plugin_id,
                declaration.source.display()
            ))
        })?;
        let existing = source_registry
            .registry
            .get(&declaration.plugin_id)
            .cloned();
        match existing {
            Some(record)
                if source_registry
                    .registry
                    .declaration_source(&declaration.plugin_id)
                    .is_some_and(|owner| owner != declaration_ref) =>
            {
                if !record.is_tombstoned() {
                    return Err(PluginHostConfigError::InvalidConfig(format!(
                        "dynamic plugin '{}' has live lifecycle state owned by {} but is now declared by {}; remove it through the plugin control plane before moving the declaration",
                        declaration.plugin_id,
                        source_registry
                            .registry
                            .declaration_source(&declaration.plugin_id)
                            .expect("nonmatching lifecycle owner was checked"),
                        declaration.source.display()
                    )));
                }
                let record = validated_record_from_manifest(
                    manifest,
                    manifest_ref,
                    None,
                    &source_registry.state_path,
                    &policy,
                    &trust,
                )?;
                revive_registry_record_for_new_owner(
                    source_registry,
                    &declaration.plugin_id,
                    record,
                )?;
                source_registry
                    .registry
                    .set_declaration_source(&declaration.plugin_id, declaration_ref)?;
            }
            Some(_) => {
                if source_registry
                    .registry
                    .declaration_source(&declaration.plugin_id)
                    .is_none()
                {
                    source_registry
                        .registry
                        .set_declaration_source(&declaration.plugin_id, declaration_ref.clone())?;
                }
                refresh_registry_record(
                    source_registry,
                    &declaration.plugin_id,
                    manifest,
                    manifest_ref,
                    &policy,
                    &trust,
                )?;
            }
            None => {
                let record = validated_record_from_manifest(
                    manifest,
                    manifest_ref,
                    None,
                    &source_registry.state_path,
                    &policy,
                    &trust,
                )?;
                let validation = record.status.validation.clone();
                source_registry.registry.add(record)?;
                source_registry
                    .registry
                    .update_validation_status(&declaration.plugin_id, validation)?;
                source_registry
                    .registry
                    .set_declaration_source(&declaration.plugin_id, declaration_ref)?;
            }
        }
    }

    // Parse and reconcile every source before the first durable write. Each individual sibling
    // state file is then replaced atomically, matching the CLI control-plane contract.
    for state_path in &declared_state_paths {
        let registry = registries.get(state_path).ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin lifecycle transaction for {} disappeared before persistence",
                state_path.display()
            ))
        })?;
        save_registry(registry)?;
    }

    let mut enabled_plugins = Vec::new();
    for declaration in &resolved.dynamic_plugins {
        let state_path = source_to_state.get(&declaration.source).ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin '{}' resolved from {} but no matching lifecycle scope exists",
                declaration.plugin_id,
                declaration.source.display()
            ))
        })?;
        let source_registry = registries.get(state_path).ok_or_else(|| {
            PluginHostConfigError::InvalidConfig(format!(
                "dynamic plugin '{}' resolved from {} but no matching lifecycle transaction exists",
                declaration.plugin_id,
                declaration.source.display()
            ))
        })?;
        let record = source_registry
            .registry
            .get(&declaration.plugin_id)
            .ok_or_else(|| {
                PluginHostConfigError::InvalidConfig(format!(
                    "dynamic plugin '{}' has no reconciled lifecycle record in {}",
                    declaration.plugin_id,
                    source_registry.state_path.display()
                ))
            })?;
        if record.is_tombstoned() || !record.spec.enabled {
            continue;
        }
        if matches!(
            &record.load,
            DynamicPluginLoadContract::Worker(load) if load.runtime == WorkerRuntime::Python
        ) && record.status.validation.environment == DynamicPluginCheckState::Invalid
        {
            let message = record
                .status
                .last_error
                .as_ref()
                .filter(|failure| failure.code == "environment_failed")
                .map(|failure| failure.message.clone())
                .unwrap_or_else(|| {
                    format!(
                        "enabled Python worker dynamic plugin '{}' has an invalid lifecycle-managed environment",
                        declaration.plugin_id
                    )
                });
            return Err(PluginHostConfigError::InvalidConfig(format!(
                "enabled Python worker dynamic plugin '{}' has an invalid lifecycle-managed environment: {message}",
                declaration.plugin_id,
            )));
        }
        enabled_plugins.push(reconciled_enabled_plugin(declaration, record)?);
    }
    Ok(ReconciledPluginLifecycle { enabled_plugins })
}

/// Reconciles lifecycle state, snapshots enabled plugins, and builds a core activation plan.
pub fn prepare_plugin_host_activation(
    resolved: ResolvedPluginFileConfiguration,
) -> Result<PluginHostActivationPlan> {
    let reconciled = reconcile_plugin_lifecycle(&resolved)?;
    let mut dynamic_plugins = Vec::with_capacity(reconciled.enabled_plugins.len());
    for plugin in reconciled.enabled_plugins {
        let snapshot = DynamicPluginActivationSnapshot::create(
            &plugin.manifest_ref,
            &plugin.plugin_id,
            plugin.kind,
            plugin.environment_ref.as_deref(),
            &resolved.dynamic_plugin_policy,
        )?;
        let spec = DynamicPluginActivationSpec {
            plugin_id: plugin.plugin_id,
            kind: plugin.kind,
            manifest_ref: snapshot.activation_manifest_ref(),
            environment_ref: snapshot.activation_environment_ref().map(str::to_owned),
            config: plugin.config,
        };
        let resource: Arc<dyn DynamicPluginActivationResource> = snapshot;
        dynamic_plugins.push(PlannedDynamicPluginActivation { spec, resource });
    }
    Ok(PluginHostActivationPlan {
        config: resolved.config,
        dynamic_plugins,
        diagnostics: resolved.diagnostics,
    })
}

fn reconciled_enabled_plugin(
    declaration: &ResolvedDynamicPluginConfig,
    record: &DynamicPluginRecord,
) -> Result<ReconciledDynamicPlugin> {
    let manifest_ref = record.source.manifest_ref.clone().ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin '{}' has no manifest_ref in lifecycle state",
            record.metadata.id
        ))
    })?;
    Ok(ReconciledDynamicPlugin {
        plugin_id: record.metadata.id.clone(),
        kind: record.metadata.kind,
        lifecycle_generation: record.metadata.generation,
        manifest_ref,
        environment_ref: record.source.environment_ref.clone(),
        config: declaration.config.clone(),
        source: declaration.source.clone(),
    })
}

fn lifecycle_declaration_ref(source: &Path) -> Result<String> {
    Ok(pin_plugin_config_path(source)?
        .to_string_lossy()
        .into_owned())
}

fn validated_record_from_manifest(
    manifest: DynamicPluginManifest,
    manifest_ref: String,
    environment_ref: Option<String>,
    state_path: &Path,
    policy: &EvaluatedDynamicPluginHostPolicy,
    trust: &EvaluatedDynamicPluginTrust,
) -> Result<DynamicPluginRecord> {
    let (environment, environment_error) =
        match validate_environment_state(&manifest, state_path, environment_ref.as_deref()) {
            Ok(environment) => (environment, None),
            Err(error) => (
                DynamicPluginCheckState::Invalid,
                environment_ref.as_ref().map(|_| error.to_string()),
            ),
        };
    let mut record = manifest.into_record(Some(manifest_ref))?;
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
                environment_error,
            )
        });
    Ok(record)
}

fn refresh_registry_record(
    source: &mut SourceRegistry,
    plugin_id: &str,
    manifest: DynamicPluginManifest,
    manifest_ref: String,
    policy: &EvaluatedDynamicPluginHostPolicy,
    trust: &EvaluatedDynamicPluginTrust,
) -> Result<()> {
    let existing = source.registry.get(plugin_id).cloned().ok_or_else(|| {
        PluginHostConfigError::InvalidConfig(format!(
            "dynamic plugin '{plugin_id}' disappeared during lifecycle reconciliation"
        ))
    })?;
    let refreshed = validated_record_from_manifest(
        manifest,
        manifest_ref,
        existing.source.environment_ref.clone(),
        &source.state_path,
        policy,
        trust,
    )?;
    source
        .registry
        .refresh_manifest_record(plugin_id, refreshed)?;
    Ok(())
}

fn revive_registry_record_for_new_owner(
    source: &mut SourceRegistry,
    plugin_id: &str,
    record: DynamicPluginRecord,
) -> Result<()> {
    if source
        .registry
        .get(plugin_id)
        .is_some_and(|existing| !existing.is_tombstoned())
    {
        return Err(PluginHostConfigError::InvalidConfig(format!(
            "refusing to replace live lifecycle state for dynamic plugin '{plugin_id}'"
        )));
    }
    let validation = record.status.validation.clone();
    source.registry.add(record)?;
    source
        .registry
        .update_validation_status(plugin_id, validation)?;
    Ok(())
}

fn environment_last_error(
    plugin_id: &str,
    environment: DynamicPluginCheckState,
    environment_ref: Option<&str>,
    detail: Option<String>,
) -> Option<DynamicPluginFailure> {
    (environment == DynamicPluginCheckState::Invalid).then(|| DynamicPluginFailure {
        phase: DynamicPluginFailurePhase::Validation,
        code: "environment_failed".into(),
        message: detail.unwrap_or_else(|| {
            environment_ref.map_or_else(
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
            )
        }),
    })
}

fn save_registry(source: &SourceRegistry) -> Result<()> {
    save_locked_lifecycle_state(&source._lock, &source.registry)
}

#[cfg(test)]
#[path = "../tests/unit/lifecycle.rs"]
mod tests;
