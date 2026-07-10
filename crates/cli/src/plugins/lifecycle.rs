// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginCompatibility, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginKind, DynamicPluginLoadContract, DynamicPluginManifest,
    DynamicPluginManifestLoad, DynamicPluginRecord, DynamicPluginValidationStatus, WorkerRuntime,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::config::{
    MAX_BOOTSTRAP_IDENTITY_FILE_BYTES, PluginsAddCommand, PluginsDisableCommand,
    PluginsEnableCommand, PluginsInspectCommand, PluginsListCommand, PluginsRemoveCommand,
    PluginsValidateCommand, ResolvedConfig, ResolvedDynamicPluginConfig, ServerArgs,
    load_bounded_dynamic_plugin_manifest_bytes, read_bounded_regular_file, resolve_plugins_config,
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
    ENVIRONMENT_ATTESTATION_FILE, MANAGED_ENVIRONMENTS_DIR, ProcessPythonEnvironmentCommandRunner,
    PythonEnvironmentCommandRunner, environment_state, provision_python_environment,
    read_environment_attestation, remove_managed_environment, validate_python_entrypoint_artifact,
    verify_environment_attestation,
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

#[cfg(test)]
pub(crate) fn attest_test_python_environment(
    environment: &Path,
    source_artifact_sha256: &str,
) -> Result<(), String> {
    self::environment::write_environment_attestation(environment, source_artifact_sha256)
}

#[cfg(test)]
pub(crate) fn reset_test_python_environment_digest_calls() {
    self::environment::reset_environment_tree_digest_calls();
}

#[cfg(test)]
pub(crate) fn test_python_environment_digest_calls() -> usize {
    self::environment::environment_tree_digest_calls()
}

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
    pub(crate) lifecycle_generation: u64,
    pub(crate) manifest_ref: Option<String>,
    pub(crate) environment_ref: Option<String>,
    pub(crate) config: Map<String, Value>,
    pub(crate) activation_snapshot: Option<Arc<DynamicPluginActivationSnapshot>>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DynamicPluginActivationSnapshot {
    root: PathBuf,
    original_manifest_ref: String,
    identity_manifest: PathBuf,
    activation_manifest: PathBuf,
    activation_environment_ref: Option<String>,
    identity_files: HashMap<PathBuf, PathBuf>,
    closure_digest: String,
    verification_digest: String,
}

impl DynamicPluginActivationSnapshot {
    fn create(
        manifest_ref: &str,
        expected_plugin_id: &str,
        expected_kind: DynamicPluginKind,
        environment_ref: Option<&str>,
        host_policy: &crate::plugins::policy::DynamicPluginHostPolicy,
    ) -> Result<Arc<Self>, CliError> {
        let (mut manifest, original_manifest_ref, manifest_bytes) =
            load_bounded_dynamic_plugin_manifest_bytes(manifest_ref)?;
        if manifest.plugin.id.trim() != expected_plugin_id || manifest.plugin.kind != expected_kind
        {
            return Err(CliError::Config(format!(
                "dynamic plugin manifest identity changed before activation for '{expected_plugin_id}'"
            )));
        }
        let policy = evaluate_dynamic_plugin_host_policy(host_policy, &manifest);
        validate_python_entrypoint_artifact(&manifest, &original_manifest_ref)
            .map_err(CliError::Config)?;

        let root = std::env::temp_dir().join(format!(
            "nemo-relay-plugin-snapshot-{}",
            uuid::Uuid::now_v7().simple()
        ));
        fs::create_dir(&root).map_err(|error| {
            CliError::Config(format!(
                "failed to create dynamic plugin activation snapshot {}: {error}",
                root.display()
            ))
        })?;
        let mut root_guard = SnapshotRootGuard(Some(root.clone()));
        #[cfg(unix)]
        fs::set_permissions(&root, {
            use std::os::unix::fs::PermissionsExt;
            fs::Permissions::from_mode(0o700)
        })
        .map_err(|error| {
            CliError::Config(format!(
                "failed to protect dynamic plugin activation snapshot {}: {error}",
                root.display()
            ))
        })?;

        let identity_manifest = root.join("identity-manifest.toml");
        fs::write(&identity_manifest, &manifest_bytes).map_err(|error| {
            CliError::Config(format!(
                "failed to write dynamic plugin activation snapshot {}: {error}",
                identity_manifest.display()
            ))
        })?;
        let original_manifest_path = PathBuf::from(&original_manifest_ref);
        let manifest_directory = original_manifest_path
            .parent()
            .ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin manifest {} has no parent directory",
                    original_manifest_path.display()
                ))
            })?
            .to_path_buf();
        let runtime_root = root.join("runtime");
        let mut budget = SnapshotBudget::default();
        let mut copied_files = HashMap::new();
        copy_snapshot_directory(
            &manifest_directory,
            &runtime_root,
            &mut copied_files,
            &mut budget,
            false,
            &mut Vec::new(),
        )?;
        let declared_artifact = manifest
            .source
            .as_ref()
            .and_then(|source| source.artifact.as_deref())
            .map(|artifact| fs::canonicalize(resolve_manifest_relative_path(&original_manifest_path, artifact)))
            .transpose()
            .map_err(|error| {
                CliError::Config(format!(
                    "failed to normalize dynamic plugin artifact for '{expected_plugin_id}': {error}"
                ))
            })?;
        let mut identity_files = HashMap::new();

        match &mut manifest.load {
            DynamicPluginManifestLoad::RustDynamic(load) => {
                if let Some(library) = load.library.as_deref() {
                    let (logical, _, copied) = copy_snapshot_file(
                        &root,
                        &original_manifest_path,
                        library,
                        "library",
                        &mut copied_files,
                        &mut budget,
                    )?;
                    identity_files
                        .entry(logical)
                        .or_insert_with(|| copied.clone());
                    load.library = Some(copied.to_string_lossy().into_owned());
                }
            }
            DynamicPluginManifestLoad::Worker(load)
                if matches!(
                    load.runtime,
                    Some(WorkerRuntime::Rust | WorkerRuntime::Command)
                ) =>
            {
                if let Some(entrypoint) = load.entrypoint.as_deref() {
                    let (logical, canonical, copied) = copy_snapshot_file(
                        &root,
                        &original_manifest_path,
                        entrypoint,
                        "entrypoint",
                        &mut copied_files,
                        &mut budget,
                    )?;
                    if declared_artifact.as_ref() != Some(&canonical) {
                        return Err(CliError::Config(format!(
                            "command worker dynamic plugin '{expected_plugin_id}' must declare its load.entrypoint as the integrity-checked source.artifact"
                        )));
                    }
                    identity_files
                        .entry(logical)
                        .or_insert_with(|| copied.clone());
                    load.entrypoint = Some(copied.to_string_lossy().into_owned());
                }
            }
            DynamicPluginManifestLoad::Worker(_) => {}
        }

        if let Some(source) = manifest.source.as_mut()
            && let Some(artifact) = source.artifact.as_deref()
        {
            let (logical, _, copied) = copy_snapshot_file(
                &root,
                &original_manifest_path,
                artifact,
                "artifact",
                &mut copied_files,
                &mut budget,
            )?;
            identity_files.insert(logical, copied.clone());
            source.artifact = Some(copied.to_string_lossy().into_owned());
        }
        if let Some(integrity) = manifest.integrity.as_mut()
            && let Some(signature) = integrity.signature.as_deref()
        {
            let (logical, _, copied) = copy_snapshot_file(
                &root,
                &original_manifest_path,
                signature,
                "signature",
                &mut copied_files,
                &mut budget,
            )?;
            identity_files.insert(logical, copied.clone());
            integrity.signature = Some(copied.to_string_lossy().into_owned());
        }

        let activation_environment_ref = if matches!(
            &manifest.load,
            DynamicPluginManifestLoad::Worker(load)
                if load.runtime == Some(WorkerRuntime::Python)
        ) {
            let environment = environment_ref.ok_or_else(|| {
                CliError::Config(format!(
                    "Python worker dynamic plugin '{expected_plugin_id}' has no managed environment"
                ))
            })?;
            let source_artifact_sha256 = trusted_source_artifact_sha256(&manifest)?;
            let environment = PathBuf::from(environment);
            verify_environment_attestation(&environment, source_artifact_sha256)
                .map_err(CliError::Config)?;
            let environment_name = environment.file_name().ok_or_else(|| {
                CliError::Config(format!(
                    "managed Python environment {} has no lifecycle environment name",
                    environment.display()
                ))
            })?;
            let copied_environment = root.join(MANAGED_ENVIRONMENTS_DIR).join(environment_name);
            copy_snapshot_directory(
                &environment,
                &copied_environment,
                &mut copied_files,
                &mut budget,
                true,
                &mut Vec::new(),
            )?;
            verify_environment_attestation(&copied_environment, source_artifact_sha256)
                .map_err(CliError::Config)?;
            Some(copied_environment.to_string_lossy().into_owned())
        } else {
            None
        };

        let activation_manifest = runtime_root.join("relay-plugin.toml");
        let rendered = toml::to_string(&manifest).map_err(|error| {
            CliError::Config(format!(
                "failed to encode dynamic plugin activation snapshot for '{expected_plugin_id}': {error}"
            ))
        })?;
        if rendered.len() as u64 > MAX_BOOTSTRAP_IDENTITY_FILE_BYTES {
            return Err(CliError::Config(format!(
                "dynamic plugin activation manifest for '{expected_plugin_id}' exceeds the {MAX_BOOTSTRAP_IDENTITY_FILE_BYTES}-byte activation snapshot budget"
            )));
        }
        fs::write(&activation_manifest, rendered).map_err(|error| {
            CliError::Config(format!(
                "failed to write dynamic plugin activation manifest {}: {error}",
                activation_manifest.display()
            ))
        })?;

        let trust = evaluate_dynamic_plugin_trust(
            &manifest,
            activation_manifest.to_string_lossy().as_ref(),
            &policy,
        );
        if !policy.policy_satisfied {
            return Err(CliError::Config(format!(
                "dynamic plugin '{expected_plugin_id}' activation snapshot violates host policy"
            )));
        }
        if let Some(failure) = trust.failure() {
            return Err(CliError::Config(
                failure.display(expected_plugin_id).to_string(),
            ));
        }

        let closure_digest = snapshot_tree_digest(&root, true)?;
        let verification_digest = snapshot_tree_digest(&root, false)?;
        #[cfg(unix)]
        protect_snapshot_tree(&root)?;
        #[cfg(windows)]
        protect_snapshot_tree(&root)?;
        root_guard.0 = None;
        Ok(Arc::new(Self {
            root,
            original_manifest_ref,
            identity_manifest,
            activation_manifest,
            activation_environment_ref,
            identity_files,
            closure_digest,
            verification_digest,
        }))
    }

    pub(crate) fn activation_manifest_ref(&self) -> String {
        self.activation_manifest.to_string_lossy().into_owned()
    }

    pub(crate) fn activation_environment_ref(&self) -> Option<&str> {
        self.activation_environment_ref.as_deref()
    }

    pub(crate) fn closure_digest(&self) -> &str {
        &self.closure_digest
    }

    pub(crate) fn verify_current(&self) -> Result<(), CliError> {
        let actual = snapshot_tree_digest(&self.root, false)?;
        if actual == self.verification_digest {
            Ok(())
        } else {
            Err(CliError::Config(format!(
                "dynamic plugin activation snapshot {} changed before code load",
                self.root.display()
            )))
        }
    }

    pub(crate) fn original_manifest_ref(&self) -> &str {
        &self.original_manifest_ref
    }

    pub(crate) fn identity_manifest(&self) -> &Path {
        &self.identity_manifest
    }

    pub(crate) fn identity_file(&self, logical_path: &Path) -> Option<&Path> {
        self.identity_files.get(logical_path).map(PathBuf::as_path)
    }
}

struct SnapshotRootGuard(Option<PathBuf>);

impl Drop for SnapshotRootGuard {
    fn drop(&mut self) {
        if let Some(root) = self.0.take() {
            make_snapshot_removable(&root);
            let _ = fs::remove_dir_all(root);
        }
    }
}

impl Drop for DynamicPluginActivationSnapshot {
    fn drop(&mut self) {
        make_snapshot_removable(&self.root);
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn copy_snapshot_file(
    root: &Path,
    manifest_path: &Path,
    reference: &str,
    label: &str,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
) -> Result<(PathBuf, PathBuf, PathBuf), CliError> {
    let logical = resolve_manifest_relative_path(manifest_path, reference);
    let canonical = fs::canonicalize(&logical).map_err(|error| {
        CliError::Config(format!(
            "failed to normalize dynamic plugin {label} {}: {error}",
            logical.display()
        ))
    })?;
    if let Some(copied) = copied_files.get(&canonical)
        && !matches!(label, "library" | "entrypoint")
    {
        return Ok((logical, canonical, copied.clone()));
    }
    if matches!(label, "library" | "entrypoint") {
        let manifest_directory = manifest_path
            .parent()
            .and_then(|parent| fs::canonicalize(parent).ok());
        if manifest_directory
            .as_ref()
            .is_some_and(|directory| canonical.starts_with(directory))
            && let Some(copied) = copied_files.get(&canonical)
        {
            // The manifest directory is copied as a complete closure before declared paths are
            // rewritten, so in-tree load targets already retain adjacent resources.
            return Ok((logical, canonical, copied.clone()));
        }
    }
    let external = root.join(format!("external-{label}"));
    if matches!(label, "library" | "entrypoint") {
        let parent = canonical.parent().ok_or_else(|| {
            CliError::Config(format!(
                "dynamic plugin {label} {} has no parent directory",
                canonical.display()
            ))
        })?;
        copy_snapshot_directory(
            parent,
            &external,
            copied_files,
            budget,
            false,
            &mut Vec::new(),
        )?;
    } else {
        fs::create_dir_all(&external).map_err(|error| CliError::Config(error.to_string()))?;
        let destination = external.join(canonical.file_name().unwrap_or_default());
        copy_snapshot_regular_file(&canonical, &destination, copied_files, budget, label)?;
    }
    let copied = copied_files.get(&canonical).cloned().ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin {label} {} was not included in its activation snapshot",
            canonical.display()
        ))
    })?;
    Ok((logical, canonical, copied))
}

const MAX_SNAPSHOT_FILES: usize = 100_000;
const MAX_SNAPSHOT_DEPTH: usize = 128;

#[derive(Default)]
struct SnapshotBudget {
    entries: usize,
    bytes: u64,
}

impl SnapshotBudget {
    fn record(&mut self, path: &Path, bytes: usize) -> Result<(), CliError> {
        self.record_entries(path, 1)?;
        self.record_bytes(path, bytes)
    }

    fn record_entries(&mut self, path: &Path, count: usize) -> Result<(), CliError> {
        self.entries = self.entries.saturating_add(count);
        if self.entries > MAX_SNAPSHOT_FILES {
            return Err(CliError::Config(format!(
                "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_FILES}-entry activation snapshot budget at {}",
                path.display()
            )));
        }
        Ok(())
    }

    fn record_bytes(&mut self, path: &Path, bytes: usize) -> Result<(), CliError> {
        self.bytes = self.bytes.saturating_add(bytes as u64);
        if self.bytes > MAX_BOOTSTRAP_IDENTITY_FILE_BYTES {
            return Err(CliError::Config(format!(
                "dynamic plugin runtime closure exceeds the {MAX_BOOTSTRAP_IDENTITY_FILE_BYTES}-byte activation snapshot budget at {}",
                path.display()
            )));
        }
        Ok(())
    }

    fn record_directory(&mut self, path: &Path) -> Result<(), CliError> {
        self.record_entries(path, 1)
    }
}

fn copy_snapshot_directory(
    source: &Path,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
) -> Result<(), CliError> {
    budget.record_directory(source)?;
    copy_snapshot_directory_contents(
        source,
        destination,
        copied_files,
        budget,
        skip_python_cache,
        ancestors,
    )
}

fn copy_snapshot_directory_contents(
    source: &Path,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
) -> Result<(), CliError> {
    if ancestors.len() >= MAX_SNAPSHOT_DEPTH {
        return Err(CliError::Config(format!(
            "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_DEPTH}-directory traversal depth at {}",
            source.display()
        )));
    }
    let canonical = fs::canonicalize(source).map_err(|error| {
        CliError::Config(format!(
            "failed to normalize dynamic plugin runtime directory {}: {error}",
            source.display()
        ))
    })?;
    if ancestors.contains(&canonical) {
        return Err(CliError::Config(format!(
            "dynamic plugin runtime closure contains a directory symlink cycle at {}",
            source.display()
        )));
    }
    ancestors.push(canonical.clone());
    fs::create_dir_all(destination).map_err(|error| {
        CliError::Config(format!(
            "failed to create dynamic plugin snapshot directory {}: {error}",
            destination.display()
        ))
    })?;
    let mut entries = bounded_runtime_directory_entries(
        &canonical,
        MAX_SNAPSHOT_FILES.saturating_sub(budget.entries),
    )?;
    budget.record_entries(source, entries.len())?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let source_path = entry.path();
        if skip_python_cache
            && (entry.file_name() == "__pycache__"
                || source_path.extension().and_then(|value| value.to_str()) == Some("pyc"))
        {
            continue;
        }
        let destination_path = destination.join(entry.file_name());
        let metadata = fs::symlink_metadata(&source_path)
            .map_err(|error| CliError::Config(error.to_string()))?;
        let resolved = if metadata.file_type().is_symlink() {
            fs::canonicalize(&source_path).map_err(|error| {
                CliError::Config(format!(
                    "failed to resolve dynamic plugin runtime symlink {}: {error}",
                    source_path.display()
                ))
            })?
        } else {
            source_path.clone()
        };
        let resolved_metadata =
            fs::metadata(&resolved).map_err(|error| CliError::Config(error.to_string()))?;
        if resolved_metadata.is_dir() {
            copy_snapshot_directory_contents(
                &resolved,
                &destination_path,
                copied_files,
                budget,
                skip_python_cache,
                ancestors,
            )?;
        } else if resolved_metadata.is_file() {
            // A macOS venv's `bin/python` is normally an absolute symlink to the managed
            // interpreter.  Dereferencing that link while snapshotting turns the interpreter
            // into a standalone file whose @rpath no longer points at libpython, so the worker
            // exits before it can create its socket. Preserve only these launcher links; all
            // other runtime symlinks remain dereferenced to keep the activation snapshot
            // self-contained and deterministic.
            #[cfg(unix)]
            if metadata.file_type().is_symlink() && is_python_venv_launcher(&source_path) {
                let target = fs::read_link(&source_path).map_err(|error| {
                    CliError::Config(format!(
                        "failed to read Python venv launcher symlink {}: {error}",
                        source_path.display()
                    ))
                })?;
                if let Some(parent) = destination_path.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|error| CliError::Config(error.to_string()))?;
                }
                std::os::unix::fs::symlink(&target, &destination_path).map_err(|error| {
                    CliError::Config(format!(
                        "failed to preserve Python venv launcher symlink {}: {error}",
                        destination_path.display()
                    ))
                })?;
                copied_files.insert(resolved, destination_path);
                continue;
            }
            copy_snapshot_regular_file(
                &resolved,
                &destination_path,
                copied_files,
                budget,
                "runtime file",
            )?;
        } else {
            return Err(CliError::Config(format!(
                "dynamic plugin runtime entry {} must resolve to a regular file or directory",
                source_path.display()
            )));
        }
    }
    ancestors.pop();
    Ok(())
}

#[cfg(unix)]
fn is_python_venv_launcher(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    parent.file_name() == Some(std::ffi::OsStr::new("bin"))
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "python" || name.starts_with("python3"))
}

fn copy_snapshot_regular_file(
    source: &Path,
    destination: &Path,
    copied_files: &mut HashMap<PathBuf, PathBuf>,
    budget: &mut SnapshotBudget,
    description: &str,
) -> Result<(), CliError> {
    let bytes = read_bounded_regular_file(source, &format!("dynamic plugin {description}"))
        .map_err(CliError::Config)?;
    budget.record_bytes(source, bytes.len())?;
    fs::write(destination, bytes).map_err(|error| {
        CliError::Config(format!(
            "failed to write dynamic plugin snapshot file {}: {error}",
            destination.display()
        ))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(source)
            .map_err(|error| CliError::Config(error.to_string()))?
            .permissions()
            .mode();
        fs::set_permissions(destination, fs::Permissions::from_mode(mode))
            .map_err(|error| CliError::Config(error.to_string()))?;
    }
    copied_files.insert(source.to_path_buf(), destination.to_path_buf());
    Ok(())
}

fn resolve_manifest_relative_path(manifest_path: &Path, reference: &str) -> PathBuf {
    let path = PathBuf::from(reference);
    if path.is_absolute() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

#[cfg(unix)]
fn protect_snapshot_tree(root: &Path) -> Result<(), CliError> {
    use std::os::unix::fs::PermissionsExt;
    for entry in fs::read_dir(root).map_err(|error| CliError::Config(error.to_string()))? {
        let path = entry
            .map_err(|error| CliError::Config(error.to_string()))?
            .path();
        let metadata =
            fs::symlink_metadata(&path).map_err(|error| CliError::Config(error.to_string()))?;
        if metadata.is_dir() {
            protect_snapshot_tree(&path)?;
            continue;
        }
        if metadata.file_type().is_symlink() {
            continue;
        }
        let mode = metadata.permissions().mode() & !0o222;
        fs::set_permissions(&path, fs::Permissions::from_mode(mode))
            .map_err(|error| CliError::Config(error.to_string()))?;
    }
    fs::set_permissions(root, fs::Permissions::from_mode(0o500))
        .map_err(|error| CliError::Config(error.to_string()))
}

#[cfg(windows)]
fn protect_snapshot_tree(root: &Path) -> Result<(), CliError> {
    for entry in fs::read_dir(root).map_err(|error| CliError::Config(error.to_string()))? {
        let path = entry
            .map_err(|error| CliError::Config(error.to_string()))?
            .path();
        let metadata =
            fs::symlink_metadata(&path).map_err(|error| CliError::Config(error.to_string()))?;
        if metadata.is_dir() {
            protect_snapshot_tree(&path)?;
        } else if !metadata.file_type().is_symlink() {
            let mut permissions = metadata.permissions();
            permissions.set_readonly(true);
            fs::set_permissions(&path, permissions)
                .map_err(|error| CliError::Config(error.to_string()))?;
        }
    }
    Ok(())
}

fn snapshot_tree_digest(root: &Path, stable_identity: bool) -> Result<String, CliError> {
    let mut files = Vec::new();
    let mut entries = 0_usize;
    collect_snapshot_files(root, root, &mut files, None, &mut entries)?;
    files.sort();
    let mut digest = Sha256::new();
    let mut budget = SnapshotBudget::default();
    for relative in files {
        if stable_identity {
            let activation_manifest = Path::new("runtime").join("relay-plugin.toml");
            let is_python_environment_content = relative.starts_with(MANAGED_ENVIRONMENTS_DIR)
                && relative.file_name() != Some(std::ffi::OsStr::new(ENVIRONMENT_ATTESTATION_FILE));
            if relative == activation_manifest || is_python_environment_content {
                continue;
            }
        }
        let path = root.join(&relative);
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            CliError::Config(format!(
                "failed to inspect dynamic plugin activation snapshot entry {}: {error}",
                path.display()
            ))
        })?;
        if metadata.file_type().is_symlink() {
            let target = fs::read_link(&path).map_err(|error| {
                CliError::Config(format!(
                    "failed to read dynamic plugin activation snapshot symlink {}: {error}",
                    path.display()
                ))
            })?;
            let target = target.as_os_str().as_encoded_bytes();
            budget.record(&path, target.len())?;
            update_snapshot_entry_digest(
                &mut digest,
                &relative,
                SnapshotEntryKind::Symlink,
                target,
            );
        } else {
            let bytes = read_bounded_regular_file(&path, "dynamic plugin activation snapshot file")
                .map_err(CliError::Config)?;
            budget.record(&path, bytes.len())?;
            update_snapshot_entry_digest(&mut digest, &relative, SnapshotEntryKind::File, &bytes);
        }
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub(crate) fn dynamic_plugin_runtime_closure_digest(
    manifest_ref: &str,
    environment_ref: Option<&str>,
) -> Result<String, CliError> {
    let (manifest, normalized_manifest_ref, manifest_bytes) =
        load_bounded_dynamic_plugin_manifest_bytes(manifest_ref)?;
    let manifest_path = PathBuf::from(normalized_manifest_ref);
    let manifest_directory = manifest_path.parent().ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin manifest {} has no parent directory",
            manifest_path.display()
        ))
    })?;
    let mut closure = RuntimeClosureSources::default();
    closure.record_bytes(PathBuf::from("identity-manifest.toml"), manifest_bytes)?;
    collect_runtime_closure_directory(
        manifest_directory,
        Path::new("runtime"),
        false,
        &mut Vec::new(),
        &mut closure,
    )?;

    let declared_artifact = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
        .map(|artifact| fs::canonicalize(resolve_manifest_relative_path(&manifest_path, artifact)))
        .transpose()
        .map_err(|error| {
            CliError::Config(format!(
                "failed to normalize dynamic plugin artifact for '{}': {error}",
                manifest.plugin.id
            ))
        })?;
    match &manifest.load {
        DynamicPluginManifestLoad::RustDynamic(load) => {
            if let Some(library) = load.library.as_deref() {
                collect_declared_runtime_closure_file(
                    &manifest_path,
                    library,
                    "library",
                    &mut closure,
                )?;
            }
        }
        DynamicPluginManifestLoad::Worker(load)
            if matches!(
                load.runtime,
                Some(WorkerRuntime::Rust | WorkerRuntime::Command)
            ) =>
        {
            if let Some(entrypoint) = load.entrypoint.as_deref() {
                let canonical_entrypoint =
                    fs::canonicalize(resolve_manifest_relative_path(&manifest_path, entrypoint))
                        .map_err(|error| {
                            CliError::Config(format!(
                                "failed to normalize dynamic plugin entrypoint for '{}': {error}",
                                manifest.plugin.id
                            ))
                        })?;
                if declared_artifact.as_ref() != Some(&canonical_entrypoint) {
                    return Err(CliError::Config(format!(
                        "command worker dynamic plugin '{}' must declare its load.entrypoint as the integrity-checked source.artifact",
                        manifest.plugin.id
                    )));
                }
                collect_declared_runtime_closure_file(
                    &manifest_path,
                    entrypoint,
                    "entrypoint",
                    &mut closure,
                )?;
            }
        }
        DynamicPluginManifestLoad::Worker(load) if load.runtime == Some(WorkerRuntime::Python) => {
            let environment_ref = environment_ref.ok_or_else(|| {
                CliError::Config(format!(
                    "Python worker dynamic plugin '{}' has no managed environment",
                    manifest.plugin.id
                ))
            })?;
            let environment = Path::new(environment_ref);
            let source_artifact_sha256 = trusted_source_artifact_sha256(&manifest)?;
            read_environment_attestation(environment, source_artifact_sha256)
                .map_err(CliError::Config)?;
            let environment_name = environment.file_name().ok_or_else(|| {
                CliError::Config(format!(
                    "managed Python environment {} has no lifecycle environment name",
                    environment.display()
                ))
            })?;
            closure.record_file(
                Path::new(MANAGED_ENVIRONMENTS_DIR)
                    .join(environment_name)
                    .join(ENVIRONMENT_ATTESTATION_FILE),
                environment.join(ENVIRONMENT_ATTESTATION_FILE),
            )?;
        }
        DynamicPluginManifestLoad::Worker(_) => {}
    }
    if let Some(artifact) = manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
    {
        collect_declared_runtime_closure_file(&manifest_path, artifact, "artifact", &mut closure)?;
    }
    if let Some(signature) = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.signature.as_deref())
    {
        collect_declared_runtime_closure_file(
            &manifest_path,
            signature,
            "signature",
            &mut closure,
        )?;
    }

    closure.digest()
}

fn trusted_source_artifact_sha256(manifest: &DynamicPluginManifest) -> Result<&str, CliError> {
    manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .map(str::trim)
        .filter(|digest| !digest.is_empty())
        .ok_or_else(|| {
            CliError::Config(format!(
                "Python worker dynamic plugin '{}' requires integrity.sha256 to bind its complete installed environment to the trusted source artifact",
                manifest.plugin.id
            ))
        })
}

enum RuntimeClosureSource {
    File(PathBuf),
    Bytes(Vec<u8>),
}

#[derive(Default)]
struct RuntimeClosureSources {
    files: BTreeMap<PathBuf, RuntimeClosureSource>,
    copied_files: HashMap<PathBuf, PathBuf>,
    entries: usize,
}

impl RuntimeClosureSources {
    fn record_file(&mut self, relative: PathBuf, source: PathBuf) -> Result<(), CliError> {
        self.entries = self.entries.saturating_add(1);
        self.record_reserved_file(relative, source);
        self.enforce_file_budget()
    }

    fn record_reserved_file(&mut self, relative: PathBuf, source: PathBuf) {
        self.files
            .insert(relative.clone(), RuntimeClosureSource::File(source.clone()));
        self.copied_files.insert(source, relative);
    }

    fn record_bytes(&mut self, relative: PathBuf, bytes: Vec<u8>) -> Result<(), CliError> {
        self.entries = self.entries.saturating_add(1);
        self.files
            .insert(relative, RuntimeClosureSource::Bytes(bytes));
        self.enforce_file_budget()
    }

    fn enforce_file_budget(&self) -> Result<(), CliError> {
        if self.entries > MAX_SNAPSHOT_FILES {
            Err(CliError::Config(format!(
                "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_FILES}-entry activation snapshot budget"
            )))
        } else {
            Ok(())
        }
    }

    fn record_entry(&mut self) -> Result<(), CliError> {
        self.entries = self.entries.saturating_add(1);
        self.enforce_file_budget()
    }

    fn record_entries(&mut self, count: usize) -> Result<(), CliError> {
        self.entries = self.entries.saturating_add(count);
        self.enforce_file_budget()
    }

    fn digest(self) -> Result<String, CliError> {
        let activation_manifest = Path::new("runtime").join("relay-plugin.toml");
        let mut digest = Sha256::new();
        let mut budget = SnapshotBudget::default();
        for (relative, source) in self.files {
            if relative == activation_manifest {
                continue;
            }
            let bytes = match source {
                RuntimeClosureSource::File(path) => {
                    read_bounded_regular_file(&path, "dynamic plugin runtime closure file")
                        .map_err(CliError::Config)?
                }
                RuntimeClosureSource::Bytes(bytes) => bytes,
            };
            budget.record(&relative, bytes.len())?;
            update_snapshot_entry_digest(&mut digest, &relative, SnapshotEntryKind::File, &bytes);
        }
        Ok(digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

#[derive(Clone, Copy)]
enum SnapshotEntryKind {
    File = 0,
    Symlink = 1,
}

fn update_snapshot_entry_digest(
    digest: &mut Sha256,
    relative: &Path,
    kind: SnapshotEntryKind,
    payload: &[u8],
) {
    let relative = relative.as_os_str().as_encoded_bytes();
    digest.update([kind as u8]);
    digest.update((relative.len() as u64).to_le_bytes());
    digest.update(relative);
    digest.update((payload.len() as u64).to_le_bytes());
    digest.update(payload);
}

fn collect_declared_runtime_closure_file(
    manifest_path: &Path,
    reference: &str,
    label: &str,
    closure: &mut RuntimeClosureSources,
) -> Result<(), CliError> {
    let logical = resolve_manifest_relative_path(manifest_path, reference);
    let canonical = fs::canonicalize(&logical).map_err(|error| {
        CliError::Config(format!(
            "failed to normalize dynamic plugin {label} {}: {error}",
            logical.display()
        ))
    })?;
    if closure.copied_files.contains_key(&canonical) && !matches!(label, "library" | "entrypoint") {
        return Ok(());
    }
    if matches!(label, "library" | "entrypoint") {
        let manifest_directory = manifest_path
            .parent()
            .and_then(|parent| fs::canonicalize(parent).ok());
        if manifest_directory
            .as_ref()
            .is_some_and(|directory| canonical.starts_with(directory))
            && closure.copied_files.contains_key(&canonical)
        {
            return Ok(());
        }
        let parent = canonical.parent().ok_or_else(|| {
            CliError::Config(format!(
                "dynamic plugin {label} {} has no parent directory",
                canonical.display()
            ))
        })?;
        return collect_runtime_closure_directory(
            parent,
            Path::new(&format!("external-{label}")),
            false,
            &mut Vec::new(),
            closure,
        );
    }
    let file_name = canonical.file_name().ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin {label} {} has no file name",
            canonical.display()
        ))
    })?;
    closure.record_file(
        Path::new(&format!("external-{label}")).join(file_name),
        canonical,
    )
}

fn collect_runtime_closure_directory(
    source: &Path,
    destination: &Path,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
    closure: &mut RuntimeClosureSources,
) -> Result<(), CliError> {
    closure.record_entry()?;
    collect_runtime_closure_directory_contents(
        source,
        destination,
        skip_python_cache,
        ancestors,
        closure,
    )
}

fn collect_runtime_closure_directory_contents(
    source: &Path,
    destination: &Path,
    skip_python_cache: bool,
    ancestors: &mut Vec<PathBuf>,
    closure: &mut RuntimeClosureSources,
) -> Result<(), CliError> {
    if ancestors.len() >= MAX_SNAPSHOT_DEPTH {
        return Err(CliError::Config(format!(
            "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_DEPTH}-directory traversal depth at {}",
            source.display()
        )));
    }
    let canonical = fs::canonicalize(source).map_err(|error| {
        CliError::Config(format!(
            "failed to normalize dynamic plugin runtime directory {}: {error}",
            source.display()
        ))
    })?;
    if ancestors.contains(&canonical) {
        return Err(CliError::Config(format!(
            "dynamic plugin runtime closure contains a directory symlink cycle at {}",
            source.display()
        )));
    }
    ancestors.push(canonical.clone());
    let mut entries = bounded_runtime_directory_entries(
        &canonical,
        MAX_SNAPSHOT_FILES.saturating_sub(closure.entries),
    )?;
    closure.record_entries(entries.len())?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let source_path = entry.path();
        if skip_python_cache
            && (entry.file_name() == "__pycache__"
                || source_path.extension().and_then(|value| value.to_str()) == Some("pyc"))
        {
            continue;
        }
        let metadata = fs::symlink_metadata(&source_path)
            .map_err(|error| CliError::Config(error.to_string()))?;
        let resolved = if metadata.file_type().is_symlink() {
            fs::canonicalize(&source_path).map_err(|error| {
                CliError::Config(format!(
                    "failed to resolve dynamic plugin runtime symlink {}: {error}",
                    source_path.display()
                ))
            })?
        } else {
            source_path.clone()
        };
        let resolved_metadata =
            fs::metadata(&resolved).map_err(|error| CliError::Config(error.to_string()))?;
        let relative = destination.join(entry.file_name());
        if resolved_metadata.is_dir() {
            collect_runtime_closure_directory_contents(
                &resolved,
                &relative,
                skip_python_cache,
                ancestors,
                closure,
            )?;
        } else if resolved_metadata.is_file() {
            closure.record_reserved_file(relative, resolved);
        } else {
            return Err(CliError::Config(format!(
                "dynamic plugin runtime entry {} must resolve to a regular file or directory",
                source_path.display()
            )));
        }
    }
    ancestors.pop();
    Ok(())
}

fn bounded_runtime_directory_entries(
    directory: &Path,
    remaining_entries: usize,
) -> Result<Vec<fs::DirEntry>, CliError> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(directory).map_err(|error| CliError::Config(error.to_string()))? {
        if entries.len() >= remaining_entries {
            return Err(CliError::Config(format!(
                "dynamic plugin runtime closure exceeds the {MAX_SNAPSHOT_FILES}-entry activation snapshot budget at {}",
                directory.display()
            )));
        }
        entries.push(entry.map_err(|error| CliError::Config(error.to_string()))?);
    }
    Ok(entries)
}

fn collect_snapshot_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
    logical_depth: Option<usize>,
    entries: &mut usize,
) -> Result<(), CliError> {
    if let Some(depth) = logical_depth
        && depth >= MAX_SNAPSHOT_DEPTH
    {
        return Err(CliError::Config(format!(
            "dynamic plugin activation snapshot exceeds the {MAX_SNAPSHOT_DEPTH}-directory traversal depth at {}",
            directory.display()
        )));
    }
    let resets_child_depth =
        directory == root || directory == root.join(environment::MANAGED_ENVIRONMENTS_DIR);
    for entry in fs::read_dir(directory).map_err(|error| CliError::Config(error.to_string()))? {
        *entries = entries.saturating_add(1);
        if *entries > MAX_SNAPSHOT_FILES {
            return Err(CliError::Config(format!(
                "dynamic plugin activation snapshot exceeds the {MAX_SNAPSHOT_FILES}-entry verification budget at {}",
                directory.display()
            )));
        }
        let path = entry
            .map_err(|error| CliError::Config(error.to_string()))?
            .path();
        let metadata =
            fs::symlink_metadata(&path).map_err(|error| CliError::Config(error.to_string()))?;
        if metadata.is_dir() {
            let child_depth = if resets_child_depth {
                0
            } else {
                logical_depth.unwrap_or(0).saturating_add(1)
            };
            collect_snapshot_files(root, &path, files, Some(child_depth), entries)?;
        } else {
            files.push(
                path.strip_prefix(root)
                    .map_err(|error| CliError::Config(error.to_string()))?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

fn make_snapshot_removable(root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(root, fs::Permissions::from_mode(0o700));
    }
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            make_snapshot_removable(&path);
        } else {
            #[cfg(windows)]
            if !metadata.file_type().is_symlink() {
                let mut permissions = metadata.permissions();
                permissions.set_readonly(false);
                let _ = fs::set_permissions(&path, permissions);
            }
        }
    }
}

pub(crate) fn active_dynamic_plugin_components(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<Vec<ActiveDynamicPluginComponent>, CliError> {
    active_dynamic_plugin_components_inner(explicit, resolved, true)
}

pub(crate) fn active_dynamic_plugin_components_for_identity(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
) -> Result<Vec<ActiveDynamicPluginComponent>, CliError> {
    let scopes = load_scoped_registries(explicit)?;
    active_dynamic_plugin_components_from_scopes(&scopes, resolved, false)
}

fn active_dynamic_plugin_components_inner(
    explicit: Option<&PathBuf>,
    resolved: &ResolvedConfig,
    create_activation_snapshots: bool,
) -> Result<Vec<ActiveDynamicPluginComponent>, CliError> {
    let scopes = load_and_hydrate_scopes(explicit, resolved)?;
    active_dynamic_plugin_components_from_scopes(&scopes, resolved, create_activation_snapshots)
}

fn active_dynamic_plugin_components_from_scopes(
    scopes: &[ScopedRegistry],
    resolved: &ResolvedConfig,
    create_activation_snapshots: bool,
) -> Result<Vec<ActiveDynamicPluginComponent>, CliError> {
    let host_config_by_id = host_config_by_id(resolved);
    let mut components = Vec::new();

    for resolved_plugin in &resolved.dynamic_plugins {
        let Some(record) = scopes
            .iter()
            .find(|scope| scope.plugins_toml_path == resolved_plugin.source)
            .and_then(|scope| scope.registry.get(&resolved_plugin.plugin_id))
        else {
            return Err(CliError::Config(format!(
                "dynamic plugin '{}' is present in resolved config but not lifecycle state",
                resolved_plugin.plugin_id
            )));
        };
        if record.is_tombstoned() || !record.spec.enabled {
            continue;
        }
        let host_config = host_config_by_id.get(&record.metadata.id).ok_or_else(|| {
            CliError::Config(format!(
                "dynamic plugin '{}' is enabled but has no resolved host config",
                record.metadata.id
            ))
        })?;
        let manifest_ref = match record.metadata.kind {
            DynamicPluginKind::RustDynamic => Some(manifest_ref_from_record(record)?),
            DynamicPluginKind::Worker => record.source.manifest_ref.clone(),
        };
        let activation_snapshot = if create_activation_snapshots {
            manifest_ref
                .as_deref()
                .map(|manifest_ref| {
                    DynamicPluginActivationSnapshot::create(
                        manifest_ref,
                        &record.metadata.id,
                        record.metadata.kind,
                        record.source.environment_ref.as_deref(),
                        &resolved.dynamic_plugin_policy,
                    )
                })
                .transpose()?
        } else {
            None
        };
        components.push(ActiveDynamicPluginComponent {
            plugin_id: record.metadata.id.clone(),
            kind: record.metadata.kind,
            lifecycle_generation: record.metadata.generation,
            manifest_ref,
            environment_ref: record.source.environment_ref.clone(),
            config: host_config.config.clone(),
            activation_snapshot,
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
    crate::config::load_bounded_dynamic_plugin_manifest(&path)
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

        if let Err(error) = crate::config::load_bounded_dynamic_plugin_manifest(&manifest_ref) {
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
