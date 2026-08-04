// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use nemo_relay::plugin::PluginConfig;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

use super::*;
use crate::environment::MANAGED_ENVIRONMENTS_DIR;
use crate::resolver::resolve_plugin_files_from_paths;
use crate::state::DYNAMIC_PLUGIN_STATE_FILENAME;

fn write_native_plugin(root: &Path, id: &str) -> PathBuf {
    let directory = root.join(id);
    fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("plugin.so");
    fs::write(&artifact, b"native plugin fixture").unwrap();
    let digest = Sha256::digest(b"native plugin fixture")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest = directory.join("relay-plugin.toml");
    fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1
[plugin]
id = "{id}"
kind = "rust_dynamic"
[compat]
relay = ">=0.5,<1.0"
native_api = "v1"
[capabilities]
items = ["plugin_native"]
[defaults]
enabled = false
[load]
library = "plugin.so"
symbol = "nemo_relay_plugin_entrypoint_v1"
[source]
artifact = "plugin.so"
[integrity]
sha256 = "sha256:{digest}"
"#
        ),
    )
    .unwrap();
    manifest
}

fn write_python_plugin(root: &Path, id: &str) -> PathBuf {
    let directory = root.join(id);
    fs::create_dir_all(&directory).unwrap();
    let artifact_body = b"def main():\n    return None\n";
    fs::write(directory.join("plugin.py"), artifact_body).unwrap();
    let digest = Sha256::digest(artifact_body)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest = directory.join("relay-plugin.toml");
    fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1
[plugin]
id = "{id}"
kind = "worker"
[compat]
relay = ">=0.5,<1.0"
worker_protocol = "grpc-v1"
[capabilities]
items = ["plugin_worker"]
[defaults]
enabled = false
[source]
manifest_root = "."
artifact = "plugin.py"
[integrity]
sha256 = "sha256:{digest}"
[load]
runtime = "python"
entrypoint = "plugin:main"
"#
        ),
    )
    .unwrap();
    manifest
}

fn write_command_plugin(root: &Path, id: &str) -> PathBuf {
    let directory = root.join(id);
    fs::create_dir_all(&directory).unwrap();
    let artifact_body = b"command worker fixture\n";
    fs::write(directory.join("worker"), artifact_body).unwrap();
    let digest = Sha256::digest(artifact_body)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest = directory.join("relay-plugin.toml");
    fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1
[plugin]
id = "{id}"
kind = "worker"
[compat]
relay = ">=0.5,<1.0"
worker_protocol = "grpc-v1"
[capabilities]
items = ["plugin_worker"]
[defaults]
enabled = false
[load]
runtime = "command"
entrypoint = "worker"
[source]
artifact = "worker"
[integrity]
sha256 = "sha256:{digest}"
"#
        ),
    )
    .unwrap();
    manifest
}

fn write_plugin_declaration(config: &Path, manifest: &Path) {
    fs::write(
        config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
}

fn mutate_state(config: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let state_path = config.parent().unwrap().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let mut state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&state_path).unwrap()).unwrap();
    mutate(&mut state);
    fs::write(&state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();
}

fn state_record<'a>(state: &'a serde_json::Value, plugin_id: &str) -> &'a serde_json::Value {
    state["records"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["metadata"]["id"] == plugin_id)
        .unwrap()
}

#[test]
fn missing_lifecycle_record_is_hydrated_disabled() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "fixture");
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    let reconciled = reconcile_plugin_lifecycle(&resolved).unwrap();
    assert!(reconciled.enabled_plugins.is_empty());
    let state: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap(),
    )
    .unwrap();
    assert_eq!(state["records"][0]["spec"]["enabled"], false);
}

#[test]
fn same_directory_sources_share_state_without_collapsing_ownership() {
    let temp = tempdir().unwrap();
    let custom_manifest = write_native_plugin(&temp.path().join("artifacts"), "custom-owner");
    let default_manifest = write_native_plugin(&temp.path().join("artifacts"), "default-owner");
    let custom = temp.path().join("custom.toml");
    let default = temp.path().join("plugins.toml");
    write_plugin_declaration(&custom, &custom_manifest);
    write_plugin_declaration(&default, &default_manifest);

    let resolved =
        resolve_plugin_files_from_paths([custom.clone(), default.clone()], None).unwrap();
    let reconciled = reconcile_plugin_lifecycle(&resolved).unwrap();

    assert!(reconciled.enabled_plugins.is_empty());
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let state: serde_json::Value = serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
    assert_eq!(state["records"].as_array().unwrap().len(), 2);
    assert_eq!(
        state["declaration_sources"]["custom-owner"],
        custom.canonicalize().unwrap().to_string_lossy().as_ref()
    );
    assert_eq!(
        state["declaration_sources"]["default-owner"],
        default.canonicalize().unwrap().to_string_lossy().as_ref()
    );
    assert!(
        state_record(&state, "custom-owner")["spec"]["config_ref"].is_null(),
        "physical declaration ownership must not consume the public config_ref field"
    );
}

#[test]
fn moving_same_id_requires_removal_before_hydrating_new_owner_disabled() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "moved-owner");
    let custom = temp.path().join("custom.toml");
    let default = temp.path().join("plugins.toml");
    write_plugin_declaration(&custom, &manifest);
    fs::write(&default, "version = 1\n").unwrap();

    let first = resolve_plugin_files_from_paths([custom.clone(), default.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&first).unwrap();
    mutate_state(&custom, |state| {
        state["records"][0]["spec"]["enabled"] = true.into()
    });

    fs::write(&custom, "version = 1\n").unwrap();
    write_plugin_declaration(&default, &manifest);
    let moved = resolve_plugin_files_from_paths([custom.clone(), default.clone()], None).unwrap();
    let error = reconcile_plugin_lifecycle(&moved).unwrap_err();

    assert!(error.to_string().contains("live lifecycle state owned by"));
    assert!(error.to_string().contains("control plane"));
    let state_path = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let unchanged: serde_json::Value =
        serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
    let unchanged_record = state_record(&unchanged, "moved-owner");
    assert_eq!(unchanged_record["spec"]["present"], true);
    assert_eq!(unchanged_record["spec"]["enabled"], true);
    assert_eq!(
        unchanged["declaration_sources"]["moved-owner"],
        custom.canonicalize().unwrap().to_string_lossy().as_ref()
    );

    mutate_state(&custom, |state| {
        state["records"][0]["spec"]["present"] = false.into();
        state["records"][0]["spec"]["enabled"] = false.into();
    });
    let reconciled = reconcile_plugin_lifecycle(&moved).unwrap();

    assert!(reconciled.enabled_plugins.is_empty());
    let state: serde_json::Value = serde_json::from_slice(&fs::read(state_path).unwrap()).unwrap();
    let record = state_record(&state, "moved-owner");
    assert_eq!(record["spec"]["present"], true);
    assert_eq!(record["spec"]["enabled"], false);
    assert_eq!(
        state["declaration_sources"]["moved-owner"],
        default.canonicalize().unwrap().to_string_lossy().as_ref()
    );
    assert!(
        !temp
            .path()
            .join(".custom.toml.dynamic-plugins.json")
            .exists()
    );
}

#[test]
fn legacy_owner_claim_preserves_enablement_and_non_path_config_ref() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "legacy-owner");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();
    mutate_state(&config, |state| {
        state.as_object_mut().unwrap().remove("declaration_sources");
        state["records"][0]["spec"]["enabled"] = true.into();
        state["records"][0]["spec"]["config_ref"] = "plugins.acme.guardrails.pii".into();
    });

    let reconciled = reconcile_plugin_lifecycle(&resolved).unwrap();

    assert_eq!(reconciled.enabled_plugins.len(), 1);
    let state: serde_json::Value =
        serde_json::from_slice(&fs::read(temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap())
            .unwrap();
    assert_eq!(
        state_record(&state, "legacy-owner")["spec"]["config_ref"],
        "plugins.acme.guardrails.pii"
    );
    assert_eq!(
        state["declaration_sources"]["legacy-owner"],
        config.canonicalize().unwrap().to_string_lossy().as_ref()
    );
}

#[test]
fn manifest_identity_change_after_resolution_fails_without_state_write() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "resolved-id");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    let state_path = config.parent().unwrap().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    assert!(!state_path.exists());

    let changed = fs::read_to_string(&manifest)
        .unwrap()
        .replace("id = \"resolved-id\"", "id = \"reloaded-id\"");
    fs::write(&manifest, changed).unwrap();

    let error = reconcile_plugin_lifecycle(&resolved).unwrap_err();
    assert!(matches!(error, PluginHostConfigError::InvalidConfig(_)));
    assert!(error.to_string().contains("changed identity"));
    assert!(error.to_string().contains("resolved-id"));
    assert!(error.to_string().contains("reloaded-id"));
    assert!(!state_path.exists());
}

#[test]
fn no_dynamic_declarations_produce_static_plan() {
    let resolved = ResolvedPluginFileConfiguration {
        config: PluginConfig::default(),
        runtime_value: None,
        dynamic_plugins: Vec::new(),
        dynamic_plugin_policy: Default::default(),
        diagnostics: Vec::new(),
        contributing_sources: Vec::new(),
        selected_sources: Vec::new(),
        had_input: true,
    };
    let plan = prepare_plugin_host_activation(resolved).unwrap();
    assert!(plan.dynamic_plugins.is_empty());
}

#[test]
fn static_only_plan_ignores_orphan_lifecycle_state_without_locking_it() {
    let temp = tempdir().unwrap();
    let config = temp.path().join("plugins.toml");
    let state = temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME);
    let lock = temp.path().join(".dynamic-plugins.lock");
    fs::write(&config, "").unwrap();
    fs::write(&state, "not valid lifecycle state").unwrap();
    let original_state = fs::read(&state).unwrap();

    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    let plan = prepare_plugin_host_activation(resolved).unwrap();

    assert!(plan.dynamic_plugins.is_empty());
    assert!(!lock.exists());
    assert_eq!(fs::read(state).unwrap(), original_state);
}

#[test]
fn enabled_and_tombstoned_state_are_joined_only_to_their_source() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "stateful");
    let first_dir = temp.path().join("first");
    let second_dir = temp.path().join("second");
    fs::create_dir_all(&first_dir).unwrap();
    fs::create_dir_all(&second_dir).unwrap();
    let first = first_dir.join("plugins.toml");
    let second = second_dir.join("plugins.toml");
    write_plugin_declaration(&first, &manifest);
    write_plugin_declaration(&second, &manifest);

    let first_resolved = resolve_plugin_files_from_paths([first.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&first_resolved).unwrap();
    mutate_state(&first, |state| {
        state["records"][0]["spec"]["enabled"] = true.into()
    });
    let first_enabled = reconcile_plugin_lifecycle(&first_resolved).unwrap();
    assert_eq!(first_enabled.enabled_plugins.len(), 1);

    let second_resolved = resolve_plugin_files_from_paths([second.clone()], None).unwrap();
    let second_disabled = reconcile_plugin_lifecycle(&second_resolved).unwrap();
    assert!(second_disabled.enabled_plugins.is_empty());

    mutate_state(&first, |state| {
        state["records"][0]["spec"]["present"] = false.into();
        state["records"][0]["spec"]["enabled"] = false.into();
    });
    let tombstoned = reconcile_plugin_lifecycle(&first_resolved).unwrap();
    assert!(tombstoned.enabled_plugins.is_empty());
}

#[test]
fn unowned_legacy_record_in_another_selected_source_blocks_rehydration() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(&temp.path().join("plugin"), "cross-source-live");
    let user_dir = temp.path().join("user");
    let project_dir = temp.path().join("project");
    fs::create_dir_all(&user_dir).unwrap();
    fs::create_dir_all(&project_dir).unwrap();
    let user = user_dir.join("plugins.toml");
    let project = project_dir.join("plugins.toml");
    write_plugin_declaration(&user, &manifest);

    let first = resolve_plugin_files_from_paths([user.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&first).unwrap();
    mutate_state(&user, |state| {
        state.as_object_mut().unwrap().remove("declaration_sources");
        state["records"][0]["spec"]["enabled"] = true.into();
    });
    fs::write(&user, "version = 1\n").unwrap();
    write_plugin_declaration(&project, &manifest);

    let layered = resolve_plugin_files_from_paths([user.clone(), project.clone()], None).unwrap();
    let error = reconcile_plugin_lifecycle(&layered).unwrap_err();

    assert!(error.to_string().contains("cross-source-live"));
    assert!(error.to_string().contains("outside its declaring source"));
    assert!(error.to_string().contains(".dynamic-plugins.json"));
    assert!(!project_dir.join(DYNAMIC_PLUGIN_STATE_FILENAME).exists());
}

#[test]
fn cross_directory_move_requires_removal_before_new_owner_hydration() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(&temp.path().join("plugin"), "cross-source-move");
    let user_dir = temp.path().join("user");
    let project_dir = temp.path().join("project");
    fs::create_dir_all(&user_dir).unwrap();
    fs::create_dir_all(&project_dir).unwrap();
    let user = user_dir.join("plugins.toml");
    let project = project_dir.join("plugins.toml");
    write_plugin_declaration(&user, &manifest);

    let first = resolve_plugin_files_from_paths([user.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&first).unwrap();
    mutate_state(&user, |state| {
        state["records"][0]["spec"]["enabled"] = true.into()
    });
    fs::write(&user, "version = 1\n").unwrap();
    write_plugin_declaration(&project, &manifest);

    let moved = resolve_plugin_files_from_paths([user.clone(), project.clone()], None).unwrap();
    let error = reconcile_plugin_lifecycle(&moved).unwrap_err();

    assert!(error.to_string().contains("live lifecycle state outside"));
    let unchanged: serde_json::Value =
        serde_json::from_slice(&fs::read(user_dir.join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap())
            .unwrap();
    assert_eq!(
        state_record(&unchanged, "cross-source-move")["spec"]["present"],
        true
    );
    assert_eq!(
        state_record(&unchanged, "cross-source-move")["spec"]["enabled"],
        true
    );

    mutate_state(&user, |state| {
        state["records"][0]["spec"]["present"] = false.into();
        state["records"][0]["spec"]["enabled"] = false.into();
    });
    let reconciled = reconcile_plugin_lifecycle(&moved).unwrap();

    assert!(reconciled.enabled_plugins.is_empty());
    let old_state: serde_json::Value =
        serde_json::from_slice(&fs::read(user_dir.join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap())
            .unwrap();
    assert_eq!(
        state_record(&old_state, "cross-source-move")["spec"]["present"],
        false
    );
    assert_eq!(
        old_state["declaration_sources"]["cross-source-move"],
        user.canonicalize().unwrap().to_string_lossy().as_ref()
    );
    let new_state: serde_json::Value =
        serde_json::from_slice(&fs::read(project_dir.join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap())
            .unwrap();
    assert_eq!(
        state_record(&new_state, "cross-source-move")["spec"]["enabled"],
        false
    );
    assert_eq!(
        new_state["declaration_sources"]["cross-source-move"],
        project.canonicalize().unwrap().to_string_lossy().as_ref()
    );
}

#[test]
fn corrupt_lifecycle_state_fails_before_activation_planning() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "corrupt");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    fs::write(
        temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME),
        b"{not-json",
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    let error = reconcile_plugin_lifecycle(&resolved).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid dynamic plugin registry state")
    );
}

#[test]
fn lifecycle_parse_errors_do_not_disclose_state_values() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "redacted-state");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    let secret = "do-not-leak-lifecycle-value";
    fs::write(
        temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME),
        format!(r#"{{"schema_version":1,"records":"{secret}"}}"#),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    let error = reconcile_plugin_lifecycle(&resolved).unwrap_err();
    assert!(!error.to_string().contains(secret));
}

#[test]
fn policy_and_trust_failures_are_persisted_without_loading_disabled_code() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "blocked");
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[plugins.policy.defaults]\nallowed = false\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    let reconciled = reconcile_plugin_lifecycle(&resolved).unwrap();
    assert!(reconciled.enabled_plugins.is_empty());
    let state: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        state["records"][0]["status"]["validation"]["policy_satisfied"],
        "invalid"
    );
    assert_eq!(
        state["records"][0]["status"]["last_error"]["code"],
        "policy_blocked"
    );
}

#[test]
fn python_worker_without_managed_environment_is_not_provisioned() {
    let temp = tempdir().unwrap();
    let manifest = write_python_plugin(temp.path(), "python-worker");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    let reconciled = reconcile_plugin_lifecycle(&resolved).unwrap();
    assert!(reconciled.enabled_plugins.is_empty());
    assert!(!temp.path().join(MANAGED_ENVIRONMENTS_DIR).exists());
    let state: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join(DYNAMIC_PLUGIN_STATE_FILENAME)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        state["records"][0]["status"]["validation"]["environment"],
        "invalid"
    );
}

#[cfg(unix)]
#[test]
fn lifecycle_save_failure_occurs_before_snapshot_or_code_load() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempdir().unwrap();
    let plugin_root = temp.path().join("plugin-root");
    let config_root = temp.path().join("config-root");
    fs::create_dir_all(&plugin_root).unwrap();
    fs::create_dir_all(&config_root).unwrap();
    let manifest = write_native_plugin(&plugin_root, "save-failure");
    let config = config_root.join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    fs::set_permissions(&config_root, fs::Permissions::from_mode(0o500)).unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    let error = match prepare_plugin_host_activation(resolved) {
        Ok(_) => panic!("read-only configuration directory unexpectedly accepted a state save"),
        Err(error) => error,
    };
    fs::set_permissions(&config_root, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(error.to_string().contains("lifecycle state"));
}

#[test]
fn enabled_plugin_plan_uses_a_retained_snapshot() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "snapshot");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();
    mutate_state(&config, |state| {
        state["records"][0]["spec"]["enabled"] = true.into()
    });
    let plan = prepare_plugin_host_activation(resolved).unwrap();
    assert_eq!(plan.dynamic_plugins.len(), 1);
    let planned = &plan.dynamic_plugins[0];
    assert_ne!(Path::new(&planned.spec.manifest_ref), manifest.as_path());
    planned.resource.verify().unwrap();
}

#[test]
fn declaration_manifest_refreshes_stale_lifecycle_load_metadata() {
    let temp = tempdir().unwrap();
    let first_manifest = write_native_plugin(&temp.path().join("v1"), "same-id");
    let second_manifest = write_native_plugin(&temp.path().join("v2"), "same-id");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &first_manifest);
    let first = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&first).unwrap();
    mutate_state(&config, |state| {
        state["records"][0]["spec"]["enabled"] = true.into();
        state["records"][0]["metadata"]["generation"] = 7.into();
    });

    write_plugin_declaration(&config, &second_manifest);
    let second = resolve_plugin_files_from_paths([config], None).unwrap();
    let reconciled = reconcile_plugin_lifecycle(&second).unwrap();
    assert_eq!(reconciled.enabled_plugins.len(), 1);
    assert_eq!(reconciled.enabled_plugins[0].lifecycle_generation, 7);
    assert_eq!(
        PathBuf::from(&reconciled.enabled_plugins[0].manifest_ref),
        second_manifest.canonicalize().unwrap()
    );
}

#[test]
fn declaration_kind_change_drives_the_activation_plan() {
    let temp = tempdir().unwrap();
    let native_manifest = write_native_plugin(&temp.path().join("native"), "same-kind-id");
    let worker_manifest = write_command_plugin(&temp.path().join("worker"), "same-kind-id");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &native_manifest);
    let first = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&first).unwrap();
    mutate_state(&config, |state| {
        state["records"][0]["spec"]["enabled"] = true.into()
    });

    write_plugin_declaration(&config, &worker_manifest);
    let second = resolve_plugin_files_from_paths([config], None).unwrap();
    let reconciled = reconcile_plugin_lifecycle(&second).unwrap();
    assert_eq!(
        reconciled.enabled_plugins[0].kind,
        DynamicPluginKind::Worker
    );
    assert_eq!(
        PathBuf::from(&reconciled.enabled_plugins[0].manifest_ref),
        worker_manifest.canonicalize().unwrap()
    );
    let plan = prepare_plugin_host_activation(second).unwrap();
    assert_eq!(plan.dynamic_plugins[0].spec.kind, DynamicPluginKind::Worker);
}

#[test]
fn reconciliation_cannot_overwrite_a_concurrent_enable_transaction() {
    let temp = tempdir().unwrap();
    let manifest = write_native_plugin(temp.path(), "concurrent-enable");
    let config = temp.path().join("plugins.toml");
    write_plugin_declaration(&config, &manifest);
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();

    let state_path = sibling_lifecycle_state_path(&config.canonicalize().unwrap());
    let control_plane_lock = lock_lifecycle_state(&state_path).unwrap();
    let mut control_plane_registry = read_locked_lifecycle_state(&control_plane_lock).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let background_resolved = resolved.clone();
    let reconcile = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = reconcile_plugin_lifecycle(&background_resolved);
        finished_tx.send(result).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(matches!(
        finished_rx.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));

    control_plane_registry.enable("concurrent-enable").unwrap();
    save_locked_lifecycle_state(&control_plane_lock, &control_plane_registry).unwrap();
    drop(control_plane_lock);

    let reconciled = finished_rx.recv().unwrap().unwrap();
    reconcile.join().unwrap();
    assert_eq!(reconciled.enabled_plugins.len(), 1);
    assert_eq!(reconciled.enabled_plugins[0].plugin_id, "concurrent-enable");
}
