// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use super::*;
use crate::config::{
    PluginsAddCommand, PluginsDisableCommand, PluginsEnableCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsRemoveCommand, PluginsScopeArgs, PluginsValidateCommand, ServerArgs,
};

struct CurrentDirGuard {
    original: PathBuf,
}

impl CurrentDirGuard {
    fn enter(path: &Path) -> Self {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self { original }
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.original).unwrap();
    }
}

fn write_dynamic_manifest(dir: &Path, plugin_id: &str) -> PathBuf {
    let manifest_path = dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "{plugin_id}"
kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "{plugin_id}.plugin:register"
"#
        ),
    )
    .unwrap();
    manifest_path
}

#[test]
fn add_registers_dynamic_plugin_in_project_plugins_toml() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(rendered.contains("[[plugins.dynamic]]"));
    assert!(rendered.contains("relay-plugin.toml"));

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.guardrail");
}

#[test]
fn add_rejects_duplicate_dynamic_plugin_ids() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let error = add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("already registered"));
}

#[test]
fn list_and_inspect_render_discovered_dynamic_plugins() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let records = collect_records(&scopes, false);
    let list = render_list(&records, &host_config_by_id);
    assert!(list.contains("acme.guardrail"));
    assert!(list.contains("absent"));
    assert!(list.contains("false"));

    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("plugin record");
    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(entry.record.source.manifest_ref.clone().unwrap())
            .map_err(|error| CliError::Config(error.to_string()))
            .unwrap();
    let inspect = render_inspect(
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.guardrail"),
    );
    assert!(inspect.contains("id: acme.guardrail"));
    assert!(inspect.contains("kind: worker"));
    assert!(inspect.contains("host_config: absent"));
    assert!(inspect.contains("load.entrypoint: acme.guardrail.plugin:register"));
}

#[test]
fn validate_renders_summary_for_path_and_id_targets() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap();

    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let path_summary = render_validation_summary(&manifest, &manifest_ref, None, None);
    assert!(path_summary.contains("Dynamic plugin 'acme.guardrail' is valid."));

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let id_summary = render_validation_summary(
        &manifest,
        &manifest_ref,
        Some(
            &find_record_by_id(&scopes, "acme.guardrail")
                .unwrap()
                .expect("plugin record"),
        ),
        host_config_by_id.get("acme.guardrail"),
    );
    assert!(id_summary.contains("host_config: absent"));
    assert!(id_summary.contains("desired.enabled: false"));

    let missing_validate = validate(
        PluginsValidateCommand {
            target: "missing.plugin".into(),
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_validate.contains("not registered"));

    let missing_inspect = inspect(
        PluginsInspectCommand {
            id: "missing.plugin".into(),
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_inspect.contains("not registered"));

    assert_eq!(
        list(
            PluginsListCommand::default(),
            &crate::config::ServerArgs::default()
        )
        .unwrap(),
        ()
    );
}

#[test]
fn enable_disable_and_remove_persist_lifecycle_state() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    enable(
        PluginsEnableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let enabled = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("enabled record");
    assert!(enabled.record.spec.enabled);

    disable(
        PluginsDisableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let disabled = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("disabled record");
    assert!(!disabled.record.spec.enabled);

    remove(
        PluginsRemoveCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    assert!(resolved.dynamic_plugins.is_empty());
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let removed = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("removed record");
    assert!(removed.record.is_tombstoned());
}

#[test]
fn add_with_explicit_config_uses_sibling_plugins_and_state_files() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join("custom-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.explicit");

    let server = ServerArgs {
        config: Some(config_dir.join("gateway.toml")),
        ..ServerArgs::default()
    };

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs::default(),
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = config_dir.join("plugins.toml");
    let state_path = config_dir.join("dynamic-plugins.json");
    assert!(plugins_toml.exists());
    assert!(state_path.exists());

    let resolved = resolve_plugins_config(server.config.as_ref()).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.explicit");

    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.explicit")
        .unwrap()
        .expect("explicit-scope record");
    assert_eq!(entry.scope.label(), "explicit");
    assert_eq!(entry.plugins_toml_path, plugins_toml);
    assert_eq!(entry.state_path, state_path);
}

#[test]
fn hydrate_bootstraps_registry_records_from_existing_dynamic_plugin_refs() {
    let _lock = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.bootstrap");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);

    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.bootstrap")
        .unwrap()
        .expect("hydrated record");
    assert_eq!(entry.scope.label(), "project");
    assert_eq!(entry.record.metadata.id, "acme.bootstrap");
    assert!(entry.record.spec.present);
    assert!(!entry.record.spec.enabled);
    let canonical_manifest_path = std::fs::canonicalize(&manifest_path).unwrap();
    assert_eq!(
        entry.record.source.manifest_ref.as_deref(),
        Some(canonical_manifest_path.to_string_lossy().as_ref())
    );
}
