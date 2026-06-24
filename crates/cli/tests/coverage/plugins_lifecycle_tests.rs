// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::{ffi::OsString, sync::MutexGuard};

use super::*;
use crate::config::{
    PluginsAddCommand, PluginsDisableCommand, PluginsEnableCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsRemoveCommand, PluginsScopeArgs, PluginsValidateCommand, ServerArgs,
};
use crate::error::PluginLifecycleFailureKind;

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

struct EnvScope {
    _guard: MutexGuard<'static, ()>,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn hermetic(temp: &tempfile::TempDir) -> Self {
        let xdg = temp.path().join("xdg");
        std::fs::create_dir_all(&xdg).unwrap();
        Self::set(&[
            ("HOME", Some(temp.path().as_os_str())),
            ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
        ])
    }

    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in values {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        Self {
            _guard: guard,
            values: previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (key, value) in self.values.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
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
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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
fn add_rejects_scope_flags_when_explicit_config_is_set() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join("custom-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.explicit-conflict");

    let server = ServerArgs {
        config: Some(config_dir.join("gateway.toml")),
        ..ServerArgs::default()
    };

    let error = add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("--config cannot be combined"));
}

#[test]
fn list_and_inspect_render_discovered_dynamic_plugins() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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
    let list = PluginListView {
        records: &records,
        host_config_by_id: &host_config_by_id,
    }
    .to_string();
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
    let inspect = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.guardrail"),
    }
    .to_string();
    let inspect_value: serde_yaml::Value = serde_yaml::from_str(&inspect).unwrap();
    assert_eq!(
        inspect_value["metadata"]["id"].as_str(),
        Some("acme.guardrail")
    );
    assert_eq!(inspect_value["metadata"]["kind"].as_str(), Some("worker"));
    assert_eq!(inspect_value["host_config_status"].as_str(), Some("absent"));
    assert!(
        inspect_value["source"]["manifest_ref"]
            .as_str()
            .unwrap()
            .contains("relay-plugin.toml")
    );
    assert_eq!(
        inspect_value["load"]["entrypoint"].as_str(),
        Some("acme.guardrail.plugin:register")
    );
}

#[test]
fn validate_renders_summary_for_path_and_id_targets() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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
    let path_summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: None,
        host_config: None,
    }
    .to_string();
    assert!(path_summary.contains("Dynamic plugin 'acme.guardrail' is valid."));

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("plugin record");
    let id_summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: Some(&entry),
        host_config: host_config_by_id.get("acme.guardrail"),
    }
    .to_string();
    assert!(id_summary.contains("host_config: absent"));
    assert!(id_summary.contains("desired.enabled: false"));

    let missing_validate = validate(
        PluginsValidateCommand {
            target: "missing.plugin".into(),
            json: false,
        },
        &crate::config::ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_validate.contains("not registered"));

    let missing_inspect = inspect(
        PluginsInspectCommand {
            id: "missing.plugin".into(),
            json: false,
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
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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

    let all_records = collect_records(&scopes, true);
    let host_config_by_id = host_config_by_id(&resolved);
    let all_list = PluginListView {
        records: &all_records,
        host_config_by_id: &host_config_by_id,
    }
    .to_string();
    assert!(all_list.contains("acme.guardrail"));
    assert!(all_list.contains("tombstoned"));

    let error = enable(
        PluginsEnableCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .expect_err("tombstoned plugin should not enable");
    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            message,
            ..
        } => assert!(message.contains("tombstoned")),
        other => panic!("unexpected tombstone enable error: {other}"),
    }
}

#[test]
fn add_with_explicit_config_uses_sibling_plugins_and_state_files() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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
    let state_path = config_dir.join(".dynamic-plugins.json");
    assert!(plugins_toml.exists());
    assert!(state_path.exists());

    let resolved = resolve_plugins_config(server.config.as_ref()).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.explicit");

    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.explicit")
        .unwrap()
        .expect("explicit-scope record");
    assert_eq!(entry.scope.to_string(), "explicit");
    assert_eq!(entry.plugins_toml_path, plugins_toml);
    assert_eq!(entry.state_path, state_path);
}

#[test]
fn hydrate_bootstraps_registry_records_from_existing_dynamic_plugin_refs() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
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
    assert_eq!(entry.scope.to_string(), "project");
    assert_eq!(entry.record.metadata.id, "acme.bootstrap");
    assert!(entry.record.spec.present);
    assert!(!entry.record.spec.enabled);
    let canonical_manifest_path = std::fs::canonicalize(&manifest_path).unwrap();
    assert_eq!(
        entry.record.source.manifest_ref.as_deref(),
        Some(canonical_manifest_path.to_string_lossy().as_ref())
    );
}

#[test]
fn add_can_revive_tombstoned_records() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.revive");
    let server = crate::config::ServerArgs::default();

    add(
        PluginsAddCommand {
            scope: PluginsScopeArgs {
                project: true,
                ..PluginsScopeArgs::default()
            },
            path: plugin_dir.clone(),
        },
        &server,
    )
    .unwrap();

    remove(
        PluginsRemoveCommand {
            id: "acme.revive".into(),
        },
        &server,
    )
    .unwrap();

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

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let revived = find_record_by_id(&scopes, "acme.revive")
        .unwrap()
        .expect("revived record");
    assert!(!revived.record.is_tombstoned());
    assert!(revived.record.spec.present);
}

#[test]
fn json_helpers_emit_stable_success_and_failure_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.json");
    let server = ServerArgs::default();

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

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let records = collect_records(&scopes, false);
    let entry = find_record_by_id(&scopes, "acme.json")
        .unwrap()
        .expect("json record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let list_value = serde_json::to_value(responses::list_success(
        "plugins list",
        None,
        &records,
        &host_config_by_id,
    ))
    .unwrap();
    assert_eq!(list_value["schema_version"], serde_json::json!(1));
    assert_eq!(list_value["ok"], serde_json::json!(true));
    assert_eq!(list_value["data"][0]["id"], serde_json::json!("acme.json"));

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.json",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.json"),
    ))
    .unwrap();
    assert_eq!(inspect_value["data"]["id"], serde_json::json!("acme.json"));
    assert_eq!(
        inspect_value["data"]["source"]["manifest_ref"],
        serde_json::json!(manifest_ref)
    );
    assert_eq!(
        inspect_value["data"]["host_config"],
        serde_json::Value::Null
    );

    let validate_value = serde_json::to_value(responses::validate_success(
        responses::ValidateResponseInput {
            command: "plugins validate",
            target: Some("acme.json"),
            target_kind: "plugin_id",
            resolved_plugin_id: Some("acme.json"),
            manifest: &manifest,
            manifest_ref: &manifest_ref,
            entry: Some(&entry),
            host_config: host_config_by_id.get("acme.json"),
        },
    ))
    .unwrap();
    assert_eq!(
        validate_value["data"]["target_kind"],
        serde_json::json!("plugin_id")
    );
    assert_eq!(validate_value["data"]["valid"], serde_json::json!(true));

    let failure = serde_json::to_value(responses::failure(
        "plugins inspect",
        Some("missing.plugin"),
        PluginLifecycleFailureKind::NotFound,
        "missing plugin",
    ))
    .unwrap();
    assert_eq!(failure["ok"], serde_json::json!(false));
    assert_eq!(failure["error"]["code"], serde_json::json!("not_found"));
}

#[test]
fn remove_tolerates_unreadable_non_target_manifest_entries() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let broken_dir = temp.path().join("plugins").join("broken");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&broken_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = ServerArgs::default();

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

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy(),
            broken_dir.join("missing.toml").to_string_lossy()
        ),
    )
    .unwrap();

    remove(
        PluginsRemoveCommand {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(!rendered.contains("acme.guardrail"));
    assert!(rendered.contains("missing.toml"));
}

#[test]
fn remove_reports_malformed_dynamic_plugin_containers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugins_toml = temp.path().join("plugins.toml");

    std::fs::write(&plugins_toml, "[plugins]\ndynamic = \"oops\"\n").unwrap();
    let error = remove_dynamic_plugin_reference(&plugins_toml, "acme.guardrail", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugins.dynamic must be an array of tables"));

    std::fs::write(&plugins_toml, "plugins = \"oops\"\n").unwrap();
    let error = remove_dynamic_plugin_reference(&plugins_toml, "acme.guardrail", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("[plugins] must be a table"));
}

#[test]
fn append_reports_malformed_dynamic_plugin_containers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugins_toml = temp.path().join("plugins.toml");

    std::fs::write(&plugins_toml, "plugins = \"oops\"\n").unwrap();
    let error = append_dynamic_plugin_reference(&plugins_toml, "/tmp/plugin/relay-plugin.toml")
        .unwrap_err()
        .to_string();
    assert!(error.contains("[plugins] must be a table"));

    std::fs::write(&plugins_toml, "[plugins]\ndynamic = \"oops\"\n").unwrap();
    let error = append_dynamic_plugin_reference(&plugins_toml, "/tmp/plugin/relay-plugin.toml")
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugins.dynamic must be an array of tables"));
}

#[test]
fn remove_matches_relative_target_manifest_refs_without_loading_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let config_dir = temp.path().join(".nemo-relay");
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();

    let manifest_path = plugin_dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrail"
kind = "worker"
"#,
    )
    .unwrap();

    let plugins_toml = config_dir.join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        "[[plugins.dynamic]]\nmanifest = \"../plugins/acme/relay-plugin.toml\"\n",
    )
    .unwrap();

    std::fs::remove_file(&manifest_path).unwrap();

    let removed = remove_dynamic_plugin_reference(
        &plugins_toml,
        "acme.guardrail",
        Some("../plugins/acme/relay-plugin.toml"),
    )
    .unwrap();
    assert!(removed);
    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(!rendered.contains("relay-plugin.toml"));
}

#[test]
fn inspect_redacts_host_config_values() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.redacted");
    let server = ServerArgs::default();

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

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\nconfig = {{ api_key = \"secret-token\", region = \"us-west-2\" }}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.redacted")
        .unwrap()
        .expect("redacted record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let inspect_output = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.redacted"),
    }
    .to_string();
    assert!(!inspect_output.contains("secret-token"));
    let inspect_output: serde_yaml::Value = serde_yaml::from_str(&inspect_output).unwrap();
    assert_eq!(
        inspect_output["host_config"]["api_key"].as_str(),
        Some("<redacted>")
    );
    assert_eq!(
        inspect_output["host_config"]["region"].as_str(),
        Some("<redacted>")
    );

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.redacted",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.redacted"),
    ))
    .unwrap();
    assert_eq!(
        inspect_value["data"]["host_config"]["api_key"],
        serde_json::json!("<redacted>")
    );
    assert_eq!(
        inspect_value["data"]["host_config"]["region"],
        serde_json::json!("<redacted>")
    );
    assert_eq!(inspect_value["data"]["host_config_status"], "present");
}

#[test]
fn inspect_distinguishes_empty_host_config_from_missing_host_config() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.empty-config");
    let server = ServerArgs::default();

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

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\nconfig = {{}}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.empty-config")
        .unwrap()
        .expect("empty-config record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let inspect_output = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.empty-config"),
    }
    .to_string();
    let inspect_output: serde_yaml::Value = serde_yaml::from_str(&inspect_output).unwrap();
    assert_eq!(
        inspect_output["host_config_status"].as_str(),
        Some("present")
    );
    assert_eq!(
        inspect_output["host_config"]
            .as_mapping()
            .expect("empty host config should render as an object")
            .len(),
        0
    );

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.empty-config",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.empty-config"),
    ))
    .unwrap();
    assert_eq!(inspect_value["data"]["host_config_status"], "present");
    assert_eq!(
        inspect_value["data"]["host_config"]
            .as_object()
            .expect("empty host config should serialize as an object")
            .len(),
        0
    );
}
