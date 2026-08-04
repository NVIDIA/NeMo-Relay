// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use nemo_relay::plugin::PluginComponentSpec;
use tempfile::tempdir;

use super::*;

fn write_manifest(root: &Path, id: &str) -> PathBuf {
    let plugin = root.join(id);
    fs::create_dir_all(&plugin).unwrap();
    let artifact = plugin.join("libplugin.so");
    fs::write(&artifact, b"artifact").unwrap();
    let manifest = plugin.join("relay-plugin.toml");
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
library = "libplugin.so"
symbol = "nemo_relay_plugin_entrypoint_v1"
[source]
artifact = "libplugin.so"
[integrity]
sha256 = "sha256:placeholder"
"#
        ),
    )
    .unwrap();
    manifest
}

#[test]
fn selected_path_replaces_only_user_layer() {
    let temp = tempdir().unwrap();
    let selected = temp.path().join("selected/plugins.toml");
    let project = temp.path().join("project/.nemo-relay/plugins.toml");
    let system = temp.path().join("system/plugins.toml");
    fs::create_dir_all(selected.parent().unwrap()).unwrap();
    fs::create_dir_all(project.parent().unwrap()).unwrap();
    fs::create_dir_all(system.parent().unwrap()).unwrap();
    fs::write(&selected, "version = 1\n").unwrap();
    fs::write(&project, "version = 1\n").unwrap();
    fs::write(&system, "version = 1\n").unwrap();
    let options = PluginFileResolveOptions {
        plugin_config_path: Some(selected.clone()),
        current_dir: Some(temp.path().join("project")),
        user_config_dir: Some(temp.path().join("ignored-user")),
        system_config_path: system.clone(),
    };
    assert_eq!(options.selected_paths(), vec![selected, project, system]);
}

#[test]
fn dynamic_manifest_is_source_relative_and_duplicates_are_fatal() {
    let temp = tempdir().unwrap();
    let first_manifest = write_manifest(temp.path(), "same");
    let first = temp.path().join("one/plugins.toml");
    let second = temp.path().join("two/plugins.toml");
    fs::create_dir_all(first.parent().unwrap()).unwrap();
    fs::create_dir_all(second.parent().unwrap()).unwrap();
    fs::write(
        &first,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            first_manifest.to_string_lossy()
        ),
    )
    .unwrap();
    fs::write(
        &second,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            first_manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let error = resolve_plugin_files_from_paths([first, second], None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("duplicate dynamic plugin id 'same'")
    );
}

#[test]
fn missing_declared_manifest_is_not_silently_ignored() {
    let temp = tempdir().unwrap();
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        "[[plugins.dynamic]]\nmanifest = \"missing/relay-plugin.toml\"\n",
    )
    .unwrap();
    let error = resolve_plugin_files_from_paths([config], None).unwrap_err();
    assert!(matches!(error, PluginHostConfigError::NotFound { .. }));
}

#[test]
fn missing_selected_layer_falls_through_to_project_and_system() {
    let temp = tempdir().unwrap();
    let project_root = temp.path().join("project");
    let project = project_root.join(".nemo-relay/plugins.toml");
    let system = temp.path().join("system/plugins.toml");
    fs::create_dir_all(project.parent().unwrap()).unwrap();
    fs::create_dir_all(system.parent().unwrap()).unwrap();
    fs::write(&project, "version = 1\n").unwrap();
    fs::write(&system, "version = 1\n").unwrap();
    let resolved = resolve_plugin_files(
        None,
        PluginFileResolveOptions {
            plugin_config_path: Some(temp.path().join("missing/plugins.toml")),
            current_dir: Some(project_root),
            user_config_dir: Some(temp.path().join("ambient-user")),
            system_config_path: system.clone(),
        },
    )
    .unwrap();
    assert_eq!(
        resolved.contributing_sources,
        vec![
            dunce::canonicalize(&project).unwrap(),
            dunce::canonicalize(&system).unwrap()
        ]
    );
}

#[test]
fn user_only_selection_suppresses_project_but_retains_system() {
    let temp = tempdir().unwrap();
    let selected = temp.path().join("selected/plugins.toml");
    let system = temp.path().join("system/plugins.toml");
    fs::create_dir_all(selected.parent().unwrap()).unwrap();
    fs::create_dir_all(system.parent().unwrap()).unwrap();
    fs::write(&selected, "version = 1\n").unwrap();
    fs::write(&system, "version = 1\n").unwrap();
    let resolved = resolve_plugin_files(
        None,
        PluginFileResolveOptions {
            plugin_config_path: Some(selected.clone()),
            current_dir: None,
            user_config_dir: None,
            system_config_path: system.clone(),
        },
    )
    .unwrap();
    assert_eq!(
        resolved.contributing_sources,
        vec![
            dunce::canonicalize(&selected).unwrap(),
            dunce::canonicalize(&system).unwrap()
        ]
    );
}

#[test]
fn static_layers_and_caller_overlay_share_core_merge_semantics() {
    let temp = tempdir().unwrap();
    let lower = temp.path().join("lower.toml");
    let higher = temp.path().join("higher.toml");
    fs::write(
        &lower,
        r#"version = 1
[[components]]
kind = "fixture"
[components.config]
lower = 1
list = ["lower"]
[[components]]
kind = "disabled"
enabled = false
"#,
    )
    .unwrap();
    fs::write(
        &higher,
        r#"version = 1
[[components]]
kind = "fixture"
[components.config]
higher = 2
list = ["higher"]
"#,
    )
    .unwrap();
    let mut caller_component = PluginComponentSpec::new("fixture");
    caller_component
        .config
        .insert("caller".into(), serde_json::json!(3));
    caller_component
        .config
        .insert("list".into(), serde_json::json!(["caller"]));
    let resolved = resolve_plugin_files_from_paths(
        [lower, higher],
        Some(PluginConfig {
            components: vec![caller_component],
            ..PluginConfig::default()
        }),
    )
    .unwrap();
    assert_eq!(resolved.config.components.len(), 1);
    let config = &resolved.config.components[0].config;
    assert_eq!(config["lower"], 1);
    assert_eq!(config["higher"], 2);
    assert_eq!(config["caller"], 3);
    assert_eq!(
        config["list"],
        serde_json::json!(["caller", "higher", "lower"])
    );
}

#[test]
fn relative_and_absolute_manifest_references_resolve_canonically() {
    let temp = tempdir().unwrap();
    let relative_manifest = write_manifest(temp.path(), "relative");
    let absolute_manifest = write_manifest(temp.path(), "absolute");
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = \"relative/relay-plugin.toml\"\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
            absolute_manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 2);
    assert_eq!(
        PathBuf::from(&resolved.dynamic_plugins[0].manifest_ref),
        relative_manifest.canonicalize().unwrap()
    );
    assert_eq!(
        PathBuf::from(&resolved.dynamic_plugins[1].manifest_ref),
        absolute_manifest.canonicalize().unwrap()
    );
}

#[test]
fn dynamic_only_diagnostics_are_redacted() {
    let temp = tempdir().unwrap();
    write_manifest(temp.path(), "redacted");
    let config = temp.path().join("plugins.toml");
    let secret = "never-report-this-token";
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = \"redacted/relay-plugin.toml\"\nconfig = {{ api_key = \"{secret}\" }}\n"
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    assert_eq!(
        resolved.contributing_sources,
        vec![dunce::canonicalize(&config).unwrap()]
    );
    assert!(
        resolved
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "plugin.configuration_inherited")
    );
    assert!(
        resolved
            .diagnostics
            .iter()
            .all(|diagnostic| !diagnostic.message.contains(secret))
    );
}

#[cfg(unix)]
#[test]
fn physical_source_aliases_are_deduplicated_after_pinning() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let manifest = write_manifest(temp.path(), "aliased");
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let alias = temp.path().join("plugins-alias.toml");
    symlink(&config, &alias).unwrap();

    let resolved = resolve_plugin_files_from_paths([alias.clone(), config.clone()], None).unwrap();
    assert_eq!(
        resolved.contributing_sources,
        vec![dunce::canonicalize(&config).unwrap()]
    );
    assert_eq!(resolved.contributing_selected_sources, vec![config.clone()]);
    assert_eq!(resolved.dynamic_plugins.len(), 1);

    let aliased = resolve_plugin_files_from_paths([config, alias.clone()], None).unwrap();
    assert_eq!(aliased.contributing_selected_sources, vec![alias]);
}

#[test]
fn malformed_toml_errors_do_not_disclose_configuration_values() {
    let temp = tempdir().unwrap();
    let config = temp.path().join("plugins.toml");
    let secret = "do-not-leak-this-token";
    fs::write(
        &config,
        format!("[[components]]\nkind = \"fixture\"\nconfig = {{ token = \"{secret} }}\n"),
    )
    .unwrap();
    let error = resolve_plugin_files_from_paths([config], None).unwrap_err();
    assert!(!error.to_string().contains(secret));
}

#[cfg(unix)]
#[test]
fn special_file_plugin_configuration_is_rejected_without_reading_it() {
    let error = resolve_plugin_files_from_paths([PathBuf::from("/dev/zero")], None)
        .expect_err("a character device must not be read as plugins.toml");
    assert!(error.to_string().contains("must be a regular file"));
}
