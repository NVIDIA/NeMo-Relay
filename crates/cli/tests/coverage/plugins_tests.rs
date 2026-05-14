// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::config::{global_plugin_config_path, project_plugin_config_path};
use nemo_flow::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND;
use nemo_flow::plugin::{ConfigPolicy, PluginComponentSpec, PluginConfig};

#[test]
fn target_scope_defaults_to_user_and_rejects_conflicts() {
    assert_eq!(
        target_scope(&PluginsEditCommand::default()).unwrap(),
        TargetScope::User
    );
    assert_eq!(
        target_scope(&PluginsEditCommand {
            project: true,
            ..PluginsEditCommand::default()
        })
        .unwrap(),
        TargetScope::Project
    );
    assert_eq!(
        target_scope(&PluginsEditCommand {
            global: true,
            ..PluginsEditCommand::default()
        })
        .unwrap(),
        TargetScope::Global
    );

    let error = target_scope(&PluginsEditCommand {
        user: true,
        project: true,
        ..PluginsEditCommand::default()
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("choose only one"), "error was: {error}");
}

#[test]
fn typed_editor_model_contains_observability_sections() {
    let schema = ObservabilityConfig::editor_schema();
    let atof = schema.field("atof").unwrap().schema().unwrap();
    let atif = schema.field("atif").unwrap().schema().unwrap();
    let openinference = schema.field("openinference").unwrap().schema().unwrap();
    assert!(atof.fields.iter().any(|field| field.name == "mode"));
    assert!(
        atif.fields
            .iter()
            .any(|field| field.name == "filename_template")
    );
    assert!(
        openinference
            .fields
            .iter()
            .any(|field| field.name == "endpoint")
    );
}

#[test]
fn plugin_menu_uses_setup_theme_markers() {
    let theme = ColorfulTheme::default();
    let lines = render_menu(
        &theme,
        "plugins.toml",
        &[MenuItem::new("First"), MenuItem::new("Second")],
        0,
    );
    let rendered = lines.join("\n");

    assert!(rendered.contains('?'));
    assert!(rendered.contains('›'));
    assert!(rendered.contains('❯'));
    assert!(rendered.contains("↑/↓"));
    assert!(!rendered.contains("> First"));
}

#[test]
fn plugin_menu_marks_configured_sections_and_fields() {
    let mut observability = ObservabilityConfig::default();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    let mode = atof.schema().unwrap().field("mode").unwrap();
    let output_directory = atof.schema().unwrap().field("output_directory").unwrap();

    assert!(!section_configured(&observability, atof));
    ensure_section(&mut observability, atof);
    assert!(section_configured(&observability, atof));
    assert!(!section_field_configured(&observability, atof, mode).unwrap());
    assert!(!section_field_configured(&observability, atof, output_directory).unwrap());

    set_section_field(&mut observability, atof, "output_directory", json!("logs")).unwrap();
    assert!(section_field_configured(&observability, atof, output_directory).unwrap());
    assert!(configured_label(true, "Edit ATOF").contains('✓'));
    assert!(!configured_label(false, "Edit ATIF").contains('✓'));
}

#[test]
fn editor_model_renders_valid_observability_plugin_config() {
    let mut config = PluginConfig::default();
    ensure_observability_component(&mut config).unwrap();
    let mut observability = component_observability_config(&config).unwrap();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    toggle_section(&mut observability, atof);
    set_section_field(&mut observability, atof, "output_directory", json!("logs")).unwrap();
    set_section_field(&mut observability, atof, "filename", json!("events.jsonl")).unwrap();
    store_observability_config(&mut config, &observability).unwrap();

    validate_config(&config).unwrap();
}

#[test]
fn typed_editor_serializes_explicit_observability_overrides() {
    let mut observability = ObservabilityConfig::default();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    toggle_section(&mut observability, atof);
    set_section_field(&mut observability, atof, "output_directory", json!("logs")).unwrap();

    let map = observability_config_map(&observability).unwrap();
    let atof = map
        .get("atof")
        .and_then(Value::as_object)
        .expect("atof section is serialized");
    assert_eq!(atof.get("enabled"), Some(&Value::Bool(true)));
    assert_eq!(atof.get("output_directory"), Some(&json!("logs")));
    assert_eq!(atof.get("mode"), Some(&json!("append")));
    assert!(map.contains_key("policy"));
}

#[test]
fn typed_editor_serializes_disabled_section_override() {
    let mut observability = ObservabilityConfig::default();
    let atif = ObservabilityConfig::editor_schema().field("atif").unwrap();
    toggle_section(&mut observability, atif);
    toggle_section(&mut observability, atif);

    let map = observability_config_map(&observability).unwrap();
    let atif = map
        .get("atif")
        .and_then(Value::as_object)
        .expect("disabled atif section is serialized");
    assert_eq!(atif.get("enabled"), Some(&Value::Bool(false)));
    assert_eq!(
        atif.get("filename_template"),
        Some(&json!("nemo-flow-atif-{session_id}.json"))
    );
}

#[test]
fn editor_save_preserves_unknown_observability_fields() {
    let mut config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "version": 1,
                "future_top_level": "preserve",
                "atof": {
                    "enabled": true,
                    "output_directory": "old-logs",
                    "future_atof_field": "preserve"
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };
    let mut observability = component_observability_config(&config).unwrap();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    remove_section_field(&mut observability, atof, "output_directory").unwrap();
    set_section_field(&mut observability, atof, "filename", json!("events.jsonl")).unwrap();

    store_observability_config(&mut config, &observability).unwrap();

    let component = observability_component(&config).unwrap();
    assert_eq!(
        component.config.get("future_top_level"),
        Some(&json!("preserve"))
    );
    let atof_config = component
        .config
        .get("atof")
        .and_then(Value::as_object)
        .unwrap();
    assert_eq!(
        atof_config.get("future_atof_field"),
        Some(&json!("preserve"))
    );
    assert_eq!(atof_config.get("filename"), Some(&json!("events.jsonl")));
    assert!(!atof_config.contains_key("output_directory"));
}

#[test]
fn component_enablement_and_summary_track_config_state() {
    let mut config = PluginConfig::default();
    ensure_observability_component(&mut config).unwrap();
    let mut observability = component_observability_config(&config).unwrap();

    assert!(component_enabled(&config));
    assert_eq!(
        observability_summary(&config, &observability),
        "component enabled, sections none"
    );

    set_component_enabled(&mut config, false);
    let atif = ObservabilityConfig::editor_schema().field("atif").unwrap();
    toggle_section(&mut observability, atif);

    assert!(!component_enabled(&config));
    assert_eq!(
        observability_summary(&config, &observability),
        "component disabled, sections ATIF"
    );
}

#[test]
fn reset_selected_field_accounts_for_section_toggle_offset() {
    let mut observability = ObservabilityConfig::default();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    let fields = atof.schema().unwrap().fields;

    set_section_field(&mut observability, atof, "output_directory", json!("logs")).unwrap();
    assert!(
        section_field_value(&observability, atof, "output_directory")
            .unwrap()
            .is_some()
    );

    let output_directory_index = fields
        .iter()
        .position(|field| field.name == "output_directory")
        .unwrap();
    assert!(
        reset_selected_field(&mut observability, atof, fields, output_directory_index + 1,)
            .unwrap()
    );
    assert_eq!(
        section_field_value(&observability, atof, "output_directory").unwrap(),
        None
    );
    assert!(!reset_selected_field(&mut observability, atof, fields, 0).unwrap());
}

#[test]
fn read_plugin_config_handles_missing_and_invalid_files() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("plugins.toml");
    let config = read_plugin_config(&missing).unwrap();
    assert!(config.components.is_empty());

    std::fs::write(&missing, "components = [\n").unwrap();
    let error = read_plugin_config(&missing).unwrap_err().to_string();
    assert!(error.contains("invalid plugin TOML"), "error was: {error}");
}

#[test]
fn write_plugin_config_prunes_defaults_and_round_trips() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    let mut config = PluginConfig::default();
    ensure_observability_component(&mut config).unwrap();

    write_plugin_config(&path, &config).unwrap();

    let rendered = std::fs::read_to_string(&path).unwrap();
    assert!(rendered.contains("kind = \"observability\""));
    assert!(!rendered.contains("enabled = true"));
    let round_tripped = read_plugin_config(&path).unwrap();
    assert_eq!(round_tripped.components.len(), 1);
    assert_eq!(round_tripped.components[0].kind, OBSERVABILITY_PLUGIN_KIND);
}

#[test]
fn prune_plugin_defaults_removes_default_policy_and_enabled_true_only() {
    let mut value = json!({
        "version": 1,
        "policy": ConfigPolicy::default(),
        "components": [
            { "kind": "observability", "enabled": true, "config": {} },
            { "kind": "other", "enabled": false, "config": {} }
        ]
    });

    prune_plugin_defaults(&mut value);

    let object = value.as_object().unwrap();
    assert!(!object.contains_key("policy"));
    let components = object["components"].as_array().unwrap();
    assert!(!components[0].as_object().unwrap().contains_key("enabled"));
    assert_eq!(components[1]["enabled"], json!(false));
}

#[test]
fn validate_config_reports_plugin_diagnostics() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "version": 1,
                "atof": {
                    "enabled": true,
                    "mode": "not-a-mode"
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };

    let error = validate_config(&config).unwrap_err().to_string();

    assert!(
        error.contains("plugin validation failed"),
        "error was: {error}"
    );
    assert!(error.contains("ATOF mode"), "error was: {error}");
}

#[test]
fn display_helpers_render_scalars_json_and_defaults() {
    assert_eq!(display_value(&json!("logs")), "logs");
    assert_eq!(display_value(&json!(true)), "true");
    assert_eq!(display_value(&json!(7)), "7");
    assert_eq!(display_value(&json!({ "a": 1 })), r#"{"a":1}"#);

    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    let mode = atof.schema().unwrap().field("mode").unwrap();
    assert_eq!(
        display_field_value(atof, mode, &json!("append")),
        "append (default)"
    );
    assert_eq!(
        display_field_value(atof, mode, &json!("overwrite")),
        "overwrite"
    );
}

#[test]
fn target_path_resolves_project_and_global_without_user_env() {
    let cwd = std::env::current_dir().unwrap();

    assert_eq!(
        target_path(TargetScope::Project).unwrap(),
        project_plugin_config_path(&cwd)
    );
    assert_eq!(
        target_path(TargetScope::Global).unwrap(),
        global_plugin_config_path()
    );
}
