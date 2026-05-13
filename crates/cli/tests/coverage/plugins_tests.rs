// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
