// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::configuration::{
    PluginsScopeArgs, global_plugin_config_path, project_plugin_config_path,
    user_plugin_config_path,
};
use nemo_relay::config_editor::{EditorConfig, EditorSchema};
use nemo_relay::observability::plugin_component::{OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig};
use nemo_relay::plugin::{ConfigPolicy, PluginComponentSpec, PluginConfig};
use nemo_relay::plugins::nemo_guardrails::component::{
    LocalBackendConfig, NEMO_GUARDRAILS_PLUGIN_KIND, NeMoGuardrailsConfig, RemoteBackendConfig,
};
use nemo_relay_adaptive::AdaptiveConfig;
use nemo_relay_adaptive::plugin_component::ADAPTIVE_PLUGIN_KIND;
use nemo_relay_pii_redaction::component::{PII_REDACTION_PLUGIN_KIND, PiiRedactionConfig};
use serde_json::Map;
use std::path::PathBuf;

fn write_editor_dynamic_manifest(
    dir: &Path,
    plugin_id: &str,
    name: Option<&str>,
    schema: Option<&Value>,
) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let name = name
        .map(|name| format!("name = {name:?}\n"))
        .unwrap_or_default();
    let schema_capability = if schema.is_some() {
        ", \"config_schema\""
    } else {
        ""
    };
    let schema_section = if schema.is_some() {
        "\n[config_schema]\npath = \"config.schema.json\"\n"
    } else {
        ""
    };
    let manifest = dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1

[plugin]
id = "{plugin_id}"
{name}kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"{schema_capability}]
{schema_section}
[load]
runtime = "python"
entrypoint = "fixture.plugin:register"
"#
        ),
    )
    .unwrap();
    if let Some(schema) = schema {
        std::fs::write(
            dir.join("config.schema.json"),
            serde_json::to_vec_pretty(schema).unwrap(),
        )
        .unwrap();
    }
    manifest
}

fn adaptive_component_config(agent_id: &str) -> serde_json::Map<String, Value> {
    json!({
        "agent_id": agent_id,
        "state": {
            "backend": {
                "kind": "in_memory",
                "config": {}
            }
        },
        "telemetry": {
            "learners": ["tool_parallelism"]
        },
        "adaptive_hints": {
            "priority": 100,
            "break_chain": false,
            "inject_header": true,
            "inject_body_path": "nvext.agent_hints"
        }
    })
    .as_object()
    .unwrap()
    .clone()
}

fn guardrails_component_config(config_id: &str) -> serde_json::Map<String, Value> {
    json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": config_id
        }
    })
    .as_object()
    .unwrap()
    .clone()
}

fn local_guardrails_component_config(config_path: &str) -> serde_json::Map<String, Value> {
    json!({
        "mode": "local",
        "input": false,
        "output": false,
        "config_path": config_path,
        "tool_input": true,
        "tool_output": true,
        "local": {
            "python_module": "custom_guardrails"
        }
    })
    .as_object()
    .unwrap()
    .clone()
}

fn local_llm_guardrails_component_config(config_yaml: &str) -> serde_json::Map<String, Value> {
    json!({
        "mode": "local",
        "codec": "openai_chat",
        "input": true,
        "output": true,
        "config_yaml": config_yaml,
        "colang_content": "define flow noop\n  pass",
        "local": {
            "python_module": "custom_guardrails"
        }
    })
    .as_object()
    .unwrap()
    .clone()
}

#[test]
fn target_scope_defaults_to_user_and_rejects_conflicts() {
    assert_eq!(
        target_scope(&PluginsScopeArgs::default()).unwrap(),
        TargetScope::User
    );
    assert_eq!(
        target_scope(&PluginsScopeArgs {
            project: true,
            ..PluginsScopeArgs::default()
        })
        .unwrap(),
        TargetScope::Project
    );
    assert_eq!(
        target_scope(&PluginsScopeArgs {
            global: true,
            ..PluginsScopeArgs::default()
        })
        .unwrap(),
        TargetScope::Global
    );

    let error = target_scope(&PluginsScopeArgs {
        user: true,
        project: true,
        ..PluginsScopeArgs::default()
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
fn typed_editor_model_contains_adaptive_options() {
    let schema = AdaptiveConfig::editor_schema();
    assert!(!schema.fields.iter().any(|field| field.name == "version"));
    let agent_id = schema.field("agent_id").unwrap();
    assert_eq!(agent_id.label, "fallback_agent_id");

    let state = schema.field("state").unwrap().schema().unwrap();
    let backend = state.field("backend").unwrap().schema().unwrap();
    assert_eq!(
        backend.field("kind").unwrap().enum_values,
        &["in_memory", "redis"]
    );
    assert_eq!(backend.field("config").unwrap().kind, EditorFieldKind::Json);

    let telemetry = schema.field("telemetry").unwrap().schema().unwrap();
    assert_eq!(
        telemetry.field("learners").unwrap().kind,
        EditorFieldKind::Json
    );

    let acg = schema.field("acg").unwrap().schema().unwrap();
    let thresholds = acg.field("stability_thresholds").unwrap().schema().unwrap();
    assert_eq!(
        thresholds.field("stable_threshold").unwrap().kind,
        EditorFieldKind::Float
    );
}

#[test]
fn typed_editor_model_contains_nemo_guardrails_options() {
    let schema = NeMoGuardrailsConfig::editor_schema();
    assert!(!schema.fields.iter().any(|field| field.name == "version"));
    assert_eq!(
        schema.field("mode").unwrap().enum_values,
        &["remote", "local"]
    );
    assert_eq!(schema.field("codec").unwrap().kind, EditorFieldKind::Enum);
    assert_eq!(
        schema.field("input").unwrap().kind,
        EditorFieldKind::Boolean
    );
    assert_eq!(
        schema.field("priority").unwrap().kind,
        EditorFieldKind::Integer
    );

    let remote = schema.field("remote").unwrap().schema().unwrap();
    assert_eq!(
        remote.field("timeout_millis").unwrap().kind,
        EditorFieldKind::Integer
    );
    assert_eq!(
        remote.field("headers").unwrap().kind,
        EditorFieldKind::StringMap
    );

    let local = schema.field("local").unwrap().schema().unwrap();
    assert_eq!(
        local.field("python_module").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        local.field("python_executable").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        schema.field("config_path").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        schema.field("config_yaml").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        schema.field("colang_content").unwrap().kind,
        EditorFieldKind::String
    );

    let request_defaults = schema.field("request_defaults").unwrap().schema().unwrap();
    let rails = request_defaults.field("rails").unwrap().schema().unwrap();
    assert_eq!(
        rails.field("tool_input").unwrap().kind,
        EditorFieldKind::Json
    );
}

#[test]
fn typed_editor_model_contains_pii_redaction_options() {
    let schema = PiiRedactionConfig::editor_schema();
    assert!(!schema.fields.iter().any(|field| field.name == "version"));
    assert_eq!(
        schema.field("mode").unwrap().enum_values,
        &["builtin", "local_model"]
    );
    assert_eq!(schema.field("codec").unwrap().kind, EditorFieldKind::Enum);
    assert_eq!(schema.field("mark").unwrap().kind, EditorFieldKind::Boolean);
    assert_eq!(
        schema.field("tool_output").unwrap().kind,
        EditorFieldKind::Boolean
    );

    let builtin = schema.field("builtin").unwrap().schema().unwrap();
    assert_eq!(builtin.field("action").unwrap().kind, EditorFieldKind::Enum);
    assert!(
        builtin
            .field("action")
            .unwrap()
            .enum_values
            .contains(&"redact")
    );
    assert_eq!(
        builtin.field("target_paths").unwrap().kind,
        EditorFieldKind::Json
    );
    assert_eq!(
        builtin.field("detector").unwrap().kind,
        EditorFieldKind::Enum
    );
    assert!(
        builtin
            .field("detector")
            .unwrap()
            .enum_values
            .contains(&"jwt")
    );
    assert!(
        builtin
            .field("detector")
            .unwrap()
            .enum_values
            .contains(&"aws_access_key_id")
    );
    assert_eq!(
        builtin.field("replacement").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        builtin.field("mask_char").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        builtin.field("unmasked_prefix").unwrap().kind,
        EditorFieldKind::Integer
    );
    assert_eq!(
        builtin.field("unmasked_suffix").unwrap().kind,
        EditorFieldKind::Integer
    );

    let local = schema.field("local").unwrap().schema().unwrap();
    assert_eq!(
        local.field("backend").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        local.field("model_id").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        local.field("detector_profile").unwrap().kind,
        EditorFieldKind::String
    );
    assert_eq!(
        local.field("allow_network").unwrap().kind,
        EditorFieldKind::Boolean
    );
    assert_eq!(
        local.field("max_latency_ms").unwrap().kind,
        EditorFieldKind::Integer
    );
}

#[test]
fn plugin_menu_uses_setup_theme_markers() {
    let theme = ColorfulTheme::default();
    let lines = render_menu_for_size(
        &theme,
        "plugins.toml",
        &[MenuItem::new("First"), MenuItem::new("Second")],
        0,
        24,
        80,
    );
    let rendered = lines.join("\n");

    assert!(rendered.contains('?'));
    assert!(rendered.contains('›'));
    assert!(rendered.contains('❯'));
    assert!(rendered.contains("↑/↓"));
    assert!(!rendered.contains("> First"));
}

#[test]
fn plugin_menu_builds_ordered_component_actions() {
    let temp = tempfile::tempdir().unwrap();
    let config = PluginConfig::default();
    let components = editable_components(&config).unwrap();

    let dynamic = vec![(
        "Example Dynamic".to_owned(),
        "dynamic; config absent; schema fields".to_owned(),
    )];
    let (items, actions) =
        plugin_menu_items(&components, &dynamic, &temp.path().join("plugins.toml"));
    let plain_labels = items
        .iter()
        .map(|item| console::strip_ansi_codes(&item.label).into_owned())
        .collect::<Vec<_>>();

    assert_eq!(items.len(), actions.len());
    assert!(plain_labels[0].starts_with("Observability [on] —"));
    assert!(
        plain_labels
            .iter()
            .any(|label| { label.starts_with("NeMo Guardrails [off] —") })
    );
    assert_eq!(
        plain_labels[components.len()],
        "Example Dynamic — dynamic; config absent; schema fields"
    );
    assert_eq!(plain_labels.len(), components.len() + dynamic.len() + 3);
    assert!(matches!(actions[0], MenuAction::EditComponent(0)));
    assert!(matches!(
        actions[components.len()],
        MenuAction::EditDynamic(0)
    ));
    assert!(matches!(actions[actions.len() - 3], MenuAction::Preview));
    assert!(matches!(actions[actions.len() - 2], MenuAction::Save));
    assert!(matches!(actions[actions.len() - 1], MenuAction::Cancel));
}

#[test]
fn component_menu_contains_toggle_fields_and_back() {
    let config = PluginConfig::default();
    let components = editable_components(&config).unwrap();
    let component = &components[0];

    let (items, actions) = component_menu_items(component);
    let labels = items
        .iter()
        .map(|item| console::strip_ansi_codes(&item.label).into_owned())
        .collect::<Vec<_>>();

    assert_eq!(labels[0], "Toggle component [on]");
    assert_eq!(labels[1], "  Edit ATOF");
    assert_eq!(labels.last().unwrap(), "Back [q]");
    assert!(matches!(actions[0], ComponentMenuAction::Toggle));
    assert!(matches!(actions[1], ComponentMenuAction::EditField(0)));
    assert!(matches!(actions.last(), Some(ComponentMenuAction::Back)));
}

#[test]
fn component_enablement_shortcuts_clear_and_reset_differ() {
    let config = PluginConfig::default();
    let mut components = editable_components(&config).unwrap();

    let observability = components
        .iter_mut()
        .find(|component| component.label() == "Observability")
        .unwrap();
    observability.set_enabled(false);
    reset_component_menu_item(observability, Some(ComponentMenuAction::Toggle)).unwrap();
    assert!(observability.enabled());

    clear_component_menu_item(observability, Some(ComponentMenuAction::Toggle)).unwrap();
    assert!(!observability.enabled());

    let adaptive = components
        .iter_mut()
        .find(|component| component.label() == "Adaptive")
        .unwrap();
    adaptive.set_enabled(true);
    reset_component_menu_item(adaptive, Some(ComponentMenuAction::Toggle)).unwrap();
    assert!(!adaptive.enabled());
}

#[test]
fn component_field_clear_only_removes_optional_fields() {
    let config = PluginConfig::default();
    let mut components = editable_components(&config).unwrap();

    let observability = components
        .iter_mut()
        .find(|component| component.label() == "Observability")
        .unwrap();
    let atof_index = observability
        .fields()
        .iter()
        .position(|field| field.name == "atof")
        .unwrap();
    let atof = observability.fields()[atof_index];
    assert!(atof.optional);
    let EditableComponent::Observability(state) = observability else {
        unreachable!();
    };
    state.config.atof = Some(Default::default());
    assert!(observability.field_configured(atof));
    clear_component_menu_item(
        observability,
        Some(ComponentMenuAction::EditField(atof_index)),
    )
    .unwrap();
    assert!(!observability.field_configured(atof));

    let pii_redaction = components
        .iter_mut()
        .find(|component| component.label() == "PII Redaction")
        .unwrap();
    let input_index = pii_redaction
        .fields()
        .iter()
        .position(|field| field.name == "input")
        .unwrap();
    let input = pii_redaction.fields()[input_index];
    assert!(!input.optional);
    let EditableComponent::PiiRedaction(state) = pii_redaction else {
        unreachable!();
    };
    state.config.input = false;
    assert!(pii_redaction.field_configured(input));
    clear_component_menu_item(
        pii_redaction,
        Some(ComponentMenuAction::EditField(input_index)),
    )
    .unwrap();
    assert!(pii_redaction.field_configured(input));
    let EditableComponent::PiiRedaction(state) = &*pii_redaction else {
        unreachable!();
    };
    assert!(!state.config.input);
    reset_component_menu_item(
        pii_redaction,
        Some(ComponentMenuAction::EditField(input_index)),
    )
    .unwrap();
    assert!(!pii_redaction.field_configured(input));
}

#[test]
fn menu_viewport_keeps_selection_visible_and_pages() {
    let first = menu_viewport(20, 0, 8);
    assert_eq!((first.start, first.end, first.page_size), (0, 4, 4));
    assert!(first.indicators);

    let middle = menu_viewport(20, 10, 8);
    assert!(middle.start <= 10 && middle.end > 10);
    assert_eq!(middle.end - middle.start, middle.page_size);

    let last = menu_viewport(20, 19, 8);
    assert_eq!(last.end, 20);
    assert!(last.start <= 19);
}

#[test]
fn menu_selection_navigation_wraps_pages_and_jumps() {
    assert_eq!(menu_selection_after_key(&Key::ArrowUp, 0, 10, 4), Some(9));
    assert_eq!(menu_selection_after_key(&Key::Char('k'), 4, 10, 4), Some(3));
    assert_eq!(menu_selection_after_key(&Key::ArrowDown, 9, 10, 4), Some(0));
    assert_eq!(menu_selection_after_key(&Key::Char('j'), 4, 10, 4), Some(5));
    assert_eq!(menu_selection_after_key(&Key::PageUp, 5, 10, 4), Some(1));
    assert_eq!(menu_selection_after_key(&Key::PageUp, 2, 10, 4), Some(0));
    assert_eq!(menu_selection_after_key(&Key::PageDown, 5, 10, 4), Some(9));
    assert_eq!(menu_selection_after_key(&Key::Home, 5, 10, 4), Some(0));
    assert_eq!(menu_selection_after_key(&Key::End, 5, 10, 4), Some(9));
    assert_eq!(menu_selection_after_key(&Key::Enter, 5, 10, 4), None);
    assert_eq!(menu_selection_after_key(&Key::PageDown, 0, 0, 4), None);
}

#[test]
fn menu_renderer_respects_terminal_height_and_width() {
    let theme = ColorfulTheme::default();
    let items = (0..20)
        .map(|index| MenuItem::new(format!("A very long menu item number {index}")))
        .collect::<Vec<_>>();
    let lines = render_menu_for_size(&theme, "plugins.toml", &items, 10, 8, 24);

    assert!(lines.len() <= 8);
    assert!(
        lines
            .iter()
            .all(|line| console::measure_text_width(line) <= 24)
    );
    assert!(lines.iter().any(|line| line.contains("↑")));
    assert!(lines.iter().any(|line| line.contains("↓")));
}

#[test]
fn menu_renderer_bounds_tiny_terminals_and_sanitizes_physical_rows() {
    let theme = ColorfulTheme::default();
    let items = [
        MenuItem::new("safe\nsecond row\r\u{1b}[31mred"),
        MenuItem::new("next"),
    ];

    for rows in [0, 1, 2] {
        let lines = render_menu_for_size(&theme, "unsafe\nprompt\u{1b}[2J", &items, 0, rows, 80);
        assert!(lines.len() <= rows.max(1), "rows={rows}: {lines:?}");
        assert!(lines.iter().all(|line| {
            !line
                .chars()
                .any(|character| matches!(character, '\n' | '\r' | '\u{1b}'))
        }));
    }
}

#[test]
fn menu_response_index_tracks_selected_and_shortcut_positions() {
    assert_eq!(menu_response_index(&MenuResponse::Selected(3)), Some(3));
    assert_eq!(
        menu_response_index(&MenuResponse::Shortcut(MenuShortcut::Reset, 4)),
        Some(4)
    );
    assert_eq!(menu_response_index(&MenuResponse::Cancel), None);
}

#[test]
fn plugin_cancellation_paths_share_message() {
    let errors = [
        cancelled_error(),
        menu_error(std::io::Error::from(std::io::ErrorKind::Interrupted)),
        editor_error(dialoguer::Error::IO(std::io::Error::from(
            std::io::ErrorKind::UnexpectedEof,
        ))),
    ];

    for error in errors {
        match error {
            CliError::Config(message) => {
                assert_eq!(message, PLUGIN_EDIT_CANCELLED_MESSAGE);
            }
            other => panic!("expected config error, got {other:?}"),
        }
    }
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
    let mut observability = component_observability_state(&config).unwrap();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    toggle_section(&mut observability.config, atof);
    set_section_field(
        &mut observability.config,
        atof,
        "output_directory",
        json!("logs"),
    )
    .unwrap();
    set_section_field(
        &mut observability.config,
        atof,
        "filename",
        json!("events.jsonl"),
    )
    .unwrap();
    store_observability_state(&mut config, &observability).unwrap();

    validate_config(&config).unwrap();
}

#[test]
fn editor_model_adds_disabled_adaptive_component() {
    let mut config = PluginConfig::default();

    ensure_adaptive_component(&mut config).unwrap();

    let component = config
        .components
        .iter()
        .find(|component| component.kind == ADAPTIVE_PLUGIN_KIND)
        .unwrap();
    assert_eq!(component.kind, ADAPTIVE_PLUGIN_KIND);
    assert!(!component.enabled);
    assert!(!component.config.contains_key("version"));
    assert!(component.config.contains_key("policy"));
}

#[test]
fn editor_model_reads_missing_nemo_guardrails_component_as_disabled_default() {
    let config = PluginConfig::default();

    let guardrails = component_nemo_guardrails_state(&config).unwrap();

    assert!(!guardrails.enabled);
    assert!(
        !config
            .components
            .iter()
            .any(|component| component.kind == NEMO_GUARDRAILS_PLUGIN_KIND)
    );
    assert_eq!(guardrails.config.mode, "remote");
    assert!(!nemo_guardrails_configured(&guardrails.config));
    assert_eq!(
        nemo_guardrails_summary(&guardrails),
        "component disabled, fields none"
    );
    assert!(!guardrails.should_store(nemo_guardrails_configured(&guardrails.config)));
}

#[test]
fn editor_save_persists_disabled_nemo_guardrails_policy_only_edits() {
    let mut config = PluginConfig::default();
    let mut guardrails = component_nemo_guardrails_state(&config).unwrap();
    let policy = NeMoGuardrailsConfig::editor_schema()
        .field("policy")
        .unwrap();

    set_section_field(
        &mut guardrails.config,
        policy,
        "unknown_field",
        json!("ignore"),
    )
    .unwrap();
    guardrails.mark_config_touched();

    assert!(!guardrails.enabled);
    assert!(!nemo_guardrails_configured(&guardrails.config));

    store_nemo_guardrails_state(&mut config, &guardrails).unwrap();

    let component = config
        .components
        .iter()
        .find(|component| component.kind == NEMO_GUARDRAILS_PLUGIN_KIND)
        .unwrap();
    assert!(!component.enabled);
    assert_eq!(component.config["policy"]["unknown_field"], json!("ignore"));
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
        Some(&json!("nemo-relay-atif-{session_id}.json"))
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
    let mut observability = component_observability_state(&config).unwrap();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    remove_section_field(&mut observability.config, atof, "output_directory").unwrap();
    set_section_field(
        &mut observability.config,
        atof,
        "filename",
        json!("events.jsonl"),
    )
    .unwrap();

    store_observability_state(&mut config, &observability).unwrap();

    let component = config
        .components
        .iter()
        .find(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
        .unwrap();
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
fn editor_save_preserves_unknown_adaptive_fields_and_all_sections() {
    let mut config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: ADAPTIVE_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "version": 1,
                "future_top_level": "preserve",
                "state": {
                    "future_state": "preserve",
                    "backend": {
                        "kind": "in_memory",
                        "config": {},
                        "future_backend": "preserve"
                    }
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };
    let mut adaptive = component_adaptive_state(&config).unwrap();
    let schema = AdaptiveConfig::editor_schema();
    let state = schema.field("state").unwrap();
    let telemetry = schema.field("telemetry").unwrap();
    let adaptive_hints = schema.field("adaptive_hints").unwrap();
    let tool_parallelism = schema.field("tool_parallelism").unwrap();
    let acg = schema.field("acg").unwrap();

    set_struct_field(&mut adaptive.config, "agent_id", json!("planner")).unwrap();
    set_section_field(
        &mut adaptive.config,
        state,
        "backend",
        json!({
            "kind": "redis",
            "config": {
                "url": "redis://127.0.0.1/",
                "key_prefix": "adaptive:"
            }
        }),
    )
    .unwrap();
    set_section_field(
        &mut adaptive.config,
        telemetry,
        "learners",
        json!(["tool_parallelism", "acg"]),
    )
    .unwrap();
    set_section_field(
        &mut adaptive.config,
        telemetry,
        "subscriber_name",
        json!("adaptive"),
    )
    .unwrap();
    set_section_field(
        &mut adaptive.config,
        adaptive_hints,
        "inject_body_path",
        json!("nvext.agent_hints"),
    )
    .unwrap();
    set_section_field(
        &mut adaptive.config,
        tool_parallelism,
        "mode",
        json!("inject_hints"),
    )
    .unwrap();
    set_section_field(&mut adaptive.config, acg, "provider", json!("anthropic")).unwrap();
    set_section_field(
        &mut adaptive.config,
        acg,
        "stability_thresholds",
        json!({
            "stable_threshold": 0.9,
            "semi_stable_threshold": 0.4,
            "min_observations_for_full_confidence": 10
        }),
    )
    .unwrap();

    store_adaptive_state(&mut config, &adaptive).unwrap();

    let component = config
        .components
        .iter()
        .find(|component| component.kind == ADAPTIVE_PLUGIN_KIND)
        .unwrap();
    assert!(!component.config.contains_key("version"));
    assert_eq!(
        component.config.get("future_top_level"),
        Some(&json!("preserve"))
    );
    let state = component.config["state"].as_object().unwrap();
    assert_eq!(state.get("future_state"), Some(&json!("preserve")));
    let backend = state["backend"].as_object().unwrap();
    assert_eq!(backend.get("kind"), Some(&json!("redis")));
    assert_eq!(backend.get("future_backend"), Some(&json!("preserve")));
    assert_eq!(backend["config"]["key_prefix"], json!("adaptive:"));
    assert_eq!(
        component.config["telemetry"]["learners"],
        json!(["tool_parallelism", "acg"])
    );
    assert_eq!(
        component.config["adaptive_hints"]["inject_body_path"],
        json!("nvext.agent_hints")
    );
    assert_eq!(
        component.config["tool_parallelism"]["mode"],
        json!("inject_hints")
    );
    assert_eq!(
        component.config["acg"]["stability_thresholds"]["stable_threshold"],
        json!(0.9)
    );
}

#[test]
fn editor_save_preserves_unknown_nemo_guardrails_fields_and_sections() {
    let mut config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "version": 1,
                "future_top_level": "preserve",
                "mode": "remote",
                "codec": "openai_chat",
                "remote": {
                    "endpoint": "http://old.example.test",
                    "config_id": "old",
                    "future_remote": "preserve"
                },
                "request_defaults": {
                    "future_defaults": "preserve",
                    "rails": {
                        "input": true,
                        "future_rails": "preserve"
                    }
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };
    let mut guardrails = component_nemo_guardrails_state(&config).unwrap();
    let schema = NeMoGuardrailsConfig::editor_schema();
    let remote = schema.field("remote").unwrap();
    let request_defaults = schema.field("request_defaults").unwrap();

    set_struct_field(&mut guardrails.config, "codec", json!("openai_chat")).unwrap();
    set_section_field(
        &mut guardrails.config,
        remote,
        "endpoint",
        json!("http://localhost:8000"),
    )
    .unwrap();
    set_section_field(
        &mut guardrails.config,
        remote,
        "config_id",
        json!("default"),
    )
    .unwrap();
    set_section_field(
        &mut guardrails.config,
        request_defaults,
        "context",
        json!({"tenant": "docs"}),
    )
    .unwrap();

    guardrails.set_enabled(false);
    store_nemo_guardrails_state(&mut config, &guardrails).unwrap();

    let component = config
        .components
        .iter()
        .find(|component| component.kind == NEMO_GUARDRAILS_PLUGIN_KIND)
        .unwrap();
    assert!(!component.enabled);
    assert!(!component.config.contains_key("version"));
    assert_eq!(
        component.config.get("future_top_level"),
        Some(&json!("preserve"))
    );
    let remote = component.config["remote"].as_object().unwrap();
    assert_eq!(
        remote.get("endpoint"),
        Some(&json!("http://localhost:8000"))
    );
    assert_eq!(remote.get("future_remote"), Some(&json!("preserve")));
    let request_defaults = component.config["request_defaults"].as_object().unwrap();
    assert_eq!(
        request_defaults.get("future_defaults"),
        Some(&json!("preserve"))
    );
    assert_eq!(request_defaults["context"], json!({"tenant": "docs"}));
    assert_eq!(request_defaults["rails"]["future_rails"], json!("preserve"));
}

#[test]
fn editor_save_preserves_unknown_pii_redaction_fields_and_prunes_version() {
    let mut config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: PII_REDACTION_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "version": 1,
                "future_top_level": "preserve",
                "mode": "builtin",
                "codec": "openai_chat",
                "builtin": {
                    "action": "mask",
                    "detector": "email",
                    "target_paths": ["/message"],
                    "future_builtin": "preserve"
                },
                "local": {
                    "future_local": "preserve"
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };

    let mut pii_redaction = component_pii_redaction_state(&config).unwrap();
    let schema = PiiRedactionConfig::editor_schema();
    let builtin = schema.field("builtin").unwrap();

    set_struct_field(&mut pii_redaction.config, "mode", json!("builtin")).unwrap();
    set_struct_field(&mut pii_redaction.config, "codec", json!("openai_chat")).unwrap();
    set_section_field(
        &mut pii_redaction.config,
        builtin,
        "action",
        json!("redact"),
    )
    .unwrap();
    set_section_field(
        &mut pii_redaction.config,
        builtin,
        "detector",
        json!("bearer_token"),
    )
    .unwrap();
    set_section_field(
        &mut pii_redaction.config,
        builtin,
        "replacement",
        json!("[REDACTED]"),
    )
    .unwrap();

    pii_redaction.set_enabled(false);
    store_pii_redaction_state(&mut config, &pii_redaction).unwrap();

    let component = config
        .components
        .iter()
        .find(|component| component.kind == PII_REDACTION_PLUGIN_KIND)
        .unwrap();
    assert!(!component.enabled);
    assert!(!component.config.contains_key("version"));
    assert_eq!(
        component.config.get("future_top_level"),
        Some(&json!("preserve"))
    );
    let builtin = component.config["builtin"].as_object().unwrap();
    assert_eq!(builtin.get("action"), Some(&json!("redact")));
    assert_eq!(builtin.get("detector"), Some(&json!("bearer_token")));
    assert_eq!(builtin.get("future_builtin"), Some(&json!("preserve")));
    let local = component.config["local"].as_object().unwrap();
    assert_eq!(local.get("future_local"), Some(&json!("preserve")));
}

#[test]
fn adaptive_config_field_reset_handles_optional_and_default_fields() {
    let mut adaptive = AdaptiveConfig {
        agent_id: Some("planner".into()),
        acg: Some(Default::default()),
        ..AdaptiveConfig::default()
    };
    let schema = AdaptiveConfig::editor_schema();

    reset_config_field(&mut adaptive, schema.field("agent_id").unwrap()).unwrap();
    reset_config_field(&mut adaptive, schema.field("acg").unwrap()).unwrap();

    assert!(adaptive.agent_id.is_none());
    assert!(adaptive.acg.is_none());
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct OptionalSectionHarness {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    optional: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent: Option<Value>,
}

fn optional_section_without_default(name: &'static str) -> EditorFieldSpec {
    EditorFieldSpec {
        name,
        label: name,
        kind: EditorFieldKind::Section,
        enum_values: &[],
        optional: true,
        nested_schema: None,
        nested_default: None,
    }
}

fn depth_root_schema() -> &'static EditorSchema {
    static FIELDS: [EditorFieldSpec; 1] = [EditorFieldSpec {
        name: "middle",
        label: "middle",
        kind: EditorFieldKind::Section,
        enum_values: &[],
        optional: false,
        nested_schema: Some(depth_middle_schema),
        nested_default: None,
    }];
    static SCHEMA: EditorSchema = EditorSchema { fields: &FIELDS };
    &SCHEMA
}

fn depth_middle_schema() -> &'static EditorSchema {
    static FIELDS: [EditorFieldSpec; 1] = [EditorFieldSpec {
        name: "leaf",
        label: "leaf",
        kind: EditorFieldKind::Section,
        enum_values: &[],
        optional: false,
        nested_schema: Some(depth_leaf_schema),
        nested_default: None,
    }];
    static SCHEMA: EditorSchema = EditorSchema { fields: &FIELDS };
    &SCHEMA
}

fn depth_leaf_schema() -> &'static EditorSchema {
    static FIELDS: [EditorFieldSpec; 1] = [EditorFieldSpec {
        name: "name",
        label: "name",
        kind: EditorFieldKind::String,
        enum_values: &[],
        optional: false,
        nested_schema: None,
        nested_default: None,
    }];
    static SCHEMA: EditorSchema = EditorSchema { fields: &FIELDS };
    &SCHEMA
}

#[test]
fn reset_section_clears_optional_section_without_default() {
    let section = optional_section_without_default("optional");
    let mut config = OptionalSectionHarness {
        optional: Some(json!({})),
        parent: None,
    };

    reset_section(&mut config, section);

    assert!(config.optional.is_none());
}

#[test]
fn nested_edit_empty_optional_section_without_default_clears_field() {
    let optional = optional_section_without_default("optional");
    let parent = optional_section_without_default("parent");
    let child = optional_section_without_default("child");
    let mut config = OptionalSectionHarness {
        optional: Some(json!({ "old": true })),
        parent: Some(json!({
            "child": {},
            "kept": true
        })),
    };

    store_edited_config_section(&mut config, optional, json!({})).unwrap();
    assert!(config.optional.is_none());

    store_edited_section_field(&mut config, parent, child, json!({})).unwrap();
    let parent = config.parent.as_ref().unwrap().as_object().unwrap();
    assert!(!parent.contains_key("child"));
    assert_eq!(parent.get("kept"), Some(&json!(true)));

    let mut value = json!({ "child": {}, "kept": true });
    store_edited_value_section(&mut value, child, json!({}));
    let value = value.as_object().unwrap();
    assert!(!value.contains_key("child"));
    assert_eq!(value.get("kept"), Some(&json!(true)));
}

#[test]
fn recursive_value_defaults_flow_from_parent_objects() {
    let middle = depth_root_schema().field("middle").unwrap();
    let leaf = depth_middle_schema().field("leaf").unwrap();
    let default = json!({
        "middle": {
            "leaf": {
                "name": "from-parent"
            }
        }
    });

    let middle_default = value_field_default(Some(&default), middle).unwrap();
    assert_eq!(
        value_field_default(Some(&middle_default), leaf),
        Some(json!({ "name": "from-parent" }))
    );

    let mut root_value = json!({ "middle": { "leaf": { "name": "custom" } } });
    reset_value_field(&mut root_value, middle, Some(&default));
    assert_eq!(root_value["middle"]["leaf"]["name"], json!("from-parent"));

    let mut middle_value = json!({ "leaf": { "name": "custom" } });
    reset_value_field(&mut middle_value, leaf, Some(&middle_default));
    assert_eq!(middle_value["leaf"]["name"], json!("from-parent"));
}

#[test]
fn merge_known_editor_object_recurses_through_arbitrary_section_depth() {
    let schema = depth_root_schema();
    let mut existing = json!({
        "middle": {
            "leaf": {
                "name": "old",
                "future_leaf": "preserve"
            },
            "future_middle": "preserve"
        },
        "future_root": "preserve"
    })
    .as_object()
    .unwrap()
    .clone();
    let edited = json!({
        "middle": {
            "leaf": {
                "name": "new"
            }
        }
    })
    .as_object()
    .unwrap()
    .clone();

    merge_known_editor_object(&mut existing, edited, &nested_editor_keys(schema), schema);

    let middle = existing.get("middle").unwrap().as_object().unwrap();
    let leaf = middle.get("leaf").unwrap().as_object().unwrap();
    assert_eq!(leaf.get("name"), Some(&json!("new")));
    assert_eq!(leaf.get("future_leaf"), Some(&json!("preserve")));
    assert_eq!(middle.get("future_middle"), Some(&json!("preserve")));
    assert_eq!(existing.get("future_root"), Some(&json!("preserve")));
}

#[test]
fn observability_config_field_reset_clears_optional_section() {
    let mut observability = ObservabilityConfig::default();
    let atof = ObservabilityConfig::editor_schema().field("atof").unwrap();
    toggle_section(&mut observability, atof);

    reset_config_field(&mut observability, atof).unwrap();

    assert!(observability.atof.is_none());
}

#[test]
fn adaptive_summary_tracks_component_and_configured_fields() {
    let mut config = PluginConfig::default();
    ensure_adaptive_component(&mut config).unwrap();
    let mut adaptive = component_adaptive_state(&config).unwrap();

    assert_eq!(
        adaptive_summary(&adaptive),
        "component disabled, fields none"
    );

    adaptive.set_enabled(true);
    set_struct_field(&mut adaptive.config, "agent_id", json!("planner")).unwrap();
    let adaptive_hints = AdaptiveConfig::editor_schema()
        .field("adaptive_hints")
        .unwrap();
    set_section_field(
        &mut adaptive.config,
        adaptive_hints,
        "inject_header",
        json!(true),
    )
    .unwrap();

    assert_eq!(
        adaptive_summary(&adaptive),
        "component enabled, fields fallback_agent_id, adaptive_hints"
    );
}

#[test]
fn nemo_guardrails_summary_tracks_component_and_configured_fields() {
    let config = PluginConfig::default();
    let mut guardrails = component_nemo_guardrails_state(&config).unwrap();

    assert_eq!(
        nemo_guardrails_summary(&guardrails),
        "component disabled, fields none"
    );

    guardrails.set_enabled(true);
    set_struct_field(&mut guardrails.config, "codec", json!("openai_chat")).unwrap();
    let remote = NeMoGuardrailsConfig::editor_schema()
        .field("remote")
        .unwrap();
    set_section_field(
        &mut guardrails.config,
        remote,
        "endpoint",
        json!("http://localhost:8000"),
    )
    .unwrap();

    assert!(nemo_guardrails_configured(&guardrails.config));
    assert_eq!(
        nemo_guardrails_summary(&guardrails),
        "component enabled, fields codec, remote"
    );
    assert!(guardrails.should_store(nemo_guardrails_configured(&guardrails.config)));

    let existing = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: false,
            config: guardrails_component_config("existing"),
        }],
        ..PluginConfig::default()
    };
    let mut existing = component_nemo_guardrails_state(&existing).unwrap();
    reset_config_field(
        &mut existing.config,
        NeMoGuardrailsConfig::editor_schema()
            .field("remote")
            .unwrap(),
    )
    .unwrap();
    assert!(existing.should_store(nemo_guardrails_configured(&existing.config)));
}

#[test]
fn component_enablement_and_summary_track_config_state() {
    let mut config = PluginConfig::default();
    ensure_observability_component(&mut config).unwrap();
    let mut observability = component_observability_state(&config).unwrap();

    assert!(observability.enabled);
    assert_eq!(
        observability_summary(&observability),
        "component enabled, sections none"
    );

    observability.set_enabled(false);
    let atif = ObservabilityConfig::editor_schema().field("atif").unwrap();
    toggle_section(&mut observability.config, atif);

    assert!(!observability.enabled);
    assert_eq!(
        observability_summary(&observability),
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

    std::fs::write(&missing, "components = \"not-a-list\"").unwrap();
    let error = read_plugin_config(&missing).unwrap_err().to_string();
    assert!(
        error.contains("invalid plugin config"),
        "error was: {error}"
    );
}

#[test]
fn plugin_config_document_preserves_host_sections_and_dynamic_config_presence() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(
        &path,
        r#"version = 1
host_setting = "preserve-me"

[[components]]
kind = "observability"
enabled = true
extension_field = "preserve-component-extension"
config = {}

[plugins.policy]
allow_unsigned = true

[[plugins.dynamic]]
manifest = "./first/relay-plugin.toml"

[[plugins.dynamic]]
manifest = "./second/relay-plugin.toml"
config = {}

[host.extra]
value = "preserve-host-section"
"#,
    )
    .unwrap();

    let mut document = PluginConfigDocument::read(&path).unwrap();
    assert_eq!(document.path(), path);
    assert_eq!(document.config().components.len(), 1);
    let entries = document.dynamic_entries().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].index, 0);
    assert_eq!(entries[0].manifest, "./first/relay-plugin.toml");
    assert_eq!(
        entries[0].manifest_path,
        temp.path().join("first/relay-plugin.toml")
    );
    assert_eq!(entries[0].config, None);
    assert_eq!(entries[1].config, Some(Map::new()));

    document.config_mut().components[0].enabled = false;
    document
        .set_dynamic_config(0, Map::from_iter([("token".to_owned(), json!("secret"))]))
        .unwrap();
    document.remove_dynamic_config(1).unwrap();

    let mut redacted = document.clone();
    redacted
        .set_dynamic_config(
            0,
            Map::from_iter([("token".to_owned(), json!("<redacted>"))]),
        )
        .unwrap();
    assert!(redacted.render().unwrap().contains("<redacted>"));
    assert!(!document.render().unwrap().contains("<redacted>"));

    document.write().unwrap();
    let rendered = std::fs::read_to_string(&path).unwrap();
    let root = rendered.parse::<toml::Table>().unwrap();
    assert_eq!(root["host_setting"].as_str(), Some("preserve-me"));
    assert_eq!(
        root["host"]["extra"]["value"].as_str(),
        Some("preserve-host-section")
    );
    let component = &root["components"].as_array().unwrap()[0];
    assert_eq!(component["enabled"].as_bool(), Some(false));
    assert_eq!(
        component["extension_field"].as_str(),
        Some("preserve-component-extension")
    );
    assert_eq!(
        root["plugins"]["policy"]["allow_unsigned"].as_bool(),
        Some(true)
    );
    let dynamic = root["plugins"]["dynamic"].as_array().unwrap();
    assert_eq!(
        dynamic[0]["manifest"].as_str(),
        Some("./first/relay-plugin.toml")
    );
    assert_eq!(dynamic[0]["config"]["token"].as_str(), Some("secret"));
    assert!(dynamic[1].get("config").is_none());
}

#[test]
fn dynamic_config_array_resize_preserves_toml_native_values() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(
        &path,
        r#"[[plugins.dynamic]]
manifest = "./plugin/relay-plugin.toml"
config = { records = [{ observed_at = 1979-05-27T07:32:00Z, label = "existing" }] }
"#,
    )
    .unwrap();

    let mut document = PluginConfigDocument::read(&path).unwrap();
    let original = document.dynamic_entries().unwrap()[0]
        .config
        .clone()
        .unwrap();
    let mut grown = original.clone();
    grown
        .get_mut("records")
        .unwrap()
        .as_array_mut()
        .unwrap()
        .push(json!({"label": "new"}));
    document
        .patch_dynamic_config(0, Some(&original), Some(grown.clone()))
        .unwrap();

    let grown_root = document.render().unwrap().parse::<toml::Table>().unwrap();
    let grown_records =
        grown_root["plugins"]["dynamic"].as_array().unwrap()[0]["config"]["records"]
            .as_array()
            .unwrap();
    assert_eq!(grown_records.len(), 2);
    assert!(grown_records[0]["observed_at"].is_datetime());

    let mut shrunk = grown.clone();
    shrunk
        .get_mut("records")
        .unwrap()
        .as_array_mut()
        .unwrap()
        .truncate(1);
    document
        .patch_dynamic_config(0, Some(&grown), Some(shrunk))
        .unwrap();

    let shrunk_root = document.render().unwrap().parse::<toml::Table>().unwrap();
    let shrunk_records =
        shrunk_root["plugins"]["dynamic"].as_array().unwrap()[0]["config"]["records"]
            .as_array()
            .unwrap();
    assert_eq!(shrunk_records.len(), 1);
    assert!(shrunk_records[0]["observed_at"].is_datetime());
}

#[test]
fn dynamic_editor_loads_document_local_plugins_and_redacts_schema_secrets() {
    let temp = tempfile::tempdir().unwrap();
    write_editor_dynamic_manifest(
        &temp.path().join("plugins/structured"),
        "acme.structured",
        Some("Structured Plugin"),
        Some(&json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "token": {"type": "string", "writeOnly": true},
                "retries": {"type": "integer", "default": 3},
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "password": {"type": "string", "writeOnly": true}
                        }
                    }
                }
            }
        })),
    );
    write_editor_dynamic_manifest(&temp.path().join("plugins/raw"), "acme.raw", None, None);
    write_editor_dynamic_manifest(
        &temp.path().join("plugins/absent"),
        "acme.absent",
        None,
        Some(&json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"optional": {"type": "string"}}
        })),
    );
    write_editor_dynamic_manifest(&temp.path().join("plugins/empty"), "acme.empty", None, None);
    let path = temp.path().join("plugins.toml");
    std::fs::write(
        &path,
        r#"[[plugins.dynamic]]
manifest = "./plugins/structured/relay-plugin.toml"
config = { token = "super-secret", retries = 5, records = [{ password = "nested-secret" }], observed_at = 1979-05-27T07:32:00Z, unknown = { nested = "keep" } }

[[plugins.dynamic]]
manifest = "./plugins/raw/relay-plugin.toml"
config = { local_time = 1979-05-27T07:32:00Z }

[[plugins.dynamic]]
manifest = "./plugins/absent/relay-plugin.toml"

[[plugins.dynamic]]
manifest = "./plugins/empty/relay-plugin.toml"
config = {}
"#,
    )
    .unwrap();

    let document = PluginConfigDocument::read(&path).unwrap();
    let mut states = load_dynamic_plugin_states(&document).unwrap();
    assert_eq!(states.len(), 4);
    assert_eq!(states[0].label(), "Structured Plugin (acme.structured)");
    assert_eq!(states[1].label(), "acme.raw");
    assert!(states[0].menu_summary().contains("schema fields"));
    assert!(states[1].menu_summary().contains("configured"));
    assert!(states[2].menu_summary().contains("config absent"));
    assert!(states[3].menu_summary().contains("explicit empty config"));
    assert!(states[0].top_level_field_uses_hidden_json("records"));
    assert_eq!(
        states[0]
            .restore_raw_field_for_test(&["records".to_owned()])
            .unwrap(),
        json!([{ "password": "nested-secret" }])
    );
    let labels = states[0].top_level_field_labels();
    assert!(labels.iter().any(|label| label.contains("<redacted>")));
    assert!(labels.iter().all(|label| !label.contains("super-secret")));

    states[2].clear_top_level_field("optional");
    states[2].reset_top_level_field("optional").unwrap();
    assert_eq!(states[2].config(), None);

    let mut preview = document.clone();
    for state in &states {
        state.apply_to_document(&mut preview, true).unwrap();
    }
    let rendered = preview.render().unwrap();
    assert!(rendered.contains("<redacted>"));
    assert!(!rendered.contains("super-secret"));
    assert!(!rendered.contains("nested-secret"));
    let preview_root = rendered.parse::<toml::Table>().unwrap();
    assert!(
        preview_root["plugins"]["dynamic"].as_array().unwrap()[0]["config"]["observed_at"]
            .is_datetime()
    );
    assert!(
        preview_root["plugins"]["dynamic"].as_array().unwrap()[1]["config"]["local_time"]
            .is_datetime()
    );
    assert!(
        preview_root["plugins"]["dynamic"].as_array().unwrap()[2]
            .get("config")
            .is_none()
    );

    states[0].reset_top_level_field("retries").unwrap();
    assert_eq!(states[0].config().unwrap().get("retries"), Some(&json!(3)));
    assert_eq!(
        states[0].config().unwrap().get("unknown"),
        Some(&json!({"nested": "keep"}))
    );
    let mut touched = document.clone();
    states[0].apply_to_document(&mut touched, false).unwrap();
    assert_eq!(
        touched.dynamic_entries().unwrap()[0]
            .config
            .as_ref()
            .unwrap()
            .get("unknown"),
        Some(&json!({"nested": "keep"}))
    );
    let touched_root = touched.render().unwrap().parse::<toml::Table>().unwrap();
    assert!(
        touched_root["plugins"]["dynamic"].as_array().unwrap()[0]["config"]["observed_at"]
            .is_datetime()
    );
    states[0].clear_top_level_field("token");
    states[0].clear_top_level_field("retries");
    states[0].clear_top_level_field("unknown");
    states[0].clear_top_level_field("observed_at");
    states[0].clear_top_level_field("records");
    assert_eq!(states[0].config(), Some(&Map::new()));

    states[0].reset();
    let mut persisted = document.clone();
    for state in &states {
        state.apply_to_document(&mut persisted, false).unwrap();
    }
    let entries = persisted.dynamic_entries().unwrap();
    assert_eq!(entries[0].config, None);
    assert_eq!(entries[2].config, None);
    assert_eq!(entries[3].config, Some(Map::new()));
    let persisted_root = persisted.render().unwrap().parse::<toml::Table>().unwrap();
    assert!(
        persisted_root["plugins"]["dynamic"].as_array().unwrap()[1]["config"]["local_time"]
            .is_datetime()
    );
}

#[test]
fn dynamic_editor_rejects_unreadable_declared_schema() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    write_editor_dynamic_manifest(
        &plugin_dir,
        "acme.missing-schema",
        None,
        Some(&json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object"
        })),
    );
    std::fs::remove_file(plugin_dir.join("config.schema.json")).unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(
        &path,
        "[[plugins.dynamic]]\nmanifest = \"./plugin/relay-plugin.toml\"\n",
    )
    .unwrap();

    let document = PluginConfigDocument::read(&path).unwrap();
    let error = load_dynamic_plugin_states(&document)
        .unwrap_err()
        .to_string();
    assert!(error.contains("acme.missing-schema"), "{error}");
    assert!(error.contains("failed to read schema"), "{error}");
}

#[test]
fn plugin_config_document_reports_invalid_dynamic_entries_and_indexes() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(&path, "[plugins]\ndynamic = \"not-an-array\"\n").unwrap();
    let mut document = PluginConfigDocument::read(&path).unwrap();
    assert!(
        document
            .dynamic_entries()
            .unwrap_err()
            .to_string()
            .contains("plugins.dynamic must be an array")
    );
    assert!(
        document
            .remove_dynamic_config(0)
            .unwrap_err()
            .to_string()
            .contains("plugins.dynamic must be an array")
    );

    std::fs::write(&path, "[[plugins.dynamic]]\nmanifest = \"plugin.toml\"\n").unwrap();
    let mut document = PluginConfigDocument::read(&path).unwrap();
    assert!(
        document
            .set_dynamic_config(1, Map::new())
            .unwrap_err()
            .to_string()
            .contains("index 1 is out of range")
    );
}

#[test]
fn write_plugin_config_prunes_defaults_and_round_trips() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    let mut config = PluginConfig::default();
    ensure_observability_component(&mut config).unwrap();
    config.components.push(PluginComponentSpec {
        kind: ADAPTIVE_PLUGIN_KIND.to_string(),
        enabled: true,
        config: adaptive_component_config("cli-roundtrip"),
    });
    config.components.push(PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: false,
        config: guardrails_component_config("cli-roundtrip"),
    });

    write_plugin_config(&path, &config).unwrap();

    let rendered = std::fs::read_to_string(&path).unwrap();
    assert!(rendered.contains("kind = \"observability\""));
    assert!(rendered.contains("kind = \"adaptive\""));
    assert!(rendered.contains("kind = \"nemo_guardrails\""));
    assert!(!rendered.contains("enabled = true"));
    let round_tripped = read_plugin_config(&path).unwrap();
    assert_eq!(round_tripped.components.len(), 3);
    assert_eq!(round_tripped.components[0].kind, OBSERVABILITY_PLUGIN_KIND);
    let adaptive = round_tripped
        .components
        .iter()
        .find(|component| component.kind == ADAPTIVE_PLUGIN_KIND)
        .unwrap();
    assert_eq!(
        adaptive.config.get("agent_id"),
        Some(&json!("cli-roundtrip"))
    );
    let adaptive_hints = adaptive
        .config
        .get("adaptive_hints")
        .and_then(Value::as_object)
        .unwrap();
    assert_eq!(
        adaptive_hints.get("inject_body_path"),
        Some(&json!("nvext.agent_hints"))
    );
    let guardrails = round_tripped
        .components
        .iter()
        .find(|component| component.kind == NEMO_GUARDRAILS_PLUGIN_KIND)
        .unwrap();
    assert!(!guardrails.enabled);
    assert_eq!(
        guardrails.config["remote"]["config_id"],
        json!("cli-roundtrip")
    );
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

    let mut scalar = json!("unchanged");
    prune_plugin_defaults(&mut scalar);
    assert_eq!(scalar, json!("unchanged"));

    let mut object = serde_json::Map::from_iter([("policy".to_string(), json!({"unknown": true}))]);
    remove_default_field(
        &mut object,
        "missing",
        serde_json::to_value(ConfigPolicy::default()).unwrap(),
    );
    assert!(object.contains_key("policy"));

    let mut nested = json!({"a": 1, "b": 2});
    remove_matching_defaults(&mut nested, &json!({"a": 1, "c": 3}));
    assert_eq!(nested, json!({"b": 2}));

    let mut non_object = json!(["unchanged"]);
    remove_matching_defaults(&mut non_object, &json!({"a": 1}));
    assert_eq!(non_object, json!(["unchanged"]));
}

#[test]
fn print_preview_renders_default_plugin_config() {
    print_preview(&PluginConfig::default()).unwrap();
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
fn validate_config_accepts_adaptive_component() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: ADAPTIVE_PLUGIN_KIND.to_string(),
            enabled: true,
            config: adaptive_component_config("cli-validation"),
        }],
        ..PluginConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn validate_config_accepts_nemo_guardrails_component() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: guardrails_component_config("cli-validation"),
        }],
        ..PluginConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn validate_config_accepts_local_tool_only_nemo_guardrails_component() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: local_guardrails_component_config("./rails"),
        }],
        ..PluginConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn validate_config_accepts_pii_redaction_component() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: PII_REDACTION_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "mode": "builtin",
                "codec": "openai_chat",
                "input": true,
                "output": true,
                "builtin": {
                    "action": "redact",
                    "detector": "email"
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };

    validate_config(&config).unwrap();
}

#[test]
fn validate_config_rejects_local_nemo_guardrails_request_defaults() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "mode": "local",
                "codec": "openai_chat",
                "input": true,
                "output": true,
                "config_yaml": "models: []",
                "request_defaults": {
                    "context": {"tenant": "demo"}
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };

    let error = validate_config(&config).unwrap_err().to_string();
    assert!(error.contains("request_defaults"), "error was: {error}");
    assert!(error.contains("local mode"), "error was: {error}");
}

#[test]
fn validate_config_rejects_local_nemo_guardrails_multiple_config_sources() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "mode": "local",
                "config_path": "./rails",
                "config_yaml": "models: []"
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };

    let error = validate_config(&config).unwrap_err().to_string();
    assert!(
        error.contains("exactly one of config_path or config_yaml"),
        "error was: {error}"
    );
}

#[test]
fn validate_config_rejects_local_nemo_guardrails_colang_without_yaml() {
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: json!({
                "mode": "local",
                "config_path": "./rails",
                "colang_content": "define flow noop\n  pass"
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        ..PluginConfig::default()
    };

    let error = validate_config(&config).unwrap_err().to_string();
    assert!(
        error.contains("colang_content can only be used with config_yaml"),
        "error was: {error}"
    );
}

#[test]
fn nemo_guardrails_config_map_prunes_default_version() {
    let map = nemo_guardrails_config_map(&NeMoGuardrailsConfig {
        codec: Some("openai_chat".into()),
        remote: Some(RemoteBackendConfig {
            endpoint: Some("http://localhost:8000".into()),
            config_id: Some("default".into()),
            ..RemoteBackendConfig::default()
        }),
        ..NeMoGuardrailsConfig::default()
    })
    .unwrap();

    assert!(!map.contains_key("version"));
    assert_eq!(map.get("codec"), Some(&json!("openai_chat")));
    assert_eq!(map["remote"]["config_id"], json!("default"));
}

#[test]
fn write_plugin_config_round_trips_local_nemo_guardrails_component() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: local_guardrails_component_config("./rails"),
        }],
        ..PluginConfig::default()
    };

    write_plugin_config(&path, &config).unwrap();

    let rendered = std::fs::read_to_string(&path).unwrap();
    assert!(rendered.contains("mode = \"local\""));
    assert!(rendered.contains("config_path = \"./rails\""));
    assert!(rendered.contains("tool_input = true"));
    assert!(rendered.contains("python_module = \"custom_guardrails\""));

    let round_tripped = read_plugin_config(&path).unwrap();
    let guardrails = round_tripped
        .components
        .iter()
        .find(|component| component.kind == NEMO_GUARDRAILS_PLUGIN_KIND)
        .unwrap();
    assert!(guardrails.enabled);
    assert_eq!(guardrails.config["mode"], json!("local"));
    assert_eq!(guardrails.config["config_path"], json!("./rails"));
    assert_eq!(guardrails.config["tool_input"], json!(true));
    assert_eq!(
        guardrails.config["local"]["python_module"],
        json!("custom_guardrails")
    );
}

#[test]
fn nemo_guardrails_config_map_serializes_local_mode_fields() {
    let map = nemo_guardrails_config_map(&NeMoGuardrailsConfig {
        mode: "local".into(),
        config_path: Some("./rails".into()),
        tool_input: true,
        tool_output: true,
        local: Some(LocalBackendConfig {
            python_module: Some("custom_guardrails".into()),
            python_executable: Some("/opt/python/bin/python3".into()),
            python_path: None,
        }),
        ..NeMoGuardrailsConfig::default()
    })
    .unwrap();

    assert!(!map.contains_key("version"));
    assert_eq!(map.get("mode"), Some(&json!("local")));
    assert_eq!(map.get("config_path"), Some(&json!("./rails")));
    assert_eq!(map.get("tool_input"), Some(&json!(true)));
    assert_eq!(map["local"]["python_module"], json!("custom_guardrails"));
    assert_eq!(
        map["local"]["python_executable"],
        json!("/opt/python/bin/python3")
    );
}

#[test]
fn write_plugin_config_round_trips_local_llm_nemo_guardrails_component() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    let config = PluginConfig {
        components: vec![PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: true,
            config: local_llm_guardrails_component_config("models: []"),
        }],
        ..PluginConfig::default()
    };

    write_plugin_config(&path, &config).unwrap();

    let rendered = std::fs::read_to_string(&path).unwrap();
    assert!(rendered.contains("mode = \"local\""));
    assert!(rendered.contains("codec = \"openai_chat\""));
    assert!(rendered.contains("input = true"));
    assert!(rendered.contains("output = true"));
    assert!(rendered.contains("config_yaml = \"models: []\""));

    let round_tripped = read_plugin_config(&path).unwrap();
    let guardrails = round_tripped
        .components
        .iter()
        .find(|component| component.kind == NEMO_GUARDRAILS_PLUGIN_KIND)
        .unwrap();
    assert_eq!(guardrails.config["mode"], json!("local"));
    assert_eq!(guardrails.config["codec"], json!("openai_chat"));
    assert_eq!(guardrails.config["input"], json!(true));
    assert_eq!(guardrails.config["output"], json!(true));
    assert_eq!(guardrails.config["config_yaml"], json!("models: []"));
    assert_eq!(
        guardrails.config["colang_content"],
        json!("define flow noop\n  pass")
    );
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
fn parse_float_value_rejects_non_finite_numbers() {
    let field = EditorFieldSpec {
        name: "stable_threshold",
        label: "Stable threshold",
        kind: EditorFieldKind::Float,
        enum_values: &[],
        optional: false,
        nested_schema: None,
        nested_default: None,
    };

    assert_eq!(parse_float_value(&field, "0.75").unwrap(), json!(0.75));

    for value in ["inf", "-inf", "NaN"] {
        let error = parse_float_value(&field, value).unwrap_err().to_string();
        assert!(
            error.contains("stable_threshold must be a finite number"),
            "error was: {error}"
        );
        assert!(error.contains(value), "error was: {error}");
    }
}

#[test]
fn target_path_resolves_project_and_global_without_user_env() {
    let _cwd = crate::test_support::CwdTestScope::locked();
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

#[test]
fn target_path_resolves_user_scope_from_xdg_and_reports_missing_home() {
    let guard = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let previous_home = std::env::var_os("HOME");
    let previous_xdg = std::env::var_os("XDG_CONFIG_HOME");
    let previous_userprofile = std::env::var_os("USERPROFILE");
    let temp = tempfile::tempdir().unwrap();

    // SAFETY: This test holds the process-wide environment mutex while overriding env vars.
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", temp.path());
        std::env::remove_var("HOME");
        std::env::remove_var("USERPROFILE");
    }
    assert_eq!(
        target_path(TargetScope::User).unwrap(),
        user_plugin_config_path().unwrap()
    );

    // SAFETY: The mutex is still held while clearing env vars for the error branch.
    unsafe {
        std::env::remove_var("XDG_CONFIG_HOME");
        std::env::remove_var("HOME");
        std::env::remove_var("USERPROFILE");
    }
    let error = target_path(TargetScope::User).unwrap_err().to_string();
    assert!(
        error.contains("cannot determine user config directory"),
        "error was: {error}"
    );

    // SAFETY: Restore the original process environment before releasing the mutex.
    unsafe {
        match previous_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        match previous_xdg {
            Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        match previous_userprofile {
            Some(value) => std::env::set_var("USERPROFILE", value),
            None => std::env::remove_var("USERPROFILE"),
        }
    }
    drop(guard);
}
