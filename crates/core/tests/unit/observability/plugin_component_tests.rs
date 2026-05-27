// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the built-in observability plugin component.

use super::*;
use crate::api::event::{BaseEvent, EventCategory, ScopeEvent};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::scope::{PopScopeParams, PushScopeParams};
use crate::config_editor::{EditorConfig, EditorFieldKind};
#[cfg(feature = "schema")]
use crate::plugin::plugin_config_schema;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
    list_plugin_kinds, lookup_plugin, validate_plugin_config,
};
use serde_json::json;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn plugin_config(config: Json) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![component(config)],
        policy: Default::default(),
    }
}

#[test]
fn editor_schema_tracks_observability_config_types() {
    let schema = ObservabilityConfig::editor_schema();
    let atof = schema.field("atof").expect("atof section");
    assert_eq!(atof.label, "ATOF");
    assert_eq!(atof.kind, EditorFieldKind::Section);
    assert!(atof.optional);

    let atof_schema = atof.schema().expect("atof editor schema");
    let mode = atof_schema.field("mode").expect("atof mode field");
    assert_eq!(mode.kind, EditorFieldKind::Enum);
    assert_eq!(mode.enum_values, &["append", "overwrite"]);

    let otlp = schema
        .field("openinference")
        .expect("openinference section")
        .schema()
        .expect("openinference editor schema");
    let headers = otlp.field("headers").expect("headers field");
    assert_eq!(headers.kind, EditorFieldKind::StringMap);
}

fn push_agent(name: &str) -> crate::api::scope::ScopeHandle {
    crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Agent)
            .input(json!({"agent": name}))
            .build(),
    )
    .unwrap()
}

fn push_function(name: &str) -> crate::api::scope::ScopeHandle {
    crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Function)
            .input(json!({"function": name}))
            .build(),
    )
    .unwrap()
}

fn pop(handle: &crate::api::scope::ScopeHandle) {
    crate::api::scope::pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"done": handle.name}))
            .build(),
    )
    .unwrap();
}

#[cfg(feature = "schema")]
fn schema_has_property(schema: &Json, name: &str) -> bool {
    schema_property(schema, name).is_some()
}

#[cfg(feature = "schema")]
fn schema_property_has_enum(schema: &Json, name: &str, expected: &[&str]) -> bool {
    schema_property(schema, name)
        .and_then(|property| property.get("enum"))
        .and_then(Json::as_array)
        .is_some_and(|values| {
            expected
                .iter()
                .all(|expected| values.iter().any(|value| value == *expected))
        })
}

#[cfg(feature = "schema")]
fn schema_property_has_default(schema: &Json, name: &str, expected: Json) -> bool {
    schema_property(schema, name)
        .and_then(|property| property.get("default"))
        .is_some_and(|default| default == &expected)
}

#[cfg(feature = "schema")]
fn schema_property<'a>(schema: &'a Json, name: &str) -> Option<&'a Json> {
    match schema {
        Json::Object(object) => {
            if let Some(property) = object
                .get("properties")
                .and_then(Json::as_object)
                .and_then(|properties| properties.get(name))
            {
                return Some(property);
            }
            object
                .values()
                .find_map(|value| schema_property(value, name))
        }
        Json::Array(values) => values.iter().find_map(|value| schema_property(value, name)),
        _ => None,
    }
}

#[test]
fn default_config_and_component_conversion_cover_public_shape() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let defaults = ObservabilityConfig::default();
    assert_eq!(defaults.version, 1);
    assert!(defaults.atof.is_none());
    assert!(defaults.atif.is_none());
    assert!(defaults.opentelemetry.is_none());
    assert!(defaults.openinference.is_none());

    let atof = AtofSectionConfig::default();
    assert!(!atof.enabled);
    assert_eq!(atof.mode, "append");
    assert!(atof.output_directory.is_none());
    assert!(atof.filename.is_none());

    let atif = AtifSectionConfig::default();
    assert!(!atif.enabled);
    assert_eq!(atif.agent_name, "NeMo Relay");
    assert_eq!(atif.agent_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(atif.model_name, "unknown");
    assert_eq!(atif.filename_template, "nemo-relay-atif-{session_id}.json");

    let otlp = OtlpSectionConfig::default();
    assert!(!otlp.enabled);
    assert_eq!(otlp.transport, "http_binary");
    assert_eq!(otlp.service_name, "nemo-relay");
    assert_eq!(otlp.timeout_millis, 3_000);

    let generic: PluginComponentSpec = ComponentSpec::new(ObservabilityConfig {
        atof: Some(atof),
        atif: Some(atif),
        opentelemetry: Some(otlp.clone()),
        openinference: Some(otlp),
        ..ObservabilityConfig::default()
    })
    .into();
    assert_eq!(generic.kind, OBSERVABILITY_PLUGIN_KIND);
    assert!(generic.enabled);
    assert_eq!(generic.config["version"], json!(1));
    assert_eq!(generic.config["atif"]["agent_name"], json!("NeMo Relay"));
}

#[cfg(feature = "schema")]
#[test]
fn schema_contains_every_supported_observability_option() {
    let schema = observability_config_schema();
    for field in [
        "version",
        "atof",
        "atif",
        "opentelemetry",
        "openinference",
        "policy",
        "enabled",
        "output_directory",
        "filename",
        "mode",
        "agent_name",
        "agent_version",
        "model_name",
        "tool_definitions",
        "extra",
        "filename_template",
        "transport",
        "endpoint",
        "headers",
        "resource_attributes",
        "service_name",
        "service_namespace",
        "service_version",
        "instrumentation_scope",
        "timeout_millis",
        "unknown_component",
        "unknown_field",
        "unsupported_value",
    ] {
        assert!(
            schema_has_property(&schema, field),
            "schema missing property `{field}`:\n{}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }
    assert!(schema_property_has_enum(
        &schema,
        "mode",
        &["append", "overwrite"]
    ));
    assert!(schema_property_has_enum(
        &schema,
        "transport",
        &["http_binary", "grpc"]
    ));
    assert!(schema_property_has_default(
        &schema,
        "mode",
        json!("append")
    ));
    assert!(schema_property_has_default(
        &schema,
        "transport",
        json!("http_binary")
    ));
}

#[cfg(feature = "schema")]
#[test]
fn plugin_schema_contains_generic_plugin_surface() {
    let schema = plugin_config_schema();
    for field in [
        "version",
        "components",
        "policy",
        "kind",
        "enabled",
        "config",
    ] {
        assert!(
            schema_has_property(&schema, field),
            "plugin schema missing property `{field}`"
        );
    }
}

#[test]
fn built_in_registration_is_automatic() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    assert!(list_plugin_kinds().contains(&OBSERVABILITY_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(OBSERVABILITY_PLUGIN_KIND).is_some());

    let config = plugin_config(json!({}));
    assert!(!validate_plugin_config(&config).has_errors());
}

#[test]
fn explicit_registration_helpers_are_idempotent_and_reversible() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    assert!(register_observability_component().is_ok());
    assert!(register_observability_component().is_ok());
    assert!(deregister_observability_component());
    assert!(!deregister_observability_component());
    register_observability_component().unwrap();
}

#[test]
fn empty_and_disabled_config_register_nothing() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let config = plugin_config(json!({
        "atof": {"enabled": false, "mode": "overwrite"},
        "atif": {"enabled": false},
        "opentelemetry": {"enabled": false, "transport": "grpc"},
        "openinference": {"enabled": false, "transport": "grpc"}
    }));
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    let state = global_context();
    assert!(state.read().unwrap().event_subscribers.is_empty());
}

#[test]
fn disabled_file_sections_do_not_create_files() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-disabled-files");

    let config = plugin_config(json!({
        "atof": {
            "enabled": false,
            "output_directory": dir,
            "filename": "events.jsonl"
        },
        "atif": {
            "enabled": false,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    let agent = push_agent("disabled-agent");
    pop(&agent);
    clear_plugin_configuration().unwrap();

    assert!(!dir.join("events.jsonl").exists());
    assert!(!dir.join(format!("trajectory-{}.json", agent.uuid)).exists());
}

#[test]
fn duplicate_component_is_rejected_as_singleton() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let config = PluginConfig {
        version: 1,
        components: vec![component(json!({})), component(json!({}))],
        policy: Default::default(),
    };
    let report = validate_plugin_config(&config);
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
}

#[test]
fn unknown_fields_and_bad_values_follow_policy() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let warn_report = validate_plugin_config(&plugin_config(json!({
        "atof": {"bogus": true, "mode": "invalid"},
        "atif": {"filename_template": "missing-session"}
    })));
    assert!(warn_report.has_errors());
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.unknown_field")
    );
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("mode"))
    );
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("filename_template"))
    );

    let ignore_report = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "ignore", "unsupported_value": "ignore"},
        "atof": {"bogus": true, "mode": "invalid"},
        "atif": {"filename_template": "missing-session"}
    })));
    assert!(!ignore_report.has_errors());
    assert!(ignore_report.diagnostics.is_empty());
}

#[test]
fn invalid_shapes_and_strict_policy_are_reported() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let invalid_shape = validate_plugin_config(&plugin_config(json!({
        "version": "one",
    })));
    assert!(invalid_shape.has_errors());
    assert!(
        invalid_shape
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.invalid_plugin_config")
    );

    let unsupported_version = validate_plugin_config(&plugin_config(json!({
        "version": 2,
    })));
    assert!(unsupported_version.has_errors());
    assert!(unsupported_version.diagnostics.iter().any(|diag| diag.code
        == "observability.unsupported_config_version"
        && diag.field.as_deref() == Some("version")));

    let strict_unknown = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "error"},
        "opentelemetry": {"unexpected": true}
    })));
    assert!(strict_unknown.has_errors());
    assert!(
        strict_unknown
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.unknown_field"
                && diag.component.as_deref() == Some("opentelemetry")
                && diag.field.as_deref() == Some("unexpected"))
    );

    let strict_bad_transport = validate_plugin_config(&plugin_config(json!({
        "openinference": {"enabled": true, "transport": "udp"}
    })));
    assert!(strict_bad_transport.has_errors());
    assert!(
        strict_bad_transport
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("transport"))
    );
}

#[test]
fn initialization_fails_for_invalid_enabled_file_exporters() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-invalid-exporters");
    let not_a_directory = dir.join("not-a-directory");
    fs::write(&not_a_directory, "file").unwrap();

    let invalid_atof = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "atof": {
            "enabled": true,
            "mode": "invalid",
            "output_directory": dir,
            "filename": "events.jsonl"
        }
    }));
    let error = futures::executor::block_on(initialize_plugins(invalid_atof)).unwrap_err();
    assert!(error.to_string().contains("ATOF mode"));

    let invalid_atif_template = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "single-file.json"
        }
    }));
    let error = futures::executor::block_on(initialize_plugins(invalid_atif_template)).unwrap_err();
    assert!(error.to_string().contains("filename_template"));

    let invalid_path = plugin_config(json!({
        "atof": {
            "enabled": true,
            "output_directory": not_a_directory,
            "filename": "events.jsonl"
        }
    }));
    let error = futures::executor::block_on(initialize_plugins(invalid_path)).unwrap_err();
    assert!(error.to_string().contains("registration failed"));

    let invalid_otel_transport = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "opentelemetry": {
            "enabled": true,
            "transport": "udp"
        }
    }));
    let error =
        futures::executor::block_on(initialize_plugins(invalid_otel_transport)).unwrap_err();
    assert!(error.to_string().contains("OpenTelemetry transport"));

    let invalid_openinference_transport = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "openinference": {
            "enabled": true,
            "transport": "udp"
        }
    }));
    let error = futures::executor::block_on(initialize_plugins(invalid_openinference_transport))
        .unwrap_err();
    assert!(error.to_string().contains("OpenInference transport"));
}

#[test]
fn atof_enabled_writes_jsonl_and_teardown_flushes() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atof");

    let config = plugin_config(json!({
        "atof": {
            "enabled": true,
            "output_directory": dir,
            "filename": "events.jsonl",
            "mode": "overwrite"
        }
    }));
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    {
        let state = global_context();
        let names = state
            .read()
            .unwrap()
            .event_subscribers
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["__nemo_relay_plugin__observability__atof"]);
    }

    let agent = push_agent("atof-agent");
    crate::api::scope::event(
        crate::api::scope::EmitMarkEventParams::builder()
            .name("checkpoint")
            .parent(&agent)
            .data(json!({"step": 1}))
            .build(),
    )
    .unwrap();
    pop(&agent);
    clear_plugin_configuration().unwrap();

    let content = fs::read_to_string(dir.join("events.jsonl")).unwrap();
    let lines = content.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("\"kind\":\"scope\""));
    assert!(lines[1].contains("\"kind\":\"mark\""));
    assert!(lines[2].contains("\"scope_category\":\"end\""));
}

#[test]
fn atif_defaults_create_one_file_per_top_level_agent() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-defaults");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir
        }
    }));
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    let first = push_agent("first-agent");
    let nested = push_agent("nested-agent");
    pop(&nested);
    pop(&first);

    let second = push_agent("second-agent");
    pop(&second);
    clear_plugin_configuration().unwrap();

    let first_path = dir.join(format!("nemo-relay-atif-{}.json", first.uuid));
    let second_path = dir.join(format!("nemo-relay-atif-{}.json", second.uuid));
    assert!(first_path.exists());
    assert!(second_path.exists());

    let first_json: Json = serde_json::from_str(&fs::read_to_string(first_path).unwrap()).unwrap();
    let second_json: Json =
        serde_json::from_str(&fs::read_to_string(second_path).unwrap()).unwrap();

    assert_eq!(first_json["session_id"], first.uuid.to_string());
    assert_eq!(first_json["agent"]["name"], "NeMo Relay");
    assert_eq!(first_json["agent"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(first_json["agent"]["model_name"], "unknown");
    let first_serialized = first_json.to_string();
    assert!(first_serialized.contains("first-agent"));
    assert!(first_serialized.contains("nested-agent"));
    assert!(!first_serialized.contains("second-agent"));

    let second_serialized = second_json.to_string();
    assert!(second_serialized.contains("second-agent"));
    assert!(!second_serialized.contains("first-agent"));
}

#[test]
fn atif_completed_top_level_agent_is_evicted_after_write() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-evict");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("evicted-agent");
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir.clone()),
        ..AtifSectionConfig::default()
    })));

    let start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent.uuid)
            .parent_uuid(root_uuid)
            .name("evicted-agent")
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    manager
        .lock()
        .unwrap()
        .observe_global(&start_event, "__test__", Arc::clone(&manager));
    {
        let dispatcher = manager.lock().unwrap();
        assert!(dispatcher.agents.contains_key(&agent.uuid));
        assert!(dispatcher.scope_subscribers.contains_key(&agent.uuid));
    }

    let end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent.uuid)
            .parent_uuid(root_uuid)
            .name("evicted-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    let pending_write = manager
        .lock()
        .unwrap()
        .observe_scope(&end_event, agent.uuid)
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{}.json", agent.uuid));
    assert!(!path.exists());
    write_atif_file(&pending_write).unwrap();
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(agent.uuid, Ok(()));
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let dispatcher = manager.lock().unwrap();
    assert!(dispatcher.last_error.is_none());
    assert!(!dispatcher.agents.contains_key(&agent.uuid));
    assert!(!dispatcher.scope_subscribers.contains_key(&agent.uuid));
    assert!(path.exists());
    drop(dispatcher);
    pop(&agent);
}

#[test]
fn atif_dispatcher_records_failed_agent_writes() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-write-error");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("failed-write-agent");
    let dispatcher = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir),
        ..AtifSectionConfig::default()
    })));

    let start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent.uuid)
            .parent_uuid(root_uuid)
            .name("failed-write-agent")
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    dispatcher
        .lock()
        .unwrap()
        .observe_global(&start_event, "__test__", Arc::clone(&dispatcher));

    let mut dispatcher = dispatcher.lock().unwrap();
    let error = dispatcher
        .finish_agent_write(agent.uuid, Err(std::io::Error::other("disk full")))
        .unwrap_err();
    assert_eq!(error.to_string(), "disk full");
    assert_eq!(dispatcher.last_error.as_deref(), Some("disk full"));
    assert!(dispatcher.last_error_result().is_err());
    drop(dispatcher);
    pop(&agent);
}

#[test]
fn atif_dispatcher_default_output_path_uses_current_directory() {
    let dispatcher = AtifDispatcher::new(AtifSectionConfig::default());
    assert_eq!(
        dispatcher.output_path("session-1"),
        std::env::current_dir()
            .unwrap()
            .join("nemo-relay-atif-session-1.json")
    );
}

#[test]
fn atif_explicit_options_and_open_agent_teardown_are_written() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-explicit");

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "agent_name": "custom-agent",
            "agent_version": "9.9.9",
            "model_name": "demo-model",
            "tool_definitions": [{"name": "search"}],
            "extra": {"team": "runtime"},
            "output_directory": dir,
            "filename_template": "custom-{session_id}.atif.json"
        }
    }));
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    let ignored = push_function("not-an-agent");
    pop(&ignored);
    let agent = push_agent("open-agent");
    clear_plugin_configuration().unwrap();

    let path = dir.join(format!("custom-{}.atif.json", agent.uuid));
    assert!(path.exists());
    let value: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(value["agent"]["name"], "custom-agent");
    assert_eq!(value["agent"]["version"], "9.9.9");
    assert_eq!(value["agent"]["model_name"], "demo-model");
    assert_eq!(value["agent"]["tool_definitions"][0]["name"], "search");
    assert_eq!(value["agent"]["extra"]["team"], "runtime");
    assert!(fs::read_dir(dir).unwrap().count() == 1);
    pop(&agent);
}

#[test]
fn atif_rejects_unsafe_template_and_ignores_non_top_level_agents() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-errors");

    let invalid_template = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "single-file.json"
        }
    }));
    assert!(validate_plugin_config(&invalid_template).has_errors());
    assert!(futures::executor::block_on(initialize_plugins(invalid_template)).is_err());

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    let function = push_function("top-level-function");
    let nested_agent = push_agent("nested-under-function");
    pop(&nested_agent);
    pop(&function);
    clear_plugin_configuration().unwrap();

    assert_eq!(fs::read_dir(dir).unwrap().count(), 0);
}

#[test]
fn otlp_sections_register_inferred_subscribers_with_full_config() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let config = plugin_config(json!({
        "opentelemetry": {
            "enabled": true,
            "transport": "http_binary",
            "endpoint": "http://127.0.0.1:4318/v1/traces",
            "headers": {"authorization": "token"},
            "resource_attributes": {"deployment.environment": "test"},
            "service_name": "otel-service",
            "service_namespace": "agents",
            "service_version": "1.2.3",
            "instrumentation_scope": "test-otel",
            "timeout_millis": 1
        },
        "openinference": {
            "enabled": true,
            "transport": "http_binary",
            "endpoint": "http://127.0.0.1:4318/v1/traces",
            "headers": {"authorization": "token"},
            "resource_attributes": {"deployment.environment": "test"},
            "service_name": "oi-service",
            "service_namespace": "agents",
            "service_version": "1.2.3",
            "instrumentation_scope": "test-openinference",
            "timeout_millis": 1
        }
    }));
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins(config)).unwrap();

    let state = global_context();
    let names = state
        .read()
        .unwrap()
        .event_subscribers
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert!(names.contains(&"__nemo_relay_plugin__observability__opentelemetry".to_string()));
    assert!(names.contains(&"__nemo_relay_plugin__observability__openinference".to_string()));
    clear_plugin_configuration().unwrap();
}
