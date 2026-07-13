// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the built-in observability plugin component.

use super::*;
use crate::api::event::{BaseEvent, EventCategory, MarkEvent, ScopeEvent};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::scope::{PopScopeParams, PushScopeParams};
use crate::api::subscriber::scope_deregister_subscriber;
use crate::config_editor::{EditorConfig, EditorFieldKind};
#[cfg(feature = "schema")]
use crate::plugin::plugin_config_schema;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins_exact,
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
    let endpoints = atof_schema
        .field("endpoints")
        .expect("atof endpoints field");
    assert_eq!(endpoints.kind, EditorFieldKind::Json);
    assert!(endpoints.optional);

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
    schema_property_matches(schema, name, &|property| {
        property
            .get("enum")
            .and_then(Json::as_array)
            .is_some_and(|values| {
                expected
                    .iter()
                    .all(|expected| values.iter().any(|value| value == *expected))
            })
    })
}

#[cfg(feature = "schema")]
fn schema_property_has_default(schema: &Json, name: &str, expected: Json) -> bool {
    schema_property_matches(schema, name, &|property| {
        property
            .get("default")
            .is_some_and(|default| default == &expected)
    })
}

#[cfg(feature = "schema")]
fn schema_property_matches(schema: &Json, name: &str, predicate: &impl Fn(&Json) -> bool) -> bool {
    match schema {
        Json::Object(object) => {
            if object
                .get("properties")
                .and_then(Json::as_object)
                .and_then(|properties| properties.get(name))
                .is_some_and(predicate)
            {
                return true;
            }
            object
                .values()
                .any(|value| schema_property_matches(value, name, predicate))
        }
        Json::Array(values) => values
            .iter()
            .any(|value| schema_property_matches(value, name, predicate)),
        _ => false,
    }
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

    let parsed_atof: AtofSectionConfig = serde_json::from_value(json!({
        "endpoints": [{"url": "http://localhost/events"}]
    }))
    .unwrap();
    assert_eq!(parsed_atof.endpoints[0].transport, "http_post");
    assert_eq!(parsed_atof.endpoints[0].field_name_policy, "preserve");

    let atif = AtifSectionConfig::default();
    assert!(!atif.enabled);
    assert_eq!(atif.agent_name, "NeMo Relay");
    assert_eq!(atif.agent_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(atif.model_name, "unknown");
    assert_eq!(atif.filename_template, "nemo-relay-atif-{session_id}.json");

    let otlp = OtlpSectionConfig::default();
    assert!(!otlp.enabled);
    assert_eq!(otlp.mark_projection, MarkProjection::Inherit);
    assert_eq!(otlp.mark_exclude_names, vec!["llm.chunk"]);
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

#[test]
fn mark_projection_parses_for_otlp_and_rejects_unknown_values() {
    let otlp: OtlpSectionConfig = serde_json::from_value(json!({
        "mark_projection": "tool"
    }))
    .unwrap();
    assert_eq!(otlp.mark_projection, MarkProjection::Tool);

    let event: OtlpSectionConfig = serde_json::from_value(json!({
        "mark_projection": "event",
        "mark_exclude_names": []
    }))
    .unwrap();
    assert_eq!(event.mark_projection, MarkProjection::Event);
    assert!(event.mark_exclude_names.is_empty());

    let custom_exclusions: OtlpSectionConfig = serde_json::from_value(json!({
        "mark_projection": "tool",
        "mark_exclude_names": ["notification", "hook_mark"]
    }))
    .unwrap();
    assert_eq!(
        custom_exclusions.mark_exclude_names,
        vec!["notification", "hook_mark"]
    );

    let error = serde_json::from_value::<OtlpSectionConfig>(json!({
        "mark_projection": "span"
    }))
    .unwrap_err();
    assert!(error.to_string().contains("unknown variant `span`"));

    let report = validate_plugin_config(&plugin_config(json!({
        "opentelemetry": {"mark_projection": "span"}
    })));
    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| diagnostic.code
        == "observability.invalid_plugin_config"
        && diagnostic.message.contains("unknown variant `span`")));

    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {"mark_projection": "tool"}
    })));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "observability.unknown_field"
            && diagnostic.field.as_deref() == Some("mark_projection")
    }));
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
        "endpoints",
        "agent_name",
        "agent_version",
        "model_name",
        "mark_projection",
        "mark_exclude_names",
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
    assert!(schema_property_has_enum(
        &schema,
        "mark_projection",
        &["inherit", "event", "tool"]
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
fn atof_endpoint_validation_rejects_bad_values() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "atof": {
            "enabled": true,
            "endpoints": [
                {"url": "", "transport": "http_post"},
                {"url": "http://localhost/events", "transport": "bogus"},
                {"url": "http://localhost/events", "transport": "ndjson", "timeout_millis": 0},
                {"url": "not a url", "transport": "http_post"},
                {"url": "http://localhost/events", "transport": "http_post", "field_name_policy": "bogus"}
            ]
        }
    })));

    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("endpoints[0].url") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("endpoints[1].transport") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("endpoints[2].timeout_millis") })
    );
    #[cfg(feature = "atof-streaming")]
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("endpoints[3].url") })
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| { diag.field.as_deref() == Some("endpoints[4].field_name_policy") })
    );
}

#[test]
fn build_atof_endpoint_config_maps_headers_timeout_and_rejects_transport() {
    let mut headers = std::collections::HashMap::new();
    headers.insert("authorization".to_string(), "token".to_string());
    let config = build_atof_endpoint_config(
        2,
        AtofEndpointSectionConfig {
            url: "ws://127.0.0.1:47632/events".into(),
            transport: "websocket".into(),
            headers: headers.clone(),
            header_env: std::collections::HashMap::from([(
                "x-api-key".into(),
                "SWITCHYARD_API_KEY".into(),
            )]),
            timeout_millis: 123,
            field_name_policy: "replace_dots".into(),
        },
    )
    .unwrap();

    assert_eq!(config.url, "ws://127.0.0.1:47632/events");
    assert_eq!(
        config.transport,
        crate::observability::atof::AtofEndpointTransport::Websocket
    );
    assert_eq!(config.headers, headers);
    assert_eq!(
        config.header_env.get("x-api-key").map(String::as_str),
        Some("SWITCHYARD_API_KEY")
    );
    assert_eq!(config.timeout_millis, 123);
    assert_eq!(
        config.field_name_policy,
        crate::observability::atof::AtofEndpointFieldNamePolicy::ReplaceDots
    );

    let error = build_atof_endpoint_config(
        3,
        AtofEndpointSectionConfig {
            url: "http://127.0.0.1:47632/events".into(),
            transport: "smtp".into(),
            headers: std::collections::HashMap::new(),
            header_env: std::collections::HashMap::new(),
            timeout_millis: 3_000,
            field_name_policy: "preserve".into(),
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("endpoints[3].transport"));

    let error = build_atof_endpoint_config(
        4,
        AtofEndpointSectionConfig {
            url: "http://127.0.0.1:47632/events".into(),
            transport: "http_post".into(),
            headers: std::collections::HashMap::new(),
            header_env: std::collections::HashMap::new(),
            timeout_millis: 3_000,
            field_name_policy: "bogus".into(),
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("endpoints[4].field_name_policy"));
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
    let error = futures::executor::block_on(initialize_plugins_exact(invalid_atof)).unwrap_err();
    assert!(error.to_string().contains("ATOF mode"));

    let invalid_atif_template = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "single-file.json"
        }
    }));
    let error =
        futures::executor::block_on(initialize_plugins_exact(invalid_atif_template)).unwrap_err();
    assert!(error.to_string().contains("filename_template"));

    let invalid_path = plugin_config(json!({
        "atof": {
            "enabled": true,
            "output_directory": not_a_directory,
            "filename": "events.jsonl"
        }
    }));
    let error = futures::executor::block_on(initialize_plugins_exact(invalid_path)).unwrap_err();
    assert!(error.to_string().contains("registration failed"));

    let invalid_otel_transport = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "opentelemetry": {
            "enabled": true,
            "transport": "udp"
        }
    }));
    let error =
        futures::executor::block_on(initialize_plugins_exact(invalid_otel_transport)).unwrap_err();
    assert!(error.to_string().contains("OpenTelemetry transport"));

    let invalid_openinference_transport = plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "openinference": {
            "enabled": true,
            "transport": "udp"
        }
    }));
    let error =
        futures::executor::block_on(initialize_plugins_exact(invalid_openinference_transport))
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    assert_eq!(first_json["schema_version"], "ATIF-v1.7");
    assert_eq!(first_json["trajectory_id"], first.uuid.to_string());
    assert_eq!(
        first_json["subagent_trajectories"][0]["trajectory_id"],
        nested.uuid.to_string()
    );
    assert_eq!(
        first_json["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"],
        nested.uuid.to_string()
    );
    let first_serialized = first_json.to_string();
    assert!(first_serialized.contains("first-agent"));
    assert!(first_serialized.contains("nested-agent"));
    assert!(!first_serialized.contains("second-agent"));

    let second_serialized = second_json.to_string();
    assert!(second_serialized.contains("second-agent"));
    assert!(!second_serialized.contains("first-agent"));
}

#[test]
fn atif_routes_global_descendant_events_by_parent_uuid() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-global-descendant");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let agent = push_agent("root-agent");
    let agent_uuid = agent.uuid;
    let child_uuid = Uuid::now_v7();
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir.clone()),
        ..AtifSectionConfig::default()
    })));
    let empty_storage: Arc<Vec<Arc<AtifRemoteStorage>>> = Arc::new(Vec::new());

    let start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent_uuid)
            .parent_uuid(root_uuid)
            .name("root-agent")
            .metadata(json!({"session_id": "root-session"}))
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &start_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(agent_uuid)
            .name("child-agent")
            .metadata(json!({"session_id": "child-session"}))
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_start_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(agent_uuid)
            .name("child-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_end_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(agent_uuid)
            .parent_uuid(root_uuid)
            .name("root-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    let (pending_write, targets) = manager
        .lock()
        .unwrap()
        .observe_global(
            &end_event,
            "__test__",
            Arc::clone(&manager),
            Arc::clone(&empty_storage),
        )
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{agent_uuid}.json"));
    let results = write_atif(&pending_write, empty_storage.as_slice(), &targets);
    for (label, result) in &results {
        assert!(result.is_ok(), "{label:?}: {result:?}");
    }
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(agent_uuid, results);
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let value: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(value["trajectory_id"], agent_uuid.to_string());
    assert_eq!(
        value["subagent_trajectories"][0]["session_id"],
        "child-session"
    );
    assert_eq!(
        value["subagent_trajectories"][0]["trajectory_id"],
        child_uuid.to_string()
    );
    assert_eq!(
        value["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"],
        child_uuid.to_string()
    );
    pop(&agent);
}

#[test]
fn atif_writes_openclaw_child_only_fallback_without_mark_steps() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    let dir = temp_dir("observability-atif-openclaw-child-fallback");
    let root_uuid = crate::api::runtime::current_scope_stack()
        .read()
        .unwrap()
        .root_uuid();
    let child_uuid = Uuid::now_v7();
    let child_mark_uuid = Uuid::now_v7();
    let manager = Arc::new(Mutex::new(AtifDispatcher::new(AtifSectionConfig {
        enabled: true,
        output_directory: Some(dir.clone()),
        ..AtifSectionConfig::default()
    })));
    let empty_storage: Arc<Vec<Arc<AtifRemoteStorage>>> = Arc::new(Vec::new());

    let child_start_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(root_uuid)
            .name("worker-agent")
            .metadata(json!({
                "session_id": "child-session",
                "nemo_relay_scope_role": "subagent"
            }))
            .build(),
        ScopeCategory::Start,
        vec![],
        EventCategory::agent(),
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_start_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_mark_event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(child_mark_uuid)
            .parent_uuid(child_uuid)
            .name("worker-started")
            .data(json!({"status": "started"}))
            .build(),
        None,
        None,
    ));
    assert!(
        manager
            .lock()
            .unwrap()
            .observe_global(
                &child_mark_event,
                "__test__",
                Arc::clone(&manager),
                Arc::clone(&empty_storage),
            )
            .is_none()
    );

    let child_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(child_uuid)
            .parent_uuid(root_uuid)
            .name("worker-agent")
            .build(),
        ScopeCategory::End,
        vec![],
        EventCategory::agent(),
        None,
    ));
    let (pending_write, targets) = manager
        .lock()
        .unwrap()
        .observe_global(
            &child_end_event,
            "__test__",
            Arc::clone(&manager),
            Arc::clone(&empty_storage),
        )
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{child_uuid}.json"));
    let results = write_atif(&pending_write, empty_storage.as_slice(), &targets);
    for (label, result) in &results {
        assert!(result.is_ok(), "{label:?}: {result:?}");
    }
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(child_uuid, results);
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let value: Json = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(value["trajectory_id"], child_uuid.to_string());
    assert!(value["steps"].as_array().unwrap().is_empty());
    assert!(
        value.get("subagent_trajectories").is_none() || value["subagent_trajectories"].is_null()
    );
    assert!(!value.to_string().contains("subagent_trajectory_ref"));
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
    let empty_storage: Arc<Vec<Arc<AtifRemoteStorage>>> = Arc::new(Vec::new());
    manager.lock().unwrap().observe_global(
        &start_event,
        "__test__",
        Arc::clone(&manager),
        Arc::clone(&empty_storage),
    );
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
    let (pending_write, targets) = manager
        .lock()
        .unwrap()
        .observe_scope(&end_event, agent.uuid)
        .unwrap();
    let path = dir.join(format!("nemo-relay-atif-{}.json", agent.uuid));
    assert!(!path.exists());
    let results = write_atif(&pending_write, empty_storage.as_slice(), &targets);
    for (label, result) in &results {
        assert!(result.is_ok(), "{label:?}: {result:?}");
    }
    let scope_subscriber = manager
        .lock()
        .unwrap()
        .complete_scope_write(agent.uuid, results);
    if let Some((scope_uuid, name)) = scope_subscriber {
        let _ = scope_deregister_subscriber(&scope_uuid, &name);
    }

    let dispatcher = manager.lock().unwrap();
    assert!(dispatcher.fatal_error.is_none());
    assert!(dispatcher.sink_errors.is_empty());
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
    dispatcher.lock().unwrap().observe_global(
        &start_event,
        "__test__",
        Arc::clone(&dispatcher),
        Arc::new(Vec::new()),
    );

    let mut dispatcher = dispatcher.lock().unwrap();
    let scope_subscriber = dispatcher.complete_scope_write(
        agent.uuid,
        vec![(SinkLabel::Local, Err(std::io::Error::other("disk full")))],
    );
    assert!(scope_subscriber.is_some());
    assert_eq!(
        dispatcher
            .sink_errors
            .get(&SinkLabel::Local)
            .map(String::as_str),
        Some("disk full")
    );
    assert!(dispatcher.last_error_result().is_ok());
    drop(dispatcher);
    pop(&agent);
}

#[test]
fn write_atif_reports_missing_local_path_and_unregistered_remote_sink() {
    let agent_uuid = Uuid::now_v7();
    let write = PendingAtifWrite {
        agent_uuid,
        session_id: agent_uuid.to_string(),
        filename: "trajectory.json".into(),
        local_path: None,
        payload: b"{}".to_vec(),
    };

    let results = write_atif(&write, &[], &[SinkLabel::Local, SinkLabel::Remote(0)]);

    assert_eq!(results.len(), 2);
    assert!(
        results[0]
            .1
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("no output path")
    );
    let remote_error = results[1].1.as_ref().unwrap_err().to_string();
    #[cfg(feature = "object-store")]
    assert!(remote_error.contains("storage[0]"));
    #[cfg(not(feature = "object-store"))]
    assert!(remote_error.contains("ATIF storage support is not enabled in this build"));
}

#[test]
fn atif_dispatcher_default_output_path_uses_current_directory() {
    let dispatcher = AtifDispatcher::new(AtifSectionConfig::default());
    let (filename, local_path) = dispatcher.prepare_destination("session-1");
    assert_eq!(filename, "nemo-relay-atif-session-1.json");
    assert_eq!(
        local_path.unwrap(),
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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    assert!(futures::executor::block_on(initialize_plugins_exact(invalid_template)).is_err());

    let config = plugin_config(json!({
        "atif": {
            "enabled": true,
            "output_directory": dir,
            "filename_template": "trajectory-{session_id}.json"
        }
    }));
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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
    futures::executor::block_on(initialize_plugins_exact(config)).unwrap();

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

#[test]
fn atif_storage_defaults_to_empty() {
    let config = AtifSectionConfig::default();
    assert!(config.storage.is_empty());
}

#[test]
fn atif_storage_section_parses_s3_variant() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [{
            "type": "s3",
            "bucket": "my-bucket",
            "key_prefix": "openshell/"
        }]
    }))
    .expect("valid storage section should parse");
    assert_eq!(parsed.storage.len(), 1);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(_) => panic!("expected s3 storage"),
        AtifStorageConfig::S3(s3) => {
            assert_eq!(s3.bucket, "my-bucket");
            assert_eq!(s3.key_prefix.as_deref(), Some("openshell/"));
        }
    }
}

#[test]
fn atif_storage_section_parses_http_variant() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [{
            "type": "http",
            "endpoint": "https://example.com/atif",
            "timeout_millis": 1500,
            "headers": {"x-static": "value"},
            "header_env": {"authorization": "NEMO_RELAY_ATIF_HTTP_TOKEN"}
        }]
    }))
    .expect("valid HTTP storage section should parse");
    assert_eq!(parsed.storage.len(), 1);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(http) => {
            assert_eq!(http.endpoint, "https://example.com/atif");
            assert_eq!(http.timeout_millis, 1500);
            assert_eq!(
                http.headers.get("x-static").map(String::as_str),
                Some("value")
            );
            assert_eq!(
                http.header_env.get("authorization").map(String::as_str),
                Some("NEMO_RELAY_ATIF_HTTP_TOKEN")
            );
        }
        AtifStorageConfig::S3(_) => panic!("expected HTTP storage"),
    }
}

#[test]
fn atif_storage_section_rejects_single_table() {
    let err = serde_json::from_value::<AtifSectionConfig>(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": {
            "type": "s3",
            "bucket": "my-bucket"
        }
    }))
    .expect_err("storage must be a list");
    assert!(
        err.to_string().contains("invalid type"),
        "unexpected error: {err}"
    );
}

#[test]
fn atif_storage_section_parses_array_of_tables() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [
            {"type": "s3", "bucket": "primary", "key_prefix": "p/"},
            {"type": "http", "endpoint": "http://127.0.0.1:3000/atif"}
        ]
    }))
    .expect("array-of-tables form should parse");
    assert_eq!(parsed.storage.len(), 2);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(_) => panic!("expected s3 storage"),
        AtifStorageConfig::S3(s3) => assert_eq!(s3.bucket, "primary"),
    }
    match &parsed.storage[1] {
        AtifStorageConfig::Http(http) => {
            assert_eq!(http.endpoint, "http://127.0.0.1:3000/atif");
        }
        AtifStorageConfig::S3(_) => panic!("expected HTTP storage"),
    }
}

#[test]
fn atif_storage_section_parses_empty_array() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": []
    }))
    .expect("empty array should parse");
    assert!(parsed.storage.is_empty());
}

#[test]
fn atif_storage_unknown_backend_type_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "carrier-pigeon", "bucket": "ignored"}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "observability.invalid_plugin_config")
    );
}

#[test]
fn disabled_atif_storage_config_does_not_report_feature_disabled() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": false,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "s3", "bucket": "configured-but-disabled"}]
        }
    })));

    assert!(
        !report.diagnostics.iter().any(|diag| {
            diag.code == "observability.feature_disabled"
                && diag.field.as_deref() == Some("storage")
        }),
        "disabled ATIF storage should not report feature-disabled diagnostics: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_empty_bucket_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "s3", "bucket": "  "}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].bucket"))
    );
}

#[test]
fn atif_storage_diagnostics_carry_sink_index() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [
                {"type": "s3", "bucket": "ok"},
                {"type": "s3", "bucket": ""}
            ]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[1].bucket")),
        "diagnostic should point at the second entry: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_empty_http_endpoint_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "http", "endpoint": "  "}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].endpoint")),
        "expected diagnostic for empty endpoint: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_malformed_http_endpoint_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{"type": "http", "endpoint": "ftp://example.com/atif"}]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].endpoint")),
        "expected diagnostic for malformed endpoint: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_timeout_must_be_positive() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "timeout_millis": 0
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].timeout_millis")),
        "expected diagnostic for non-positive timeout: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_invalid_literal_header_name_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "headers": {"bad header": "value"}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].headers.bad header")),
        "expected diagnostic for invalid header name: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_invalid_literal_header_value_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "headers": {"x-bad": "bad\nvalue"}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].headers.x-bad")),
        "expected diagnostic for invalid header value: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_missing_env_is_rejected() {
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_MISSING_ZZZZ";
    // SAFETY: tests in this binary do not concurrently observe this uniquely
    // named env var, so removing it is safe.
    unsafe {
        std::env::remove_var(var_name);
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": var_name}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].header_env.authorization")),
        "expected diagnostic for missing header env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_empty_env_is_rejected() {
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_EMPTY_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": var_name}
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].header_env.authorization")),
        "expected diagnostic for empty header env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_whitespace_name_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": " NEMO_RELAY_TEST_ATIF_HTTP_HEADER "}
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report.diagnostics.iter().any(|diag| diag.field.as_deref()
            == Some("storage[0].header_env.authorization")
            && diag.message.contains("surrounding whitespace")),
        "expected diagnostic for whitespace header env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_http_header_env_present_env_is_accepted() {
    let var_name = "NEMO_RELAY_TEST_ATIF_HTTP_HEADER_OK_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "Bearer test-token");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "http",
                "endpoint": "https://example.com/atif",
                "header_env": {"authorization": var_name}
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(
        !report.has_errors(),
        "validation should pass when header env var is set: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_editor_field_is_optional_json() {
    let schema = AtifSectionConfig::editor_schema();
    let storage = schema.field("storage").expect("storage editor field");
    assert_eq!(storage.kind, EditorFieldKind::Json);
    assert!(storage.optional);
}

#[test]
fn atif_storage_s3_parses_full_credential_block() {
    let parsed: AtifSectionConfig = serde_json::from_value(json!({
        "enabled": true,
        "filename_template": "trajectory-{session_id}.json",
        "storage": [{
            "type": "s3",
            "bucket": "my-bucket",
            "key_prefix": "openshell/",
            "access_key_id": "AKIAEXAMPLE",
            "secret_access_key_var": "MY_SECRET_VAR",
            "session_token_var": "MY_TOKEN_VAR",
            "region": "us-west-2",
            "endpoint_url": "https://s3.example.com",
            "allow_http": false
        }]
    }))
    .expect("full credential block should parse");
    assert_eq!(parsed.storage.len(), 1);
    match &parsed.storage[0] {
        AtifStorageConfig::Http(_) => panic!("expected s3 storage"),
        AtifStorageConfig::S3(s3) => {
            assert_eq!(s3.bucket, "my-bucket");
            assert_eq!(s3.key_prefix.as_deref(), Some("openshell/"));
            assert_eq!(s3.access_key_id.as_deref(), Some("AKIAEXAMPLE"));
            assert_eq!(s3.secret_access_key_var.as_deref(), Some("MY_SECRET_VAR"));
            assert_eq!(s3.session_token_var.as_deref(), Some("MY_TOKEN_VAR"));
            assert_eq!(s3.region.as_deref(), Some("us-west-2"));
            assert_eq!(s3.endpoint_url.as_deref(), Some("https://s3.example.com"));
            assert_eq!(s3.allow_http, Some(false));
        }
    }
}

#[test]
fn atif_storage_secret_var_missing_env_is_rejected() {
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_MISSING_ZZZZ";
    // SAFETY: tests in this binary do not concurrently observe this uniquely
    // named env var, so removing it is safe.
    unsafe {
        std::env::remove_var(var_name);
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": var_name
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].secret_access_key_var")),
        "expected diagnostic for missing env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_secret_var_empty_env_is_rejected() {
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_EMPTY_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": var_name
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].secret_access_key_var")),
        "expected diagnostic for empty env var: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_secret_var_present_env_is_accepted() {
    let var_name = "NEMO_RELAY_TEST_S3_SECRET_OK_ZZZZ";
    // SAFETY: this uniquely named env var is only touched by this test.
    unsafe {
        std::env::set_var(var_name, "secret-value");
    }
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": var_name
            }]
        }
    })));
    // SAFETY: cleanup of test-only env var.
    unsafe {
        std::env::remove_var(var_name);
    }
    assert!(
        !report.has_errors(),
        "validation should pass when env var is set: {:?}",
        report.diagnostics
    );
}

#[test]
fn atif_storage_secret_var_empty_name_is_rejected() {
    let report = validate_plugin_config(&plugin_config(json!({
        "atif": {
            "enabled": true,
            "filename_template": "trajectory-{session_id}.json",
            "storage": [{
                "type": "s3",
                "bucket": "my-bucket",
                "secret_access_key_var": "   "
            }]
        }
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("storage[0].secret_access_key_var")),
        "expected diagnostic for empty var name: {:?}",
        report.diagnostics
    );
}

#[test]
#[cfg(feature = "object-store")]
fn atif_storage_private_helpers_resolve_env_and_key_prefix_branches() {
    let missing = "NEMO_RELAY_TEST_ATIF_HELPER_MISSING_ZZZZ";
    let empty = "NEMO_RELAY_TEST_ATIF_HELPER_EMPTY_ZZZZ";
    let secret = "NEMO_RELAY_TEST_ATIF_HELPER_SECRET_ZZZZ";
    let token = "NEMO_RELAY_TEST_ATIF_HELPER_TOKEN_ZZZZ";
    // SAFETY: these uniquely named variables are only touched by this test.
    unsafe {
        std::env::remove_var(missing);
        std::env::set_var(empty, "");
        std::env::set_var(secret, "secret-value");
        std::env::set_var(token, "token-value");
    }

    assert_eq!(resolve_env_var_field("field", None).unwrap(), None);
    assert!(
        resolve_env_var_field("field", Some(" padded "))
            .unwrap_err()
            .to_string()
            .contains("must be the name of an environment variable")
    );
    assert!(
        resolve_env_var_field("field", Some(missing))
            .unwrap_err()
            .to_string()
            .contains("is not set")
    );
    assert!(
        resolve_env_var_field("field", Some(empty))
            .unwrap_err()
            .to_string()
            .contains("set but empty")
    );
    assert_eq!(
        resolve_env_var_field("field", Some(secret)).unwrap(),
        Some("secret-value".to_string())
    );

    assert_eq!(normalize_storage_key_prefix(None), "");
    assert_eq!(
        normalize_storage_key_prefix(Some("  nested/path  ")),
        "nested/path/"
    );
    assert_eq!(
        normalize_storage_key_prefix(Some("nested/path/")),
        "nested/path/"
    );

    let overrides = S3BuilderOverrides::resolve(
        3,
        &S3StorageConfig {
            bucket: "bucket".into(),
            key_prefix: Some("prefix".into()),
            access_key_id: Some("access".into()),
            secret_access_key_var: Some(secret.into()),
            session_token_var: Some(token.into()),
            region: Some("us-west-2".into()),
            endpoint_url: Some("http://127.0.0.1:9000".into()),
            allow_http: Some(true),
        },
    )
    .unwrap();
    assert_eq!(overrides.access_key_id.as_deref(), Some("access"));
    assert_eq!(overrides.secret_access_key.as_deref(), Some("secret-value"));
    assert_eq!(overrides.session_token.as_deref(), Some("token-value"));
    assert_eq!(overrides.region.as_deref(), Some("us-west-2"));
    assert_eq!(
        overrides.endpoint_url.as_deref(),
        Some("http://127.0.0.1:9000")
    );
    assert_eq!(overrides.allow_http, Some(true));
    let _builder = overrides.apply(object_store::aws::AmazonS3Builder::from_env());

    // SAFETY: cleanup of test-only env vars.
    unsafe {
        std::env::remove_var(empty);
        std::env::remove_var(secret);
        std::env::remove_var(token);
    }
}
