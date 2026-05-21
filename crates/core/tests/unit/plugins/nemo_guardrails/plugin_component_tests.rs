// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the planned NeMo Guardrails plugin component contract.

use super::*;
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::config_editor::{EditorConfig, EditorFieldKind};
#[cfg(feature = "schema")]
use crate::plugin::plugin_config_schema;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
    list_plugin_kinds, lookup_plugin, validate_plugin_config,
};
use serde_json::json;

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    let _ = deregister_nemo_guardrails_component();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn ensure_registered() {
    register_nemo_guardrails_component().unwrap();
}

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn disabled_component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: false,
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

fn remote_valid_config() -> Json {
    json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "safety-default"
        }
    })
}

#[test]
fn editor_schema_tracks_nemo_guardrails_config_types() {
    let schema = NeMoGuardrailsConfig::editor_schema();
    let mode = schema.field("mode").expect("mode field");
    assert_eq!(mode.kind, EditorFieldKind::Enum);
    assert_eq!(mode.enum_values, &["remote", "local"]);

    let remote = schema.field("remote").expect("remote section");
    assert_eq!(remote.kind, EditorFieldKind::Section);
    assert!(remote.optional);

    let remote_schema = remote.schema().expect("remote editor schema");
    let headers = remote_schema.field("headers").expect("headers field");
    assert_eq!(headers.kind, EditorFieldKind::StringMap);

    let request_defaults = schema
        .field("request_defaults")
        .expect("request_defaults section");
    assert_eq!(request_defaults.kind, EditorFieldKind::Section);
    assert!(request_defaults.optional);

    let request_defaults_schema = request_defaults
        .schema()
        .expect("request_defaults editor schema");
    let rails = request_defaults_schema.field("rails").expect("rails field");
    assert_eq!(rails.kind, EditorFieldKind::Section);

    let rails_schema = rails.schema().expect("request rails editor schema");
    let retrieval = rails_schema.field("retrieval").expect("retrieval field");
    assert_eq!(retrieval.kind, EditorFieldKind::Json);
}

#[test]
fn default_config_and_component_conversion_cover_public_shape() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let defaults = NeMoGuardrailsConfig::default();
    assert_eq!(defaults.version, 1);
    assert_eq!(defaults.mode, "remote");
    assert!(defaults.input);
    assert!(defaults.output);
    assert!(!defaults.tool_input);
    assert!(!defaults.tool_output);
    assert_eq!(defaults.priority, 100);
    assert!(defaults.remote.is_none());
    assert!(defaults.local.is_none());
    assert!(defaults.request_defaults.is_none());

    let remote = RemoteBackendConfig::default();
    assert_eq!(remote.timeout_millis, 3_000);
    assert!(remote.headers.is_empty());
    assert!(remote.config_ids.is_empty());

    let generic: PluginComponentSpec = ComponentSpec::new(NeMoGuardrailsConfig {
        remote: Some(RemoteBackendConfig {
            endpoint: Some("http://localhost:8000".into()),
            config_id: Some("default".into()),
            ..RemoteBackendConfig::default()
        }),
        ..NeMoGuardrailsConfig::default()
    })
    .into();
    assert_eq!(generic.kind, NEMO_GUARDRAILS_PLUGIN_KIND);
    assert!(generic.enabled);
    assert_eq!(generic.config["mode"], json!("remote"));
    assert_eq!(generic.config["remote"]["config_id"], json!("default"));
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

#[cfg(feature = "schema")]
#[test]
fn schema_contains_every_supported_nemo_guardrails_option() {
    let schema = nemo_guardrails_config_schema();
    for field in [
        "version",
        "mode",
        "config_path",
        "config_yaml",
        "colang_content",
        "codec",
        "input",
        "output",
        "tool_input",
        "tool_output",
        "priority",
        "remote",
        "local",
        "request_defaults",
        "policy",
        "endpoint",
        "config_id",
        "config_ids",
        "headers",
        "timeout_millis",
        "python_module",
        "context",
        "rails",
        "llm_params",
        "llm_output",
        "output_vars",
        "log",
        "retrieval",
        "dialog",
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
        &["remote", "local"]
    ));
    assert!(schema_property_has_enum(
        &schema,
        "codec",
        &["openai_chat", "openai_responses", "anthropic_messages"]
    ));
    assert!(schema_property_has_default(
        &schema,
        "mode",
        json!("remote")
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
fn registration_is_explicit_not_automatic() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    assert!(!list_plugin_kinds().contains(&NEMO_GUARDRAILS_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_none());

    ensure_registered();
    assert!(list_plugin_kinds().contains(&NEMO_GUARDRAILS_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some());

    ensure_registered();
    assert!(lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some());
    assert!(deregister_nemo_guardrails_component());
    assert!(!deregister_nemo_guardrails_component());
}

#[test]
fn disabled_component_validates_and_initializes_without_runtime_work() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let config = PluginConfig {
        version: 1,
        components: vec![disabled_component(remote_valid_config())],
        policy: Default::default(),
    };
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins(config)).unwrap();
}

#[test]
fn duplicate_component_is_rejected_as_singleton() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let config = PluginConfig {
        version: 1,
        components: vec![
            component(remote_valid_config()),
            component(remote_valid_config()),
        ],
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
fn invalid_shapes_and_values_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let invalid_shape = validate_plugin_config(&plugin_config(json!({
        "version": "one",
    })));
    assert!(invalid_shape.has_errors());
    assert!(
        invalid_shape
            .diagnostics
            .iter()
            .any(|diag| diag.code == "nemo_guardrails.invalid_plugin_config")
    );

    let local_missing_source = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "codec": "openai_chat",
    })));
    assert!(local_missing_source.has_errors());
    assert!(local_missing_source.diagnostics.iter().any(|diag| {
        diag.message
            .contains("exactly one of config_path or config_yaml is required in local mode")
    }));

    let local_bad_colang = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_path": "./rails",
        "colang_content": "define flow x",
        "codec": "openai_chat",
    })));
    assert!(local_bad_colang.has_errors());
    assert!(
        local_bad_colang
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("colang_content can only be used"))
    );

    let remote_missing_identity = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000"},
    })));
    assert!(remote_missing_identity.has_errors());
    assert!(remote_missing_identity.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote mode requires remote.config_id or remote.config_ids")
    }));

    let remote_conflicting_ids = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "one",
            "config_ids": ["two"]
        },
    })));
    assert!(remote_conflicting_ids.has_errors());
    assert!(remote_conflicting_ids.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote.config_id and remote.config_ids cannot be used together")
    }));

    let missing_codec = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(missing_codec.has_errors());
    assert!(
        missing_codec
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("codec"))
    );

    let bad_codec = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_agents",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(bad_codec.has_errors());
    assert!(bad_codec.diagnostics.iter().any(|diag| {
        diag.message
            .contains("codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'")
    }));

    let remote_empty_fields = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "",
            "config_id": "",
            "config_ids": ["default", ""]
        }
    })));
    assert!(remote_empty_fields.has_errors());
    assert!(
        remote_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.endpoint"))
    );
    assert!(
        remote_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.config_id"))
    );
    assert!(
        remote_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.config_ids[1]"))
    );

    let remote_local_mix = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "config_path": "./rails",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        },
        "local": {"python_module": "nemoguardrails"}
    })));
    assert!(remote_local_mix.has_errors());
    assert!(
        remote_local_mix
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("local"))
    );
    assert!(remote_local_mix.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote mode uses remote config identity")
    }));

    let no_surfaces = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_path": "./rails",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": false
    })));
    assert!(no_surfaces.has_errors());
    assert!(
        no_surfaces
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("at least one Guardrails surface"))
    );

    let local_empty_fields = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_yaml": "",
        "colang_content": "",
        "codec": "openai_chat",
        "local": {"python_module": ""}
    })));
    assert!(local_empty_fields.has_errors());
    assert!(
        local_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("config_yaml"))
    );
    assert!(
        local_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("colang_content"))
    );
    assert!(
        local_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("local.python_module"))
    );

    let invalid_request_defaults = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        },
        "request_defaults": {
            "context": true,
            "llm_params": [],
            "log": "verbose",
            "output_vars": 7,
            "rails": {
                "retrieval": [""]
            }
        }
    })));
    assert!(invalid_request_defaults.has_errors());
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.context"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.llm_params"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.log"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.output_vars"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.rails.retrieval[0]"))
    );
}

#[test]
fn unknown_fields_follow_policy() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let warn_report = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"},
        "bogus": true
    })));
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "nemo_guardrails.unknown_field")
    );

    let nested_warn_report = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"},
        "request_defaults": {
            "rails": {
                "bogus": true
            }
        }
    })));
    assert!(
        nested_warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.component.as_deref() == Some("request_defaults.rails"))
    );

    let ignored = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "ignore", "unsupported_value": "ignore"},
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"},
        "bogus": true
    })));
    assert!(!ignored.has_errors());
    assert!(ignored.diagnostics.is_empty());
}

#[test]
fn enabled_initialization_fails_fast_until_backend_exists() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let error =
        futures::executor::block_on(initialize_plugins(plugin_config(remote_valid_config())))
            .unwrap_err();

    match error {
        crate::plugin::PluginError::RegistrationFailed(message) => {
            assert!(message.contains("not implemented yet"));
        }
        other => panic!("unexpected error: {other}"),
    }
}
