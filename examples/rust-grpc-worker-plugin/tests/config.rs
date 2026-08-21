// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_rust_grpc_worker_plugin_example::{
    DocumentationWorker, default_example_config, validate_example_config,
};
use nemo_relay_worker::{PluginContext, WorkerPlugin};
use serde_json::{Value as Json, json};

#[test]
fn shared_configuration_is_valid() {
    let diagnostics = validate_example_config(&json!({
        "tag": "documentation",
        "observe": { "enabled": true, "redact_keys": ["secret"] },
        "requests": {
            "enabled": true,
            "mode": "enforce",
            "blocked_tools": ["dangerous_tool"],
            "blocked_models": ["restricted-model"],
            "header_name": "x-nemo-relay-plugin",
            "header_value": "documentation",
            "priority": 20,
            "break_chain": false
        },
        "execution": { "enabled": true, "priority": 30, "emit_pending_marks": true },
        "runtime": { "emit_marks": true, "emit_isolated_scope": true },
        "registration_control": {
            "enabled": false,
            "kinds": ["subscriber"],
            "registration_name": "documentation-controlled-subscriber",
            "reason": "disabled by documentation plugin"
        }
    }));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn unsupported_mode_is_rejected() {
    let diagnostics = validate_example_config(&json!({
        "requests": { "mode": "maybe" }
    }));
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "examples.rust_grpc_worker.unsupported_mode")
    );
}

#[test]
fn unknown_field_produces_diagnostic() {
    let diagnostics = validate_example_config(&json!({
        "requests": { "mystery": true }
    }));

    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "examples.rust_grpc_worker.unknown_field")
    );
}

#[test]
fn wrong_types_are_rejected() {
    let diagnostics = validate_example_config(&json!({
        "requests": { "priority": "high" }
    }));
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "examples.rust_grpc_worker.invalid_config")
    );
}

#[test]
fn registration_control_parse_errors_are_rejected() {
    for config in [
        json!({ "registration_control": { "enabled": "yes" } }),
        json!({ "registration_control": { "kinds": ["unsupported"] } }),
    ] {
        let diagnostics = validate_example_config(&config);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "examples.rust_grpc_worker.invalid_config"
                && diagnostic.field.is_none()
        }));
    }
}

#[test]
fn empty_headers_are_reported_at_their_individual_fields() {
    for (config, field) in [
        (
            json!({ "requests": { "header_name": "" } }),
            "requests.header_name",
        ),
        (
            json!({ "requests": { "header_value": "" } }),
            "requests.header_value",
        ),
    ] {
        let diagnostics = validate_example_config(&config);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "examples.rust_grpc_worker.invalid_header"
                && diagnostic.field.as_deref() == Some(field)
        }));
    }
}

#[test]
fn invalid_registration_control_is_reported_at_its_field() {
    for (config, field) in [
        (
            json!({ "registration_control": { "kinds": [] } }),
            "registration_control.kinds",
        ),
        (
            json!({ "registration_control": { "registration_name": "" } }),
            "registration_control.registration_name",
        ),
        (
            json!({ "registration_control": { "reason": "" } }),
            "registration_control.reason",
        ),
    ] {
        let diagnostics = validate_example_config(&config);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "examples.rust_grpc_worker.invalid_registration_control"
                && diagnostic.field.as_deref() == Some(field)
        }));
    }
}

#[test]
fn register_rejects_invalid_registration_control() {
    let mut context = PluginContext::new();
    let error = DocumentationWorker
        .register(
            &mut context,
            &json!({ "registration_control": { "enabled": true, "reason": "" } }),
        )
        .expect_err("register should validate the raw configuration");

    assert!(
        error
            .to_string()
            .contains("registration_control.reason must not be empty")
    );
}

#[test]
fn schema_contains_every_feature_group() {
    let schema: Json = serde_json::from_str(include_str!("../config.schema.json"))
        .expect("schema should be valid JSON");
    let fields = schema["properties"].as_object().expect("properties object");
    assert_eq!(schema["additionalProperties"], Json::Bool(true));
    assert_eq!(fields.len(), 6);
    for field in [
        "tag",
        "observe",
        "requests",
        "execution",
        "runtime",
        "registration_control",
    ] {
        assert!(fields.contains_key(field));
    }
}

#[test]
fn schema_defaults_match_the_runtime_defaults() {
    let schema: Json = serde_json::from_str(include_str!("../config.schema.json"))
        .expect("schema should be valid JSON");
    assert_schema_defaults(&schema, &default_example_config(), "");
}

fn assert_schema_defaults(schema: &Json, value: &Json, path: &str) {
    if let Some(expected) = schema.get("default") {
        assert_eq!(value, expected, "default mismatch at {path}");
    }
    let Some(properties) = schema.get("properties").and_then(Json::as_object) else {
        return;
    };
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("runtime default at {path} must be an object"));
    for (name, child_schema) in properties {
        let child_value = object
            .get(name)
            .unwrap_or_else(|| panic!("runtime default is missing {path}{name}"));
        assert_schema_defaults(child_schema, child_value, &format!("{path}{name}."));
    }
}

#[test]
fn manifest_uses_the_rust_worker_load_contract() {
    let manifest = include_str!("../relay-plugin.toml");
    assert!(manifest.contains("relay = \">=0.8.0,<1.0\""));
    assert!(manifest.contains("worker_protocol = \"grpc-v1\""));
    assert!(manifest.contains("runtime = \"rust\""));
    assert!(manifest.contains("entrypoint = \"target/debug/<platform-worker-file>\""));
    assert!(!manifest.contains("command ="));
}
