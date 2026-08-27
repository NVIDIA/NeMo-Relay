// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_rust_native_plugin_example::validate_example_config;
use serde_json::{Map, Value as Json, json};

fn object(value: Json) -> Map<String, Json> {
    value.as_object().expect("test config is an object").clone()
}

#[test]
fn shared_configuration_is_valid() {
    let diagnostics = validate_example_config(&object(json!({
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
            "allowed_registration_name": "documentation-observed-subscriber",
            "reason": "disabled by documentation plugin"
        },
        "executor": { "worker_threads": 2 }
    })));

    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn enabled_registration_control_requires_distinct_targets() {
    let diagnostics = validate_example_config(&object(json!({
        "registration_control": {
            "enabled": true,
            "registration_name": "same-target",
            "allowed_registration_name": "same-target"
        }
    })));

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "examples.rust_native_policy.invalid_registration_control"
            && diagnostic.field.as_deref()
                == Some("registration_control.allowed_registration_name")
    }));

    let disabled_diagnostics = validate_example_config(&object(json!({
        "registration_control": {
            "enabled": false,
            "registration_name": "same-target",
            "allowed_registration_name": "same-target"
        }
    })));
    assert!(disabled_diagnostics.is_empty(), "{disabled_diagnostics:?}");
}

#[test]
fn unsupported_mode_is_rejected() {
    let diagnostics = validate_example_config(&object(json!({
        "requests": { "mode": "maybe" }
    })));

    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "examples.rust_native_policy.unsupported_mode")
    );
}

#[test]
fn unknown_field_produces_diagnostic() {
    let diagnostics = validate_example_config(&object(json!({
        "requests": { "mystery": true }
    })));

    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "examples.rust_native_policy.unknown_field")
    );
}

#[test]
fn wrong_types_are_rejected() {
    let diagnostics = validate_example_config(&object(json!({
        "requests": { "priority": "high" }
    })));
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "examples.rust_native_policy.invalid_config")
    );
}

#[test]
fn registration_control_parse_errors_are_rejected() {
    for config in [
        json!({ "registration_control": { "enabled": "yes" } }),
        json!({ "registration_control": { "kinds": ["unsupported"] } }),
    ] {
        let diagnostics = validate_example_config(&object(config));
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "examples.rust_native_policy.invalid_config"
                && diagnostic.field.is_none()
        }));
    }
}

#[test]
fn invalid_values_are_rejected_at_their_fields() {
    for (config, code, field) in [
        (
            json!({ "executor": { "worker_threads": 0 } }),
            "examples.rust_native_policy.invalid_executor",
            "executor.worker_threads",
        ),
        (
            json!({ "tag": "" }),
            "examples.rust_native_policy.empty_tag",
            "tag",
        ),
        (
            json!({ "requests": { "header_name": "" } }),
            "examples.rust_native_policy.invalid_header",
            "requests.header_name",
        ),
        (
            json!({ "registration_control": { "kinds": [] } }),
            "examples.rust_native_policy.invalid_registration_control",
            "registration_control.kinds",
        ),
        (
            json!({ "registration_control": { "registration_name": "" } }),
            "examples.rust_native_policy.invalid_registration_control",
            "registration_control.registration_name",
        ),
        (
            json!({ "registration_control": { "allowed_registration_name": "" } }),
            "examples.rust_native_policy.invalid_registration_control",
            "registration_control.allowed_registration_name",
        ),
        (
            json!({ "registration_control": { "reason": "" } }),
            "examples.rust_native_policy.invalid_registration_control",
            "registration_control.reason",
        ),
    ] {
        let diagnostics = validate_example_config(&object(config));
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == code && diagnostic.field.as_deref() == Some(field)
        }));
    }
}

#[test]
fn schema_declares_every_feature_group_and_executor() {
    let schema: Json = serde_json::from_str(include_str!("../config.schema.json"))
        .expect("example schema should be valid JSON");
    let properties = schema["properties"]
        .as_object()
        .expect("schema properties should be an object");

    for field in [
        "tag",
        "observe",
        "requests",
        "execution",
        "runtime",
        "registration_control",
        "executor",
    ] {
        assert!(properties.contains_key(field), "schema is missing {field}");
    }
    assert_eq!(schema["additionalProperties"], Json::Bool(false));
}
