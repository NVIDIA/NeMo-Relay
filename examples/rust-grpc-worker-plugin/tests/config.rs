// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_rust_grpc_worker_plugin_example::validate_example_config;
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
        "runtime": { "emit_marks": true, "emit_isolated_scope": true }
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
fn schema_contains_every_feature_group() {
    let schema: Json = serde_json::from_str(include_str!("../config.schema.json"))
        .expect("schema should be valid JSON");
    let fields = schema["properties"].as_object().expect("properties object");
    assert_eq!(schema["additionalProperties"], Json::Bool(false));
    assert_eq!(fields.len(), 5);
    for field in ["tag", "observe", "requests", "execution", "runtime"] {
        assert!(fields.contains_key(field));
    }
}

#[test]
fn manifest_uses_the_rust_worker_load_contract() {
    let manifest = include_str!("../relay-plugin.toml");
    assert!(manifest.contains("worker_protocol = \"grpc-v1\""));
    assert!(manifest.contains("runtime = \"rust\""));
    assert!(manifest.contains("entrypoint = \"target/debug/<platform-worker-file>\""));
    assert!(!manifest.contains("command ="));
}
