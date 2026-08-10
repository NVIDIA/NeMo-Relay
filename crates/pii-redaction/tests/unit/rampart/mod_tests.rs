// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the Rampart plugin contract and configuration.

use super::*;

fn valid_config() -> Map<String, Json> {
    let Json::Object(config) = serde_json::to_value(RampartPiiConfig {
        model_path: "/tmp/rampart".into(),
        target_path_patterns: vec!["/messages/*/content".into()],
        ..RampartPiiConfig::default()
    })
    .unwrap() else {
        unreachable!()
    };
    config
}

#[test]
fn validates_explicit_model_and_content_paths() {
    assert!(validate_rampart_pii_config(&valid_config()).is_empty());

    let mut config = valid_config();
    config.insert("model_path".into(), Json::String("relative/model".into()));
    config.insert(
        "target_path_patterns".into(),
        serde_json::json!(["/messages/pre*fix/content"]),
    );
    let diagnostics = validate_rampart_pii_config(&config);
    assert!(
        diagnostics
            .iter()
            .any(|item| item.field.as_deref() == Some("model_path"))
    );
    assert!(
        diagnostics
            .iter()
            .any(|item| item.field.as_deref() == Some("target_path_patterns"))
    );
}

#[test]
fn accepts_every_builtin_provider_codec() {
    for codec in supported_codec_names() {
        let mut config = valid_config();
        config.insert("codec".into(), Json::String(codec.into()));
        assert!(
            validate_rampart_pii_config(&config).is_empty(),
            "built-in codec {codec} should be accepted"
        );
    }
}

#[test]
fn bounds_the_configurable_window_budget() {
    assert_eq!(RampartPiiConfig::default().max_windows_per_payload, 4);

    let mut config = valid_config();
    config.insert(
        "max_windows_per_payload".into(),
        Json::from(MAX_WINDOWS_PER_PAYLOAD),
    );
    assert!(validate_rampart_pii_config(&config).is_empty());

    config.insert(
        "max_windows_per_payload".into(),
        Json::from(MAX_WINDOWS_PER_PAYLOAD + 1),
    );
    assert!(
        validate_rampart_pii_config(&config)
            .iter()
            .any(|item| item.field.as_deref() == Some("max_windows_per_payload"))
    );

    config.insert("max_windows_per_payload".into(), Json::from(0_usize));
    assert!(
        validate_rampart_pii_config(&config)
            .iter()
            .any(|item| item.field.as_deref() == Some("max_windows_per_payload"))
    );
}

#[test]
fn validates_trajectory_preset_without_explicit_paths() {
    let Json::Object(config) = serde_json::to_value(RampartPiiConfig {
        model_path: "/tmp/rampart".into(),
        preset: Some("trajectory_context".into()),
        ..RampartPiiConfig::default()
    })
    .unwrap() else {
        unreachable!()
    };

    assert!(validate_rampart_pii_config(&config).is_empty());
}

#[test]
fn rejects_ambiguous_or_unsupported_preset_configuration() {
    let mut config = valid_config();
    config.insert("preset".into(), Json::String("trajectory_context".into()));
    assert!(
        validate_rampart_pii_config(&config)
            .iter()
            .any(|item| item.field.as_deref() == Some("preset"))
    );

    config.remove("target_path_patterns");
    config.insert("preset".into(), Json::String("unknown".into()));
    assert!(
        validate_rampart_pii_config(&config)
            .iter()
            .any(|item| item.field.as_deref() == Some("preset"))
    );

    config.remove("preset");
    config.insert(
        "custom_mark_payload_policy".into(),
        Json::String("redact_all_leaves".into()),
    );
    let diagnostics = validate_rampart_pii_config(&config);
    assert!(
        diagnostics
            .iter()
            .any(|item| item.field.as_deref() == Some("custom_mark_payload_policy"))
    );
}

#[test]
fn activation_invariants_remain_errors_when_policy_warns() {
    let cases = [
        (
            "target_paths",
            serde_json::json!(["messages/0/content"]),
            "target_paths entries",
        ),
        (
            "target_paths",
            serde_json::json!([""]),
            "target_paths entries",
        ),
        (
            "target_path_patterns",
            serde_json::json!([""]),
            "target_path_patterns entries",
        ),
        ("min_score", serde_json::json!(1.1), "min_score must"),
    ];

    for (field, value, expected) in cases {
        let mut config = valid_config();
        config.insert(
            "policy".into(),
            serde_json::json!({"unsupported_value": "warn"}),
        );
        config.insert(field.into(), value);
        let diagnostics = validate_rampart_pii_config(&config);
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic.level == DiagnosticLevel::Warning
                    && diagnostic.field.as_deref() == Some(field)
            }),
            "expected a warning for {field}: {diagnostics:?}"
        );

        let parsed = parse_config(&config).expect("diagnostic input should deserialize");
        let error = enforce_activation_invariants(&parsed)
            .expect_err("unsafe configuration must fail activation");
        assert!(
            error.to_string().contains(expected),
            "unexpected registration error for {field}: {error}"
        );
    }
}
