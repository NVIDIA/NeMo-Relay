// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn selector_diagnostics_report_effectiveness_without_values() {
    use nemo_relay::diagnostics::{
        ComponentRegistration, DiagnosticQuery, Diagnostics, DiagnosticsSettings,
        ExtensionRegistration,
    };
    use serde_json::json;

    let diagnostics = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let recorder = diagnostics
        .register_component(ComponentRegistration::new("pii_redaction.selector", &[]))
        .unwrap();
    let code = recorder
        .register_extension(ExtensionRegistration {
            namespace: "pii_redaction",
            name: "selector_evaluated",
            description: "Counts of selected nodes, selected string leaves, and changed values",
            allowed_classes: &[],
            outcome: DiagnosticOutcome::Success,
        })
        .unwrap()
        .unwrap();
    let payload = json!({"messages": [{"content": [{"text": "alice@example.com"}]}]});
    for (paths, globs, expected, detail) in [
        (
            vec!["/messages/*/content/*/text"],
            vec![],
            "alice@example.com",
            "matched_paths=0 eligible_string_leaves=0 affected_values=0",
        ),
        (
            vec!["/messages/0/content"],
            vec![],
            "alice@example.com",
            "matched_paths=1 eligible_string_leaves=0 affected_values=0",
        ),
        (
            vec![],
            vec!["/messages/*/content/*/text"],
            "[REDACTED]",
            "matched_paths=1 eligible_string_leaves=1 affected_values=1",
        ),
        (
            vec![],
            vec![],
            "[REDACTED]",
            "matched_paths=6 eligible_string_leaves=1 affected_values=1",
        ),
    ] {
        let backend = CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "redact".into(),
                detector: Some("email".into()),
                target_paths: paths.into_iter().map(str::to_owned).collect(),
                target_path_globs: globs.into_iter().map(str::to_owned).collect(),
                ..BuiltinBackendConfig::default()
            },
            None,
        )
        .unwrap()
        .with_match_diagnostics(recorder.clone(), code);
        let result = backend.sanitize_json_preorder_dfs(payload.clone());
        assert_eq!(result["messages"][0]["content"][0]["text"], expected);
        let snapshot = diagnostics
            .snapshot(&DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap();
        assert_eq!(
            snapshot
                .events
                .last()
                .unwrap()
                .detail
                .as_ref()
                .unwrap()
                .message,
            detail
        );
        assert!(
            !serde_json::to_string(&snapshot)
                .unwrap()
                .contains("alice@example.com")
        );
    }

    let backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "remove".into(),
            target_paths: vec!["/messages/0/content/0/text".into()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap()
    .with_match_diagnostics(recorder, code);
    let result = backend.sanitize_json_preorder_dfs(payload);
    assert!(result["messages"][0]["content"][0].get("text").is_none());
    let snapshot = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(
        snapshot
            .events
            .last()
            .unwrap()
            .detail
            .as_ref()
            .unwrap()
            .message,
        "matched_paths=1 eligible_string_leaves=1 affected_values=1"
    );
}

#[test]
fn builtin_compilation_rejects_invalid_selectors_and_incompatible_preset_options() {
    use serde_json::json;

    for (config, expected) in [
        (
            json!({"target_paths": ["not-a-pointer"]}),
            "target_paths[0]",
        ),
        (
            json!({"target_path_globs": ["/invalid~escape"]}),
            "target_path_globs[0]",
        ),
        (json!({"preset": "unknown"}), "unsupported builtin preset"),
        (
            json!({"preset": "trajectory_context", "pattern": "secret"}),
            "cannot be combined",
        ),
        (
            json!({"preset": "trajectory_context", "custom_mark_payload_policy": "unknown"}),
            "unsupported custom-mark payload policy",
        ),
        (
            json!({"metric_string_attribute_allowlist": {"region": ["safe"]}}),
            "requires builtin.preset",
        ),
        (
            json!({"custom_mark_payload_policy": "drop"}),
            "requires builtin.preset",
        ),
        (
            json!({"detector": "unknown"}),
            "unsupported builtin.detector",
        ),
        (json!({"action": "unknown"}), "unsupported builtin.action"),
        (
            json!({"action": "redact"}),
            "builtin.pattern or builtin.detector is required",
        ),
        (
            json!({"action": "regex_replace", "pattern": "["}),
            "invalid builtin matcher regex",
        ),
        (
            json!({"pattern": "secret", "detector": "email"}),
            "cannot both be set",
        ),
    ] {
        let config: BuiltinBackendConfig = serde_json::from_value(config).unwrap();
        let error = CompiledBuiltinBackend::new(config, None)
            .err()
            .expect("invalid configuration must not compile");
        assert!(matches!(error, PluginError::InvalidConfig(_)));
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error}"
        );
    }
    let error = CompiledBuiltinBackend::new(
        BuiltinBackendConfig::default(),
        Some("unknown-codec".into()),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("unsupported codec"));
}

#[test]
fn target_path_matcher_deduplicates_equivalent_selectors() {
    let matcher = TargetPathMatcher::new(
        &["/prompt".to_string(), "/prompt".to_string()],
        &[
            "/prompt".to_string(),
            "/messages/*/content".to_string(),
            "/messages/*/content".to_string(),
        ],
    )
    .unwrap();

    assert_eq!(matcher.selectors.len(), 2);
    assert!(matcher.matches(&["prompt".to_string()]));
    assert!(matcher.matches(&[
        "messages".to_string(),
        "0".to_string(),
        "content".to_string(),
    ]));
}

#[test]
fn detectors_ignore_embedded_api_key_prefixes_and_redact_hex_aws_secrets() {
    let api_key_backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "redact".to_string(),
            detector: Some("api_key".to_string()),
            target_paths: vec!["/value".to_string()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let api_key_result = api_key_backend.sanitize_json_preorder_dfs(serde_json::json!({
        "value": "task-management risk-assessment network-topology sk-abcdef123456"
    }));
    assert_eq!(
        api_key_result["value"],
        "task-management risk-assessment network-topology [REDACTED]"
    );

    let aws_backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "redact".to_string(),
            detector: Some("aws_secret_access_key".to_string()),
            target_paths: vec!["/value".to_string()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let aws_result = aws_backend.sanitize_json_preorder_dfs(serde_json::json!({
            "value": "sha 0123456789abcdef0123456789abcdef01234567 key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        }));
    assert_eq!(aws_result["value"], "sha [REDACTED] key [REDACTED]");
}

#[test]
fn regex_replace_expands_numbered_and_named_captures() {
    let numbered_backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("(token)-(\\d+)".to_string()),
            replacement: Some("$1-[REDACTED]".to_string()),
            target_paths: vec!["/value".to_string()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        numbered_backend.sanitize_json_preorder_dfs(serde_json::json!({
            "value": "token-123"
        }))["value"],
        "token-[REDACTED]"
    );

    let named_backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "regex_replace".to_string(),
            pattern: Some("(?<kind>token)-(?<value>\\d+)".to_string()),
            replacement: Some("${kind}-[REDACTED]".to_string()),
            target_paths: vec!["/value".to_string()],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        named_backend.sanitize_json_preorder_dfs(serde_json::json!({
            "value": "token-123"
        }))["value"],
        "token-[REDACTED]"
    );
}
