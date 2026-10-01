// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
