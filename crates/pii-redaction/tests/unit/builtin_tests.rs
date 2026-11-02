// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn selector_diagnostics_report_effectiveness_without_values() {
    use nemo_relay_diagnostics::{
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
            vec!["/messages/**/text", "/**/text", "/messages/**/**/text"],
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
    .with_match_diagnostics(recorder.clone(), code);
    let result = backend.sanitize_json_preorder_dfs(payload.clone());
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

    let result = backend.sanitize_json_preorder_dfs(
        json!({"messages": [{"content": [{"text": {"nested": "alice@example.com"}}]}]}),
    );
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
        "matched_paths=1 eligible_string_leaves=0 affected_values=1"
    );

    let backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "remove".into(),
            target_paths: vec![
                "/messages/0/content".into(),
                "/messages/0/content/0/text".into(),
            ],
            ..BuiltinBackendConfig::default()
        },
        None,
    )
    .unwrap()
    .with_match_diagnostics(recorder.clone(), code);
    let result = backend.sanitize_json_preorder_dfs(payload);
    assert!(result["messages"][0].get("content").is_none());
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
        "matched_paths=2 eligible_string_leaves=1 affected_values=1"
    );

    // A single recursive selector can hide its own descendant matches when a
    // selected container is removed. Counts describe the original document.
    let backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "remove".into(),
            target_path_globs: vec!["/messages/**".into()],
            ..Default::default()
        },
        None,
    )
    .unwrap()
    .with_match_diagnostics(recorder, code);
    for (payload, expected_counts) in [
        (
            json!({"messages": [{"content": [{"text": "alice@example.com"}]}]}),
            "matched_paths=5 eligible_string_leaves=1 affected_values=1",
        ),
        (
            json!({"messages": {}}),
            "matched_paths=1 eligible_string_leaves=0 affected_values=1",
        ),
    ] {
        assert_eq!(backend.sanitize_json_preorder_dfs(payload), json!({}));
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
            expected_counts
        );
        assert!(
            !serde_json::to_string(&snapshot)
                .unwrap()
                .contains("alice@example.com")
        );
    }
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
    assert!(!matcher.may_hide_descendant);
    assert!(matcher.matches(&["prompt".to_string()]));
    assert!(matcher.matches(&[
        "messages".to_string(),
        "0".to_string(),
        "content".to_string(),
    ]));
}

#[test]
fn overlapping_selectors_require_full_document_counts() {
    let overlapping = TargetPathMatcher::new(
        &["/messages/0/content".into()],
        &["/messages/*/content/*/text".into()],
    )
    .unwrap();
    assert!(overlapping.may_hide_descendant);

    let unrelated = TargetPathMatcher::new(
        &["/metadata/owner".into()],
        &["/messages/*/content/*/text".into()],
    )
    .unwrap();
    assert!(!unrelated.may_hide_descendant);
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

#[test]
fn recursive_globs_select_zero_or_more_whole_segments() {
    use serde_json::json;

    let payload = json!({
        "text": "root",
        "messages": {
            "text": "direct",
            "A": {"content": [{"text": "nested"}, {"deep": {"text": "deeper"}}]},
            "B": {"text": "sibling"}
        },
        "other": {"text": "outside"}
    });
    let matcher = TargetPathMatcher::new(&[], &["/messages/**/text".into()]).unwrap();
    let paths = matcher.matching_json_pointer_paths(&payload);
    assert_eq!(paths.len(), 4);
    for pointer in [
        "/messages/text",
        "/messages/A/content/0/text",
        "/messages/A/content/1/deep/text",
        "/messages/B/text",
    ] {
        assert!(
            paths.contains(&json_pointer_segments(pointer).unwrap()),
            "{pointer}"
        );
    }
    assert!(matcher.may_match_descendant(&[]));
    assert!(matcher.may_match_descendant(&["messages".into(), "text".into()]));
    assert!(!matcher.may_match_descendant(&["other".into()]));

    for (glob, path, expected) in [
        ("/**", "", true),
        ("/**", "/a/b", true),
        ("/messages/**", "/messages", true),
        ("/messages/*/**", "/messages", false),
        ("/messages/*/**", "/messages/A", true),
        ("/**/text", "/text", true),
        ("/**/text", "/context", false),
        ("/a/**/b/**/text", "/a/b/text", true),
        ("/a/**/b/**/text", "/a/x/b/y/z/text", true),
        ("/a/**/**/text", "/a/text", true),
        ("/a/**/text", "/ab/text", false),
        ("/messages/*/text", "/messages/A/deep/text", false),
        ("/**/a~1b/~0key", "/nested/a~1b/~0key", true),
        ("/a/**/", "/a/", true),
        ("/a/**/", "/a", false),
        ("/content**", "/content**", true),
        ("/content**", "/content", false),
        ("/***", "/***", true),
    ] {
        let matcher = TargetPathMatcher::new(&[], &[glob.into()]).unwrap();
        assert_eq!(
            matcher.matches(&json_pointer_segments(path).unwrap()),
            expected,
            "{glob} vs {path}"
        );
    }
    // Exact selectors retain literal `*` and `**` object keys.
    let literal = TargetPathMatcher::new(&["/*/**".into()], &[]).unwrap();
    assert!(literal.matches(&["*".into(), "**".into()]));
    assert!(!literal.matches(&["a".into(), "b".into()]));

    let backend = CompiledBuiltinBackend::new(
        BuiltinBackendConfig {
            action: "redact".into(),
            pattern: Some("direct|nested|deeper|sibling".into()),
            target_path_globs: vec!["/messages/**/text".into()],
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        backend.sanitize_json_preorder_dfs(payload),
        json!({
            "text": "root",
            "messages": {
                "text": "[REDACTED]",
                "A": {"content": [{"text": "[REDACTED]"}, {"deep": {"text": "[REDACTED]"}}]},
                "B": {"text": "[REDACTED]"}
            },
            "other": {"text": "outside"}
        })
    );
}

#[test]
fn recursive_globs_agree_with_exhaustive_reference_matching() {
    // An independent reference explores every possible `**` length. Restrict
    // its inputs to small patterns; the production matcher must not do this.
    fn reference(pattern: &[&str], path: &[String], descendant: bool) -> bool {
        if path.is_empty() {
            return if descendant {
                !pattern.is_empty()
            } else {
                pattern.iter().all(|segment| *segment == "**")
            };
        }
        match pattern.split_first() {
            None => false,
            Some((&"**", rest)) => {
                reference(rest, path, descendant) || reference(pattern, &path[1..], descendant)
            }
            Some((segment, rest)) => {
                (*segment == "*" || *segment == path[0]) && reference(rest, &path[1..], descendant)
            }
        }
    }
    fn sequences(alphabet: &[&str], max: usize) -> Vec<Vec<String>> {
        let mut sequences = vec![vec![]];
        let mut level = vec![vec![]];
        for _ in 0..max {
            level = level
                .iter()
                .flat_map(|prefix| {
                    alphabet.iter().map(|value| {
                        let mut next = prefix.clone();
                        next.push((*value).into());
                        next
                    })
                })
                .collect();
            sequences.extend(level.clone());
        }
        sequences
    }
    let paths = sequences(&["a", "b", ""], 4);
    for pattern in sequences(&["a", "b", "*", "**"], 4) {
        let refs: Vec<_> = pattern.iter().map(String::as_str).collect();
        let pointer = if pattern.is_empty() {
            String::new()
        } else {
            format!("/{}", pattern.join("/"))
        };
        let matcher = TargetPathMatcher::new(&[], &[pointer]).unwrap();
        for path in &paths {
            assert_eq!(
                matcher.matches(path),
                reference(&refs, path, false),
                "pattern={pattern:?} path={path:?}"
            );
            assert_eq!(
                matcher.may_match_descendant(path),
                reference(&refs, path, true),
                "prefix pattern={pattern:?} path={path:?}"
            );
        }
    }
    // Many recursive segments followed by a failing suffix must finish without
    // exploring the combinatorial number of possible segment assignments.
    let pointer = format!("/{}/missing", ["**/a"; 64].join("/"));
    let matcher = TargetPathMatcher::new(&[], &[pointer]).unwrap();
    assert!(!matcher.matches(&vec!["a".into(); 256]));

    // A nearly matching literal suffix after `**` exercises repeated retries,
    // rather than only the many-globstar case above.
    let pointer = format!("/**/{}/missing", ["a"; 64].join("/"));
    let matcher = TargetPathMatcher::new(&[], &[pointer]).unwrap();
    assert!(!matcher.matches(&vec!["a".into(); 512]));
}

#[test]
fn recursive_globs_preserve_containers_only_when_excluded_from_selection() {
    use serde_json::json;
    for (glob, expected) in [
        ("/messages/**", json!({"keep": 1})),
        ("/messages/*/**", json!({"messages": {}, "keep": 1})),
        ("/**", Json::Null),
    ] {
        let backend = CompiledBuiltinBackend::new(
            BuiltinBackendConfig {
                action: "remove".into(),
                target_path_globs: vec![glob.into()],
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(
            backend.sanitize_json_preorder_dfs(
                json!({"messages": {"A": {"text": "secret"}}, "keep": 1})
            ),
            expected
        );
    }
}
