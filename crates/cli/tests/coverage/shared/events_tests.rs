// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use super::AgentKind;
use super::json_path::{string_at, string_at_any, value_at, value_at_any};

#[test]
fn agent_kinds_use_stable_runtime_metadata_names() {
    assert_eq!(AgentKind::Codex.as_str(), "codex");
    assert_eq!(AgentKind::ClaudeCode.as_str(), "claude-code");
    assert_eq!(AgentKind::Hermes.as_str(), "hermes");
    assert_eq!(AgentKind::Gateway.as_str(), "gateway");
}

#[test]
fn values_follow_nested_paths_and_return_owned_data() {
    let payload = json!({"request": {"id": "request-1", "enabled": true}});

    let extracted = value_at(&payload, &["request", "id"]);
    assert_eq!(extracted, Some(json!("request-1")));
    assert_eq!(value_at(&payload, &["request", "missing"]), None);
    assert_eq!(
        value_at_any(&payload, &[&["missing"], &["request", "enabled"]]),
        Some(json!(true))
    );
}

#[test]
fn string_helpers_accept_scalars_and_skip_empty_or_structured_values() {
    let payload = json!({
        "empty": "",
        "number": 42,
        "enabled": false,
        "object": {"id": "nested"}
    });

    assert_eq!(string_at(&payload, &["number"]), Some("42".into()));
    assert_eq!(string_at(&payload, &["enabled"]), Some("false".into()));
    assert_eq!(string_at(&payload, &["empty"]), None);
    assert_eq!(string_at(&payload, &["object"]), None);
    assert_eq!(
        string_at_any(&payload, &[&["empty"], &["object"], &["object", "id"]]),
        Some("nested".into())
    );
}
