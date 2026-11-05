// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderValue;
use serde_json::{Value, json};

use super::*;

fn tool_event(agent_kind: AgentKind, tool_name: &str, result: Value) -> ToolEvent {
    ToolEvent {
        session_id: "session".into(),
        agent_kind,
        event_name: "PostToolUse".into(),
        tool_call_id: "tool-1".into(),
        tool_name: tool_name.into(),
        subagent_id: None,
        arguments: Value::Null,
        result,
        status: Some("success".into()),
        payload: json!({}),
        metadata: json!({}),
    }
}

#[test]
fn owns_anthropic_gateway_providers_only() {
    assert!(owns_gateway_provider("anthropic.messages"));
    assert!(owns_gateway_provider("anthropic.count_tokens"));
    assert!(!owns_gateway_provider("openai.responses"));
}

#[test]
fn session_id_from_headers_reads_claude_native_header() {
    let mut headers = HeaderMap::new();
    assert_eq!(session_id_from_headers(&headers), None);

    headers.insert(
        "x-claude-code-session-id",
        HeaderValue::from_static("claude-session"),
    );
    assert_eq!(
        session_id_from_headers(&headers).as_deref(),
        Some("claude-session")
    );
}

#[test]
fn test_request_class_known_value_preserves_class_and_internal_status() {
    for (value, expected, internal) in [
        ("main", RequestClass::Main, false),
        ("subagent", RequestClass::Subagent, false),
        ("workflow", RequestClass::Workflow, false),
        ("compaction", RequestClass::Compaction, true),
        ("auxiliary", RequestClass::Auxiliary, true),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-claude-code-request-class",
            HeaderValue::from_static(value),
        );
        let actual = request_class(&headers);
        assert_eq!(actual, expected);
        assert_eq!(actual.as_str(), value);
        assert_eq!(actual.is_internal(), internal);
    }
}

#[test]
fn test_request_class_missing_or_noncanonical_value_is_unknown() {
    let mut headers = HeaderMap::new();
    assert_eq!(request_class(&headers), RequestClass::Unknown);
    for value in [
        "",
        "Auxiliary",
        " auxiliary",
        "auxiliary ",
        "auxiliary,main",
        "auxiliary, auxiliary",
        "side_reply",
        "future_class",
    ] {
        headers.insert(
            "x-claude-code-request-class",
            HeaderValue::from_static(value),
        );
        assert_eq!(request_class(&headers), RequestClass::Unknown, "{value:?}");
    }
    headers.insert(
        "x-claude-code-request-class",
        HeaderValue::from_bytes(&[0xff]).unwrap(),
    );
    assert_eq!(request_class(&headers), RequestClass::Unknown);
    assert_eq!(RequestClass::Unknown.as_str(), "unknown");
    assert!(!RequestClass::Unknown.is_internal());
}

#[test]
fn test_request_class_repeated_value_is_unknown() {
    for second in ["auxiliary", "main"] {
        let mut headers = HeaderMap::new();
        headers.append(
            "x-claude-code-request-class",
            HeaderValue::from_static("auxiliary"),
        );
        headers.append(
            "x-claude-code-request-class",
            HeaderValue::from_static(second),
        );
        assert_eq!(request_class(&headers), RequestClass::Unknown);
    }
}

#[test]
fn completed_subagent_from_agent_tool_accepts_known_result_keys() {
    for (key, expected) in [
        ("agentId", "agent-id"),
        ("agent_id", "agent-id-snake"),
        ("subagentId", "subagent-id"),
        ("subagent_id", "subagent-id-snake"),
    ] {
        let event = tool_event(
            AgentKind::ClaudeCode,
            "Agent",
            json!({ key: expected, "status": "completed" }),
        );
        assert_eq!(
            completed_subagent_from_agent_tool(&event).as_deref(),
            Some(expected),
            "key {key} should close the matching subagent"
        );
    }
}

#[test]
fn completed_subagent_from_agent_tool_rejects_async_launch_results() {
    for status in ["async_launched", "started", "running", "in-progress"] {
        let event = tool_event(
            AgentKind::ClaudeCode,
            "Agent",
            json!({
                "agentId": "worker",
                "status": status
            }),
        );
        assert_eq!(
            completed_subagent_from_agent_tool(&event),
            None,
            "status {status} is not a terminal worker result"
        );
    }
}

#[test]
fn completed_subagent_from_agent_tool_requires_terminal_result_evidence() {
    assert_eq!(
        completed_subagent_from_agent_tool(&tool_event(
            AgentKind::ClaudeCode,
            "Agent",
            json!({ "agentId": "worker" }),
        )),
        None
    );
    assert_eq!(
        completed_subagent_from_agent_tool(&tool_event(
            AgentKind::ClaudeCode,
            "Agent",
            json!({
                "agentId": "worker",
                "totalDurationMs": 123
            }),
        ))
        .as_deref(),
        Some("worker")
    );
}

#[test]
fn completed_subagent_from_agent_tool_rejects_unrelated_tools_and_agents() {
    assert_eq!(
        completed_subagent_from_agent_tool(&tool_event(
            AgentKind::Codex,
            "Agent",
            json!({ "agentId": "worker" }),
        )),
        None
    );
    assert_eq!(
        completed_subagent_from_agent_tool(&tool_event(
            AgentKind::ClaudeCode,
            "Read",
            json!({ "agentId": "worker" }),
        )),
        None
    );
    assert_eq!(
        completed_subagent_from_agent_tool(&tool_event(
            AgentKind::ClaudeCode,
            "Agent",
            json!({ "status": "completed" }),
        )),
        None
    );
}
