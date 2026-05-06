// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::json;

use super::*;
use crate::adapters::{claude_code, codex, cursor};

#[test]
fn maps_claude_canonical_tool_payload() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/workspace",
            "hook_event_name": "PreToolUse",
            "tool_use_id": "toolu-1",
            "tool_name": "Read",
            "tool_input": { "file_path": "README.md" }
        }),
        &headers,
    );
    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert_eq!(event.session_id, "claude-session");
            assert_eq!(event.tool_call_id, "toolu-1");
            assert_eq!(event.tool_name, "Read");
            assert_eq!(event.arguments, json!({ "file_path": "README.md" }));
            assert_eq!(
                event.metadata["transcript_path"],
                json!("/tmp/transcript.jsonl")
            );
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert_eq!(outcome.response["continue"], json!(true));
    assert_eq!(
        outcome.response["hookSpecificOutput"],
        json!({
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow"
        })
    );
}

#[test]
fn maps_claude_post_tool_failure_with_canonical_fields() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "PostToolUseFailure",
            "tool_use_id": "toolu-1",
            "tool_name": "Bash",
            "tool_input": { "command": "false" },
            "tool_response": { "stderr": "failed" }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.tool_call_id, "toolu-1");
            assert_eq!(event.tool_name, "Bash");
            assert_eq!(event.result, json!({ "stderr": "failed" }));
            assert_eq!(event.status.as_deref(), Some("error"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_cursor_subagent_and_permission_response() {
    let headers = HeaderMap::new();
    let outcome = cursor::adapt(
        json!({
            "session_id": "cursor-session",
            "project_dir": "/repo",
            "user_email": "dev@example.com",
            "hook_event_name": "beforeShellExecution",
            "subagent": { "id": "worker" },
            "tool_call_id": "shell-1",
            "tool_name": "shell",
            "input": { "command": "cargo test" }
        }),
        &headers,
    );
    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert_eq!(event.session_id, "cursor-session");
            assert_eq!(event.subagent_id.as_deref(), Some("worker"));
            assert_eq!(event.metadata["project_dir"], json!("/repo"));
            assert_eq!(event.metadata["user_email"], json!("dev@example.com"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert_eq!(outcome.response["permission"], json!("allow"));
}

#[test]
fn keeps_codex_response_unwrapped() {
    let headers = HeaderMap::new();
    let outcome = codex::adapt(
        json!({
            "session_id": "codex-session",
            "hook_event_name": "sessionStart"
        }),
        &headers,
    );
    assert!(matches!(
        outcome.events[0],
        NormalizedEvent::AgentStarted(_)
    ));
    assert_eq!(outcome.response, json!({}));
}

#[test]
fn normalizes_mark_style_events_and_header_session_ids() {
    let mut headers = HeaderMap::new();
    headers.insert("x-nemo-flow-session-id", "header-session".parse().unwrap());
    headers.insert("x-nemo-flow-config-profile", "coverage".parse().unwrap());

    for (event_name, expected) in [
        ("UserPromptSubmit", "prompt"),
        ("afterAgentResponse", "response"),
        ("PreCompact", "compact"),
        ("Notification", "notification"),
        ("Unrecognized.Event", "hook"),
    ] {
        let outcome = cursor::adapt(
            json!({
                "eventName": event_name,
                "model": "model-a",
                "cwd": "/repo"
            }),
            &headers,
        );
        let session = match &outcome.events[0] {
            NormalizedEvent::PromptSubmitted(event) if expected == "prompt" => event,
            NormalizedEvent::AgentResponse(event) if expected == "response" => event,
            NormalizedEvent::Compaction(event) if expected == "compact" => event,
            NormalizedEvent::Notification(event) if expected == "notification" => event,
            NormalizedEvent::HookMark(event) if expected == "hook" => event,
            event => panic!("unexpected event for {event_name}: {event:?}"),
        };
        assert_eq!(session.session_id, "header-session");
        assert_eq!(session.metadata["model"], json!("model-a"));
        assert_eq!(session.metadata["cwd"], json!("/repo"));
        assert_eq!(
            session.metadata["sidecar_config_profile"],
            json!("coverage")
        );
    }
}

#[test]
fn extracts_tool_fields_from_fallback_payload_shapes() {
    let headers = HeaderMap::new();
    let outcome = codex::adapt(
        json!({
            "conversationId": "conversation-1",
            "event": "toolEnded",
            "tool": { "id": "tool-id", "name": "Shell" },
            "arguments": { "cmd": "pwd" },
            "result": { "stdout": "/repo" },
            "permission": "allow"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.session_id, "conversation-1");
            assert_eq!(event.tool_call_id, "tool-id");
            assert_eq!(event.tool_name, "Shell");
            assert_eq!(event.arguments, json!({ "cmd": "pwd" }));
            assert_eq!(event.result, json!({ "stdout": "/repo" }));
            assert_eq!(event.status.as_deref(), Some("allow"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn generated_ids_are_used_when_payload_omits_identifiers() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "hook_event_name": "PreToolUse",
            "tool_input": { "name": "Read", "file_path": "Cargo.toml" }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert!(event.session_id.starts_with("hook-"));
            assert!(event.tool_call_id.starts_with("tool-"));
            assert_eq!(event.tool_name, "Read");
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn stop_responses_preserve_vendor_shapes() {
    let headers = HeaderMap::new();
    let claude = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "Stop"
        }),
        &headers,
    );
    assert!(matches!(claude.events[0], NormalizedEvent::AgentEnded(_)));
    assert_eq!(claude.response["stopReason"], Value::Null);

    let cursor = cursor::adapt(
        json!({
            "session_id": "cursor-session",
            "hook_event_name": "stop"
        }),
        &headers,
    );
    assert!(matches!(cursor.events[0], NormalizedEvent::AgentEnded(_)));
    assert_eq!(cursor.response, json!({ "continue": true }));
}
