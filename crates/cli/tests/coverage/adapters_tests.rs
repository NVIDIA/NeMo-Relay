// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::json;

use super::*;
use crate::adapters::{claude_code, codex, cursor, hermes};

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
            assert!(event.metadata.get("transcript_path").is_none());
            assert!(event.metadata.get("cwd").is_none());
            assert_eq!(
                event.payload["transcript_path"],
                json!("/tmp/transcript.jsonl")
            );
            assert_eq!(event.payload["cwd"], json!("/workspace"));
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
            "error": "failed",
            "is_interrupt": false,
            "duration_ms": 12
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.tool_call_id, "toolu-1");
            assert_eq!(event.tool_name, "Bash");
            assert_eq!(
                event.result,
                json!({ "error": "failed", "is_interrupt": false, "duration_ms": 12 })
            );
            assert_eq!(event.status.as_deref(), Some("error"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_permission_denied_as_tool_end() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "PermissionDenied",
            "tool_use_id": "toolu-denied",
            "tool_name": "Bash",
            "tool_input": { "command": "rm -rf /tmp/project" },
            "reason": "policy"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.tool_call_id, "toolu-denied");
            assert_eq!(event.status.as_deref(), Some("denied"));
            assert_eq!(event.result, json!({ "reason": "policy" }));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_subagent_canonical_agent_id() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "SubagentStart",
            "agent_id": "agent-worker-1",
            "agent_type": "general-purpose"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentStarted(event) => {
            assert_eq!(event.subagent_id, "agent-worker-1");
            assert_eq!(event.metadata["agent_type"], json!("general-purpose"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_subagent_stop() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "SubagentStop",
            "agent_id": "agent-worker-1"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentEnded(event) => {
            assert_eq!(event.subagent_id, "agent-worker-1");
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_stop_response_shape() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "Stop"
        }),
        &HeaderMap::new(),
    );

    // Claude's hook output schema rejects `null` for optional string fields like stopReason —
    // the adapter must omit them entirely (return only `{ continue: true }`).
    assert_eq!(outcome.response, json!({ "continue": true }));
    assert!(
        outcome.response.get("stopReason").is_none(),
        "stopReason must not appear in the response (Claude rejects null)"
    );
}

// Stop hook on Claude/Codex/Cursor (per-turn boundary) must yield a TurnEnded event so the
// session manager can snapshot ATIF without closing the agent scope. Codex needs this because
// it has no SessionEnd hook; Claude/Cursor get it for free for resilience.
#[test]
fn stop_hook_emits_turn_ended_for_codex() {
    let outcome = codex::adapt(
        json!({ "session_id": "codex-session", "hook_event_name": "Stop" }),
        &HeaderMap::new(),
    );
    assert!(
        outcome
            .events
            .iter()
            .any(|e| matches!(e, NormalizedEvent::TurnEnded(_))),
        "codex Stop must produce a TurnEnded event for ATIF snapshot. events: {:?}",
        outcome.events
    );
}

#[test]
fn stop_hook_emits_turn_ended_for_claude() {
    let outcome = claude_code::adapt(
        json!({ "session_id": "claude-session", "hook_event_name": "Stop" }),
        &HeaderMap::new(),
    );
    assert!(
        outcome
            .events
            .iter()
            .any(|e| matches!(e, NormalizedEvent::TurnEnded(_))),
        "claude Stop must produce a TurnEnded event for ATIF snapshot"
    );
}

// Cursor classifies `stop` as AgentEnded (its existing per-adapter rule). The TurnEnded path
// must NOT also fire there — flush_observers already writes ATIF on agent-end, and a follow-up
// snapshot on a removed session would recreate an empty session and overwrite the freshly
// written file with an empty trajectory.
#[test]
fn stop_hook_does_not_double_emit_for_cursor_agent_end() {
    let outcome = cursor::adapt(
        json!({ "session_id": "cursor-session", "hook_event_name": "stop" }),
        &HeaderMap::new(),
    );
    assert!(
        matches!(outcome.events.first(), Some(NormalizedEvent::AgentEnded(_))),
        "cursor stop must classify as AgentEnded"
    );
    assert!(
        !outcome
            .events
            .iter()
            .any(|e| matches!(e, NormalizedEvent::TurnEnded(_))),
        "cursor stop must NOT also produce TurnEnded — would double-write ATIF then wipe it"
    );
}

#[test]
fn adapter_string_lookup_accepts_scalar_values_only() {
    let payload = json!({
        "number": 7,
        "boolean": false,
        "object": { "nested": true }
    });

    assert_eq!(string_at(&payload, &["number"]).as_deref(), Some("7"));
    assert_eq!(string_at(&payload, &["boolean"]).as_deref(), Some("false"));
    assert_eq!(string_at(&payload, &["object"]), None);
}

#[test]
fn builtin_extractor_keeps_fallbacks_at_adapter_boundary() {
    let headers = HeaderMap::new();
    let payload = json!({});

    assert_eq!(
        BUILTIN_AGENT_PAYLOAD_EXTRACTOR.session_id(&payload, &headers),
        None
    );
    assert_eq!(BUILTIN_AGENT_PAYLOAD_EXTRACTOR.event_name(&payload), None);
    assert_eq!(
        BUILTIN_AGENT_PAYLOAD_EXTRACTOR.subagent_id(&payload, &headers),
        None
    );
    assert_eq!(
        BUILTIN_AGENT_PAYLOAD_EXTRACTOR.llm_hint(&payload, &headers),
        ExtractedLlmHint::default()
    );
    assert_eq!(
        BUILTIN_AGENT_PAYLOAD_EXTRACTOR.tool_call(&payload, &headers, "PreToolUse"),
        ExtractedToolCall {
            tool_call_id: None,
            tool_name: None,
            subagent_id: None,
            arguments: None,
            result: None,
            status: None,
        }
    );

    assert!(session_id(&payload, &headers).starts_with("hook-"));
    assert_eq!(event_name(&payload), "unknown");

    let event = common_tool_event(&payload, &headers, AgentKind::ClaudeCode);
    assert!(event.tool_call_id.starts_with("tool-"));
    assert_eq!(event.tool_name, "unknown_tool");
    assert_eq!(event.arguments, json!(null));
    assert_eq!(event.result, json!(null));
}

#[test]
fn builtin_extractor_reads_agent_hint_and_tool_call_fields() {
    let headers = HeaderMap::new();
    let payload = json!({
        "subagent_id": "worker-1",
        "agent": {
            "id": "agent-1",
            "type": "reviewer"
        },
        "conversationId": "conversation-1",
        "generation": { "id": "generation-1" },
        "request": { "id": "request-1" },
        "modelName": "gpt-test",
        "tool_call_id": "tool-call-1",
        "tool": { "name": "search" },
        "arguments": { "query": "needle" },
        "result": { "matches": 2 },
        "status": "success"
    });

    assert_eq!(
        BUILTIN_AGENT_PAYLOAD_EXTRACTOR.llm_hint(&payload, &headers),
        ExtractedLlmHint {
            subagent_id: Some("worker-1".into()),
            agent_id: Some("agent-1".into()),
            agent_type: Some("reviewer".into()),
            conversation_id: Some("conversation-1".into()),
            generation_id: Some("generation-1".into()),
            request_id: Some("request-1".into()),
            model: Some("gpt-test".into()),
        }
    );
    assert_eq!(
        BUILTIN_AGENT_PAYLOAD_EXTRACTOR.tool_call(&payload, &headers, "PostToolUse"),
        ExtractedToolCall {
            tool_call_id: Some("tool-call-1".into()),
            tool_name: Some("search".into()),
            subagent_id: Some("worker-1".into()),
            arguments: Some(json!({ "query": "needle" })),
            result: Some(json!({ "matches": 2 })),
            status: Some("success".into()),
        }
    );
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
            assert!(event.metadata.get("project_dir").is_none());
            assert!(event.metadata.get("user_email").is_none());
            assert_eq!(event.payload["project_dir"], json!("/repo"));
            assert_eq!(event.payload["user_email"], json!("dev@example.com"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert_eq!(outcome.response["permission"], json!("allow"));
    assert!(outcome.response.get("user_message").is_none());
    assert!(outcome.response.get("agent_message").is_none());
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
fn maps_hermes_shell_hook_tool_payload() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_tool_call",
            "tool_name": "terminal",
            "tool_input": { "command": "pwd" },
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "tool-1"
            }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert_eq!(event.agent_kind, AgentKind::Hermes);
            assert_eq!(event.session_id, "hermes-session");
            assert_eq!(event.tool_call_id, "tool-1");
            assert_eq!(event.tool_name, "terminal");
            assert_eq!(event.arguments, json!({ "command": "pwd" }));
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert_eq!(outcome.response, json!({}));
}

#[test]
fn drops_uncorrelatable_hermes_pre_tool_call() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_tool_call",
            "task_id": "task-1",
            "tool_call_id": "toolcall-1",
            "tool_name": "terminal",
            "tool_input": { "command": "pwd" }
        }),
        &headers,
    );

    assert!(outcome.events.is_empty());
    assert_eq!(outcome.response, json!({}));
}

#[test]
fn maps_hermes_subagent_child_identifiers() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "extra": {
                "child_session_id": "child-session",
                "child_subagent_id": "sa-1",
                "parent_turn_id": "turn-1"
            }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentStarted(event) => {
            assert_eq!(event.agent_kind, AgentKind::Hermes);
            assert_eq!(event.session_id, "parent-session");
            assert_eq!(event.subagent_id, "sa-1");
            assert_eq!(
                event.payload["extra"]["child_session_id"],
                json!("child-session")
            );
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_camel_case_child_subagent_identifiers() {
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "childSubagentId": "sa-camel-top"
        }),
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "extra": {
                "childSubagentId": "sa-camel-extra"
            }
        }),
    ] {
        let expected = payload
            .get("childSubagentId")
            .or_else(|| payload.pointer("/extra/childSubagentId"))
            .and_then(|value| value.as_str())
            .expect("test payload should include childSubagentId")
            .to_string();
        let outcome = hermes::adapt(payload, &headers);

        match &outcome.events[0] {
            NormalizedEvent::SubagentStarted(event) => {
                assert_eq!(event.subagent_id, expected);
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }
}

#[test]
fn maps_hermes_real_session_boundary_without_closing_per_turn_end() {
    let headers = HeaderMap::new();

    let per_turn = hermes::adapt(
        json!({
            "hook_event_name": "on_session_end",
            "session_id": "hermes-session"
        }),
        &headers,
    );
    // `on_session_end` is per-turn for hermes-agent, so it snapshots ATIF without becoming a
    // user-visible system trajectory step.
    assert_eq!(per_turn.events.len(), 1);
    assert!(matches!(per_turn.events[0], NormalizedEvent::TurnEnded(_)));

    let finalized = hermes::adapt(
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-session"
        }),
        &headers,
    );
    assert_eq!(finalized.events.len(), 1);
    assert!(matches!(
        finalized.events[0],
        NormalizedEvent::AgentEnded(_)
    ));
}

#[test]
fn maps_hermes_hook_event_name_and_subagent_from_extra_payload() {
    let outcome = hermes::adapt(
        json!({
            "session_id": "hermes-session",
            "extra": {
                "hook_event_name": "subagent_stop",
                "subagent_id": "worker-1"
            }
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentEnded(event) => {
            assert_eq!(event.event_name, "subagent_stop");
            assert_eq!(event.subagent_id, "worker-1");
            assert_eq!(event.session_id, "hermes-session");
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_api_hooks_to_llm_lifecycle() {
    let headers = HeaderMap::new();

    let started = hermes::adapt(
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 2,
                "model": "qwen",
                "provider": "custom",
                "base_url": "http://localhost:11434/v1",
                "api_mode": "chat_completions",
                "message_count": 3,
                "tool_count": 1,
                "approx_input_tokens": 12,
                "request_char_count": 456,
                "max_tokens": 1024
            }
        }),
        &headers,
    );
    match &started.events[0] {
        NormalizedEvent::LlmStarted(event) => {
            assert_eq!(event.session_id, "hermes-session");
            assert_eq!(event.api_call_id, "hermes-session:task-1:2");
            assert_eq!(event.provider, "custom");
            assert_eq!(event.model_name.as_deref(), Some("qwen"));
            assert_eq!(event.request["message_count"], json!(3));
            assert_eq!(
                event.request["fidelity"]["provider_payload_exact"],
                json!(false)
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(false));
        }
        event => panic!("unexpected event: {event:?}"),
    }

    let ended = hermes::adapt(
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 2,
                "model": "qwen",
                "response_model": "qwen",
                "provider": "custom",
                "api_duration": 0.25,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 3 }
                }
            }
        }),
        &headers,
    );
    match &ended.events[0] {
        NormalizedEvent::LlmEnded(event) => {
            assert_eq!(event.api_call_id, "hermes-session:task-1:2");
            assert_eq!(event.response["usage"]["prompt_tokens"], json!(10));
            assert_eq!(event.response["usage"]["completion_tokens"], json!(5));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_exact_api_hook_payloads_to_llm_lifecycle() {
    let headers = HeaderMap::new();

    let started = hermes::adapt(
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:2",
                "api_call_count": 2,
                "model": "qwen",
                "provider": "custom",
                "request": {
                    "method": "POST",
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "hello" }
                        ],
                        "tools": [
                            { "type": "function", "function": { "name": "search_files" } }
                        ]
                    }
                }
            }
        }),
        &headers,
    );
    match &started.events[0] {
        NormalizedEvent::LlmStarted(event) => {
            assert_eq!(event.api_call_id, "turn-1:api:2");
            assert_eq!(event.request["messages"][0]["content"], json!("hello"));
            assert_eq!(
                event.request["tools"][0]["function"]["name"],
                json!("search_files")
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(true));
            assert_eq!(
                event.metadata["fidelity_source"],
                json!("hermes_api_hooks_sanitized")
            );
        }
        event => panic!("unexpected event: {event:?}"),
    }

    let ended = hermes::adapt(
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:2",
                "api_call_count": 2,
                "model": "qwen",
                "response": {
                    "model": "qwen",
                    "finish_reason": "tool_calls",
                    "assistant_message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": "{\"query\":\"needle\"}"
                                }
                            }
                        ]
                    },
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5
                    }
                }
            }
        }),
        &headers,
    );
    match &ended.events[0] {
        NormalizedEvent::LlmEnded(event) => {
            assert_eq!(event.api_call_id, "turn-1:api:2");
            assert_eq!(event.response["tool_calls"][0]["id"], json!("call-1"));
            assert_eq!(event.response["usage"]["prompt_tokens"], json!(10));
            assert_eq!(event.metadata["provider_payload_exact"], json!(true));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_api_request_error_to_llm_end() {
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "api_request_error",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:3",
                "api_call_count": 3,
                "model": "qwen",
                "provider": "custom",
                "status_code": 502,
                "retry_count": 1,
                "max_retries": 2,
                "retryable": true,
                "reason": "upstream",
                "error": {
                    "type": "BadGateway",
                    "message": "gateway upstream error"
                }
            }
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmEnded(event) => {
            assert_eq!(event.api_call_id, "turn-1:api:3");
            assert_eq!(event.response["status_code"], json!(502));
            assert_eq!(
                event.response["error"]["message"],
                json!("gateway upstream error")
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(false));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_null_request_as_lossy_summary() {
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 4,
                "model": "qwen",
                "provider": "custom",
                "request": null,
                "message_count": 2
            }
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmStarted(event) => {
            assert_eq!(event.api_call_id, "hermes-session:task-1:4");
            assert_eq!(event.request["message_count"], json!(2));
            assert_eq!(
                event.request["fidelity"]["provider_payload_exact"],
                json!(false)
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(false));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn normalizes_mark_style_events_and_header_session_ids() {
    let mut headers = HeaderMap::new();
    headers.insert("x-nemo-relay-session-id", "header-session".parse().unwrap());
    headers.insert("x-nemo-relay-config-profile", "coverage".parse().unwrap());

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
        let (session_id, metadata) = match &outcome.events[0] {
            NormalizedEvent::PromptSubmitted(event) if expected == "prompt" => {
                (event.session_id.as_str(), &event.metadata)
            }
            NormalizedEvent::LlmHint(event) if expected == "response" => {
                (event.session_id.as_str(), &event.metadata)
            }
            NormalizedEvent::Compaction(event) if expected == "compact" => {
                (event.session_id.as_str(), &event.metadata)
            }
            NormalizedEvent::Notification(event) if expected == "notification" => {
                (event.session_id.as_str(), &event.metadata)
            }
            NormalizedEvent::HookMark(event) if expected == "hook" => {
                (event.session_id.as_str(), &event.metadata)
            }
            event => panic!("unexpected event for {event_name}: {event:?}"),
        };
        if expected == "prompt" {
            assert!(
                matches!(outcome.events.get(1), Some(NormalizedEvent::LlmHint(_))),
                "prompt hooks should also emit a private LLM hint"
            );
        }
        assert_eq!(session_id, "header-session");
        assert_eq!(metadata["model"], json!("model-a"));
        assert!(metadata.get("cwd").is_none());
        assert_eq!(metadata["gateway_config_profile"], json!("coverage"));
    }
}

#[test]
fn maps_hermes_llm_hooks_to_private_hints() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_llm_call",
            "session_id": "hermes-session",
            "model": "anthropic/claude-sonnet",
            "request_id": "req-1"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmHint(event) => {
            assert_eq!(event.session_id, "hermes-session");
            assert_eq!(event.event_name, "pre_llm_call");
            assert_eq!(event.model.as_deref(), Some("anthropic/claude-sonnet"));
            assert_eq!(event.request_id.as_deref(), Some("req-1"));
        }
        event => panic!("unexpected event: {event:?}"),
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
    assert!(matches!(claude.events[0], NormalizedEvent::LlmHint(_)));
    assert!(
        claude.response.get("stopReason").is_none(),
        "stopReason must not be present (Claude rejects null per its hook schema)"
    );

    let codex = codex::adapt(
        json!({
            "session_id": "codex-session",
            "hook_event_name": "stop"
        }),
        &headers,
    );
    assert!(matches!(codex.events[0], NormalizedEvent::LlmHint(_)));
    assert_eq!(codex.response, json!({}));

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
