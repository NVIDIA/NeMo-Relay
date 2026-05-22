// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderValue;
use nemo_relay::api::llm::LlmRequest;
use serde_json::Map;

use super::*;
use crate::model::{LlmEvent, LlmHintEvent};

fn session_event(session_id: &str, event_name: &str) -> SessionEvent {
    SessionEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: event_name.into(),
        payload: json!({ "event": event_name }),
        metadata: json!({ "event_metadata": event_name }),
    }
}

fn subagent_event(session_id: &str, event_name: &str) -> SubagentEvent {
    SubagentEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: event_name.into(),
        subagent_id: "nested-child".into(),
        payload: json!({ "event": event_name }),
        metadata: json!({ "event_metadata": event_name }),
    }
}

fn llm_hint_event(session_id: &str) -> LlmHintEvent {
    LlmHintEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: "Stop".into(),
        subagent_id: Some("payload-child".into()),
        agent_id: None,
        agent_type: Some("explorer".into()),
        conversation_id: Some("conversation-1".into()),
        generation_id: Some("generation-1".into()),
        request_id: Some("request-1".into()),
        model: Some("gpt-test".into()),
        payload: json!({ "hint": true }),
        metadata: json!({ "event_metadata": "hint" }),
    }
}

fn llm_event(session_id: &str, event_name: &str) -> LlmEvent {
    LlmEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: event_name.into(),
        api_call_id: "api-call-1".into(),
        provider: "openai.responses".into(),
        model_name: Some("gpt-test".into()),
        request: json!({ "input": "hello" }),
        response: json!({ "output_text": "hi" }),
        metadata: json!({ "event_metadata": event_name }),
    }
}

fn tool_event(session_id: &str, event_name: &str) -> ToolEvent {
    ToolEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: event_name.into(),
        tool_call_id: "tool-1".into(),
        tool_name: "exec_command".into(),
        subagent_id: Some("payload-child".into()),
        arguments: json!({ "cmd": "true" }),
        result: json!({ "ok": true }),
        status: Some("success".into()),
        payload: json!({ "tool": true }),
        metadata: json!({ "event_metadata": event_name }),
    }
}

fn aliases() -> HashMap<String, SessionAlias> {
    HashMap::from([(
        "child".into(),
        SessionAlias::new(
            "parent".into(),
            "child".into(),
            json!({ "alias_metadata": true }),
        ),
    )])
}

#[test]
fn gateway_session_id_uses_explicit_claude_then_codex_fallbacks() {
    let mut headers = HeaderMap::new();
    let codex_body = json!({
        "prompt_cache_key": "codex-thread",
        "client_metadata": { "x-codex-installation-id": "install-1" }
    });

    assert_eq!(
        gateway_session_id(&headers, &codex_body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("codex-thread")
    );

    headers.insert(
        "x-claude-code-session-id",
        HeaderValue::from_static("claude-thread"),
    );
    assert_eq!(
        gateway_session_id(&headers, &codex_body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("claude-thread")
    );

    headers.insert(
        "x-nemo-relay-session-id",
        HeaderValue::from_static("explicit-thread"),
    );
    assert_eq!(
        gateway_session_id(&headers, &codex_body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("explicit-thread")
    );
}

#[test]
fn gateway_subagent_and_identifier_helpers_respect_header_precedence() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-subagent-id",
        HeaderValue::from_static("worker-1"),
    );
    headers.insert(
        "x-nemo-relay-request-id",
        HeaderValue::from_static("request-header"),
    );
    let body = json!({
        "conversation": { "id": 42 },
        "request": { "id": "request-body" },
        "object": { "id": { "nested": true } }
    });

    assert_eq!(gateway_subagent_id(&headers).as_deref(), Some("worker-1"));
    assert_eq!(
        gateway_identifier(
            &headers,
            &body,
            "x-nemo-relay-request-id",
            &[&["request", "id"]]
        )
        .as_deref(),
        Some("request-header")
    );
    assert_eq!(
        gateway_identifier(
            &HeaderMap::new(),
            &body,
            "missing",
            &[&["conversation", "id"]]
        )
        .as_deref(),
        Some("42")
    );
    assert_eq!(
        gateway_identifier(&HeaderMap::new(), &body, "missing", &[&["object", "id"]]),
        None
    );
}

#[test]
fn agent_kind_inference_covers_known_provider_names() {
    assert_eq!(
        agent_kind_for_gateway_provider("anthropic.messages"),
        AgentKind::ClaudeCode
    );
    assert_eq!(
        agent_kind_for_gateway_provider("anthropic.count_tokens"),
        AgentKind::ClaudeCode
    );
    assert_eq!(
        agent_kind_for_gateway_provider("openai.responses"),
        AgentKind::Codex
    );
    assert_eq!(
        agent_kind_for_gateway_provider("openai.chat_completions"),
        AgentKind::Gateway
    );
}

#[test]
fn session_agent_scope_policy_skips_unbounded_coding_agent_sessions() {
    assert!(!should_emit_session_agent_scope(AgentKind::ClaudeCode));
    assert!(!should_emit_session_agent_scope(AgentKind::Codex));
    assert!(should_emit_session_agent_scope(AgentKind::Gateway));
}

#[test]
fn request_affinity_key_reads_messages_content_blocks() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "<system-reminder>\nToday is 2026-05-19.\n</system-reminder>" },
                        { "type": "text", "text": "  Analyze the python binding\n\nwith detail.  " }
                    ]
                }
            ]
        }),
    };

    assert_eq!(
        request_affinity_key(&request).as_deref(),
        Some("Analyze the python binding with detail.")
    );
}

#[test]
fn request_affinity_key_reads_chat_completion_string_messages() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                { "role": "system", "content": "You are a coding agent." },
                { "role": "user", "content": "Review the Rust CLI gateway alignment code." }
            ]
        }),
    };

    assert_eq!(
        request_affinity_key(&request).as_deref(),
        Some("Review the Rust CLI gateway alignment code.")
    );
}

#[test]
fn request_affinity_key_preserves_leading_tagged_context_text() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                {
                    "role": "user",
                    "content": "<runtime-context>\nTrace run 7.\n</runtime-context>\n<system-reminder>\nToday is 2026-05-19.\n</system-reminder>\n\nReview the gateway correlation logic."
                }
            ]
        }),
    };

    assert_eq!(
        request_affinity_key(&request).as_deref(),
        Some(
            "<runtime-context> Trace run 7. </runtime-context> <system-reminder> Today is 2026-05-19. </system-reminder> Review the gateway correlation logic."
        )
    );
}

#[test]
fn request_affinity_key_keeps_task_after_large_prefix() {
    let prefix = "volatile context ".repeat(80);
    let task = "Review the gateway correlation logic.";
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                {
                    "role": "user",
                    "content": format!("<runtime-context>{prefix}</runtime-context> {task}")
                }
            ]
        }),
    };

    let key = request_affinity_key(&request).unwrap();
    assert!(key.starts_with("<runtime-context>volatile context"));
    assert!(
        key.ends_with(task),
        "larger affinity prefixes should preserve the task text after volatile context"
    );
}

#[test]
fn request_affinity_key_preserves_fully_tagged_prompt_text() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                {
                    "role": "user",
                    "content": "<task>Review the gateway correlation logic.</task>"
                }
            ]
        }),
    };

    assert_eq!(
        request_affinity_key(&request).as_deref(),
        Some("<task>Review the gateway correlation logic.</task>")
    );
}

#[test]
fn request_affinity_key_prefers_latest_task_message_over_root_history() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                {
                    "role": "user",
                    "content": "Can you analyze the whole codebase with parallel subagents?"
                },
                {
                    "role": "assistant",
                    "content": "I will delegate the directory reviews."
                },
                {
                    "role": "user",
                    "content": "Thoroughly explore the crates/ffi directory."
                }
            ]
        }),
    };

    assert_eq!(
        request_affinity_key(&request).as_deref(),
        Some("Thoroughly explore the crates/ffi directory.")
    );
}

#[test]
fn request_affinity_key_reads_responses_input_items_and_prompt() {
    let responses_request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Analyze the Node binding architecture."
                        }
                    ]
                }
            ]
        }),
    };
    let prompt_request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "prompt": "Summarize the Go binding architecture."
        }),
    };

    assert_eq!(
        request_affinity_key(&responses_request).as_deref(),
        Some("Analyze the Node binding architecture.")
    );
    assert_eq!(
        request_affinity_key(&prompt_request).as_deref(),
        Some("Summarize the Go binding architecture.")
    );
}

#[test]
fn request_affinity_key_ignores_count_token_style_payloads() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!("// source text without a chat user task"),
    };

    assert_eq!(request_affinity_key(&request), None);
}

#[test]
fn request_affinity_key_ignores_tool_results_and_sidecar_json() {
    let tool_result = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "// SPDX-FileCopyrightText: Copyright (c) 2026\npub fn source() {}"
                        }
                    ]
                }
            ]
        }),
    };
    let sidecar_json = LlmRequest {
        headers: Map::new(),
        content: json!({
            "input": "{\"parentUuid\":\"scope\",\"childUuid\":\"scope\"}"
        }),
    };
    let sidecar_json_array = LlmRequest {
        headers: Map::new(),
        content: json!({
            "input": "  [{\"parentUuid\":\"scope\",\"childUuid\":\"scope\"}]"
        }),
    };

    assert_eq!(request_affinity_key(&tool_result), None);
    assert_eq!(request_affinity_key(&sidecar_json), None);
    assert_eq!(request_affinity_key(&sidecar_json_array), None);
}

#[test]
fn route_event_through_alias_covers_all_event_variants() {
    let aliases = aliases();
    let cases = vec![
        NormalizedEvent::AgentStarted(session_event("child", "SessionStart")),
        NormalizedEvent::AgentEnded(session_event("child", "SessionEnd")),
        NormalizedEvent::TurnEnded(session_event("child", "Stop")),
        NormalizedEvent::PromptSubmitted(session_event("child", "Prompt")),
        NormalizedEvent::Compaction(session_event("child", "Compact")),
        NormalizedEvent::Notification(session_event("child", "Notify")),
        NormalizedEvent::HookMark(session_event("child", "Mark")),
        NormalizedEvent::SubagentStarted(subagent_event("child", "SubagentStart")),
        NormalizedEvent::SubagentEnded(subagent_event("child", "SubagentEnd")),
        NormalizedEvent::LlmHint(llm_hint_event("child")),
        NormalizedEvent::LlmStarted(llm_event("child", "LlmStart")),
        NormalizedEvent::LlmEnded(llm_event("child", "LlmEnd")),
        NormalizedEvent::ToolStarted(tool_event("child", "ToolStart")),
        NormalizedEvent::ToolEnded(tool_event("child", "ToolEnd")),
    ];

    for event in cases {
        let closes_alias = matches!(
            event,
            NormalizedEvent::AgentEnded(_) | NormalizedEvent::TurnEnded(_)
        );
        let (event, finished_alias) = route_event_through_alias(event, &aliases);
        assert_eq!(event.session_id(), "parent");
        assert_eq!(
            event_metadata(&event)["alias_metadata"],
            json!(true),
            "alias metadata should be stamped on {event:?}"
        );
        assert_eq!(finished_alias.as_deref(), closes_alias.then_some("child"));

        match event {
            NormalizedEvent::AgentStarted(event) => panic!("unexpected agent start: {event:?}"),
            NormalizedEvent::AgentEnded(event) => panic!("unexpected agent end: {event:?}"),
            NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
                assert!(!event.subagent_id.is_empty());
            }
            NormalizedEvent::LlmHint(event) => {
                assert_eq!(event.subagent_id.as_deref(), Some("child"));
            }
            NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => {
                assert_eq!(event.subagent_id.as_deref(), Some("child"));
            }
            NormalizedEvent::TurnEnded(_)
            | NormalizedEvent::PromptSubmitted(_)
            | NormalizedEvent::Compaction(_)
            | NormalizedEvent::Notification(_)
            | NormalizedEvent::HookMark(_)
            | NormalizedEvent::LlmStarted(_)
            | NormalizedEvent::LlmEnded(_) => {}
        }
    }
}

#[test]
fn route_event_without_alias_is_unchanged() {
    let event = NormalizedEvent::ToolStarted(tool_event("unknown-child", "ToolStart"));
    let (routed, finished_alias) = route_event_through_alias(event.clone(), &aliases());

    assert_eq!(routed, event);
    assert_eq!(finished_alias, None);
}

#[test]
fn json_helpers_and_metadata_merge_cover_edge_shapes() {
    let payload = json!({
        "string": "value",
        "number": 7,
        "boolean": false,
        "empty": "",
        "object": { "nested": true }
    });

    assert_eq!(
        json_string_at(&payload, &[&["missing"][..], &["string"][..]]).as_deref(),
        Some("value")
    );
    assert_eq!(
        json_string_at(&payload, &[&["number"][..]]).as_deref(),
        Some("7")
    );
    assert_eq!(
        json_string_at(&payload, &[&["boolean"][..]]).as_deref(),
        Some("false")
    );
    assert_eq!(json_string_at(&payload, &[&["empty"][..]]), None);
    assert_eq!(json_string_at(&payload, &[&["object"][..]]), None);
    assert_eq!(
        json_value_at(&payload, &[&["object"][..]]),
        Some(json!({ "nested": true }))
    );

    let mut inserted = Map::new();
    insert_optional(&mut inserted, "present", Some("value"));
    insert_optional(&mut inserted, "absent", None);
    assert_eq!(inserted.get("present"), Some(&json!("value")));
    assert!(!inserted.contains_key("absent"));

    assert_eq!(
        merge_metadata(json!({ "a": 1 }), json!({ "b": 2, "c": null })),
        json!({ "a": 1, "b": 2 })
    );
    assert_eq!(
        merge_metadata(Value::Null, json!({ "a": 1 })),
        json!({ "a": 1 })
    );
    assert_eq!(
        merge_metadata(json!({ "a": 1 }), Value::Null),
        json!({ "a": 1 })
    );
    assert_eq!(
        merge_metadata(json!("left"), json!("right")),
        json!({ "metadata": "left", "extra_metadata": "right" })
    );
}

fn event_metadata(event: &NormalizedEvent) -> &Value {
    match event {
        NormalizedEvent::AgentStarted(event)
        | NormalizedEvent::AgentEnded(event)
        | NormalizedEvent::TurnEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => &event.metadata,
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            &event.metadata
        }
        NormalizedEvent::LlmHint(event) => &event.metadata,
        NormalizedEvent::LlmStarted(event) | NormalizedEvent::LlmEnded(event) => &event.metadata,
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => &event.metadata,
    }
}
