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

fn hermes_llm_event(session_id: &str, task_id: &str) -> NormalizedEvent {
    NormalizedEvent::LlmStarted(LlmEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Hermes,
        event_name: "pre_api_request".into(),
        api_call_id: format!("{session_id}:{task_id}:1"),
        provider: "custom".into(),
        model_name: Some("qwen".into()),
        request: json!({ "extra": { "task_id": task_id } }),
        response: Value::Null,
        metadata: json!({ "event_metadata": "pre_api_request" }),
    })
}

fn hermes_tool_event(task_id: &str, session_scope: Option<&str>) -> NormalizedEvent {
    let mut payload = json!({ "extra": { "task_id": task_id } });
    if let Some(session_scope) = session_scope {
        payload["extra"]["parent_session_id"] = json!(session_scope);
    }

    NormalizedEvent::ToolStarted(ToolEvent {
        session_id: task_id.into(),
        agent_kind: AgentKind::Hermes,
        event_name: "pre_tool_call".into(),
        tool_call_id: format!("{task_id}:tool-1"),
        tool_name: "read_file".into(),
        subagent_id: None,
        arguments: json!({ "path": "README.md" }),
        result: Value::Null,
        status: None,
        payload,
        metadata: json!({ "event_metadata": "pre_tool_call" }),
    })
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
fn hermes_task_session_routing_is_scoped_by_parent_session() {
    let mut state = SessionAlignmentState::default();

    state.route_event(hermes_llm_event("hermes-a", "task-1"));
    state.route_event(hermes_llm_event("hermes-b", "task-1"));

    let routed_a = state.route_event(hermes_tool_event("task-1", Some("hermes-a")));
    let NormalizedEvent::ToolStarted(routed_a) = routed_a else {
        panic!("expected routed Hermes tool event");
    };
    assert_eq!(routed_a.session_id, "hermes-a");
    assert_eq!(routed_a.metadata["hermes_task_id"], json!("task-1"));
    assert_eq!(routed_a.metadata["hermes_session_id"], json!("hermes-a"));

    let routed_b = state.route_event(hermes_tool_event("task-1", Some("hermes-b")));
    let NormalizedEvent::ToolStarted(routed_b) = routed_b else {
        panic!("expected routed Hermes tool event");
    };
    assert_eq!(routed_b.session_id, "hermes-b");
    assert_eq!(routed_b.metadata["hermes_task_id"], json!("task-1"));
    assert_eq!(routed_b.metadata["hermes_session_id"], json!("hermes-b"));
}

#[test]
fn hermes_task_session_routing_leaves_ambiguous_unscoped_task_event_unchanged() {
    let mut state = SessionAlignmentState::default();

    state.route_event(hermes_llm_event("hermes-a", "task-1"));
    state.route_event(hermes_llm_event("hermes-b", "task-1"));

    let routed = state.route_event(hermes_tool_event("task-1", None));
    let NormalizedEvent::ToolStarted(routed) = routed else {
        panic!("expected Hermes tool event");
    };
    assert_eq!(routed.session_id, "task-1");
    assert!(routed.metadata.get("hermes_session_id").is_none());
}

#[test]
fn gateway_session_id_uses_explicit_claude_then_codex_fallbacks() {
    let mut headers = HeaderMap::new();
    let codex_body = json!({
        "prompt_cache_key": "codex-thread",
        "client_metadata": {
            "x-codex-installation-id": "install-1",
            "session_id": "codex-session"
        },
        "session_id": "body-thread"
    });

    assert_eq!(
        gateway_session_id(&headers, &codex_body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("codex-session")
    );

    let legacy_codex_body = json!({
        "prompt_cache_key": "legacy-codex-thread",
        "client_metadata": { "x-codex-installation-id": "install-1" }
    });
    assert_eq!(
        gateway_session_id(
            &headers,
            &legacy_codex_body,
            GatewayRouteKind::OpenAiResponses
        )
        .as_deref(),
        Some("legacy-codex-thread")
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
fn gateway_session_id_accepts_openai_body_session_id_fallback() {
    let headers = HeaderMap::new();

    assert_eq!(
        gateway_session_id(
            &headers,
            &json!({ "session_id": " body-session " }),
            GatewayRouteKind::OpenAiChatCompletions,
        )
        .as_deref(),
        Some("body-session")
    );
    assert_eq!(
        gateway_session_id(
            &headers,
            &json!({ "session_id": "body-session" }),
            GatewayRouteKind::AnthropicMessages,
        ),
        None
    );
    assert_eq!(
        gateway_session_id(
            &headers,
            &json!({ "session_id": "" }),
            GatewayRouteKind::OpenAiChatCompletions,
        ),
        None
    );
    assert_eq!(
        gateway_session_id(
            &headers,
            &json!({ "session_id": 42 }),
            GatewayRouteKind::OpenAiResponses,
        ),
        None
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

    assert_eq!(
        gateway_subagent_id(&headers, &body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("worker-1")
    );
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
fn gateway_subagent_id_reads_codex_0142_client_metadata() {
    let body = json!({
        "client_metadata": {
            "x-codex-installation-id": "install-1",
            "x-openai-subagent": "collab_spawn",
            "session_id": "root-session",
            "thread_id": "child-thread"
        }
    });

    assert_eq!(
        gateway_subagent_id(&HeaderMap::new(), &body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("child-thread")
    );
    assert_eq!(
        gateway_subagent_id(
            &HeaderMap::new(),
            &body,
            GatewayRouteKind::OpenAiChatCompletions
        ),
        None
    );

    let internal = json!({
        "client_metadata": {
            "x-codex-installation-id": "install-1",
            "x-openai-subagent": "review",
            "thread_id": "review-thread"
        }
    });
    assert_eq!(
        gateway_subagent_id(
            &HeaderMap::new(),
            &internal,
            GatewayRouteKind::OpenAiResponses
        ),
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
        request_affinity_key("anthropic.messages", &request).as_deref(),
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
        request_affinity_key("openai.chat_completions", &request).as_deref(),
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
        request_affinity_key("anthropic.messages", &request).as_deref(),
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

    let key = request_affinity_key("anthropic.messages", &request).unwrap();
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
        request_affinity_key("anthropic.messages", &request).as_deref(),
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
        request_affinity_key("openai.chat_completions", &request).as_deref(),
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
        request_affinity_key("openai.responses", &responses_request).as_deref(),
        Some("Analyze the Node binding architecture.")
    );
    assert_eq!(
        request_affinity_key("openai.responses", &prompt_request).as_deref(),
        Some("Summarize the Go binding architecture.")
    );
}

#[test]
fn request_affinity_key_ignores_count_token_style_payloads() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!("// source text without a chat user task"),
    };

    assert_eq!(
        request_affinity_key("anthropic.count_tokens", &request),
        None
    );
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

    assert_eq!(
        request_affinity_key("anthropic.messages", &tool_result),
        None
    );
    assert_eq!(
        request_affinity_key("openai.responses", &sidecar_json),
        None
    );
    assert_eq!(
        request_affinity_key("openai.responses", &sidecar_json_array),
        None
    );
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
    assert_eq!(
        json_string_at(
            &payload,
            &[&["object"][..], &["empty"][..], &["string"][..]]
        )
        .as_deref(),
        Some("value")
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

#[test]
fn request_affinity_key_is_route_gated_not_content_gated() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                { "role": "user", "content": "Review the Rust CLI gateway alignment code." }
            ]
        }),
    };

    // The same task text that produces a key on the chat/messages routes is ignored on routes
    // whose extractors opt out of affinity keys entirely, and on unknown provider names.
    assert!(request_affinity_key("openai.chat_completions", &request).is_some());
    assert!(request_affinity_key("anthropic.messages", &request).is_some());
    assert_eq!(request_affinity_key("openai.models", &request), None);
    assert_eq!(
        request_affinity_key("anthropic.count_tokens", &request),
        None
    );
    assert_eq!(request_affinity_key("unknown.provider", &request), None);
}

#[test]
fn request_affinity_key_returns_none_when_route_task_fields_are_missing() {
    let responses_shape = LlmRequest {
        headers: Map::new(),
        content: json!({ "input": "Review the Rust CLI gateway alignment code." }),
    };
    let messages_shape = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                { "role": "user", "content": "Review the Rust CLI gateway alignment code." }
            ]
        }),
    };
    let assistant_only = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                { "role": "assistant", "content": "I will review the gateway alignment code." }
            ]
        }),
    };

    // Chat/messages extractors read `messages`; the responses extractor reads `input`/`prompt`.
    // A body shaped for the other route yields no key instead of borrowing its fields.
    assert!(request_affinity_key("openai.responses", &responses_shape).is_some());
    assert_eq!(
        request_affinity_key("openai.chat_completions", &responses_shape),
        None
    );
    assert_eq!(
        request_affinity_key("anthropic.messages", &responses_shape),
        None
    );
    assert_eq!(
        request_affinity_key("openai.responses", &messages_shape),
        None
    );
    assert_eq!(
        request_affinity_key("anthropic.messages", &assistant_only),
        None
    );
}

#[test]
fn request_affinity_key_enforces_min_and_max_task_text_length() {
    let request_with_text = |text: String| LlmRequest {
        headers: Map::new(),
        content: json!({ "messages": [{ "role": "user", "content": text }] }),
    };

    let below_min = request_with_text("a".repeat(REQUEST_AFFINITY_KEY_MIN_CHARS - 1));
    let at_min = request_with_text("a".repeat(REQUEST_AFFINITY_KEY_MIN_CHARS));
    let above_max = request_with_text("a".repeat(REQUEST_AFFINITY_KEY_MAX_CHARS + 100));

    assert_eq!(request_affinity_key("anthropic.messages", &below_min), None);
    assert_eq!(
        request_affinity_key("anthropic.messages", &at_min).map(|key| key.chars().count()),
        Some(REQUEST_AFFINITY_KEY_MIN_CHARS)
    );
    assert_eq!(
        request_affinity_key("anthropic.messages", &above_max).map(|key| key.chars().count()),
        Some(REQUEST_AFFINITY_KEY_MAX_CHARS)
    );
}

#[test]
fn gateway_turn_input_builds_claude_prompt_for_anthropic_messages_only() {
    // Shorter than the affinity-key minimum on purpose: turn input has no length gate.
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [
                { "role": "user", "content": "Fix the CLI tests." }
            ]
        }),
    };

    assert_eq!(
        gateway_turn_input(AgentKind::ClaudeCode, "anthropic.messages", &request),
        Some(json!({ "prompt": "Fix the CLI tests." }))
    );

    // Only Claude installed mode can race the UserPromptSubmit hook, so every other agent kind
    // and route stays None.
    for agent_kind in [AgentKind::Codex, AgentKind::Hermes, AgentKind::Gateway] {
        assert_eq!(
            gateway_turn_input(agent_kind, "anthropic.messages", &request),
            None
        );
    }
    for provider in [
        "openai.responses",
        "openai.chat_completions",
        "openai.models",
        "anthropic.count_tokens",
    ] {
        assert_eq!(
            gateway_turn_input(AgentKind::ClaudeCode, provider, &request),
            None
        );
    }

    let no_messages = LlmRequest {
        headers: Map::new(),
        content: json!({ "input": "Fix the CLI tests." }),
    };
    assert_eq!(
        gateway_turn_input(AgentKind::ClaudeCode, "anthropic.messages", &no_messages),
        None
    );
}

#[test]
fn gateway_session_id_ignores_body_ids_on_header_only_routes() {
    // Body fallbacks that succeed on the OpenAI Responses/Chat Completions routes must be
    // ignored on the header-only routes.
    let body = json!({
        "session_id": "body-session",
        "prompt_cache_key": "codex-thread",
        "client_metadata": { "x-codex-installation-id": "install-1" }
    });

    for route in [
        GatewayRouteKind::AnthropicMessages,
        GatewayRouteKind::AnthropicCountTokens,
        GatewayRouteKind::OpenAiModels,
    ] {
        assert_eq!(
            gateway_session_id(&HeaderMap::new(), &body, route),
            None,
            "route {route:?} must not read body session ids"
        );

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-claude-code-session-id",
            HeaderValue::from_static("claude-thread"),
        );
        assert_eq!(
            gateway_session_id(&headers, &body, route).as_deref(),
            Some("claude-thread")
        );

        headers.insert(
            "x-nemo-relay-session-id",
            HeaderValue::from_static("explicit-thread"),
        );
        assert_eq!(
            gateway_session_id(&headers, &body, route).as_deref(),
            Some("explicit-thread")
        );
    }
}

#[test]
fn gateway_route_names_round_trip_through_provider_lookup() {
    for route in GatewayRouteKind::ALL {
        assert_eq!(
            GatewayRouteKind::from_provider_name(route.name()),
            Some(route)
        );
    }
    assert_eq!(GatewayRouteKind::from_provider_name("openai.unknown"), None);
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
