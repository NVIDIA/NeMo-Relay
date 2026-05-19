// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderValue;

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
        "x-nemo-flow-session-id",
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
        "x-nemo-flow-subagent-id",
        HeaderValue::from_static("worker-1"),
    );
    headers.insert(
        "x-nemo-flow-request-id",
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
            "x-nemo-flow-request-id",
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
        let was_agent_end = matches!(event, NormalizedEvent::AgentEnded(_));
        let (event, finished_alias) = route_event_through_alias(event, &aliases);
        assert_eq!(event.session_id(), "parent");
        assert_eq!(
            event_metadata(&event)["alias_metadata"],
            json!(true),
            "alias metadata should be stamped on {event:?}"
        );
        assert_eq!(finished_alias.as_deref(), was_agent_end.then_some("child"));

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
