// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};

use crate::adapters::{
    AdapterOutcome, ClassificationRules, classify, common_session_event, event_name, metadata,
    normalize_name, session_id, value_at,
};
use crate::model::{AgentKind, LlmEvent, NormalizedEvent};

/// Normalizes Hermes shell hook payloads without emitting control directives.
///
/// Hermes hooks are installed as shell commands and may run outside `run`, so this adapter keeps
/// responses minimal and relies on the forwarder fail-open/fail-closed setting to decide whether
/// hook delivery problems affect the invoking agent.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let event_name = event_name(&payload);
    let normalized = normalize_name(&event_name);
    if normalized == "preapirequest" {
        return AdapterOutcome {
            events: vec![crate::model::NormalizedEvent::LlmStarted(hermes_llm_event(
                &payload,
                headers,
                &event_name,
            ))],
            response: json!({}),
        };
    }
    if normalized == "postapirequest" {
        return AdapterOutcome {
            events: vec![crate::model::NormalizedEvent::LlmEnded(hermes_llm_event(
                &payload,
                headers,
                &event_name,
            ))],
            response: json!({}),
        };
    }

    let mut events = classify(
        &payload,
        headers,
        &ClassificationRules {
            kind: AgentKind::Hermes,
            agent_start: &["on_session_start", "sessionStart"],
            agent_end: &["on_session_finalize", "on_session_reset"],
            subagent_start: &["subagent_start", "subagentStart"],
            subagent_end: &["subagent_stop", "subagentStop"],
            tool_start: &["pre_tool_call", "preToolCall"],
            tool_end: &["post_tool_call", "postToolCall"],
        },
    );
    // hermes-agent fires `on_session_end` at every user-turn boundary (it is intentionally distinct
    // from `on_session_finalize`, which marks the real session close). Emit a `TurnEnded` alongside
    // the HookMark so the session manager snapshots ATIF per turn — without this, sessions that
    // never reach `on_session_finalize` (e.g., terminated via Ctrl+D before hermes-agent finalizes)
    // leave their ATIF un-flushed.
    if normalized == "onsessionend" {
        events.push(NormalizedEvent::TurnEnded(common_session_event(
            &payload,
            headers,
            AgentKind::Hermes,
        )));
    }
    AdapterOutcome {
        events,
        response: json!({}),
    }
}

fn hermes_llm_event(payload: &Value, headers: &HeaderMap, event_name: &str) -> LlmEvent {
    let session_id = session_id(payload, headers);
    let api_call_id = hermes_api_call_id(payload, &session_id);
    let provider = hermes_string_at(payload, "provider")
        .or_else(|| hermes_string_at(payload, "api_mode"))
        .unwrap_or_else(|| "hermes_api_request".to_string());
    let model_name =
        hermes_string_at(payload, "response_model").or_else(|| hermes_string_at(payload, "model"));
    let mut event_metadata = metadata(payload, headers, AgentKind::Hermes, event_name);
    if let Value::Object(ref mut object) = event_metadata {
        object.insert("api_call_id".into(), json!(api_call_id.clone()));
        object.insert("provider_payload_exact".into(), json!(false));
        object.insert("fidelity_source".into(), json!("hermes_api_hooks"));
    }
    LlmEvent {
        session_id,
        agent_kind: AgentKind::Hermes,
        event_name: event_name.to_string(),
        api_call_id,
        provider,
        model_name,
        request: hermes_llm_request(payload),
        response: hermes_llm_response(payload),
        metadata: event_metadata,
    }
}

fn hermes_api_call_id(payload: &Value, session_id: &str) -> String {
    let task_id = hermes_string_at(payload, "task_id").unwrap_or_default();
    let api_call_count = hermes_string_at(payload, "api_call_count").unwrap_or_default();
    format!("{session_id}:{task_id}:{api_call_count}")
}

fn hermes_llm_request(payload: &Value) -> Value {
    let mut object = Map::new();
    for key in [
        "task_id",
        "session_id",
        "platform",
        "model",
        "provider",
        "base_url",
        "api_mode",
        "api_call_count",
        "message_count",
        "tool_count",
        "approx_input_tokens",
        "request_char_count",
        "max_tokens",
    ] {
        if let Some(value) = hermes_value_at(payload, key) {
            object.insert(key.into(), value);
        }
    }
    object.insert(
        "fidelity".into(),
        json!({
            "provider_payload_exact": false,
            "source": "hermes_pre_api_request"
        }),
    );
    Value::Object(object)
}

fn hermes_llm_response(payload: &Value) -> Value {
    let mut object = Map::new();
    for key in [
        "task_id",
        "session_id",
        "platform",
        "model",
        "provider",
        "base_url",
        "api_mode",
        "api_call_count",
        "api_duration",
        "finish_reason",
        "message_count",
        "response_model",
        "usage",
        "assistant_content_chars",
        "assistant_tool_call_count",
    ] {
        if let Some(value) = hermes_value_at(payload, key) {
            object.insert(key.into(), value);
        }
    }
    Value::Object(object)
}

fn hermes_string_at(payload: &Value, key: &str) -> Option<String> {
    value_at(payload, &[key])
        .or_else(|| value_at(payload, &["extra", key]))
        .and_then(|value| match value {
            Value::String(value) => Some(value),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            _ => None,
        })
        .filter(|value| !value.is_empty())
}

fn hermes_value_at(payload: &Value, key: &str) -> Option<Value> {
    value_at(payload, &[key]).or_else(|| value_at(payload, &["extra", key]))
}
