// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};

use crate::adapters::{
    AdapterOutcome, ClassificationRules, HERMES_PAYLOAD_EXTRACTOR, classify, common_session_event,
    event_name, metadata, normalize_name, session_id,
};
use crate::json_path::value_at;
use crate::model::{AgentKind, LlmEvent, NormalizedEvent};

/// Normalizes Hermes shell hook payloads without emitting control directives.
///
/// Hermes hooks are installed as shell commands and may run outside `run`, so this adapter keeps
/// responses minimal and relies on the forwarder fail-open/fail-closed setting to decide whether
/// hook delivery problems affect the invoking agent.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let event_name = event_name(&payload, &HERMES_PAYLOAD_EXTRACTOR);
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
    if normalized == "apirequesterror" {
        return AdapterOutcome {
            events: vec![crate::model::NormalizedEvent::LlmEnded(hermes_llm_event(
                &payload,
                headers,
                &event_name,
            ))],
            response: json!({}),
        };
    }
    if normalized == "pretoolcall" && !hermes_pre_tool_call_is_correlatable(&payload, headers) {
        return AdapterOutcome {
            events: Vec::new(),
            response: json!({}),
        };
    }

    // `on_session_end` is a Hermes per-turn boundary, not user-visible trajectory content.
    // Emitting it as both HookMark and TurnEnded polluted ATIF with system rows whose only purpose
    // was to trigger a snapshot. Keep the snapshot signal and leave the agent scope open.
    if normalized == "onsessionend" {
        return AdapterOutcome {
            events: vec![NormalizedEvent::TurnEnded(common_session_event(
                &payload,
                headers,
                AgentKind::Hermes,
                &HERMES_PAYLOAD_EXTRACTOR,
            ))],
            response: json!({}),
        };
    }

    let events = classify(
        &payload,
        headers,
        &HERMES_PAYLOAD_EXTRACTOR,
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
    AdapterOutcome {
        events,
        response: json!({}),
    }
}

fn hermes_llm_event(payload: &Value, headers: &HeaderMap, event_name: &str) -> LlmEvent {
    let session_id = session_id(payload, headers, &HERMES_PAYLOAD_EXTRACTOR);
    let api_call_id = hermes_api_call_id(payload, &session_id);
    let provider = hermes_string_at(payload, "provider")
        .or_else(|| hermes_string_at(payload, "api_mode"))
        .unwrap_or_else(|| "hermes_api_request".to_string());
    let model_name =
        hermes_string_at(payload, "response_model").or_else(|| hermes_string_at(payload, "model"));
    let payload_exact = hermes_payload_exact(payload, event_name);
    let mut event_metadata = metadata(
        payload,
        headers,
        AgentKind::Hermes,
        event_name,
        &HERMES_PAYLOAD_EXTRACTOR,
    );
    if let Value::Object(ref mut object) = event_metadata {
        object.insert("api_call_id".into(), json!(api_call_id.clone()));
        object.insert("provider_payload_exact".into(), json!(payload_exact));
        object.insert(
            "fidelity_source".into(),
            json!(if payload_exact {
                "hermes_api_hooks_sanitized"
            } else {
                "hermes_api_hooks"
            }),
        );
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
    // Newer Hermes request-scoped hooks emit a stable per-attempt ID. Prefer it so pre, post,
    // error, tool, and approval telemetry can join without depending on turn-local counters.
    // Older Hermes payloads do not have it, so keep the synthesized ID for compatibility.
    if let Some(api_request_id) = hermes_string_at(payload, "api_request_id") {
        return api_request_id;
    }
    let task_id = hermes_string_at(payload, "task_id").unwrap_or_default();
    let api_call_count = hermes_string_at(payload, "api_call_count").unwrap_or_default();
    format!("{session_id}:{task_id}:{api_call_count}")
}

fn hermes_llm_request(payload: &Value) -> Value {
    // Prefer first-party sanitized request bodies from newer Hermes telemetry hooks. This is still
    // observer-only data: NeMo Relay is not intercepting or rewriting Hermes execution here. When the
    // exact payload is absent or was truncated by Hermes, fall back to the legacy summary shape.
    if let Some(request) = hermes_exact_request(payload) {
        return request;
    }
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
    // Prefer first-party sanitized response bodies from newer Hermes telemetry hooks. Older Hermes
    // versions only send summary fields, which remain useful for latency/token accounting but not
    // full ATIF reconstruction.
    if let Some(response) = hermes_exact_response(payload) {
        return response;
    }
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
        "status_code",
        "retry_count",
        "max_retries",
        "retryable",
        "reason",
        "error",
    ] {
        if let Some(value) = hermes_value_at(payload, key) {
            object.insert(key.into(), value);
        }
    }
    Value::Object(object)
}

fn hermes_payload_exact(payload: &Value, event_name: &str) -> bool {
    // The fallback is automatic and per-event: exact sanitized hook payloads get marked as
    // provider_payload_exact=true, while missing/truncated payloads retain the lossy summary marker.
    // Consumers can inspect these metadata fields to decide whether the trace is reconstruction
    // grade or summary-only.
    //
    // Follow-up: once the Hermes middleware branch that emits sanitized request/response hook
    // payloads is available in the smoke environment, rerun the Hermes Harbor smoke against that
    // version to validate the exact hook-telemetry path end to end. Until then, the smoke mainly
    // exercises the legacy summary fallback.
    match normalize_name(event_name).as_str() {
        "preapirequest" => hermes_exact_request(payload).is_some(),
        "postapirequest" => hermes_exact_response(payload).is_some(),
        _ => false,
    }
}

fn hermes_pre_tool_call_is_correlatable(payload: &Value, headers: &HeaderMap) -> bool {
    // Public Hermes releases can emit `pre_tool_call` with only a turn/task id. Treating that
    // `task_id` as a session opens a synthetic session that is later closed as `gateway_shutdown`.
    // Keep pre-tool spans only when they can be routed to a real session and paired with a stable
    // tool call id. The matching `post_tool_call` still records the tool result.
    has_explicit_hermes_session_id(payload, headers) && has_explicit_hermes_tool_call_id(payload)
}

fn has_explicit_hermes_session_id(payload: &Value, headers: &HeaderMap) -> bool {
    header_has_value(headers, "x-nemo-relay-session-id")
        || header_has_value(headers, "x-claude-code-session-id")
        || hermes_string_at(payload, "session_id").is_some()
        || hermes_string_at(payload, "sessionId").is_some()
        || value_at(payload, &["session", "id"]).is_some()
        || hermes_string_at(payload, "conversation_id").is_some()
        || hermes_string_at(payload, "conversationId").is_some()
        || hermes_string_at(payload, "parent_session_id").is_some()
}

fn has_explicit_hermes_tool_call_id(payload: &Value) -> bool {
    hermes_string_at(payload, "tool_call_id").is_some()
        || hermes_string_at(payload, "toolCallId").is_some()
        || hermes_string_at(payload, "tool_use_id").is_some()
        || hermes_string_at(payload, "call_id").is_some()
        || value_at(payload, &["tool", "id"]).is_some()
        || value_at(payload, &["tool_input", "id"]).is_some()
        || hermes_string_at(payload, "id").is_some()
}

fn header_has_value(headers: &HeaderMap, name: &str) -> bool {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| !value.trim().is_empty())
}

fn hermes_exact_request(payload: &Value) -> Option<Value> {
    let request = hermes_value_at(payload, "request")?;
    // Hermes bounds hook payload size before invoking plugins. A truncated payload is intentionally
    // not treated as exact, because ATIF/ATOF reconstruction would otherwise trust partial context.
    if request.is_null() || is_truncated_payload(&request) {
        return None;
    }
    request
        .get("body")
        .filter(|body| !body.is_null())
        .cloned()
        .or(Some(request))
}

fn hermes_exact_response(payload: &Value) -> Option<Value> {
    let response = hermes_value_at(payload, "response")?;
    // Same rule as requests: truncated response telemetry is useful as a diagnostic, but it is not
    // exact provider payload evidence.
    if is_truncated_payload(&response) {
        return None;
    }
    if let Some(raw_response) = response
        .get("raw_response")
        .filter(|raw_response| !raw_response.is_null() && !is_truncated_payload(raw_response))
    {
        return Some(raw_response.clone());
    }
    if response.get("choices").is_some()
        || response.get("output").is_some()
        || response.get("content").is_some()
    {
        return Some(response);
    }
    let assistant_message = response.get("assistant_message")?;
    let mut object = Map::new();
    if let Some(content) = assistant_message.get("content") {
        object.insert("content".into(), content.clone());
    }
    if let Some(tool_calls) = assistant_message.get("tool_calls") {
        object.insert("tool_calls".into(), tool_calls.clone());
    }
    if let Some(usage) = response
        .get("usage")
        .cloned()
        .or_else(|| hermes_value_at(payload, "usage"))
    {
        object.insert("usage".into(), usage);
    }
    for key in ["model", "finish_reason"] {
        if let Some(value) = response
            .get(key)
            .cloned()
            .or_else(|| hermes_value_at(payload, key))
        {
            object.insert(key.into(), value);
        }
    }
    (!object.is_empty()).then_some(Value::Object(object))
}

fn is_truncated_payload(value: &Value) -> bool {
    value
        .get("_truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
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
