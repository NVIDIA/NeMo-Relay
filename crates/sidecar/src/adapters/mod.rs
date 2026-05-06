// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod claude_code;
pub(crate) mod codex;
pub(crate) mod cursor;

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::config::header_string;
use crate::model::{AgentKind, NormalizedEvent, SessionEvent, SubagentEvent, ToolEvent};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AdapterOutcome {
    pub(crate) events: Vec<NormalizedEvent>,
    pub(crate) response: Value,
}

pub(super) struct ClassificationRules<'a> {
    kind: AgentKind,
    agent_start: &'a [&'a str],
    agent_end: &'a [&'a str],
    subagent_start: &'a [&'a str],
    subagent_end: &'a [&'a str],
    tool_start: &'a [&'a str],
    tool_end: &'a [&'a str],
}

fn session_id(payload: &Value, headers: &HeaderMap) -> String {
    header_string(headers, "x-nemo-flow-session-id")
        .or_else(|| header_string(headers, "x-claude-code-session-id"))
        .or_else(|| string_at(payload, &["session_id"]))
        .or_else(|| string_at(payload, &["sessionId"]))
        .or_else(|| string_at(payload, &["session", "id"]))
        .or_else(|| string_at(payload, &["conversation_id"]))
        .or_else(|| string_at(payload, &["conversationId"]))
        .unwrap_or_else(|| format!("hook-{}", Uuid::now_v7()))
}

fn event_name(payload: &Value) -> String {
    string_at(payload, &["hook_event_name"])
        .or_else(|| string_at(payload, &["event_name"]))
        .or_else(|| string_at(payload, &["eventName"]))
        .or_else(|| string_at(payload, &["event"]))
        .or_else(|| string_at(payload, &["type"]))
        .or_else(|| string_at(payload, &["name"]))
        .unwrap_or_else(|| "unknown".to_string())
}

fn metadata(payload: &Value, headers: &HeaderMap, kind: AgentKind, event_name: &str) -> Value {
    let mut object = Map::new();
    object.insert("agent_kind".into(), json!(kind.as_str()));
    object.insert("hook_event_name".into(), json!(event_name));
    if let Some(profile) = header_string(headers, "x-nemo-flow-config-profile") {
        object.insert("sidecar_config_profile".into(), json!(profile));
    }
    for (key, value) in [
        ("cwd", string_at(payload, &["cwd"])),
        ("transcript_path", string_at(payload, &["transcript_path"])),
        ("project_dir", string_at(payload, &["project_dir"])),
        ("user_email", string_at(payload, &["user_email"])),
        ("model", string_at(payload, &["model"])),
    ] {
        if let Some(value) = value {
            object.insert(key.into(), json!(value));
        }
    }
    Value::Object(object)
}

fn common_session_event(payload: &Value, headers: &HeaderMap, kind: AgentKind) -> SessionEvent {
    let event_name = event_name(payload);
    SessionEvent {
        session_id: session_id(payload, headers),
        agent_kind: kind,
        event_name: event_name.clone(),
        payload: payload.clone(),
        metadata: metadata(payload, headers, kind, &event_name),
    }
}

fn common_subagent_event(payload: &Value, headers: &HeaderMap, kind: AgentKind) -> SubagentEvent {
    let session = common_session_event(payload, headers, kind);
    let subagent_id = subagent_id(payload)
        .or_else(|| header_string(headers, "x-nemo-flow-subagent-id"))
        .unwrap_or_else(|| "subagent".to_string());
    SubagentEvent {
        session_id: session.session_id,
        agent_kind: kind,
        event_name: session.event_name,
        subagent_id,
        payload: session.payload,
        metadata: session.metadata,
    }
}

fn common_tool_event(payload: &Value, headers: &HeaderMap, kind: AgentKind) -> ToolEvent {
    let session = common_session_event(payload, headers, kind);
    let tool_call_id = string_at(payload, &["tool_call_id"])
        .or_else(|| string_at(payload, &["toolCallId"]))
        .or_else(|| string_at(payload, &["tool_use_id"]))
        .or_else(|| string_at(payload, &["call_id"]))
        .or_else(|| string_at(payload, &["tool", "id"]))
        .or_else(|| string_at(payload, &["tool_input", "id"]))
        .or_else(|| string_at(payload, &["id"]))
        .unwrap_or_else(|| format!("tool-{}", Uuid::now_v7()));
    let tool_name = string_at(payload, &["tool_name"])
        .or_else(|| string_at(payload, &["toolName"]))
        .or_else(|| string_at(payload, &["tool", "name"]))
        .or_else(|| string_at(payload, &["tool_input", "name"]))
        .or_else(|| string_at(payload, &["name"]))
        .unwrap_or_else(|| "unknown_tool".to_string());
    let arguments = value_at(payload, &["tool_input"])
        .or_else(|| value_at(payload, &["input"]))
        .or_else(|| value_at(payload, &["arguments"]))
        .or_else(|| value_at(payload, &["args"]))
        .unwrap_or(Value::Null);
    let result = value_at(payload, &["tool_output"])
        .or_else(|| value_at(payload, &["tool_response"]))
        .or_else(|| value_at(payload, &["output"]))
        .or_else(|| value_at(payload, &["result"]))
        .unwrap_or(Value::Null);
    let normalized_event = normalize_name(&session.event_name);
    ToolEvent {
        session_id: session.session_id,
        agent_kind: kind,
        event_name: session.event_name,
        tool_call_id,
        tool_name,
        subagent_id: subagent_id(payload)
            .or_else(|| header_string(headers, "x-nemo-flow-subagent-id")),
        arguments,
        result,
        status: string_at(payload, &["status"])
            .or_else(|| string_at(payload, &["decision"]))
            .or_else(|| string_at(payload, &["permission"]))
            .or_else(|| {
                (normalized_event.contains("failure") || normalized_event.contains("failed"))
                    .then_some("error".to_string())
            }),
        payload: session.payload,
        metadata: session.metadata,
    }
}

fn subagent_id(payload: &Value) -> Option<String> {
    string_at(payload, &["subagent_id"])
        .or_else(|| string_at(payload, &["subagentId"]))
        .or_else(|| string_at(payload, &["subagent", "id"]))
        .or_else(|| string_at(payload, &["agent", "id"]))
}

fn string_at(payload: &Value, path: &[&str]) -> Option<String> {
    value_at(payload, path).and_then(|value| match value {
        Value::String(value) => Some(value),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    })
}

fn value_at(payload: &Value, path: &[&str]) -> Option<Value> {
    let mut current = payload;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current.clone())
}

fn classify(
    payload: &Value,
    headers: &HeaderMap,
    rules: &ClassificationRules<'_>,
) -> NormalizedEvent {
    let event = event_name(payload);
    let normalized = normalize_name(&event);
    if rules
        .agent_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::AgentStarted(common_session_event(payload, headers, rules.kind))
    } else if rules
        .agent_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::AgentEnded(common_session_event(payload, headers, rules.kind))
    } else if rules
        .subagent_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::SubagentStarted(common_subagent_event(payload, headers, rules.kind))
    } else if rules
        .subagent_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::SubagentEnded(common_subagent_event(payload, headers, rules.kind))
    } else if rules
        .tool_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::ToolStarted(common_tool_event(payload, headers, rules.kind))
    } else if rules
        .tool_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::ToolEnded(common_tool_event(payload, headers, rules.kind))
    } else {
        match normalized.as_str() {
            "beforesubmitprompt" | "promptsubmitted" | "userpromptsubmit" => {
                NormalizedEvent::PromptSubmitted(common_session_event(payload, headers, rules.kind))
            }
            "afteragentresponse" | "agentresponse" | "assistantresponse" => {
                NormalizedEvent::AgentResponse(common_session_event(payload, headers, rules.kind))
            }
            "precompact" | "compaction" => {
                NormalizedEvent::Compaction(common_session_event(payload, headers, rules.kind))
            }
            "notification" => {
                NormalizedEvent::Notification(common_session_event(payload, headers, rules.kind))
            }
            _ => NormalizedEvent::HookMark(common_session_event(payload, headers, rules.kind)),
        }
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
#[path = "../../tests/coverage/adapters_tests.rs"]
mod tests;
