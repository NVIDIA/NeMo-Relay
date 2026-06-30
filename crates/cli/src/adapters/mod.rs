// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod claude_code;
pub(crate) mod codex;
pub(crate) mod hermes;

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::config::header_string;
use crate::model::{
    AgentKind, LlmHintEvent, NormalizedEvent, SessionEvent, SubagentEvent, ToolEvent,
};

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

// Derives a stable session identifier from gateway headers first, then common agent payload
// fields, and finally a v7 UUID. Header precedence lets gateway and hook-forward callers
// correlate events even when agent payload schemas omit or rename their native session field.
fn session_id(payload: &Value, headers: &HeaderMap) -> String {
    header_string(headers, "x-nemo-relay-session-id")
        .or_else(|| header_string(headers, "x-claude-code-session-id"))
        .or_else(|| session_id_from_payload(payload))
        .unwrap_or_else(|| format!("hook-{}", Uuid::now_v7()))
}

// Reads the first known session identifier payload path. Keeping the path list in one place makes
// adapter precedence explicit without nesting a long `or_else` chain in `session_id`.
fn session_id_from_payload(payload: &Value) -> Option<String> {
    [
        &["session_id"][..],
        &["sessionId"],
        &["session", "id"],
        &["conversation_id"],
        &["conversationId"],
        &["parent_session_id"],
        &["task_id"],
        &["extra", "session_id"],
        &["extra", "task_id"],
    ]
    .into_iter()
    .find_map(|path| string_at(payload, path))
}

// Reads the agent's event name from the known hook fields in order and falls back to `unknown`.
// This deliberately keeps unknown payloads observable instead of rejecting them at the adapter
// boundary, allowing the session layer to emit a generic mark event.
fn event_name(payload: &Value) -> String {
    string_at(payload, &["hook_event_name"])
        .or_else(|| string_at(payload, &["event_name"]))
        .or_else(|| string_at(payload, &["eventName"]))
        .or_else(|| string_at(payload, &["event"]))
        .or_else(|| string_at(payload, &["type"]))
        .or_else(|| string_at(payload, &["name"]))
        .or_else(|| string_at(payload, &["extra", "hook_event_name"]))
        .or_else(|| string_at(payload, &["extra", "event_name"]))
        .or_else(|| string_at(payload, &["extra", "eventName"]))
        .or_else(|| string_at(payload, &["extra", "event"]))
        .or_else(|| string_at(payload, &["extra", "type"]))
        .or_else(|| string_at(payload, &["extra", "name"]))
        .unwrap_or_else(|| "unknown".to_string())
}

// Builds shared metadata for every normalized hook event. Only stable, low-cardinality fields and
// gateway configuration hints are lifted out; the full payload remains on the event for consumers
// that need agent-specific detail.
fn metadata(payload: &Value, headers: &HeaderMap, kind: AgentKind, event_name: &str) -> Value {
    let mut object = Map::new();
    object.insert("agent_kind".into(), json!(kind.as_str()));
    object.insert("hook_event_name".into(), json!(event_name));
    if let Some(profile) = header_string(headers, "x-nemo-relay-config-profile") {
        object.insert("gateway_config_profile".into(), json!(profile));
    }
    for (key, value) in [
        ("cwd", string_at(payload, &["cwd"])),
        ("transcript_path", string_at(payload, &["transcript_path"])),
        ("project_dir", string_at(payload, &["project_dir"])),
        ("user_email", string_at(payload, &["user_email"])),
        ("model", string_at(payload, &["model"])),
        ("agent_id", string_at(payload, &["agent_id"])),
        ("agent_type", string_at(payload, &["agent_type"])),
    ] {
        if let Some(value) = value {
            object.insert(key.into(), json!(value));
        }
    }
    Value::Object(object)
}

// Creates a root session event using the common session-id and metadata extraction rules so
// lifecycle, marks, notifications, and compaction events all carry identical correlation fields.
pub(crate) fn common_session_event(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
) -> SessionEvent {
    let event_name = event_name(payload);
    SessionEvent {
        session_id: session_id(payload, headers),
        agent_kind: kind,
        event_name: event_name.clone(),
        payload: payload.clone(),
        metadata: metadata(payload, headers, kind, &event_name),
    }
}

// Creates a subagent event and tolerates sparse agent payloads by using the gateway subagent
// header and then a synthetic `subagent` id. The fallback keeps unmatched start/end events visible
// rather than dropping them when an integration lacks explicit nested-agent IDs.
fn common_subagent_event(payload: &Value, headers: &HeaderMap, kind: AgentKind) -> SubagentEvent {
    let session = common_session_event(payload, headers, kind);
    let subagent_id = subagent_id(payload)
        .or_else(|| header_string(headers, "x-nemo-relay-subagent-id"))
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

// Captures hook payloads that can help correlate nearby gateway LLM calls to the right agent or
// subagent. Multiple naming conventions are accepted because integrations expose conversation,
// generation, request, and model identifiers under different shapes.
fn common_llm_hint_event(payload: &Value, headers: &HeaderMap, kind: AgentKind) -> LlmHintEvent {
    let session = common_session_event(payload, headers, kind);
    LlmHintEvent {
        session_id: session.session_id,
        agent_kind: kind,
        event_name: session.event_name,
        subagent_id: hook_subagent_id(payload, headers),
        agent_id: first_string_at(payload, &[&["agent_id"][..], &["agent", "id"][..]]),
        agent_type: first_string_at(
            payload,
            &[
                &["agent_type"][..],
                &["agent", "type"][..],
                &["agent", "name"][..],
            ],
        ),
        conversation_id: first_string_at(
            payload,
            &[
                &["conversation_id"][..],
                &["conversationId"][..],
                &["conversation", "id"][..],
            ],
        ),
        generation_id: first_string_at(
            payload,
            &[
                &["generation_id"][..],
                &["generationId"][..],
                &["generation", "id"][..],
            ],
        ),
        request_id: first_string_at(
            payload,
            &[
                &["request_id"][..],
                &["requestId"][..],
                &["request", "id"][..],
                &["extra", "request_id"][..],
            ],
        ),
        model: first_string_at(
            payload,
            &[&["model"][..], &["model_name"][..], &["modelName"][..]],
        ),
        payload: session.payload,
        metadata: session.metadata,
    }
}

// Converts agent tool hooks into the runtime tool event shape while preserving missing fields.
// Tool IDs and names are synthesized when absent, arguments/results are searched across known
// payload shapes, and failure or permission-denied event names are reflected in status metadata.
fn common_tool_event(payload: &Value, headers: &HeaderMap, kind: AgentKind) -> ToolEvent {
    let session = common_session_event(payload, headers, kind);
    let normalized_event = normalize_name(&session.event_name);
    ToolEvent {
        session_id: session.session_id,
        agent_kind: kind,
        event_name: session.event_name,
        tool_call_id: tool_call_id(payload),
        tool_name: tool_name(payload),
        subagent_id: hook_subagent_id(payload, headers),
        arguments: tool_arguments(payload),
        result: tool_result(payload, &normalized_event),
        status: tool_status(payload, &normalized_event),
        payload: session.payload,
        metadata: session.metadata,
    }
}

// Looks up the first string across a list of payload paths. Keeping this fallback mechanic in one
// helper makes event-specific extraction code read as schema precedence rather than control flow.
fn first_string_at(payload: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| string_at(payload, path))
}

// Resolves a subagent id from payload shape first and the gateway header second. The payload wins
// because it is the agent's native ownership signal; the header exists for gateway correlation and
// sparse hook systems.
fn hook_subagent_id(payload: &Value, headers: &HeaderMap) -> Option<String> {
    subagent_id(payload).or_else(|| header_string(headers, "x-nemo-relay-subagent-id"))
}

// Resolves a tool call identifier from all known agent payload conventions before synthesizing a
// UUID-backed id. The synthetic id keeps lifecycle events recordable even when hooks omit IDs.
fn tool_call_id(payload: &Value) -> String {
    first_string_at(
        payload,
        &[
            &["tool_call_id"][..],
            &["toolCallId"][..],
            &["tool_use_id"][..],
            &["call_id"][..],
            &["extra", "tool_call_id"][..],
            &["extra", "call_id"][..],
            &["tool", "id"][..],
            &["tool_input", "id"][..],
            &["id"][..],
        ],
    )
    .unwrap_or_else(|| format!("tool-{}", Uuid::now_v7()))
}

// Resolves a human-readable tool name from the common top-level, nested tool, and tool-input
// shapes. Missing names are kept explicit as `unknown_tool` rather than inheriting event names.
fn tool_name(payload: &Value) -> String {
    first_string_at(
        payload,
        &[
            &["tool_name"][..],
            &["toolName"][..],
            &["tool", "name"][..],
            &["tool_input", "name"][..],
            &["name"][..],
        ],
    )
    .unwrap_or_else(|| "unknown_tool".to_string())
}

// Extracts tool input from the agent-specific fields that represent call arguments. A missing
// argument payload remains JSON null so downstream consumers can distinguish it from `{}`.
fn tool_arguments(payload: &Value) -> Value {
    value_at(payload, &["tool_input"])
        .or_else(|| value_at(payload, &["input"]))
        .or_else(|| value_at(payload, &["arguments"]))
        .or_else(|| value_at(payload, &["args"]))
        .unwrap_or(Value::Null)
}

// Extracts tool output from success payloads first and then failure diagnostics. Failure detail
// synthesis is last so an explicit result always wins over gateway-built diagnostic metadata.
fn tool_result(payload: &Value, normalized_event: &str) -> Value {
    value_at(payload, &["tool_output"])
        .or_else(|| value_at(payload, &["tool_response"]))
        .or_else(|| value_at(payload, &["output"]))
        .or_else(|| value_at(payload, &["result"]))
        .or_else(|| value_at(payload, &["extra", "tool_output"]))
        .or_else(|| value_at(payload, &["extra", "result"]))
        .or_else(|| event_detail_result(payload, normalized_event))
        .unwrap_or(Value::Null)
}

// Resolves explicit status fields before deriving error/denied status from event names. Derived
// status is intentionally conservative and only covers known failure or permission-denial spellings.
fn tool_status(payload: &Value, normalized_event: &str) -> Option<String> {
    first_string_at(
        payload,
        &[&["status"][..], &["decision"][..], &["permission"][..]],
    )
    .or_else(|| {
        (normalized_event.contains("failure") || normalized_event.contains("failed"))
            .then_some("error".to_string())
    })
    .or_else(|| {
        normalized_event
            .contains("permissiondenied")
            .then_some("denied".to_string())
    })
}

// Finds the most specific nested-agent identifier the gateway knows how to interpret. Agent IDs
// are accepted as subagent IDs because several hook systems use `agent` terminology for spawned
// workers rather than for the top-level coding agent.
fn subagent_id(payload: &Value) -> Option<String> {
    string_at(payload, &["subagent_id"])
        .or_else(|| string_at(payload, &["subagentId"]))
        .or_else(|| string_at(payload, &["child_subagent_id"]))
        .or_else(|| string_at(payload, &["childSubagentId"]))
        .or_else(|| string_at(payload, &["agent_id"]))
        .or_else(|| string_at(payload, &["subagent", "id"]))
        .or_else(|| string_at(payload, &["agent", "id"]))
        .or_else(|| string_at(payload, &["extra", "subagent_id"]))
        .or_else(|| string_at(payload, &["extra", "subagentId"]))
        .or_else(|| string_at(payload, &["extra", "child_subagent_id"]))
        .or_else(|| string_at(payload, &["extra", "childSubagentId"]))
        .or_else(|| string_at(payload, &["extra", "agent_id"]))
        .or_else(|| string_at(payload, &["extra", "subagent", "id"]))
        .or_else(|| string_at(payload, &["extra", "agent", "id"]))
}

// Extracts detail fields as a synthetic tool result only for failure-like hooks. Successful tool
// events without explicit output remain `null` so observers can distinguish "no output supplied"
// from "the gateway assembled diagnostic details".
fn event_detail_result(payload: &Value, normalized_event: &str) -> Option<Value> {
    let include_details = normalized_event.contains("failure")
        || normalized_event.contains("failed")
        || normalized_event.contains("permissiondenied");
    if !include_details {
        return None;
    }

    let mut object = Map::new();
    for key in ["error", "reason", "is_interrupt", "duration_ms"] {
        if let Some(value) = value_at(payload, &[key]) {
            object.insert(key.into(), value);
        }
    }
    (!object.is_empty()).then_some(Value::Object(object))
}

// Reads a nested value as a string, accepting numbers and booleans for agent schemas that encode
// identifiers or flags without string types. Empty strings are treated as absent to preserve
// fallback ordering.
fn string_at(payload: &Value, path: &[&str]) -> Option<String> {
    value_at(payload, path)
        .and_then(|value| match value {
            Value::String(value) => Some(value),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            _ => None,
        })
        .filter(|value| !value.is_empty())
}

// Returns a cloned nested JSON value using exact object-key traversal. Missing intermediate keys
// stop the lookup without error so callers can chain schema fallbacks cheaply.
fn value_at(payload: &Value, path: &[&str]) -> Option<Value> {
    let mut current = payload;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current.clone())
}

// Classifies a raw hook event into one or more normalized events.
//
// Most hook events produce a single normalized event from `classify_primary`. The exception is
// `Stop` (Claude/Codex): it emits both the existing `LlmHint` (preserving correlation for
// subsequent LLM calls) AND a `TurnEnded` so the session manager can snapshot ATIF without
// closing the agent scope. Codex 0.129 has no `SessionEnd`-equivalent hook — without this dual
// emission, codex transparent runs would never trigger an ATIF write.
//
// If the primary event is already terminal, the snapshot is skipped to avoid double-writing —
// `flush_observers` already writes ATIF on agent-end, and a follow-up `TurnEnded` on a removed
// session would recreate an empty session and overwrite the freshly-written ATIF.
fn classify(
    payload: &Value,
    headers: &HeaderMap,
    rules: &ClassificationRules<'_>,
) -> Vec<NormalizedEvent> {
    let normalized = normalize_name(&event_name(payload));
    if matches!(
        normalized.as_str(),
        "beforesubmitprompt" | "promptsubmitted" | "userpromptsubmit"
    ) {
        return vec![
            NormalizedEvent::PromptSubmitted(common_session_event(payload, headers, rules.kind)),
            NormalizedEvent::LlmHint(common_llm_hint_event(payload, headers, rules.kind)),
        ];
    }
    let primary = classify_primary(payload, headers, rules);
    if normalized == "stop" && !primary.is_terminal() {
        return vec![
            primary,
            NormalizedEvent::TurnEnded(common_session_event(payload, headers, rules.kind)),
        ];
    }
    vec![primary]
}

// Classifies a raw hook event using adapter-specific lifecycle names first and generic gateway
// names second. Unknown events are intentionally converted to hook marks, not errors, so new agent
// hook types remain observable until first-class normalization rules are added.
fn classify_primary(
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
            "afteragentresponse" | "agentresponse" | "assistantresponse" | "afteragentthought"
            | "prellmcall" | "postllmcall" | "stop" => {
                NormalizedEvent::LlmHint(common_llm_hint_event(payload, headers, rules.kind))
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

// Removes separators and case differences before comparing hook names. The gateway uses this for
// agent-specific aliases so `PostToolUse`, `post_tool_use`, and `postToolUse` converge.
fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
#[path = "../../tests/coverage/adapters_tests.rs"]
mod tests;
