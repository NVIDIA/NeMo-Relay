// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM/tool hint scoring and gateway ownership correlation.

use super::*;

pub(super) fn hint_match_score(hint: &LlmHintEvent, start: &LlmGatewayStart) -> u8 {
    let mut score = 0;
    if same_optional(hint.subagent_id.as_deref(), start.subagent_id.as_deref())
        || same_optional(hint.agent_id.as_deref(), start.subagent_id.as_deref())
    {
        score += 8;
    }
    if same_optional(
        hint.conversation_id.as_deref(),
        start.conversation_id.as_deref(),
    ) {
        score += 4;
    }
    if same_optional(
        hint.generation_id.as_deref(),
        start.generation_id.as_deref(),
    ) {
        score += 4;
    }
    if same_optional(hint.request_id.as_deref(), start.request_id.as_deref()) {
        score += 4;
    }
    if same_optional(hint.model.as_deref(), start.model_name.as_deref()) {
        score += 1;
    }
    score
}

// Extracts tool-call hints from common provider response shapes. These private hints let later
// hook-only tool events attach to the subagent that received the LLM response proposing the tool.
pub(super) fn tool_hints_from_llm_response(
    response: &Value,
    owner_subagent_id: Option<String>,
) -> Vec<ToolHint> {
    let mut hints = Vec::new();
    collect_openai_chat_tool_hints(response, owner_subagent_id.as_deref(), &mut hints);
    collect_openai_response_tool_hints(response, owner_subagent_id.as_deref(), &mut hints);
    collect_anthropic_tool_hints(response, owner_subagent_id.as_deref(), &mut hints);
    hints
}

// Collects OpenAI Chat Completions `choices[].message.tool_calls[]` entries and preserves
// stringified function arguments as parsed JSON when possible.
pub(super) fn collect_openai_chat_tool_hints(
    response: &Value,
    owner_subagent_id: Option<&str>,
    hints: &mut Vec<ToolHint>,
) {
    let Some(choices) = response.get("choices").and_then(Value::as_array) else {
        return;
    };
    for choice in choices {
        let Some(tool_calls) = choice
            .get("message")
            .and_then(|message| message.get("tool_calls"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for call in tool_calls {
            push_tool_hint(
                hints,
                call,
                owner_subagent_id,
                "openai_chat_tool_call",
                &[&["id"][..], &["call_id"][..]],
                &[&["function", "name"][..], &["name"][..]],
                &[&["function", "arguments"][..], &["arguments"][..]],
            );
        }
    }
}

// Collects OpenAI Responses output items where function-call data is usually direct on each item.
// Items without an id or name are ignored because they are too weak for ownership correlation.
pub(super) fn collect_openai_response_tool_hints(
    response: &Value,
    owner_subagent_id: Option<&str>,
    hints: &mut Vec<ToolHint>,
) {
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return;
    };
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            continue;
        }
        push_tool_hint(
            hints,
            item,
            owner_subagent_id,
            "openai_response_tool_call",
            &[&["call_id"][..], &["id"][..]],
            &[&["name"][..], &["tool_name"][..]],
            &[&["arguments"][..], &["input"][..]],
        );
    }
}

// Collects Anthropic `tool_use` blocks from top-level or nested message content arrays. Other
// content block types are skipped so text and thinking blocks never become tool hints.
pub(super) fn collect_anthropic_tool_hints(
    response: &Value,
    owner_subagent_id: Option<&str>,
    hints: &mut Vec<ToolHint>,
) {
    for content in [
        response.get("content"),
        response
            .get("message")
            .and_then(|message| message.get("content")),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_array)
    {
        for block in content {
            if json_string_at(block, &[&["type"][..]]).as_deref() == Some("tool_use") {
                push_tool_hint(
                    hints,
                    block,
                    owner_subagent_id,
                    "anthropic_tool_use",
                    &[&["id"][..], &["tool_use_id"][..]],
                    &[&["name"][..], &["tool_name"][..]],
                    &[&["input"][..], &["arguments"][..]],
                );
            }
        }
    }
}

// Appends one provider tool hint when an object carries either a tool-call id or enough
// name-plus-argument data to disambiguate common tool names. Name-only and argument-only hints are
// skipped because they over-match across unrelated tools in parallel coding-agent sessions.
pub(super) fn push_tool_hint(
    hints: &mut Vec<ToolHint>,
    object: &Value,
    owner_subagent_id: Option<&str>,
    source: &str,
    id_paths: &[&[&str]],
    name_paths: &[&[&str]],
    argument_paths: &[&[&str]],
) {
    let tool_call_id = json_string_at(object, id_paths);
    let tool_name = json_string_at(object, name_paths);
    let arguments = json_value_at(object, argument_paths)
        .map(normalize_tool_arguments)
        .unwrap_or(Value::Null);
    if tool_call_id.is_none() && (tool_name.is_none() || arguments.is_null()) {
        return;
    }
    hints.push(ToolHint {
        tool_call_id,
        tool_name,
        subagent_id: owner_subagent_id.map(ToOwned::to_owned),
        arguments,
        source: source.to_string(),
    });
}

// Scores how strongly a pending provider tool hint matches an observed hook event. A shared
// provider call id is strongest. Without an id match, require both tool name and exact arguments so
// repeated coding-agent tool names cannot claim unrelated hooks.
pub(super) fn tool_hint_match_score(hint: &ToolHint, event: &ToolEvent) -> u8 {
    let mut score = 0;
    let id_matches = same_optional(
        hint.tool_call_id.as_deref(),
        Some(event.tool_call_id.as_str()),
    );
    let name_matches = same_optional(hint.tool_name.as_deref(), Some(event.tool_name.as_str()));
    let arguments_match = !hint.arguments.is_null()
        && !event.arguments.is_null()
        && hint.arguments == event.arguments;
    if id_matches {
        score += 12;
    }
    if id_matches && name_matches {
        score += 4;
    }
    if id_matches && arguments_match {
        score += 1;
    }
    if !id_matches && name_matches && arguments_match {
        score += 5;
    }
    score
}

pub(super) fn same_optional(left: Option<&str>, right: Option<&str>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if left == right)
}

pub(super) fn owner_status_teaches_request_affinity(status: &str) -> bool {
    matches!(
        status,
        "explicit"
            | "single_hint"
            | "matched_hint"
            | "active_subagent"
            | "subagent_start"
            | "request_affinity"
    )
}

// Parses stringified tool arguments when providers encode them as JSON text. Non-JSON strings are
// preserved as strings so metadata still reflects what the provider actually returned.
pub(super) fn normalize_tool_arguments(arguments: Value) -> Value {
    match arguments {
        Value::String(raw) => serde_json::from_str(&raw).unwrap_or(Value::String(raw)),
        value => value,
    }
}

// Adds correlation status and consumed-hint identifiers to the LLM event metadata. Caller metadata
// is merged first so correlation keys win when names collide.
pub(super) fn llm_correlation_metadata(
    metadata: Value,
    status: &str,
    source: Option<&str>,
    subagent_id: Option<&str>,
    hint: Option<&LlmHintEvent>,
) -> Value {
    let mut correlation = Map::new();
    correlation.insert("llm_correlation_status".into(), json!(status));
    if let Some(source) = source {
        correlation.insert("llm_correlation_source".into(), json!(source));
    }
    if let Some(subagent_id) = subagent_id {
        correlation.insert("llm_correlation_subagent_id".into(), json!(subagent_id));
    }
    if let Some(hint) = hint {
        insert_optional(
            &mut correlation,
            "llm_correlation_conversation_id",
            hint.conversation_id.as_deref(),
        );
        insert_optional(
            &mut correlation,
            "llm_correlation_generation_id",
            hint.generation_id.as_deref(),
        );
        insert_optional(
            &mut correlation,
            "llm_correlation_request_id",
            hint.request_id.as_deref(),
        );
        insert_optional(
            &mut correlation,
            "llm_correlation_agent_type",
            hint.agent_type.as_deref(),
        );
    }
    merge_metadata(metadata, Value::Object(correlation))
}

// Adds correlation metadata to tool spans created from hook events. Consumed hints preserve the
// provider-side tool id/name and extracted arguments so ambiguous or fallback ownership can be
// debugged from emitted events.
pub(super) fn tool_correlation_metadata(
    metadata: Value,
    status: &str,
    source: Option<&str>,
    subagent_id: Option<&str>,
    hint: Option<&ToolHint>,
) -> Value {
    let mut correlation = Map::new();
    correlation.insert("tool_correlation_status".into(), json!(status));
    if let Some(source) = source {
        correlation.insert("tool_correlation_source".into(), json!(source));
    }
    if let Some(subagent_id) = subagent_id {
        correlation.insert("tool_correlation_subagent_id".into(), json!(subagent_id));
    }
    if let Some(hint) = hint {
        insert_optional(
            &mut correlation,
            "tool_correlation_tool_call_id",
            hint.tool_call_id.as_deref(),
        );
        insert_optional(
            &mut correlation,
            "tool_correlation_tool_name",
            hint.tool_name.as_deref(),
        );
        if !hint.arguments.is_null() {
            correlation.insert("tool_correlation_arguments".into(), hint.arguments.clone());
        }
    }
    merge_metadata(metadata, Value::Object(correlation))
}

// Extracts the source agent kind from any normalized event variant so newly created sessions can
// inherit the correct agent identity before an explicit agent-start hook arrives.
pub(super) fn event_agent_kind(event: &NormalizedEvent) -> AgentKind {
    match event {
        NormalizedEvent::AgentStarted(event)
        | NormalizedEvent::AgentEnded(event)
        | NormalizedEvent::TurnEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => event.agent_kind,
        NormalizedEvent::LlmHint(event) => event.agent_kind,
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            event.agent_kind
        }
        NormalizedEvent::LlmStarted(event) | NormalizedEvent::LlmEnded(event) => event.agent_kind,
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => event.agent_kind,
    }
}

// Returns a session id only when exactly one session is active. Gateway requests without explicit
// session headers use this narrow fallback to avoid cross-correlating concurrent agents.
pub(super) fn single_active_session_id(sessions: &HashMap<String, Session>) -> Option<String> {
    (sessions.len() == 1)
        .then(|| sessions.keys().next().cloned())
        .flatten()
}

// Selects a gateway session without guessing between concurrent agents. An explicit session id or
// the sole active session is safe to retain. With no sessions, the stable synthetic root preserves
// pure-proxy continuity. When multiple sessions are active and the request carries no join key,
// isolate that request in a unique short-lived root instead of cross-correlating unrelated agents.
pub(super) fn gateway_session_for_call(
    start: &LlmGatewayStart,
    sessions: &HashMap<String, Session>,
) -> (String, GatewaySessionFinish) {
    if let Some(session_id) = start.session_id.clone() {
        return (session_id, GatewaySessionFinish::Retain);
    }
    if let Some(session_id) = single_active_session_id(sessions) {
        return (session_id, GatewaySessionFinish::Retain);
    }
    if sessions.is_empty() {
        return (
            format!("{}-gateway", AgentKind::Gateway.as_str()),
            GatewaySessionFinish::Retain,
        );
    }
    (
        format!("gateway-isolated-{}", uuid::Uuid::now_v7()),
        GatewaySessionFinish::Close,
    )
}
