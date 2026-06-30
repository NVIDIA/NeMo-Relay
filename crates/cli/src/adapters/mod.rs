// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod claude_code;
pub(crate) mod codex;
pub(crate) mod cursor;
pub(crate) mod hermes;

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::config::header_string;
use crate::json_path::{
    string_at, string_at_any as first_string_at, value_at, value_at_any as first_value_at,
};
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ExtractedLlmHint {
    pub(crate) subagent_id: Option<String>,
    pub(crate) agent_id: Option<String>,
    pub(crate) agent_type: Option<String>,
    pub(crate) conversation_id: Option<String>,
    pub(crate) generation_id: Option<String>,
    pub(crate) request_id: Option<String>,
    pub(crate) model: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExtractedToolCall {
    pub(crate) tool_call_id: Option<String>,
    pub(crate) tool_name: Option<String>,
    pub(crate) subagent_id: Option<String>,
    pub(crate) arguments: Option<Value>,
    pub(crate) result: Option<Value>,
    pub(crate) status: Option<String>,
}

/// Strategy for extracting normalized facts from agent or harness hook payloads.
///
/// Implementations should return `None` for missing or untrusted fields,
/// including per-field values inside returned hint and tool-call structs. The
/// adapter layer owns compatibility fallbacks such as synthetic session IDs,
/// synthetic tool-call IDs, and `unknown_tool` names so downstream lifecycle
/// behavior remains stable for sparse payloads.
pub(crate) trait AgentPayloadExtractor {
    fn session_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String>;
    fn event_name(&self, payload: &Value) -> Option<String>;
    fn metadata(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        kind: AgentKind,
        event_name: &str,
    ) -> Value;
    fn subagent_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String>;
    fn llm_hint(&self, payload: &Value, headers: &HeaderMap) -> ExtractedLlmHint;
    fn tool_call(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        event_name: &str,
    ) -> ExtractedToolCall;
}

pub(super) struct ClaudeCodePayloadExtractor;
pub(super) struct CodexPayloadExtractor;
pub(super) struct CursorPayloadExtractor;
pub(super) struct HermesPayloadExtractor;

pub(super) static CLAUDE_CODE_PAYLOAD_EXTRACTOR: ClaudeCodePayloadExtractor =
    ClaudeCodePayloadExtractor;
pub(super) static CODEX_PAYLOAD_EXTRACTOR: CodexPayloadExtractor = CodexPayloadExtractor;
pub(super) static CURSOR_PAYLOAD_EXTRACTOR: CursorPayloadExtractor = CursorPayloadExtractor;
pub(super) static HERMES_PAYLOAD_EXTRACTOR: HermesPayloadExtractor = HermesPayloadExtractor;

impl AgentPayloadExtractor for ClaudeCodePayloadExtractor {
    fn session_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_session_id(
            headers,
            payload,
            SessionHeaderPolicy::RelayAndClaude,
            CLAUDE_SESSION_ID_PATHS,
        )
    }

    fn event_name(&self, payload: &Value) -> Option<String> {
        agent_event_name(payload, CLAUDE_EVENT_NAME_PATHS)
    }

    fn metadata(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        kind: AgentKind,
        event_name: &str,
    ) -> Value {
        agent_metadata(payload, headers, kind, event_name)
    }

    fn subagent_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_subagent_id(payload, headers, CLAUDE_SUBAGENT_ID_PATHS)
    }

    fn llm_hint(&self, payload: &Value, headers: &HeaderMap) -> ExtractedLlmHint {
        agent_llm_hint(payload, self.subagent_id(payload, headers))
    }

    fn tool_call(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        event_name: &str,
    ) -> ExtractedToolCall {
        agent_tool_call(
            payload,
            self.subagent_id(payload, headers),
            event_name,
            CLAUDE_TOOL_PATHS,
        )
    }
}

impl AgentPayloadExtractor for CodexPayloadExtractor {
    fn session_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_session_id(
            headers,
            payload,
            SessionHeaderPolicy::RelayOnly,
            CODEX_SESSION_ID_PATHS,
        )
    }

    fn event_name(&self, payload: &Value) -> Option<String> {
        agent_event_name(payload, CODEX_EVENT_NAME_PATHS)
    }

    fn metadata(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        kind: AgentKind,
        event_name: &str,
    ) -> Value {
        agent_metadata(payload, headers, kind, event_name)
    }

    fn subagent_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_subagent_id(payload, headers, CODEX_SUBAGENT_ID_PATHS)
    }

    fn llm_hint(&self, payload: &Value, headers: &HeaderMap) -> ExtractedLlmHint {
        agent_llm_hint(payload, self.subagent_id(payload, headers))
    }

    fn tool_call(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        event_name: &str,
    ) -> ExtractedToolCall {
        agent_tool_call(
            payload,
            self.subagent_id(payload, headers),
            event_name,
            CODEX_TOOL_PATHS,
        )
    }
}

impl AgentPayloadExtractor for CursorPayloadExtractor {
    fn session_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_session_id(
            headers,
            payload,
            SessionHeaderPolicy::RelayOnly,
            CURSOR_SESSION_ID_PATHS,
        )
    }

    fn event_name(&self, payload: &Value) -> Option<String> {
        agent_event_name(payload, CURSOR_EVENT_NAME_PATHS)
    }

    fn metadata(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        kind: AgentKind,
        event_name: &str,
    ) -> Value {
        agent_metadata(payload, headers, kind, event_name)
    }

    fn subagent_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_subagent_id(payload, headers, CURSOR_SUBAGENT_ID_PATHS)
    }

    fn llm_hint(&self, payload: &Value, headers: &HeaderMap) -> ExtractedLlmHint {
        agent_llm_hint(payload, self.subagent_id(payload, headers))
    }

    fn tool_call(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        event_name: &str,
    ) -> ExtractedToolCall {
        agent_tool_call(
            payload,
            self.subagent_id(payload, headers),
            event_name,
            CURSOR_TOOL_PATHS,
        )
    }
}

impl AgentPayloadExtractor for HermesPayloadExtractor {
    fn session_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        // Hermes pre-tool correlation already treats the Claude installed-mode
        // session header as explicit session evidence; extraction must accept
        // the same header so those tool hooks do not fall back to synthetic IDs.
        agent_session_id(
            headers,
            payload,
            SessionHeaderPolicy::RelayAndClaude,
            HERMES_SESSION_ID_PATHS,
        )
    }

    fn event_name(&self, payload: &Value) -> Option<String> {
        agent_event_name(payload, HERMES_EVENT_NAME_PATHS)
    }

    fn metadata(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        kind: AgentKind,
        event_name: &str,
    ) -> Value {
        agent_metadata(payload, headers, kind, event_name)
    }

    fn subagent_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_subagent_id(payload, headers, HERMES_SUBAGENT_ID_PATHS)
    }

    fn llm_hint(&self, payload: &Value, headers: &HeaderMap) -> ExtractedLlmHint {
        agent_llm_hint(payload, self.subagent_id(payload, headers))
    }

    fn tool_call(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        event_name: &str,
    ) -> ExtractedToolCall {
        agent_tool_call(
            payload,
            self.subagent_id(payload, headers),
            event_name,
            HERMES_TOOL_PATHS,
        )
    }
}

struct ToolPathSet {
    call_id: &'static [&'static [&'static str]],
    name: &'static [&'static [&'static str]],
    arguments: &'static [&'static [&'static str]],
    result: &'static [&'static [&'static str]],
    status: &'static [&'static [&'static str]],
}

#[derive(Clone, Copy)]
enum SessionHeaderPolicy {
    RelayOnly,
    RelayAndClaude,
}

const CLAUDE_SESSION_ID_PATHS: &[&[&str]] = &[
    &["session_id"],
    &["sessionId"],
    &["session", "id"],
    &["conversation_id"],
    &["conversationId"],
    &["parent_session_id"],
    &["task_id"],
    &["extra", "session_id"],
    &["extra", "task_id"],
];
const CODEX_SESSION_ID_PATHS: &[&[&str]] = &[
    &["session_id"],
    &["sessionId"],
    &["session", "id"],
    &["conversation_id"],
    &["conversationId"],
    &["parent_session_id"],
    &["task_id"],
    &["extra", "session_id"],
    &["extra", "task_id"],
];
const CURSOR_SESSION_ID_PATHS: &[&[&str]] = &[
    &["session_id"],
    &["sessionId"],
    &["session", "id"],
    &["conversation_id"],
    &["conversationId"],
    &["parent_session_id"],
    &["task_id"],
    &["extra", "session_id"],
    &["extra", "task_id"],
];
const HERMES_SESSION_ID_PATHS: &[&[&str]] = &[
    &["session_id"],
    &["sessionId"],
    &["session", "id"],
    &["conversation_id"],
    &["conversationId"],
    &["parent_session_id"],
    &["task_id"],
    &["extra", "session_id"],
    &["extra", "task_id"],
];

const CLAUDE_EVENT_NAME_PATHS: &[&[&str]] = &[
    &["hook_event_name"],
    &["event_name"],
    &["eventName"],
    &["event"],
    &["type"],
    &["name"],
    &["extra", "hook_event_name"],
    &["extra", "event_name"],
    &["extra", "eventName"],
    &["extra", "event"],
    &["extra", "type"],
    &["extra", "name"],
];
const CODEX_EVENT_NAME_PATHS: &[&[&str]] = &[
    &["hook_event_name"],
    &["event_name"],
    &["eventName"],
    &["event"],
    &["type"],
    &["name"],
    &["extra", "hook_event_name"],
    &["extra", "event_name"],
    &["extra", "eventName"],
    &["extra", "event"],
    &["extra", "type"],
    &["extra", "name"],
];
const CURSOR_EVENT_NAME_PATHS: &[&[&str]] = &[
    &["hook_event_name"],
    &["event_name"],
    &["eventName"],
    &["event"],
    &["type"],
    &["name"],
    &["extra", "hook_event_name"],
    &["extra", "event_name"],
    &["extra", "eventName"],
    &["extra", "event"],
    &["extra", "type"],
    &["extra", "name"],
];
const HERMES_EVENT_NAME_PATHS: &[&[&str]] = &[
    &["hook_event_name"],
    &["event_name"],
    &["eventName"],
    &["event"],
    &["type"],
    &["name"],
    &["extra", "hook_event_name"],
    &["extra", "event_name"],
    &["extra", "eventName"],
    &["extra", "event"],
    &["extra", "type"],
    &["extra", "name"],
];

const CLAUDE_SUBAGENT_ID_PATHS: &[&[&str]] = &[
    &["subagent_id"],
    &["subagentId"],
    &["child_subagent_id"],
    &["childSubagentId"],
    &["agent_id"],
    &["subagent", "id"],
    &["agent", "id"],
    &["extra", "subagent_id"],
    &["extra", "subagentId"],
    &["extra", "child_subagent_id"],
    &["extra", "childSubagentId"],
    &["extra", "agent_id"],
    &["extra", "subagent", "id"],
    &["extra", "agent", "id"],
];
const CODEX_SUBAGENT_ID_PATHS: &[&[&str]] = &[
    &["subagent_id"],
    &["subagentId"],
    &["child_subagent_id"],
    &["childSubagentId"],
    &["agent_id"],
    &["source", "subagent", "thread_spawn", "agent_nickname"],
    &["subagent", "id"],
    &["agent", "id"],
    &["extra", "subagent_id"],
    &["extra", "subagentId"],
    &["extra", "child_subagent_id"],
    &["extra", "childSubagentId"],
    &["extra", "agent_id"],
    &["extra", "subagent", "id"],
    &["extra", "agent", "id"],
];
const CURSOR_SUBAGENT_ID_PATHS: &[&[&str]] = &[
    &["subagent_id"],
    &["subagentId"],
    &["child_subagent_id"],
    &["childSubagentId"],
    &["agent_id"],
    &["subagent", "id"],
    &["agent", "id"],
    &["extra", "subagent_id"],
    &["extra", "subagentId"],
    &["extra", "child_subagent_id"],
    &["extra", "childSubagentId"],
    &["extra", "agent_id"],
    &["extra", "subagent", "id"],
    &["extra", "agent", "id"],
];
const HERMES_SUBAGENT_ID_PATHS: &[&[&str]] = &[
    &["child_subagent_id"],
    &["childSubagentId"],
    &["subagent_id"],
    &["subagentId"],
    &["agent_id"],
    &["subagent", "id"],
    &["agent", "id"],
    &["extra", "child_subagent_id"],
    &["extra", "childSubagentId"],
    &["extra", "subagent_id"],
    &["extra", "subagentId"],
    &["extra", "agent_id"],
    &["extra", "subagent", "id"],
    &["extra", "agent", "id"],
];

const CLAUDE_TOOL_CALL_ID_PATHS: &[&[&str]] = &[
    &["tool_use_id"],
    &["tool_call_id"],
    &["toolCallId"],
    &["call_id"],
    &["extra", "tool_call_id"],
    &["extra", "call_id"],
    &["tool", "id"],
    &["tool_input", "id"],
    &["id"],
];
const CODEX_TOOL_CALL_ID_PATHS: &[&[&str]] = &[
    &["tool_call_id"],
    &["toolCallId"],
    &["tool_use_id"],
    &["call_id"],
    &["extra", "tool_call_id"],
    &["extra", "call_id"],
    &["tool", "id"],
    &["tool_input", "id"],
    &["id"],
];
const CURSOR_TOOL_CALL_ID_PATHS: &[&[&str]] = &[
    &["tool_call_id"],
    &["toolCallId"],
    &["tool_use_id"],
    &["call_id"],
    &["extra", "tool_call_id"],
    &["extra", "call_id"],
    &["tool", "id"],
    &["tool_input", "id"],
    &["id"],
];
const HERMES_TOOL_CALL_ID_PATHS: &[&[&str]] = &[
    &["tool_call_id"],
    &["toolCallId"],
    &["tool_use_id"],
    &["call_id"],
    &["extra", "tool_call_id"],
    &["extra", "call_id"],
    &["tool", "id"],
    &["tool_input", "id"],
    &["id"],
];

const TOOL_NAME_PATHS: &[&[&str]] = &[
    &["tool_name"],
    &["toolName"],
    &["tool", "name"],
    &["tool_input", "name"],
    &["name"],
];
const TOOL_ARGUMENT_PATHS: &[&[&str]] = &[&["tool_input"], &["input"], &["arguments"], &["args"]];
const CODEX_TOOL_ARGUMENT_PATHS: &[&[&str]] =
    &[&["arguments"], &["args"], &["input"], &["tool_input"]];
const TOOL_RESULT_PATHS: &[&[&str]] = &[
    &["tool_output"],
    &["tool_response"],
    &["output"],
    &["result"],
    &["extra", "tool_output"],
    &["extra", "result"],
];
const TOOL_STATUS_PATHS: &[&[&str]] = &[&["status"], &["decision"], &["permission"]];

const CLAUDE_TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: CLAUDE_TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: TOOL_ARGUMENT_PATHS,
    result: TOOL_RESULT_PATHS,
    status: TOOL_STATUS_PATHS,
};
const CODEX_TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: CODEX_TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: CODEX_TOOL_ARGUMENT_PATHS,
    result: TOOL_RESULT_PATHS,
    status: TOOL_STATUS_PATHS,
};
const CURSOR_TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: CURSOR_TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: TOOL_ARGUMENT_PATHS,
    result: TOOL_RESULT_PATHS,
    status: TOOL_STATUS_PATHS,
};
const HERMES_TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: HERMES_TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: TOOL_ARGUMENT_PATHS,
    result: TOOL_RESULT_PATHS,
    status: TOOL_STATUS_PATHS,
};

fn agent_session_id(
    headers: &HeaderMap,
    payload: &Value,
    header_policy: SessionHeaderPolicy,
    payload_paths: &'static [&'static [&'static str]],
) -> Option<String> {
    header_string(headers, "x-nemo-relay-session-id")
        .or_else(|| match header_policy {
            SessionHeaderPolicy::RelayOnly => None,
            SessionHeaderPolicy::RelayAndClaude => {
                header_string(headers, "x-claude-code-session-id")
            }
        })
        .or_else(|| session_id_from_payload(payload, payload_paths))
}

fn agent_event_name(payload: &Value, paths: &'static [&'static [&'static str]]) -> Option<String> {
    first_string_at(payload, paths)
}

fn agent_metadata(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    event_name: &str,
) -> Value {
    let mut object = Map::new();
    object.insert("agent_kind".into(), json!(kind.as_str()));
    object.insert("hook_event_name".into(), json!(event_name));
    if let Some(profile) = header_string(headers, "x-nemo-relay-config-profile") {
        object.insert("gateway_config_profile".into(), json!(profile));
    }
    for (key, value) in [
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

fn agent_subagent_id(
    payload: &Value,
    headers: &HeaderMap,
    paths: &'static [&'static [&'static str]],
) -> Option<String> {
    first_string_at(payload, paths).or_else(|| header_string(headers, "x-nemo-relay-subagent-id"))
}

fn agent_llm_hint(payload: &Value, subagent_id: Option<String>) -> ExtractedLlmHint {
    ExtractedLlmHint {
        subagent_id,
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
    }
}

fn agent_tool_call(
    payload: &Value,
    subagent_id: Option<String>,
    event_name: &str,
    paths: &ToolPathSet,
) -> ExtractedToolCall {
    let normalized_event = normalize_name(event_name);
    ExtractedToolCall {
        tool_call_id: first_string_at(payload, paths.call_id),
        tool_name: first_string_at(payload, paths.name),
        subagent_id,
        arguments: first_value_at(payload, paths.arguments),
        result: first_value_at(payload, paths.result)
            .or_else(|| event_detail_result(payload, &normalized_event)),
        status: first_string_at(payload, paths.status)
            .or_else(|| derived_tool_status(&normalized_event)),
    }
}

// Derives a stable session identifier from gateway headers first, then common agent payload
// fields, and finally a v7 UUID. Header precedence lets gateway and hook-forward callers
// correlate events even when agent payload schemas omit or rename their native session field.
fn session_id(
    payload: &Value,
    headers: &HeaderMap,
    extractor: &dyn AgentPayloadExtractor,
) -> String {
    extractor
        .session_id(payload, headers)
        .unwrap_or_else(|| format!("hook-{}", Uuid::now_v7()))
}

// Reads the first known session identifier payload path. Keeping the path list in one place makes
// adapter precedence explicit without nesting a long `or_else` chain in `session_id`.
fn session_id_from_payload(
    payload: &Value,
    paths: &'static [&'static [&'static str]],
) -> Option<String> {
    first_string_at(payload, paths)
}

// Reads the agent's event name from the known hook fields in order and falls back to `unknown`.
// This deliberately keeps unknown payloads observable instead of rejecting them at the adapter
// boundary, allowing the session layer to emit a generic mark event.
fn event_name(payload: &Value, extractor: &dyn AgentPayloadExtractor) -> String {
    extractor
        .event_name(payload)
        .unwrap_or_else(|| "unknown".to_string())
}

// Builds shared metadata for every normalized hook event. Only stable, low-cardinality fields and
// gateway configuration hints are lifted out; the full payload remains on the event for consumers
// that need agent-specific detail.
fn metadata(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    event_name: &str,
    extractor: &dyn AgentPayloadExtractor,
) -> Value {
    extractor.metadata(payload, headers, kind, event_name)
}

// Creates a root session event using the common session-id and metadata extraction rules so
// lifecycle, marks, notifications, and compaction events all carry identical correlation fields.
pub(crate) fn common_session_event(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
) -> SessionEvent {
    let event_name = event_name(payload, extractor);
    SessionEvent {
        session_id: session_id(payload, headers, extractor),
        agent_kind: kind,
        event_name: event_name.clone(),
        payload: payload.clone(),
        metadata: metadata(payload, headers, kind, &event_name, extractor),
    }
}

// Creates a subagent event and tolerates sparse agent payloads by using the gateway subagent
// header and then a synthetic `subagent` id. The fallback keeps unmatched start/end events visible
// rather than dropping them when an integration lacks explicit nested-agent IDs.
fn common_subagent_event(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
) -> SubagentEvent {
    let session = common_session_event(payload, headers, kind, extractor);
    let subagent_id = extractor
        .subagent_id(payload, headers)
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
fn common_llm_hint_event(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
) -> LlmHintEvent {
    let session = common_session_event(payload, headers, kind, extractor);
    let hint = extractor.llm_hint(payload, headers);
    LlmHintEvent {
        session_id: session.session_id,
        agent_kind: kind,
        event_name: session.event_name,
        subagent_id: hint.subagent_id,
        agent_id: hint.agent_id,
        agent_type: hint.agent_type,
        conversation_id: hint.conversation_id,
        generation_id: hint.generation_id,
        request_id: hint.request_id,
        model: hint.model,
        payload: session.payload,
        metadata: session.metadata,
    }
}

// Converts agent tool hooks into the runtime tool event shape while preserving missing fields.
// Tool IDs and names are synthesized when absent, arguments/results are searched across known
// payload shapes, and failure or permission-denied event names are reflected in status metadata.
fn common_tool_event(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
) -> ToolEvent {
    let session = common_session_event(payload, headers, kind, extractor);
    let tool_call = extractor.tool_call(payload, headers, &session.event_name);
    ToolEvent {
        session_id: session.session_id,
        agent_kind: kind,
        event_name: session.event_name,
        tool_call_id: tool_call
            .tool_call_id
            .unwrap_or_else(|| format!("tool-{}", Uuid::now_v7())),
        tool_name: tool_call
            .tool_name
            .unwrap_or_else(|| "unknown_tool".to_string()),
        subagent_id: tool_call.subagent_id,
        arguments: tool_call.arguments.unwrap_or(Value::Null),
        result: tool_call.result.unwrap_or(Value::Null),
        status: tool_call.status,
        payload: session.payload,
        metadata: session.metadata,
    }
}

// Derives error/denied status from event names after an extractor has checked its explicit status
// fields. The derivation is intentionally conservative and only covers known failure spellings.
fn derived_tool_status(normalized_event: &str) -> Option<String> {
    {
        (normalized_event.contains("failure") || normalized_event.contains("failed"))
            .then_some("error".to_string())
    }
    .or_else(|| {
        normalized_event
            .contains("permissiondenied")
            .then_some("denied".to_string())
    })
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

// Classifies a raw hook event into one or more normalized events.
//
// Most hook events produce a single normalized event from `classify_primary`. The exception is
// `Stop` (Claude/Codex): it emits both the existing `LlmHint` (preserving correlation for
// subsequent LLM calls) AND a `TurnEnded` so the session manager can snapshot ATIF without
// closing the agent scope. Codex 0.129 has no `SessionEnd`-equivalent hook — without this dual
// emission, codex transparent runs would never trigger an ATIF write.
//
// If the primary event is already terminal (e.g., Cursor classifies `stop` as `AgentEnded`),
// the snapshot is skipped to avoid double-writing — `flush_observers` already writes ATIF on
// agent-end, and a follow-up `TurnEnded` on a removed session would recreate an empty session
// and overwrite the freshly-written ATIF with an empty trajectory.
fn classify(
    payload: &Value,
    headers: &HeaderMap,
    extractor: &dyn AgentPayloadExtractor,
    rules: &ClassificationRules<'_>,
) -> Vec<NormalizedEvent> {
    let normalized = normalize_name(&event_name(payload, extractor));
    if matches!(
        normalized.as_str(),
        "beforesubmitprompt" | "promptsubmitted" | "userpromptsubmit"
    ) {
        return vec![
            NormalizedEvent::PromptSubmitted(common_session_event(
                payload, headers, rules.kind, extractor,
            )),
            NormalizedEvent::LlmHint(common_llm_hint_event(
                payload, headers, rules.kind, extractor,
            )),
        ];
    }
    let primary = classify_primary(payload, headers, extractor, rules);
    if normalized == "stop" && !primary.is_terminal() {
        return vec![
            primary,
            NormalizedEvent::TurnEnded(common_session_event(
                payload, headers, rules.kind, extractor,
            )),
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
    extractor: &dyn AgentPayloadExtractor,
    rules: &ClassificationRules<'_>,
) -> NormalizedEvent {
    let event = event_name(payload, extractor);
    let normalized = normalize_name(&event);
    if rules
        .agent_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::AgentStarted(common_session_event(
            payload, headers, rules.kind, extractor,
        ))
    } else if rules
        .agent_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::AgentEnded(common_session_event(
            payload, headers, rules.kind, extractor,
        ))
    } else if rules
        .subagent_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::SubagentStarted(common_subagent_event(
            payload, headers, rules.kind, extractor,
        ))
    } else if rules
        .subagent_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::SubagentEnded(common_subagent_event(
            payload, headers, rules.kind, extractor,
        ))
    } else if rules
        .tool_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::ToolStarted(common_tool_event(payload, headers, rules.kind, extractor))
    } else if rules
        .tool_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::ToolEnded(common_tool_event(payload, headers, rules.kind, extractor))
    } else {
        match normalized.as_str() {
            "afteragentresponse" | "agentresponse" | "assistantresponse" | "afteragentthought"
            | "prellmcall" | "postllmcall" | "stop" => NormalizedEvent::LlmHint(
                common_llm_hint_event(payload, headers, rules.kind, extractor),
            ),
            "precompact" | "compaction" => NormalizedEvent::Compaction(common_session_event(
                payload, headers, rules.kind, extractor,
            )),
            "notification" => NormalizedEvent::Notification(common_session_event(
                payload, headers, rules.kind, extractor,
            )),
            _ => NormalizedEvent::HookMark(common_session_event(
                payload, headers, rules.kind, extractor,
            )),
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
