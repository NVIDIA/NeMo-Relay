// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod claude_code;
pub(crate) mod codex;
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
    /// Normalized events emitted from one incoming agent hook payload.
    pub(crate) events: Vec<NormalizedEvent>,
    /// Hook response body returned to the invoking agent process.
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
    /// Agent-local worker or subagent identifier, when the payload supplies one.
    pub(crate) subagent_id: Option<String>,
    /// Stable agent identifier from the hook payload, not synthesized.
    pub(crate) agent_id: Option<String>,
    /// Agent type or role reported by the harness, not synthesized.
    pub(crate) agent_type: Option<String>,
    /// Provider or harness conversation identifier used for later LLM correlation.
    pub(crate) conversation_id: Option<String>,
    /// Generation identifier used to pair hook hints with provider responses.
    pub(crate) generation_id: Option<String>,
    /// Request identifier used to pair hook hints with provider requests.
    pub(crate) request_id: Option<String>,
    /// Model name reported by the hook payload.
    pub(crate) model: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExtractedToolCall {
    /// Tool-call identifier reported by the hook payload, not synthesized.
    pub(crate) tool_call_id: Option<String>,
    /// Tool name reported by the hook payload, not synthesized.
    pub(crate) tool_name: Option<String>,
    /// Agent-local worker or subagent that owns the tool call.
    pub(crate) subagent_id: Option<String>,
    /// Tool arguments exactly as supplied by the hook payload.
    pub(crate) arguments: Option<Value>,
    /// Tool result exactly as supplied by the hook payload.
    pub(crate) result: Option<Value>,
    /// Tool status reported or conservatively derived from the hook event name.
    pub(crate) status: Option<String>,
}

/// Strategy for extracting normalized facts from agent or harness hook payloads.
///
/// The trait is organized as a small set of per-harness *deviation hooks*
/// (`session_header_policy`, `session_id_paths`, `event_name_paths`,
/// `subagent_id_paths`, `tool_paths`) plus shared *behavior* methods built on
/// top of them. Every deviation hook has a canonical default, so a harness
/// implementation overrides only the hooks where its hook payloads genuinely
/// differ and the shared behavior is written once.
///
/// Behavior methods return `None` for missing or untrusted fields. The adapter
/// layer owns compatibility fallbacks such as synthetic session IDs, synthetic
/// tool-call IDs, and `unknown_tool` names so downstream lifecycle behavior
/// remains stable for sparse payloads.
pub(crate) trait AgentPayloadExtractor {
    // -- Per-harness deviations (override only what genuinely differs) -------

    /// Whether this harness also trusts the Claude installed-mode session
    /// header (`x-claude-code-session-id`) as explicit session evidence.
    fn session_header_policy(&self) -> SessionHeaderPolicy {
        SessionHeaderPolicy::RelayAndClaude
    }

    /// Candidate payload paths for the native session identifier.
    fn session_id_paths(&self) -> &'static [&'static [&'static str]] {
        SESSION_ID_PATHS
    }

    /// Candidate payload paths for the native hook event name.
    fn event_name_paths(&self) -> &'static [&'static [&'static str]] {
        EVENT_NAME_PATHS
    }

    /// Candidate payload paths for the native subagent or worker identifier.
    fn subagent_id_paths(&self) -> &'static [&'static [&'static str]] {
        SUBAGENT_ID_PATHS
    }

    /// Tool payload paths (call id, name, arguments, result, status).
    fn tool_paths(&self) -> &'static ToolPathSet {
        TOOL_PATHS
    }

    // -- Shared behavior (derived from the deviation hooks above) ------------

    /// Extract the native session identifier for this agent payload.
    ///
    /// Returning `None` means the payload did not supply a trustworthy session
    /// id; the adapter boundary will apply compatibility fallbacks.
    fn session_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_session_id(
            headers,
            payload,
            self.session_header_policy(),
            self.session_id_paths(),
        )
    }

    /// Extract the native hook event name for this agent payload.
    ///
    /// Returning `None` keeps unknown events observable by letting the adapter
    /// boundary synthesize the generic `unknown` event name.
    fn event_name(&self, payload: &Value) -> Option<String> {
        first_string_at(payload, self.event_name_paths())
    }

    /// Build stable, low-cardinality metadata shared by normalized events.
    ///
    /// Implementations must not promote high-cardinality paths or PII into this
    /// map; consumers that need full details can read the raw event payload.
    fn metadata(
        &self,
        payload: &Value,
        headers: &HeaderMap,
        kind: AgentKind,
        event_name: &str,
    ) -> Value {
        agent_metadata(payload, headers, kind, event_name)
    }

    /// Extract the native subagent or worker identifier for this agent payload.
    ///
    /// Returning `None` means the payload did not identify a subagent; callers
    /// decide whether to synthesize a compatibility owner.
    fn subagent_id(&self, payload: &Value, headers: &HeaderMap) -> Option<String> {
        agent_subagent_id(payload, headers, self.subagent_id_paths())
    }

    /// Extract LLM-correlation hints without applying fallback values.
    fn llm_hint(&self, payload: &Value, headers: &HeaderMap) -> ExtractedLlmHint {
        agent_llm_hint(payload, self.subagent_id(payload, headers))
    }

    /// Extract tool-call facts without applying fallback identifiers or names.
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
            self.tool_paths(),
        )
    }
}

pub(super) struct ClaudeCodePayloadExtractor;
pub(super) struct CodexPayloadExtractor;
pub(super) struct HermesPayloadExtractor;

pub(super) static CLAUDE_CODE_PAYLOAD_EXTRACTOR: ClaudeCodePayloadExtractor =
    ClaudeCodePayloadExtractor;
pub(super) static CODEX_PAYLOAD_EXTRACTOR: CodexPayloadExtractor = CodexPayloadExtractor;
pub(super) static HERMES_PAYLOAD_EXTRACTOR: HermesPayloadExtractor = HermesPayloadExtractor;

/// Claude Code reports its native tool identifier as `tool_use_id`, so it uses
/// a tool path set that prefers that key. Every other hook field matches the
/// canonical defaults (including the installed-mode session-header policy).
impl AgentPayloadExtractor for ClaudeCodePayloadExtractor {
    fn tool_paths(&self) -> &'static ToolPathSet {
        CLAUDE_TOOL_PATHS
    }
}

/// Codex transparent runs forward provider tokens directly, so they must not
/// adopt the Claude installed-mode session header. They also expose a
/// Codex-native subagent nickname and send tool arguments under `arguments`.
impl AgentPayloadExtractor for CodexPayloadExtractor {
    fn session_header_policy(&self) -> SessionHeaderPolicy {
        SessionHeaderPolicy::RelayOnly
    }

    fn subagent_id_paths(&self) -> &'static [&'static [&'static str]] {
        CODEX_SUBAGENT_ID_PATHS
    }

    fn tool_paths(&self) -> &'static ToolPathSet {
        CODEX_TOOL_PATHS
    }
}

/// Hermes always runs nested under another agent, so the `child_subagent_id`
/// signal is the most reliable owner and is preferred over the generic
/// session-scoped subagent id. Session, event, and tool extraction match the
/// canonical defaults.
impl AgentPayloadExtractor for HermesPayloadExtractor {
    fn subagent_id_paths(&self) -> &'static [&'static [&'static str]] {
        HERMES_SUBAGENT_ID_PATHS
    }
}

pub(crate) struct ToolPathSet {
    call_id: &'static [&'static [&'static str]],
    name: &'static [&'static [&'static str]],
    arguments: &'static [&'static [&'static str]],
    result: &'static [&'static [&'static str]],
    status: &'static [&'static [&'static str]],
}

/// Whether an extractor accepts the Claude installed-mode session header.
#[derive(Clone, Copy)]
pub(crate) enum SessionHeaderPolicy {
    /// Trust only the NeMo Relay session header. Used by harnesses (Codex
    /// transparent runs) that forward provider tokens directly and must not
    /// inherit a Claude installed-mode session id.
    RelayOnly,
    /// Trust the NeMo Relay session header and then the Claude installed-mode
    /// `x-claude-code-session-id` header as explicit session evidence.
    RelayAndClaude,
}

/// Canonical session-id precedence. All supported harnesses share this list;
/// only their [`SessionHeaderPolicy`] differs.
const SESSION_ID_PATHS: &[&[&str]] = &[
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

/// Canonical hook event-name precedence, shared by all supported harnesses.
const EVENT_NAME_PATHS: &[&[&str]] = &[
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

/// Canonical subagent-id precedence for harnesses without a native nested-agent
/// signal of their own (Claude Code).
const SUBAGENT_ID_PATHS: &[&[&str]] = &[
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

/// Codex deviation: adds the thread-spawn nickname between the flat id keys and
/// the nested `subagent.id`/`agent.id` shapes.
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

/// Hermes deviation: prefers the `child_subagent_id` owner signal before the
/// generic session-scoped subagent id.
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

/// Claude Code deviation: its native tool identifier is `tool_use_id`, checked
/// before the generic `tool_call_id` shapes.
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

/// Canonical tool-call-id precedence for harnesses that report the generic
/// `tool_call_id` first (Codex and Hermes).
const TOOL_CALL_ID_PATHS: &[&[&str]] = &[
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

/// Canonical argument precedence for harnesses that nest tool input under
/// `tool_input` first (Claude Code and Hermes).
const TOOL_ARGUMENT_PATHS: &[&[&str]] = &[&["tool_input"], &["input"], &["arguments"], &["args"]];

/// Codex deviation: sends tool arguments under `arguments`/`args` first.
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

/// Canonical tool path set used by harnesses that report generic tool shapes
/// (Hermes). Name, result, and status precedence is shared by every harness.
const TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: TOOL_ARGUMENT_PATHS,
    result: TOOL_RESULT_PATHS,
    status: TOOL_STATUS_PATHS,
};
const CLAUDE_TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: CLAUDE_TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: TOOL_ARGUMENT_PATHS,
    result: TOOL_RESULT_PATHS,
    status: TOOL_STATUS_PATHS,
};
const CODEX_TOOL_PATHS: &ToolPathSet = &ToolPathSet {
    call_id: TOOL_CALL_ID_PATHS,
    name: TOOL_NAME_PATHS,
    arguments: CODEX_TOOL_ARGUMENT_PATHS,
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

/// Derive a stable session identifier from extracted facts and compatibility fallbacks.
///
/// Header and payload precedence lives in the selected extractor. This boundary
/// applies the final synthetic ID fallback so sparse payloads stay observable.
fn session_id(
    payload: &Value,
    headers: &HeaderMap,
    extractor: &dyn AgentPayloadExtractor,
) -> String {
    let fallback_session_id = fallback_session_id();
    session_id_with_fallback(payload, headers, extractor, &fallback_session_id)
}

fn fallback_session_id() -> String {
    format!("hook-{}", Uuid::now_v7())
}

fn session_id_with_fallback(
    payload: &Value,
    headers: &HeaderMap,
    extractor: &dyn AgentPayloadExtractor,
    fallback_session_id: &str,
) -> String {
    extractor
        .session_id(payload, headers)
        .unwrap_or_else(|| fallback_session_id.to_string())
}

/// Read the first known session identifier payload path for one agent strategy.
///
/// Keeping the path list in one place makes adapter precedence explicit without
/// nesting a long `or_else` chain in `session_id`.
fn session_id_from_payload(
    payload: &Value,
    paths: &'static [&'static [&'static str]],
) -> Option<String> {
    first_string_at(payload, paths)
}

/// Read the agent's event name and fall back to `unknown`.
///
/// Unknown payloads stay observable instead of being rejected at the adapter
/// boundary, allowing the session layer to emit a generic mark event.
fn event_name(payload: &Value, extractor: &dyn AgentPayloadExtractor) -> String {
    extractor
        .event_name(payload)
        .unwrap_or_else(|| "unknown".to_string())
}

/// Build shared metadata for a normalized hook event.
///
/// Only stable, low-cardinality fields and gateway configuration hints are
/// lifted out; the full payload remains on the event for consumers that need
/// agent-specific detail.
fn metadata(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    event_name: &str,
    extractor: &dyn AgentPayloadExtractor,
) -> Value {
    extractor.metadata(payload, headers, kind, event_name)
}

/// Create a root session event using the common extraction rules.
///
/// Lifecycle, marks, notifications, and compaction events all carry identical
/// session-id and metadata correlation fields.
pub(crate) fn common_session_event(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
) -> SessionEvent {
    let fallback_session_id = fallback_session_id();
    common_session_event_with_fallback(payload, headers, kind, extractor, &fallback_session_id)
}

fn common_session_event_with_fallback(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
    fallback_session_id: &str,
) -> SessionEvent {
    let event_name = event_name(payload, extractor);
    SessionEvent {
        session_id: session_id_with_fallback(payload, headers, extractor, fallback_session_id),
        agent_kind: kind,
        event_name: event_name.clone(),
        payload: payload.clone(),
        metadata: metadata(payload, headers, kind, &event_name, extractor),
    }
}

/// Create a subagent event from an agent hook payload.
///
/// Sparse payloads fall back through the selected extractor and then to a
/// synthetic `subagent` id, keeping unmatched start/end events visible when an
/// integration lacks explicit nested-agent IDs.
fn common_subagent_event_with_fallback(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
    fallback_session_id: &str,
) -> SubagentEvent {
    let session =
        common_session_event_with_fallback(payload, headers, kind, extractor, fallback_session_id);
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

/// Capture hook payload hints used to correlate nearby gateway LLM calls.
///
/// Multiple naming conventions are accepted because integrations expose
/// conversation, generation, request, and model identifiers under different
/// shapes.
fn common_llm_hint_event_with_fallback(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
    fallback_session_id: &str,
) -> LlmHintEvent {
    let session =
        common_session_event_with_fallback(payload, headers, kind, extractor, fallback_session_id);
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

/// Convert agent tool hooks into the runtime tool event shape.
///
/// Tool IDs and names are synthesized when absent, arguments/results are
/// searched across known payload shapes, and failure or permission-denied event
/// names are reflected in status metadata.
fn common_tool_event_with_fallback(
    payload: &Value,
    headers: &HeaderMap,
    kind: AgentKind,
    extractor: &dyn AgentPayloadExtractor,
    fallback_session_id: &str,
) -> ToolEvent {
    let session =
        common_session_event_with_fallback(payload, headers, kind, extractor, fallback_session_id);
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

/// Derive error or denied status from normalized event names.
///
/// This runs after an extractor has checked explicit status fields and remains
/// conservative by covering only known failure spellings.
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

/// Extract diagnostic detail fields as a synthetic result for failure-like hooks.
///
/// Successful tool events without explicit output remain `null` so observers can
/// distinguish "no output supplied" from "the gateway assembled diagnostic
/// details".
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

/// Classify a raw hook event into one or more normalized events.
///
/// Most hook events produce a single normalized event from `classify_primary`.
/// The exception is `Stop` for Claude Code and Codex: it emits both the
/// existing `LlmHint` and a `TurnEnded` so the session manager can snapshot ATIF
/// without closing the agent scope.
///
/// If the primary event is already terminal, the snapshot is skipped to avoid
/// double-writing and accidentally recreating an empty session.
fn classify(
    payload: &Value,
    headers: &HeaderMap,
    extractor: &dyn AgentPayloadExtractor,
    rules: &ClassificationRules<'_>,
) -> Vec<NormalizedEvent> {
    let fallback_session_id = fallback_session_id();
    let normalized = normalize_name(&event_name(payload, extractor));
    if matches!(
        normalized.as_str(),
        "beforesubmitprompt" | "promptsubmitted" | "userpromptsubmit"
    ) {
        return vec![
            NormalizedEvent::PromptSubmitted(common_session_event_with_fallback(
                payload,
                headers,
                rules.kind,
                extractor,
                &fallback_session_id,
            )),
            NormalizedEvent::LlmHint(common_llm_hint_event_with_fallback(
                payload,
                headers,
                rules.kind,
                extractor,
                &fallback_session_id,
            )),
        ];
    }
    let primary = classify_primary(payload, headers, extractor, rules, &fallback_session_id);
    if normalized == "stop" && !primary.is_terminal() {
        return vec![
            primary,
            NormalizedEvent::TurnEnded(common_session_event_with_fallback(
                payload,
                headers,
                rules.kind,
                extractor,
                &fallback_session_id,
            )),
        ];
    }
    vec![primary]
}

/// Classify a raw hook event using adapter-specific names before generic names.
///
/// Unknown events are intentionally converted to hook marks, not errors, so new
/// agent hook types remain observable until first-class normalization rules are
/// added.
fn classify_primary(
    payload: &Value,
    headers: &HeaderMap,
    extractor: &dyn AgentPayloadExtractor,
    rules: &ClassificationRules<'_>,
    fallback_session_id: &str,
) -> NormalizedEvent {
    let event = event_name(payload, extractor);
    let normalized = normalize_name(&event);
    if rules
        .agent_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::AgentStarted(common_session_event_with_fallback(
            payload,
            headers,
            rules.kind,
            extractor,
            fallback_session_id,
        ))
    } else if rules
        .agent_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::AgentEnded(common_session_event_with_fallback(
            payload,
            headers,
            rules.kind,
            extractor,
            fallback_session_id,
        ))
    } else if rules
        .subagent_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::SubagentStarted(common_subagent_event_with_fallback(
            payload,
            headers,
            rules.kind,
            extractor,
            fallback_session_id,
        ))
    } else if rules
        .subagent_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::SubagentEnded(common_subagent_event_with_fallback(
            payload,
            headers,
            rules.kind,
            extractor,
            fallback_session_id,
        ))
    } else if rules
        .tool_start
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::ToolStarted(common_tool_event_with_fallback(
            payload,
            headers,
            rules.kind,
            extractor,
            fallback_session_id,
        ))
    } else if rules
        .tool_end
        .iter()
        .any(|name| normalize_name(name) == normalized)
    {
        NormalizedEvent::ToolEnded(common_tool_event_with_fallback(
            payload,
            headers,
            rules.kind,
            extractor,
            fallback_session_id,
        ))
    } else {
        match normalized.as_str() {
            "afteragentresponse" | "agentresponse" | "assistantresponse" | "afteragentthought"
            | "prellmcall" | "postllmcall" | "stop" => {
                NormalizedEvent::LlmHint(common_llm_hint_event_with_fallback(
                    payload,
                    headers,
                    rules.kind,
                    extractor,
                    fallback_session_id,
                ))
            }
            "precompact" | "compaction" => {
                NormalizedEvent::Compaction(common_session_event_with_fallback(
                    payload,
                    headers,
                    rules.kind,
                    extractor,
                    fallback_session_id,
                ))
            }
            "notification" => NormalizedEvent::Notification(common_session_event_with_fallback(
                payload,
                headers,
                rules.kind,
                extractor,
                fallback_session_id,
            )),
            _ => NormalizedEvent::HookMark(common_session_event_with_fallback(
                payload,
                headers,
                rules.kind,
                extractor,
                fallback_session_id,
            )),
        }
    }
}

/// Remove separators and case differences before comparing hook names.
///
/// The gateway uses this for agent-specific aliases so `PostToolUse`,
/// `post_tool_use`, and `postToolUse` converge.
fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
#[path = "../../tests/coverage/adapters_tests.rs"]
mod tests;
