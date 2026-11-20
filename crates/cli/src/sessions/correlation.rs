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
    owner_subagent_id: Option<&str>,
) -> Vec<ToolHint> {
    let mut hints = Vec::new();
    collect_openai_chat_tool_hints(response, owner_subagent_id, &mut hints);
    collect_openai_response_tool_hints(response, owner_subagent_id, &mut hints);
    collect_anthropic_tool_hints(response, owner_subagent_id, &mut hints);
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

// Collects OpenAI Responses function and custom calls directly from output items.
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
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => push_tool_hint(
                hints,
                item,
                owner_subagent_id,
                "openai_response_tool_call",
                &[&["call_id"][..], &["id"][..]],
                &[&["name"][..], &["tool_name"][..]],
                &[&["arguments"][..], &["input"][..]],
            ),
            Some("custom_tool_call") => {
                let (Some(name), Some(input)) = (
                    json_string_at(item, &[&["name"][..]]),
                    item.get("input").and_then(Value::as_str),
                ) else {
                    continue;
                };
                // The item ID is not the invocation ID. Custom input is free-form
                // text even when it happens to be syntactically valid JSON.
                hints.push(ToolHint {
                    tool_call_id: json_string_at(item, &[&["call_id"][..]]),
                    tool_name: Some(name),
                    subagent_id: owner_subagent_id.map(ToOwned::to_owned),
                    arguments: Value::String(input.to_string()),
                    source: "openai_response_custom_tool_call".into(),
                });
            }
            _ => {}
        }
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
        "explicit" | "single_hint" | "matched_hint" | "active_subagent" | "request_affinity"
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
        | NormalizedEvent::TurnStarted(event)
        | NormalizedEvent::TurnEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => event.agent_kind,
        NormalizedEvent::LlmHint(event) => event.agent_kind,
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            event.agent_kind
        }
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => event.agent_kind,
    }
}

// Returns a session id only when exactly one session is active. Gateway requests without explicit
// session headers use this narrow fallback to avoid cross-correlating concurrent agents.
#[cfg(test)]
pub(super) fn single_active_session_id(sessions: &HashMap<String, Session>) -> Option<String> {
    let now = session_now();
    let mut active = sessions
        .iter()
        .filter(|(_, session)| session.is_active_or_recent(now));
    let (session_id, _) = active.next()?;
    active.next().is_none().then(|| session_id.clone())
}

// Selects a gateway session without guessing between concurrent agents. An explicit session id or
// the sole active session is safe to retain. With no sessions, the stable synthetic root preserves
// pure-proxy continuity. When multiple sessions are active and the request carries no join key,
// isolate that request in a unique short-lived root instead of cross-correlating unrelated agents.
pub(super) fn gateway_session_for_call_with_preparing(
    start: &LlmGatewayStart,
    sessions: &HashMap<String, Session>,
    preparing: &HashSet<String>,
) -> (String, GatewaySessionFinish) {
    if let Some(session_id) = start.session_id.clone() {
        return (session_id, GatewaySessionFinish::Retain);
    }
    let now = session_now();
    let mut active = sessions
        .iter()
        .filter(|(_, session)| session.is_active_or_recent(now))
        .map(|(id, _)| id)
        .chain(preparing.iter());
    if let Some(session_id) = active.next()
        && active.next().is_none()
    {
        return (session_id.clone(), GatewaySessionFinish::Retain);
    }
    if sessions.is_empty() && preparing.is_empty() {
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

#[derive(Default)]
pub(super) struct SessionCorrelation {
    pub(super) pending_llm_hints: Vec<PendingLlmHint>,
    pub(super) pending_tool_hints: Vec<PendingToolHint>,
    // Conflicting owners make an affinity key ambiguous; affinity has no time-to-live.
    pub(super) llm_request_affinity: HashMap<String, Option<String>>,
    pub(super) last_llm_owner: Option<LastLlmOwner>,
}

impl SessionCorrelation {
    pub(super) fn cleanup(&mut self, now: Instant) {
        self.pending_llm_hints
            .retain(|hint| now.duration_since(hint.inserted_at) <= LLM_HINT_TTL);
        self.pending_tool_hints
            .retain(|hint| now.duration_since(hint.inserted_at) <= TOOL_HINT_TTL);
        if self
            .last_llm_owner
            .as_ref()
            .is_some_and(|owner| now.duration_since(owner.updated_at) > LAST_OWNER_TTL)
        {
            self.last_llm_owner = None;
        }
    }

    pub(super) fn matching_hint_index(&self, start: &LlmGatewayStart) -> Option<usize> {
        let matches: Vec<_> = self
            .pending_llm_hints
            .iter()
            .enumerate()
            .filter_map(|(index, pending)| {
                let score = hint_match_score(&pending.hint, start);
                (score > 0).then_some((index, score))
            })
            .collect();
        let best_score = matches.iter().map(|(_, score)| *score).max()?;
        let best: Vec<_> = matches
            .into_iter()
            .filter(|(_, score)| *score == best_score)
            .collect();
        (best.len() == 1).then_some(best[0].0)
    }

    pub(super) fn matching_tool_hint_index(&self, event: &ToolEvent) -> Option<usize> {
        let matches: Vec<_> = self
            .pending_tool_hints
            .iter()
            .enumerate()
            .filter_map(|(index, pending)| {
                let score = tool_hint_match_score(&pending.hint, event);
                (score > 0).then_some((index, score))
            })
            .collect();
        let best_score = matches.iter().map(|(_, score)| *score).max()?;
        let best: Vec<_> = matches
            .into_iter()
            .filter(|(_, score)| *score == best_score)
            .collect();
        (best.len() == 1).then_some(best[0].0)
    }

    pub(super) fn consume_matching_tool_hint(&mut self, event: &ToolEvent) {
        if let Some(index) = self.matching_tool_hint_index(event) {
            self.pending_tool_hints.remove(index);
        }
    }

    pub(super) fn set_last_llm_owner(&mut self, subagent_id: Option<String>) {
        self.last_llm_owner = subagent_id.map(|subagent_id| LastLlmOwner {
            subagent_id,
            updated_at: session_now(),
            source: LastLlmOwnerSource::Llm,
        });
    }

    pub(super) fn set_last_tool_owner(&mut self, subagent_id: Option<String>) {
        if let Some(subagent_id) = subagent_id {
            self.last_llm_owner = Some(LastLlmOwner {
                subagent_id,
                updated_at: session_now(),
                source: LastLlmOwnerSource::Tool,
            });
        }
    }

    pub(super) fn set_last_subagent_start_owner(&mut self, subagent_id: Option<String>) {
        if let Some(subagent_id) = subagent_id {
            self.last_llm_owner = Some(LastLlmOwner {
                subagent_id,
                updated_at: session_now(),
                source: LastLlmOwnerSource::SubagentStart,
            });
        }
    }

    pub(super) fn record_request_affinity(
        &mut self,
        provider: &str,
        request: &LlmRequest,
        subagent_id: Option<&str>,
        status: &str,
    ) {
        if !owner_status_teaches_request_affinity(status) {
            return;
        }
        let Some(subagent_id) = subagent_id else {
            return;
        };
        let Some(key) = alignment::request_affinity_key(provider, request) else {
            return;
        };
        match self.llm_request_affinity.get_mut(&key) {
            Some(Some(existing)) if existing == subagent_id => {}
            // If two live subagents share the same prompt text, the key is no longer safe as a
            // discriminator. Mark it ambiguous instead of allowing either worker to claim it.
            Some(owner) => *owner = None,
            None => {
                self.llm_request_affinity
                    .insert(key, Some(subagent_id.to_string()));
            }
        }
    }

    pub(super) fn clear_root_turn(&mut self, subagents: &HashMap<String, ActiveSubagent>) {
        self.pending_llm_hints.retain(|pending| {
            pending
                .hint
                .subagent_id
                .as_ref()
                .or(pending.hint.agent_id.as_ref())
                .is_some_and(|id| subagents.contains_key(id))
        });
        self.pending_tool_hints.retain(|pending| {
            pending
                .hint
                .subagent_id
                .as_ref()
                .is_some_and(|id| subagents.contains_key(id))
        });
        self.llm_request_affinity
            .retain(|_, owner| owner.as_ref().is_some_and(|id| subagents.contains_key(id)));
        self.last_llm_owner = None;
    }

    pub(super) fn remove_subagent(&mut self, subagent_id: &str) {
        self.pending_llm_hints.retain(|pending| {
            pending
                .hint
                .subagent_id
                .as_ref()
                .or(pending.hint.agent_id.as_ref())
                .map(String::as_str)
                != Some(subagent_id)
        });
        self.pending_tool_hints
            .retain(|pending| pending.hint.subagent_id.as_deref() != Some(subagent_id));
        self.llm_request_affinity
            .retain(|_, owner| owner.as_deref() != Some(subagent_id));
        if self
            .last_llm_owner
            .as_ref()
            .is_some_and(|owner| owner.subagent_id == subagent_id)
        {
            self.last_llm_owner = None;
        }
    }
}

pub(super) enum OwnershipTarget {
    Root,
    Subagent(String),
}

impl OwnershipTarget {
    pub(super) fn subagent_id(&self) -> Option<&str> {
        match self {
            Self::Root => None,
            Self::Subagent(identifier) => Some(identifier),
        }
    }

    pub(super) fn into_subagent_id(self) -> Option<String> {
        match self {
            Self::Root => None,
            Self::Subagent(identifier) => Some(identifier),
        }
    }
}

pub(super) struct OwnershipDecision {
    pub(super) owner: OwnershipTarget,
    pub(super) evidence: OwnershipEvidence,
}

impl OwnershipDecision {
    pub(super) fn new(subagent_id: Option<String>, evidence: OwnershipEvidence) -> Self {
        Self {
            owner: subagent_id.map_or(OwnershipTarget::Root, OwnershipTarget::Subagent),
            evidence,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum HintSelection {
    Single,
    Matched,
}

#[derive(Clone, Copy)]
pub(super) enum OwnershipEvidence {
    GatewayHeader,
    HookPayload,
    RootRequestPayload,
    SingleHint,
    MatchedHint,
    SingleHintUnavailableOwner,
    MatchedHintUnavailableOwner,
    RequestAffinity,
    Sticky(LastLlmOwnerSource),
    SoleActiveSubagent,
    AgentFallback,
    AmbiguousFallback,
    HarnessInternal,
}

impl OwnershipEvidence {
    pub(super) fn hint(selection: HintSelection, owner_unavailable: bool) -> Self {
        match (selection, owner_unavailable) {
            (HintSelection::Single, false) => Self::SingleHint,
            (HintSelection::Single, true) => Self::SingleHintUnavailableOwner,
            (HintSelection::Matched, false) => Self::MatchedHint,
            (HintSelection::Matched, true) => Self::MatchedHintUnavailableOwner,
        }
    }

    pub(super) fn is_hint(self) -> bool {
        matches!(
            self,
            Self::SingleHint
                | Self::MatchedHint
                | Self::SingleHintUnavailableOwner
                | Self::MatchedHintUnavailableOwner
        )
    }

    pub(super) fn status(self) -> &'static str {
        match self {
            Self::GatewayHeader | Self::HookPayload | Self::RootRequestPayload => "explicit",
            Self::SingleHint | Self::SingleHintUnavailableOwner => "single_hint",
            Self::MatchedHint | Self::MatchedHintUnavailableOwner => "matched_hint",
            Self::RequestAffinity => "request_affinity",
            Self::Sticky(source) => source.status(),
            Self::SoleActiveSubagent => "active_subagent",
            Self::AgentFallback => "agent_fallback",
            Self::AmbiguousFallback => "ambiguous_fallback",
            Self::HarnessInternal => "harness_internal",
        }
    }

    pub(super) fn source(self) -> Option<&'static str> {
        match self {
            Self::GatewayHeader => Some("gateway_header"),
            Self::HookPayload => Some("hook_payload"),
            Self::RootRequestPayload | Self::RequestAffinity => Some("request_payload"),
            Self::Sticky(source) => source.metadata_source(),
            Self::HarnessInternal => Some("claude_code_request_class"),
            _ => None,
        }
    }
}
