// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-specific alignment and gateway-normalization helpers for the CLI.
//!
//! The session and gateway modules own generic lifecycle mechanics. This module owns the cases
//! where a coding agent's wire format needs interpretation before those generic mechanics can
//! attach LLMs/tools to the right scope or forward the request to the right upstream.

use std::collections::HashMap;

use axum::http::HeaderMap;
use nemo_relay::api::llm::LlmRequest;
use serde_json::{Map, Value, json};

use crate::config::header_string;
use crate::model::{AgentKind, LlmEvent, NormalizedEvent, SessionEvent, SubagentEvent, ToolEvent};

pub(crate) mod claude_code;
pub(crate) mod codex;
pub(crate) mod hermes;

const REQUEST_AFFINITY_KEY_MIN_CHARS: usize = 24;
const REQUEST_AFFINITY_KEY_MAX_CHARS: usize = 4096;

#[derive(Debug, Clone)]
pub(crate) enum SubagentSessionContext {
    Codex(codex::SubagentContext),
    Hermes(hermes::SubagentContext),
}

impl SubagentSessionContext {
    pub(crate) fn parent_session_id(&self) -> &str {
        match self {
            Self::Codex(context) => &context.parent_session_id,
            Self::Hermes(context) => &context.parent_session_id,
        }
    }
}

// Minimal route taxonomy used by alignment code. The gateway has richer routing state, but these
// variants are the only distinctions provider-specific correlation needs: Codex-owned OpenAI
// Responses, Claude-owned Anthropic endpoints, and other OpenAI-compatible traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GatewayRouteKind {
    OpenAiResponses,
    OpenAiChatCompletions,
    OpenAiModels,
    AnthropicMessages,
    AnthropicCountTokens,
}

// Records that a provider-created child session is really a subagent under another session. The
// session manager stores this until the child emits its terminal AgentEnded event, then removes the
// alias so future unrelated events cannot be reparented through stale state.
#[derive(Debug, Clone)]
pub(crate) struct SessionAlias {
    pub(crate) parent_session_id: String,
    pub(crate) subagent_id: String,
    // Metadata explains why this alias exists and is stamped on rewritten events. Phoenix traces
    // then stay filterable/debuggable even after the event has been moved under its parent scope.
    metadata: Value,
}

impl SessionAlias {
    // Builds the explicit child-session-to-parent-subagent mapping after an adapter has proven the
    // child session is not an independent root agent.
    pub(crate) fn new(parent_session_id: String, subagent_id: String, metadata: Value) -> Self {
        Self {
            parent_session_id,
            subagent_id,
            metadata,
        }
    }

    // Returns owned metadata because routing consumes events by value and may need to merge the
    // same alias explanation into several lifecycle events before the alias is closed.
    pub(crate) fn metadata(&self) -> Value {
        self.metadata.clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingSubagentStart {
    // The original child SessionStart is retained because promotion may happen on a later parent
    // hook or gateway request, after this hook request has already returned.
    pub(crate) event: SessionEvent,
    context: SubagentSessionContext,
}

impl PendingSubagentStart {
    pub(crate) fn parent_session_id(&self) -> &str {
        self.context.parent_session_id()
    }

    pub(crate) fn subagent_start_event(&self) -> SubagentEvent {
        subagent_start_event(&self.event, &self.context)
    }

    pub(crate) fn alias_for_child_session(&self, child_session_id: String) -> SessionAlias {
        alias_for_child_session(child_session_id, &self.context)
    }
}

// Owns all cross-session correlation state used by the session manager. Keeping aliases and
// pending child starts together makes lifecycle cleanup atomic: any code that removes stale alias
// state also removes the matching pending state before later events can observe a half-updated map.
#[derive(Debug, Default)]
pub(crate) struct SessionAlignmentState {
    aliases: HashMap<String, SessionAlias>,
    completed_aliases: HashMap<String, SessionAlias>,
    pending_subagents: HashMap<String, PendingSubagentStart>,
    hermes_task_sessions: HashMap<String, HashMap<String, String>>,
}

impl SessionAlignmentState {
    pub(crate) fn clear(&mut self) {
        self.aliases.clear();
        self.completed_aliases.clear();
        self.pending_subagents.clear();
        self.hermes_task_sessions.clear();
    }

    pub(crate) fn alias_for_session(&self, session_id: &str) -> Option<SessionAlias> {
        self.aliases.get(session_id).cloned()
    }

    #[cfg(test)]
    pub(crate) fn has_alias(&self, session_id: &str) -> bool {
        self.aliases.contains_key(session_id)
    }

    #[cfg(test)]
    pub(crate) fn has_pending_session(&self, session_id: &str) -> bool {
        self.pending_subagents.contains_key(session_id)
    }

    pub(crate) fn pending_for_session(&mut self, session_id: &str) -> Option<PendingSubagentStart> {
        self.pending_subagents.remove(session_id)
    }

    pub(crate) fn insert_pending(
        &mut self,
        child_session_id: String,
        pending: PendingSubagentStart,
    ) {
        self.pending_subagents.insert(child_session_id, pending);
    }

    pub(crate) fn remove_pending(&mut self, child_session_id: &str) {
        self.pending_subagents.remove(child_session_id);
    }

    pub(crate) fn insert_alias(&mut self, child_session_id: String, alias: SessionAlias) {
        self.aliases.insert(child_session_id, alias);
    }

    pub(crate) fn route_event(&mut self, event: NormalizedEvent) -> NormalizedEvent {
        self.record_hermes_task_session(&event);
        let event = self.route_hermes_task_event(event);
        let (event, finished_alias) = route_event_through_alias(event, &self.aliases);
        let session_id = event.session_id().to_string();
        if let Some(child_session_id) = finished_alias.as_ref() {
            // Remove aliases before terminal skip checks so a late child AgentEnd, or a child
            // TurnEnded used as a subagent completion signal, cannot leave stale reparenting state.
            if let Some(alias) = self.aliases.remove(child_session_id) {
                self.completed_aliases
                    .insert(child_session_id.clone(), alias);
            }
            self.pending_subagents.remove(child_session_id);
        }
        if matches!(&event, NormalizedEvent::AgentEnded(_)) {
            self.clear_for_ended_agent(&session_id);
        }
        event
    }

    pub(crate) fn align_explicit_subagent_end(&mut self, event: &mut NormalizedEvent) {
        let NormalizedEvent::SubagentEnded(subagent_event) = event else {
            return;
        };
        let Some(child_session_id) = hermes::child_session_id_for_subagent_event(subagent_event)
        else {
            return;
        };
        let Some(alias) = self
            .aliases
            .get(&child_session_id)
            .or_else(|| self.completed_aliases.get(&child_session_id))
            .cloned()
        else {
            return;
        };
        if subagent_event.session_id != alias.parent_session_id {
            return;
        }
        subagent_event.subagent_id = alias.subagent_id.clone();
        subagent_event.metadata = merge_metadata(subagent_event.metadata.clone(), alias.metadata());
        self.aliases.remove(&child_session_id);
        self.completed_aliases.remove(&child_session_id);
    }

    pub(crate) fn pending_for_parent(
        &mut self,
        parent_session_id: &str,
    ) -> Vec<(String, PendingSubagentStart)> {
        let child_session_ids = self
            .pending_subagents
            .iter()
            .filter_map(|(child_session_id, pending)| {
                (pending.parent_session_id() == parent_session_id)
                    .then_some(child_session_id.clone())
            })
            .collect::<Vec<_>>();
        child_session_ids
            .into_iter()
            .filter_map(|child_session_id| {
                self.pending_subagents
                    .remove(&child_session_id)
                    .map(|pending| (child_session_id, pending))
            })
            .collect()
    }

    pub(crate) fn clear_for_ended_agent(&mut self, session_id: &str) {
        self.aliases.retain(|child_session_id, alias| {
            child_session_id != session_id && alias.parent_session_id != session_id
        });
        self.completed_aliases.retain(|child_session_id, alias| {
            child_session_id != session_id && alias.parent_session_id != session_id
        });
        self.pending_subagents.retain(|child_session_id, pending| {
            child_session_id != session_id && pending.parent_session_id() != session_id
        });
        self.hermes_task_sessions.remove(session_id);
        prune_hermes_task_sessions(&mut self.hermes_task_sessions, session_id);
    }

    pub(crate) fn clear_for_ended_subagent(&mut self, parent_session_id: &str, subagent_id: &str) {
        self.aliases.retain(|child_session_id, alias| {
            child_session_id != subagent_id
                && !(alias.parent_session_id == parent_session_id
                    && alias.subagent_id == subagent_id)
        });
        self.pending_subagents.retain(|child_session_id, pending| {
            child_session_id != subagent_id
                && !(pending.parent_session_id() == parent_session_id
                    && pending.event.session_id == subagent_id)
        });
        self.hermes_task_sessions
            .retain(|session_id, _| session_id != subagent_id);
        prune_hermes_task_sessions(&mut self.hermes_task_sessions, subagent_id);
    }

    fn record_hermes_task_session(&mut self, event: &NormalizedEvent) {
        if normalized_event_agent_kind(event) != AgentKind::Hermes {
            return;
        }
        let Some(task_id) = hermes_task_id(event) else {
            return;
        };
        let session_id = event.session_id();
        if session_id == task_id {
            return;
        }
        self.hermes_task_sessions
            .entry(session_id.to_string())
            .or_default()
            .insert(task_id, session_id.to_string());
    }

    fn route_hermes_task_event(&self, event: NormalizedEvent) -> NormalizedEvent {
        let should_route = matches!(
            event,
            NormalizedEvent::ToolStarted(_) | NormalizedEvent::ToolEnded(_)
        ) && normalized_event_agent_kind(&event) == AgentKind::Hermes;
        if !should_route {
            return event;
        }

        let task_id = hermes_task_id(&event).unwrap_or_else(|| event.session_id().to_string());
        let session_scope = hermes_task_session_scope(&event);
        let Some(session_id) = self.hermes_session_for_task(&task_id, session_scope.as_deref())
        else {
            return event;
        };
        route_hermes_task_session_event(event, task_id, session_id)
    }

    fn hermes_session_for_task(
        &self,
        task_id: &str,
        session_scope: Option<&str>,
    ) -> Option<String> {
        if let Some(session_scope) = session_scope {
            return self
                .hermes_task_sessions
                .get(session_scope)
                .and_then(|tasks| tasks.get(task_id))
                .cloned();
        }

        let mut matches = self
            .hermes_task_sessions
            .values()
            .filter_map(|tasks| tasks.get(task_id).cloned());
        let session_id = matches.next()?;
        matches.next().is_none().then_some(session_id)
    }
}

fn prune_hermes_task_sessions(
    hermes_task_sessions: &mut HashMap<String, HashMap<String, String>>,
    session_id: &str,
) {
    hermes_task_sessions.values_mut().for_each(|tasks| {
        tasks.retain(|task_id, mapped_session_id| {
            task_id != session_id && mapped_session_id != session_id
        });
    });
    hermes_task_sessions.retain(|_, tasks| !tasks.is_empty());
}

// Resolves the session id for a gateway request in precedence order:
// explicit NeMo Relay header, agent-native headers, agent-specific body fallbacks, then the
// generic OpenAI-compatible `session_id` body field. Keeping the provider fallbacks behind one
// function makes a new agent integration add one small alignment adapter instead of threading
// bespoke checks through gateway request construction.
pub(crate) fn gateway_session_id(
    headers: &HeaderMap,
    body: &Value,
    route: GatewayRouteKind,
) -> Option<String> {
    header_string(headers, "x-nemo-relay-session-id")
        .or_else(|| claude_code::session_id_from_headers(headers))
        .or_else(|| codex::prompt_cache_session_id(body, route))
        .or_else(|| openai_body_session_id(body, route))
}

fn openai_body_session_id(body: &Value, route: GatewayRouteKind) -> Option<String> {
    if !matches!(
        route,
        GatewayRouteKind::OpenAiChatCompletions | GatewayRouteKind::OpenAiResponses
    ) {
        return None;
    }
    body.get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|session_id| !session_id.is_empty())
        .map(ToOwned::to_owned)
}

// Gives provider adapters a chance to select an agent-native upstream before the gateway falls
// back to configured provider bases. Codex uses this for ChatGPT OAuth tokens that target the
// ChatGPT backend instead of the public OpenAI API.
pub(crate) fn gateway_upstream_url_override(
    headers: &HeaderMap,
    route: GatewayRouteKind,
    path_and_query: &str,
    has_openai_replacement_key: bool,
) -> Option<String> {
    codex::chatgpt_upstream_url_if_needed(
        headers,
        route,
        path_and_query,
        has_openai_replacement_key,
    )
}

// Lets provider adapters remove or preserve agent-native auth material before generic provider
// auth injection runs. Codex strips ChatGPT OAuth JWTs only when an OpenAI API key is available to
// replace them.
pub(crate) fn gateway_forward_headers(
    headers: &HeaderMap,
    route: GatewayRouteKind,
    has_openai_replacement_key: bool,
) -> HeaderMap {
    codex::strip_chatgpt_oauth_for_openai_route(headers, route, has_openai_replacement_key)
}

// Reads the explicit subagent header that callers can use when the gateway request already knows
// the target worker scope. Unlike session ids, there is intentionally no body fallback here:
// subagent body fields are provider-specific and easy to confuse with tool-call payload content.
pub(crate) fn gateway_subagent_id(headers: &HeaderMap) -> Option<String> {
    header_string(headers, "x-nemo-relay-subagent-id")
}

// Resolves a correlation identifier from a dedicated header before trying known JSON body paths.
// Header precedence lets callers disambiguate requests even when provider payloads contain stale
// or differently scoped identifiers.
pub(crate) fn gateway_identifier(
    headers: &HeaderMap,
    body: &Value,
    header_name: &'static str,
    body_paths: &[&[&str]],
) -> Option<String> {
    header_string(headers, header_name).or_else(|| json_string_at(body, body_paths))
}

// Infers the owning agent for a session created by a gateway request that beat its SessionStart
// hook. This is the last chance to label the root scope correctly because exporter identities are
// baked when the scope opens.
pub(crate) fn agent_kind_for_gateway_provider(provider: &str) -> AgentKind {
    if claude_code::owns_gateway_provider(provider) {
        AgentKind::ClaudeCode
    } else if codex::owns_gateway_provider(provider) {
        AgentKind::Codex
    } else {
        AgentKind::Gateway
    }
}

// Not every harness has a reliable process/session end signal. Claude Code and Codex sessions can
// outlive a user-visible run, so the CLI represents their work with bounded turn scopes instead of
// exporting a long-lived agent scope that needs synthetic termination.
pub(crate) fn should_emit_session_agent_scope(agent_kind: AgentKind) -> bool {
    !matches!(agent_kind, AgentKind::ClaudeCode | AgentKind::Codex)
}

// Detects agent harnesses that report a child session which should become a subagent under another
// session. Codex is the first such harness: it starts child threads with parent-thread metadata.
// Future harness-specific detectors should plug in here so the session manager can stay provider
// neutral.
pub(crate) async fn subagent_session_context(
    event: &SessionEvent,
) -> Option<SubagentSessionContext> {
    codex::subagent_context(event)
        .await
        .map(SubagentSessionContext::Codex)
        .or_else(|| hermes::subagent_context(event).map(SubagentSessionContext::Hermes))
}

// Converts an AgentStarted event into a pending child-session record when a harness explicitly
// reports parentage. The caller still decides whether the child session is empty enough to promote.
pub(crate) async fn pending_subagent_start(
    event: &mut NormalizedEvent,
) -> Option<(String, PendingSubagentStart)> {
    let NormalizedEvent::AgentStarted(session_event) = event else {
        return None;
    };
    let context = subagent_session_context(session_event).await?;
    let child_session_id = session_event.session_id.clone();
    if context.parent_session_id() == child_session_id {
        return None;
    }
    session_event.metadata =
        augment_subagent_session_metadata(session_event.metadata.clone(), &context);
    Some((
        child_session_id,
        PendingSubagentStart {
            event: session_event.clone(),
            context,
        },
    ))
}

// Lets the owning alignment adapter stamp provider-specific debug fields on a child SessionStart
// before the generic session manager promotes it to a subagent.
pub(crate) fn augment_subagent_session_metadata(
    metadata: Value,
    context: &SubagentSessionContext,
) -> Value {
    match context {
        SubagentSessionContext::Codex(context) => {
            codex::augment_subagent_metadata(metadata, context)
        }
        SubagentSessionContext::Hermes(context) => {
            hermes::augment_subagent_metadata(metadata, context)
        }
    }
}

// Converts a child SessionStart into the provider-appropriate SubagentStarted event. The session
// manager only knows that a child session should be promoted; the adapter owns how to preserve the
// provider's original metadata.
pub(crate) fn subagent_start_event(
    event: &SessionEvent,
    context: &SubagentSessionContext,
) -> SubagentEvent {
    match context {
        SubagentSessionContext::Codex(context) => codex::subagent_start_event(event, context),
        SubagentSessionContext::Hermes(context) => hermes::subagent_start_event(event, context),
    }
}

// Builds the alias used to route later child-session events through the promoted parent/subagent
// pair. The adapter supplies provider-specific metadata explaining why the alias exists.
pub(crate) fn alias_for_child_session(
    child_session_id: String,
    context: &SubagentSessionContext,
) -> SessionAlias {
    match context {
        SubagentSessionContext::Codex(context) => {
            codex::alias_for_child_session(child_session_id, context)
        }
        SubagentSessionContext::Hermes(context) => {
            hermes::alias_for_child_session(child_session_id, context)
        }
    }
}

pub(crate) fn explicit_subagent_alias(
    event: &mut NormalizedEvent,
) -> Option<(String, SessionAlias)> {
    let NormalizedEvent::SubagentStarted(subagent_event) = event else {
        return None;
    };
    let explicit = hermes::explicit_subagent_alias(subagent_event)?;
    subagent_event.metadata =
        merge_metadata(subagent_event.metadata.clone(), explicit.scope_metadata);
    Some((explicit.child_session_id, explicit.alias))
}

// Recovers provider-specific metadata from a subagent scope and copies only the fields that should
// follow LLM spans. Codex contributes thread identifiers today; other harnesses can add filters
// here without changing session ownership code.
pub(crate) fn llm_owner_metadata(scope_metadata: Option<&Value>) -> Value {
    merge_metadata(
        codex::llm_owner_metadata(scope_metadata),
        hermes::llm_owner_metadata(scope_metadata),
    )
}

// Builds a provider-neutral affinity key from the user task text inside common LLM request
// formats. Coding agents often replay the same task prompt on later provider calls without a
// worker id; this key lets session correlation pair those calls with the subagent that first owned
// the task. The extractor understands Anthropic Messages, OpenAI Chat Completions, and OpenAI
// Responses shapes, and deliberately ignores raw count-token/file payloads.
pub(crate) fn request_affinity_key(request: &LlmRequest) -> Option<String> {
    let task_text = request_user_task_text(&request.content)?;
    let normalized = normalize_affinity_text(&task_text);
    (normalized.chars().count() >= REQUEST_AFFINITY_KEY_MIN_CHARS)
        .then(|| truncate_affinity_text(&normalized, REQUEST_AFFINITY_KEY_MAX_CHARS))
}

// Detects tool results that imply a subagent completed. Claude Code reports this through the
// `Agent` tool today; keeping the check here avoids leaking that tool shape into session teardown.
pub(crate) fn completed_subagent_from_tool(event: &ToolEvent) -> Option<String> {
    claude_code::completed_subagent_from_agent_tool(event)
}

// Some harnesses route child-session turn boundaries through a parent-owned subagent alias. A
// child turn end should close that subagent, not the parent turn containing all sibling work.
pub(crate) fn aliased_turn_subagent_id(event: &SessionEvent) -> Option<String> {
    json_string_at(
        &event.metadata,
        &[
            &["hermes_child_subagent_id"][..],
            &["subagent_id"][..],
            &["codex_subagent_session_id"][..],
            &["subagent_session_id"][..],
        ],
    )
}

// Routes events from an aliased child session through the parent session/subagent pair. The alias
// records why the child is not a top-level agent; this generic router only rewrites ownership and
// preserves the adapter-supplied metadata for filtering/debugging in Phoenix.
pub(crate) fn route_event_through_alias(
    event: NormalizedEvent,
    aliases: &HashMap<String, SessionAlias>,
) -> (NormalizedEvent, Option<String>) {
    let child_session_id = event.session_id().to_string();
    let Some(alias) = aliases.get(&child_session_id).cloned() else {
        return (event, None);
    };
    let metadata = alias.metadata();
    match event {
        NormalizedEvent::AgentStarted(event) => (
            NormalizedEvent::SubagentStarted(SubagentEvent {
                session_id: alias.parent_session_id,
                agent_kind: event.agent_kind,
                event_name: event.event_name,
                subagent_id: alias.subagent_id,
                payload: event.payload,
                metadata: merge_metadata(event.metadata, metadata),
            }),
            None,
        ),
        NormalizedEvent::AgentEnded(event) => (
            NormalizedEvent::SubagentEnded(SubagentEvent {
                session_id: alias.parent_session_id,
                agent_kind: event.agent_kind,
                event_name: event.event_name,
                subagent_id: alias.subagent_id,
                payload: event.payload,
                metadata: merge_metadata(event.metadata, metadata),
            }),
            Some(child_session_id),
        ),
        NormalizedEvent::TurnEnded(mut event) => {
            route_session_event(&mut event, &alias, metadata);
            (NormalizedEvent::TurnEnded(event), Some(child_session_id))
        }
        NormalizedEvent::PromptSubmitted(mut event) => {
            route_session_event(&mut event, &alias, metadata);
            (NormalizedEvent::PromptSubmitted(event), None)
        }
        NormalizedEvent::Compaction(mut event) => {
            route_session_event(&mut event, &alias, metadata);
            (NormalizedEvent::Compaction(event), None)
        }
        NormalizedEvent::Notification(mut event) => {
            route_session_event(&mut event, &alias, metadata);
            (NormalizedEvent::Notification(event), None)
        }
        NormalizedEvent::HookMark(mut event) => {
            route_session_event(&mut event, &alias, metadata);
            (NormalizedEvent::HookMark(event), None)
        }
        NormalizedEvent::SubagentStarted(mut event) => {
            route_subagent_event(&mut event, &alias, metadata);
            (NormalizedEvent::SubagentStarted(event), None)
        }
        NormalizedEvent::SubagentEnded(mut event) => {
            route_subagent_event(&mut event, &alias, metadata);
            (NormalizedEvent::SubagentEnded(event), None)
        }
        NormalizedEvent::LlmHint(mut event) => {
            event.session_id = alias.parent_session_id;
            event.subagent_id = Some(alias.subagent_id);
            event.metadata = merge_metadata(event.metadata, metadata);
            (NormalizedEvent::LlmHint(event), None)
        }
        NormalizedEvent::LlmStarted(mut event) => {
            route_llm_event(&mut event, &alias, metadata);
            (NormalizedEvent::LlmStarted(event), None)
        }
        NormalizedEvent::LlmEnded(mut event) => {
            route_llm_event(&mut event, &alias, metadata);
            (NormalizedEvent::LlmEnded(event), None)
        }
        NormalizedEvent::ToolStarted(mut event) => {
            route_tool_event(&mut event, &alias, metadata);
            (NormalizedEvent::ToolStarted(event), None)
        }
        NormalizedEvent::ToolEnded(mut event) => {
            route_tool_event(&mut event, &alias, metadata);
            (NormalizedEvent::ToolEnded(event), None)
        }
    }
}

// Rewrites session-level child events after an alias match. These events still describe the parent
// session's timeline once the child thread is known to be a subagent, so only the session id and
// debug metadata change.
fn route_session_event(event: &mut SessionEvent, alias: &SessionAlias, metadata: Value) {
    event.session_id = alias.parent_session_id.clone();
    event.metadata = merge_metadata(event.metadata.clone(), metadata);
}

// Rewrites nested subagent events that appeared inside an aliased child session. The inner
// subagent id remains intact, while the containing session id moves to the real parent session.
fn route_subagent_event(event: &mut SubagentEvent, alias: &SessionAlias, metadata: Value) {
    event.session_id = alias.parent_session_id.clone();
    event.metadata = merge_metadata(event.metadata.clone(), metadata);
}

// Rewrites hook-originated LLM events from aliased child sessions. `LlmEvent` does not have a
// first-class subagent id field, so the alias owner is stamped into metadata where the session
// manager's hook-LLM path can recover it and choose the subagent scope.
fn route_llm_event(event: &mut LlmEvent, alias: &SessionAlias, metadata: Value) {
    event.session_id = alias.parent_session_id.clone();
    event.metadata = merge_metadata(
        event.metadata.clone(),
        merge_metadata(
            metadata,
            json!({
                "llm_correlation_status": "session_alias",
                "llm_correlation_source": "session_alias",
                "llm_correlation_subagent_id": alias.subagent_id.clone(),
            }),
        ),
    );
}

// Rewrites tool calls emitted by an aliased child session so they attach under the aliased
// subagent. This is the common case for Codex child-thread tool activity that would otherwise show
// up as root-agent tool calls.
fn route_tool_event(event: &mut ToolEvent, alias: &SessionAlias, metadata: Value) {
    event.session_id = alias.parent_session_id.clone();
    event.subagent_id = Some(alias.subagent_id.clone());
    event.metadata = merge_metadata(event.metadata.clone(), metadata);
}

fn route_hermes_task_session_event(
    event: NormalizedEvent,
    task_id: String,
    session_id: String,
) -> NormalizedEvent {
    let metadata = json!({
        "session_correlation_status": "task_session_alias",
        "session_correlation_source": "hermes_task_id",
        "hermes_task_id": task_id,
        "hermes_session_id": session_id,
    });
    match event {
        NormalizedEvent::ToolStarted(mut event) => {
            event.session_id = session_id;
            event.metadata = merge_metadata(event.metadata, metadata);
            NormalizedEvent::ToolStarted(event)
        }
        NormalizedEvent::ToolEnded(mut event) => {
            event.session_id = session_id;
            event.metadata = merge_metadata(event.metadata, metadata);
            NormalizedEvent::ToolEnded(event)
        }
        event => event,
    }
}

fn normalized_event_agent_kind(event: &NormalizedEvent) -> AgentKind {
    match event {
        NormalizedEvent::AgentStarted(event)
        | NormalizedEvent::AgentEnded(event)
        | NormalizedEvent::TurnEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => event.agent_kind,
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            event.agent_kind
        }
        NormalizedEvent::LlmHint(event) => event.agent_kind,
        NormalizedEvent::LlmStarted(event) | NormalizedEvent::LlmEnded(event) => event.agent_kind,
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => event.agent_kind,
    }
}

fn hermes_task_id(event: &NormalizedEvent) -> Option<String> {
    match event {
        NormalizedEvent::AgentStarted(event)
        | NormalizedEvent::AgentEnded(event)
        | NormalizedEvent::TurnEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => {
            task_id_from_payload_and_metadata(&event.payload, &event.metadata)
        }
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            task_id_from_payload_and_metadata(&event.payload, &event.metadata)
        }
        NormalizedEvent::LlmHint(event) => {
            task_id_from_payload_and_metadata(&event.payload, &event.metadata)
        }
        NormalizedEvent::LlmStarted(event) | NormalizedEvent::LlmEnded(event) => {
            task_id_from_llm_event(event)
        }
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => {
            task_id_from_payload_and_metadata(&event.payload, &event.metadata)
        }
    }
}

fn hermes_task_session_scope(event: &NormalizedEvent) -> Option<String> {
    match event {
        NormalizedEvent::AgentStarted(event)
        | NormalizedEvent::AgentEnded(event)
        | NormalizedEvent::TurnEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => {
            session_scope_from_payload_and_metadata(&event.payload, &event.metadata)
        }
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            session_scope_from_payload_and_metadata(&event.payload, &event.metadata)
        }
        NormalizedEvent::LlmHint(event) => {
            session_scope_from_payload_and_metadata(&event.payload, &event.metadata)
        }
        NormalizedEvent::LlmStarted(event) | NormalizedEvent::LlmEnded(event) => {
            session_scope_from_llm_event(event)
        }
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => {
            session_scope_from_payload_and_metadata(&event.payload, &event.metadata)
        }
    }
}

fn task_id_from_llm_event(event: &LlmEvent) -> Option<String> {
    task_id_from_payload_and_metadata(&event.request, &event.metadata)
        .or_else(|| task_id_from_payload_and_metadata(&event.response, &event.metadata))
}

fn task_id_from_payload_and_metadata(payload: &Value, metadata: &Value) -> Option<String> {
    json_string_at(payload, TASK_ID_PATHS).or_else(|| json_string_at(metadata, TASK_ID_PATHS))
}

fn session_scope_from_llm_event(event: &LlmEvent) -> Option<String> {
    session_scope_from_payload_and_metadata(&event.request, &event.metadata)
        .or_else(|| session_scope_from_payload_and_metadata(&event.response, &event.metadata))
}

fn session_scope_from_payload_and_metadata(payload: &Value, metadata: &Value) -> Option<String> {
    json_string_at(payload, HERMES_SESSION_SCOPE_PATHS)
        .or_else(|| json_string_at(metadata, HERMES_SESSION_SCOPE_PATHS))
}

const TASK_ID_PATHS: &[&[&str]] = &[
    &["task_id"],
    &["taskId"],
    &["extra", "task_id"],
    &["extra", "taskId"],
];

const HERMES_SESSION_SCOPE_PATHS: &[&[&str]] = &[
    &["session_id"],
    &["sessionId"],
    &["session", "id"],
    &["conversation_id"],
    &["conversationId"],
    &["parent_session_id"],
    &["parentSessionId"],
    &["extra", "session_id"],
    &["extra", "sessionId"],
    &["extra", "parent_session_id"],
    &["extra", "parentSessionId"],
];

fn request_user_task_text(payload: &Value) -> Option<String> {
    payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.iter().rev().find_map(user_message_task_text))
        .or_else(|| responses_input_task_text(payload.get("input")?))
        .or_else(|| prompt_task_text(payload.get("prompt")?))
}

fn user_message_task_text(message: &Value) -> Option<String> {
    if message.get("role").and_then(Value::as_str) != Some("user") {
        return None;
    }
    content_task_text(message.get("content")?)
}

fn responses_input_task_text(input: &Value) -> Option<String> {
    match input {
        Value::String(text) => affinity_candidate_text(text),
        Value::Array(items) => items.iter().rev().find_map(user_message_task_text),
        _ => None,
    }
}

fn prompt_task_text(prompt: &Value) -> Option<String> {
    prompt.as_str().and_then(affinity_candidate_text)
}

fn content_task_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => affinity_candidate_text(text),
        Value::Array(blocks) => blocks.iter().rev().find_map(content_block_task_text),
        _ => None,
    }
}

fn content_block_task_text(block: &Value) -> Option<String> {
    if let Some(block_type) = block.get("type").and_then(Value::as_str)
        && !matches!(block_type, "text" | "input_text")
    {
        return None;
    }
    block
        .get("text")
        .and_then(Value::as_str)
        .and_then(affinity_candidate_text)
}

fn affinity_candidate_text(text: &str) -> Option<String> {
    let cleaned = text.trim();
    if cleaned.is_empty() || looks_like_json_payload(cleaned) {
        return None;
    }
    Some(cleaned.to_string())
}

fn looks_like_json_payload(text: &str) -> bool {
    let trimmed = text.trim_start();
    if !matches!(trimmed.chars().next(), Some('{' | '[')) {
        return false;
    }
    matches!(
        serde_json::from_str::<Value>(trimmed),
        Ok(Value::Object(_) | Value::Array(_))
    )
}

fn normalize_affinity_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_affinity_text(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

// Reads the first string-like value from any candidate JSON path. Scalar numbers and booleans are
// accepted for IDs because provider payloads are not always strict about identifier types.
pub(crate) fn json_string_at(payload: &Value, paths: &[&[&str]]) -> Option<String> {
    json_value_at(payload, paths)
        .and_then(|value| match value {
            Value::String(value) => Some(value),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            _ => None,
        })
        .filter(|value| !value.is_empty())
}

// Reads the first JSON value from any candidate path. The clone is intentional because extracted
// correlation data must live independently of the provider payload it was read from.
pub(crate) fn json_value_at(payload: &Value, paths: &[&[&str]]) -> Option<Value> {
    paths.iter().find_map(|path| {
        let mut current = payload;
        for key in *path {
            current = current.get(*key)?;
        }
        Some(current.clone())
    })
}

// Inserts an optional string value into a JSON object while omitting absent fields entirely. This
// keeps correlation metadata compact and avoids serializing nulls as meaningful observations.
pub(crate) fn insert_optional(object: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_string(), json!(value));
    }
}

// Merges metadata objects with right-hand values taking precedence and null right-hand fields
// ignored. Non-object values are preserved under separate keys so callers do not lose unusual
// metadata shapes supplied by configuration or hooks.
pub(crate) fn merge_metadata(left: Value, right: Value) -> Value {
    match (left, right) {
        (Value::Object(mut left), Value::Object(right)) => {
            for (key, value) in right {
                if !value.is_null() {
                    left.insert(key, value);
                }
            }
            Value::Object(left)
        }
        (Value::Null, right) => right,
        (left, Value::Null) => left,
        (left, right) => {
            let mut object = Map::new();
            object.insert("metadata".into(), left);
            object.insert("extra_metadata".into(), right);
            Value::Object(object)
        }
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/alignment_tests.rs"]
mod tests;
