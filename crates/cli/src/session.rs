// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use nemo_flow::api::llm::{
    LlmAttributes, LlmCallEndParams, LlmCallParams, LlmHandle, LlmRequest, llm_call, llm_call_end,
};
use nemo_flow::api::runtime::{ScopeStackHandle, TASK_SCOPE_STACK, create_scope_stack};
use nemo_flow::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeHandle, ScopeType,
    event as emit_mark_event, get_handle, pop_scope, push_scope,
};
use nemo_flow::api::tool::{
    ToolCallEndParams, ToolCallParams, ToolHandle, tool_call, tool_call_end,
};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::config::{GatewayConfig, SessionConfig};
use crate::error::CliError;
use crate::model::{
    AgentKind, LlmEvent, LlmHintEvent, NormalizedEvent, SessionEvent, SubagentEvent, ToolEvent,
};

const LLM_HINT_TTL: Duration = Duration::from_secs(300);
const TOOL_HINT_TTL: Duration = Duration::from_secs(300);
const LAST_OWNER_TTL: Duration = Duration::from_secs(300);

#[derive(Clone)]
pub(crate) struct SessionManager {
    inner: Arc<Mutex<HashMap<String, Session>>>,
    default_config: GatewayConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct LlmGatewayStart {
    pub(crate) session_id: Option<String>,
    pub(crate) provider: String,
    pub(crate) model_name: Option<String>,
    pub(crate) subagent_id: Option<String>,
    pub(crate) conversation_id: Option<String>,
    pub(crate) generation_id: Option<String>,
    pub(crate) request_id: Option<String>,
    pub(crate) request: LlmRequest,
    pub(crate) streaming: bool,
    pub(crate) metadata: Value,
}

/// Legacy active-LLM record kept for tests that exercise the manual `llm_call` /
/// `llm_call_end` correlation path. Production gateway traffic now uses managed execution via
/// [`SessionManager::prepare_gateway_call`].
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct ActiveLlm {
    stack: ScopeStackHandle,
    handle: LlmHandle,
    session_id: String,
    owner_subagent_id: Option<String>,
}

/// Inputs prepared by [`SessionManager::prepare_gateway_call`] for invoking the
/// runtime's managed LLM execution pipeline outside the session lock.
///
/// The session lock is released after the prep is built, so the gateway can run
/// the upstream HTTP work without blocking unrelated session activity. The
/// preserved `scope_stack` is what restores the agent/subagent scope context
/// the call was opened against when the runtime emits start/end events.
pub(crate) struct GatewayCallPrep {
    pub(crate) scope_stack: ScopeStackHandle,
    pub(crate) session_id: String,
    pub(crate) provider_name: String,
    pub(crate) request: LlmRequest,
    pub(crate) parent: Option<ScopeHandle>,
    pub(crate) attributes: LlmAttributes,
    pub(crate) metadata: Value,
    pub(crate) model_name: Option<String>,
    pub(crate) owner_subagent_id: Option<String>,
}

struct Session {
    agent_kind: AgentKind,
    session_id: String,
    scope_stack: ScopeStackHandle,
    agent_scope: Option<ScopeHandle>,
    subagents: HashMap<String, ScopeHandle>,
    subagent_stack: Vec<String>,
    llms: HashMap<String, LlmHandle>,
    tools: HashMap<String, ToolHandle>,
    pending_llm_hints: Vec<PendingLlmHint>,
    pending_tool_hints: Vec<PendingToolHint>,
    last_llm_owner: Option<LastLlmOwner>,
    config: SessionConfig,
}

#[derive(Debug, Clone)]
struct PendingLlmHint {
    hint: LlmHintEvent,
    inserted_at: Instant,
}

#[derive(Debug, Clone)]
struct PendingToolHint {
    hint: ToolHint,
    inserted_at: Instant,
}

#[derive(Debug, Clone)]
struct ToolHint {
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    subagent_id: Option<String>,
    arguments: Value,
    source: String,
}

#[derive(Debug, Clone)]
struct LastLlmOwner {
    subagent_id: Option<String>,
    updated_at: Instant,
}

struct LlmOwnerResolution {
    parent: Option<ScopeHandle>,
    subagent_id: Option<String>,
    status: &'static str,
    source: Option<String>,
    hint: Option<LlmHintEvent>,
}

struct ToolOwnerResolution {
    parent: Option<ScopeHandle>,
    subagent_id: Option<String>,
    status: &'static str,
    source: Option<String>,
    hint: Option<ToolHint>,
}

impl SessionManager {
    /// Creates an empty manager that uses the supplied gateway config as the header fallback layer.
    ///
    /// Sessions are stored behind a shared async mutex because hook requests and gateway requests
    /// may arrive concurrently and need to resolve LLM ownership against the same in-memory state.
    pub(crate) fn new(default_config: GatewayConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            default_config,
        }
    }

    /// Applies normalized hook events to their owning sessions in arrival order.
    ///
    /// Session configuration is re-read from headers for each request so installed hook commands can
    /// override metadata per invocation. Empty sessions are removed after lifecycle
    /// closure to avoid retaining stale correlation state.
    ///
    /// When an `AgentStarted` event arrives for a session that was already created by the gateway
    /// path (i.e., agent_kind is still `Gateway` because an LLM call beat the SessionStart hook),
    /// upgrade the session's agent_kind to the real one carried in the event so subsequent
    /// metadata reflects the actual agent. Note: agent-scope and observer identities are baked at
    /// scope-open time, so this upgrade applies to session metadata only — the
    /// provider-inferred kind set in `start_llm` is the primary defense.
    pub(crate) async fn apply_events(
        &self,
        headers: &HeaderMap,
        events: Vec<NormalizedEvent>,
    ) -> Result<(), CliError> {
        let mut sessions = self.inner.lock().await;
        for event in events {
            let session_id = event.session_id().to_string();
            if event.is_terminal() && !sessions.contains_key(&session_id) {
                continue;
            }
            let config = self.default_config.session_config_from_headers(headers);
            let event_kind = event_agent_kind(&event);
            let session = sessions
                .entry(session_id.clone())
                .or_insert_with(|| Session::new(session_id.clone(), event_kind, config.clone()));
            if matches!(&event, NormalizedEvent::AgentStarted(_))
                && session.agent_kind == AgentKind::Gateway
                && event_kind != AgentKind::Gateway
            {
                session.agent_kind = event_kind;
            }
            session.apply(event).await?;
            if session.agent_scope.is_none()
                && session.subagents.is_empty()
                && session.subagent_stack.is_empty()
                && session.llms.is_empty()
                && session.tools.is_empty()
            {
                sessions.remove(&session_id);
            }
        }
        Ok(())
    }

    /// Legacy manual-lifecycle entry point retained for tests that drive correlation behavior
    /// directly. Production gateway traffic uses [`Self::prepare_gateway_call`] +
    /// `llm_call_execute` / `llm_stream_call_execute` so the runtime owns start/end events.
    ///
    /// Explicit session IDs win, a single active hook session is reused as a convenience fallback,
    /// and otherwise a synthetic gateway session is created so pure proxy use still emits runtime
    /// events. When this path creates a brand-new session (i.e., a real agent's gateway request
    /// beat its SessionStart hook), the session's agent_kind is inferred from the gateway provider
    /// rather than defaulting to `Gateway`. Without this inference, the session's exported agent
    /// name (in ATIF and Phoenix scope spans) would freeze as "gateway" for the lifetime of the
    /// session, even after a SessionStart hook arrives, because observer identities are baked at
    /// scope-open time. With it, an Anthropic Messages call before SessionStart still labels the
    /// trace as `claude-code`, an OpenAI Responses call as `codex`, etc.
    #[cfg(test)]
    pub(crate) async fn start_llm(
        &self,
        headers: &HeaderMap,
        start: LlmGatewayStart,
    ) -> Result<ActiveLlm, CliError> {
        let mut sessions = self.inner.lock().await;
        let config = self.default_config.session_config_from_headers(headers);
        let session_id = start
            .session_id
            .clone()
            .or_else(|| single_active_session_id(&sessions))
            .unwrap_or_else(|| format!("{}-gateway", AgentKind::Gateway.as_str()));
        let inferred_agent_kind = agent_kind_for_gateway_provider(&start.provider);
        let session = sessions
            .entry(session_id.clone())
            .or_insert_with(|| Session::new(session_id, inferred_agent_kind, config));
        session.start_llm(start).await
    }

    /// Prepares a managed LLM execution against the right session and scope context.
    ///
    /// Resolves the session, opens the agent scope if needed, computes the parent scope and
    /// correlation metadata, and returns a [`GatewayCallPrep`]. The returned prep carries the
    /// `ScopeStackHandle` that callers must restore around `llm_call_execute` /
    /// `llm_stream_call_execute` so the runtime emits start/end events under the same agent or
    /// subagent scope the prep was opened under.
    ///
    /// The session manager lock is held only long enough to build the prep; the actual upstream
    /// HTTP and managed pipeline run outside the lock.
    pub(crate) async fn prepare_gateway_call(
        &self,
        headers: &HeaderMap,
        start: LlmGatewayStart,
    ) -> Result<GatewayCallPrep, CliError> {
        let mut sessions = self.inner.lock().await;
        let config = self.default_config.session_config_from_headers(headers);
        let session_id = start
            .session_id
            .clone()
            .or_else(|| single_active_session_id(&sessions))
            .unwrap_or_else(|| format!("{}-gateway", AgentKind::Gateway.as_str()));
        // Match `start_llm`: when this path creates a brand-new session (real agent's gateway
        // request beats its SessionStart hook), label the session by the provider so ATIF and
        // Phoenix scopes carry the agent identity instead of freezing on "gateway".
        let inferred_agent_kind = agent_kind_for_gateway_provider(&start.provider);
        let session = sessions
            .entry(session_id.clone())
            .or_insert_with(|| Session::new(session_id, inferred_agent_kind, config));
        session.prepare_gateway_call(start).await
    }

    /// Legacy manual-lifecycle close paired with [`Self::start_llm`]. Production gateway traffic
    /// no longer needs this helper because managed execution emits the end event automatically.
    ///
    /// The captured stack is restored around `llm_call_end` so asynchronous gateway body handling
    /// closes the correct scoped event even after the original request task has moved on.
    #[cfg(test)]
    pub(crate) async fn end_llm(
        &self,
        active: ActiveLlm,
        response: Value,
        metadata: Value,
    ) -> Result<(), CliError> {
        let response_for_hints = response.clone();
        let session_id = active.session_id.clone();
        let llm_id = active.handle.uuid.to_string();
        let owner_subagent_id = active.owner_subagent_id.clone();
        {
            let mut sessions = self.inner.lock().await;
            let Some(session) = sessions.get_mut(&session_id) else {
                return Ok(());
            };
            if session.llms.remove(&llm_id).is_none() {
                return Ok(());
            }
        }
        TASK_SCOPE_STACK
            .scope(active.stack, async move {
                llm_call_end(
                    LlmCallEndParams::builder()
                        .handle(&active.handle)
                        .response(response)
                        .metadata(metadata)
                        .build(),
                )
                .map_err(CliError::from)
            })
            .await?;
        let mut sessions = self.inner.lock().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.add_tool_hints_from_llm_response(response_for_hints, owner_subagent_id);
        }
        Ok(())
    }

    /// Records tool-call hints from a completed gateway response onto the owning session.
    ///
    /// The runtime owns the LLM lifecycle when the gateway uses managed execution, so the
    /// per-response tool-hint extraction that `end_llm` would normally do has to be triggered
    /// explicitly after the managed pipeline returns. Missing or already-removed sessions are
    /// silently skipped because hints are advisory.
    pub(crate) async fn record_gateway_response_hints(
        &self,
        session_id: &str,
        owner_subagent_id: Option<String>,
        response: Value,
    ) {
        let mut sessions = self.inner.lock().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.add_tool_hints_from_llm_response(response, owner_subagent_id);
        }
    }

    /// Closes every still-open session before gateway teardown.
    ///
    /// Codex transparent runs can exit without a native `SessionEnd` hook. Gateway shutdown is the
    /// last deterministic lifecycle boundary for those sessions, so close open scopes while
    /// observability plugins are still active.
    pub(crate) async fn close_all(&self, reason: &str) -> Result<(), CliError> {
        let mut sessions = {
            let mut guard = self.inner.lock().await;
            guard
                .drain()
                .map(|(_, session)| session)
                .collect::<Vec<_>>()
        };
        close_sessions_for_shutdown(&mut sessions, reason).await
    }
}

async fn close_sessions_for_shutdown(
    sessions: &mut [Session],
    reason: &str,
) -> Result<(), CliError> {
    let mut first_error = None;
    for session in sessions {
        if let Err(error) = session.close_for_shutdown(reason).await
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

impl Session {
    // Constructs per-session runtime state without creating a scope yet. The root agent scope is
    // opened lazily on the first event or gateway LLM call so sessions created from hints and pure
    // gateway traffic share the same initialization path.
    fn new(session_id: String, agent_kind: AgentKind, config: SessionConfig) -> Self {
        Self {
            agent_kind,
            session_id,
            scope_stack: create_scope_stack(),
            agent_scope: None,
            subagents: HashMap::new(),
            subagent_stack: Vec::new(),
            llms: HashMap::new(),
            tools: HashMap::new(),
            pending_llm_hints: Vec::new(),
            pending_tool_hints: Vec::new(),
            last_llm_owner: None,
            config,
        }
    }

    // Runs one normalized hook event inside this session's scope stack. Dispatch stays synchronous
    // inside the scoped closure so lifecycle ordering from each hook request is preserved exactly.
    async fn apply(&mut self, event: NormalizedEvent) -> Result<(), CliError> {
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack, async move {
                match event {
                    NormalizedEvent::AgentStarted(event) => self.start_agent(event),
                    NormalizedEvent::AgentEnded(event) => self.end_agent(event),
                    NormalizedEvent::TurnEnded(_) => Ok(()),
                    NormalizedEvent::SubagentStarted(event) => self.start_subagent(event),
                    NormalizedEvent::SubagentEnded(event) => self.end_subagent(event),
                    NormalizedEvent::LlmHint(event) => self.add_llm_hint(event),
                    NormalizedEvent::LlmStarted(event) => self.start_hook_llm(event),
                    NormalizedEvent::LlmEnded(event) => self.end_hook_llm(event),
                    NormalizedEvent::ToolStarted(event) => self.start_tool(event),
                    NormalizedEvent::ToolEnded(event) => self.end_tool(event),
                    NormalizedEvent::PromptSubmitted(event) => self.mark("prompt_submitted", event),
                    NormalizedEvent::Compaction(event) => self.mark("compaction", event),
                    NormalizedEvent::Notification(event) => self.mark("notification", event),
                    NormalizedEvent::HookMark(event) => self.mark("hook_mark", event),
                }
            })
            .await
    }

    // Legacy manual-lifecycle gateway start used by tests. Production code uses
    // `prepare_gateway_call` + managed execution.
    #[cfg(test)]
    async fn start_llm(&mut self, start: LlmGatewayStart) -> Result<ActiveLlm, CliError> {
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack.clone(), async move {
                self.ensure_agent_started(Value::Null)?;
                let mut attributes = LlmAttributes::empty();
                if start.streaming {
                    attributes |= LlmAttributes::STREAMING;
                }
                let owner = self.resolve_llm_owner(&start);
                let metadata = llm_correlation_metadata(
                    start.metadata,
                    owner.status,
                    owner.source.as_deref(),
                    owner.subagent_id.as_deref(),
                    owner.hint.as_ref(),
                );
                let handle = llm_call(
                    LlmCallParams::builder()
                        .name(start.provider.as_str())
                        .request(&start.request)
                        .parent_opt(owner.parent.as_ref())
                        .attributes(attributes)
                        .metadata(metadata)
                        .model_name_opt(start.model_name)
                        .build(),
                )?;
                let active = ActiveLlm {
                    stack,
                    handle,
                    session_id: self.session_id.clone(),
                    owner_subagent_id: owner.subagent_id,
                };
                self.llms
                    .insert(active.handle.uuid.to_string(), active.handle.clone());
                Ok(active)
            })
            .await
    }

    // Builds a managed-execution prep without creating an LlmHandle. The agent scope is opened if
    // needed and ownership/correlation metadata is computed exactly as the manual `start_llm` path
    // does. The handle and start/end events are emitted later by `llm_call_execute` /
    // `llm_stream_call_execute`, which the gateway runs outside the session lock.
    async fn prepare_gateway_call(
        &mut self,
        start: LlmGatewayStart,
    ) -> Result<GatewayCallPrep, CliError> {
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack.clone(), async move {
                self.ensure_agent_started(Value::Null)?;
                let mut attributes = LlmAttributes::empty();
                if start.streaming {
                    attributes |= LlmAttributes::STREAMING;
                }
                let owner = self.resolve_llm_owner(&start);
                let metadata = llm_correlation_metadata(
                    start.metadata,
                    owner.status,
                    owner.source.as_deref(),
                    owner.subagent_id.as_deref(),
                    owner.hint.as_ref(),
                );
                Ok(GatewayCallPrep {
                    scope_stack: stack,
                    session_id: self.session_id.clone(),
                    provider_name: start.provider,
                    request: start.request,
                    parent: owner.parent,
                    attributes,
                    metadata,
                    model_name: start.model_name,
                    owner_subagent_id: owner.subagent_id,
                })
            })
            .await
    }

    // Records an explicit top-level agent start. Repeated start hooks are idempotent because
    // `ensure_agent_started` leaves an existing agent scope open and only updates agent kind first.
    fn start_agent(&mut self, event: SessionEvent) -> Result<(), CliError> {
        self.agent_kind = event.agent_kind;
        self.ensure_agent_started(event.metadata)
    }

    // Lazily opens the root agent scope and merges metadata from config, event payload, and
    // gateway headers. Later calls are no-ops to keep duplicate hooks from nesting agent scopes.
    fn ensure_agent_started(&mut self, event_metadata: Value) -> Result<(), CliError> {
        if self.agent_scope.is_some() {
            return Ok(());
        }
        let _root = get_handle()?;
        let metadata = merge_metadata(
            merge_metadata(
                self.config.metadata.clone().unwrap_or(Value::Null),
                event_metadata,
            ),
            json!({
                "session_id": self.session_id,
                "gateway_config_profile": self.config.profile,
                "plugin_config": self.config.plugin_config,
                "gateway_mode": self.config.gateway_mode,
            }),
        );
        let scope = push_scope(
            PushScopeParams::builder()
                .name(self.agent_kind.as_str())
                .scope_type(ScopeType::Agent)
                .metadata(metadata)
                .build(),
        )?;
        self.agent_scope = Some(scope);
        Ok(())
    }

    // Closes the session in a fail-safe order: active LLMs/tools first, nested subagents from the
    // top down, correlation state, then the root agent scope.
    fn end_agent(&mut self, event: SessionEvent) -> Result<(), CliError> {
        // Duplicate agent-end hooks (e.g., hermes-agent emitting `on_session_end` more than once
        // per session) must not reopen the agent scope.
        if self.agent_scope.is_none() {
            return Ok(());
        }
        self.ensure_agent_started(event.metadata.clone())?;
        self.close_active_llms_for_agent_end()?;
        self.close_active_tools_for_agent_end()?;
        self.close_active_subagents_for_agent_end()?;
        self.clear_correlation_state();
        self.close_agent_scope(event.payload)?;
        Ok(())
    }

    async fn close_for_shutdown(&mut self, reason: &str) -> Result<(), CliError> {
        let stack = self.scope_stack.clone();
        let payload = json!({ "status": reason });
        TASK_SCOPE_STACK
            .scope(stack, async move {
                if self.agent_scope.is_none() {
                    return Ok(());
                }
                self.close_active_llms_for_agent_end()?;
                self.close_active_tools_for_agent_end()?;
                self.close_active_subagents_for_agent_end()?;
                self.clear_correlation_state();
                self.close_agent_scope(payload)?;
                Ok(())
            })
            .await
    }

    // Ends all active hook-observed LLM calls before closing their containing scopes.
    fn close_active_llms_for_agent_end(&mut self) -> Result<(), CliError> {
        let active_llms: Vec<_> = self.llms.drain().map(|(_, handle)| handle).collect();
        for handle in active_llms {
            llm_call_end(
                LlmCallEndParams::builder()
                    .handle(&handle)
                    .response(json!({ "status": "closed_by_agent_end" }))
                    .metadata(json!({ "status": "closed_by_agent_end" }))
                    .build(),
            )?;
        }
        Ok(())
    }

    // Ends all active tool calls with a synthetic close result before ending their containing scopes.
    // Draining first avoids holding mutable map state while the runtime emits lifecycle events.
    fn close_active_tools_for_agent_end(&mut self) -> Result<(), CliError> {
        let active_tools: Vec<_> = self.tools.drain().map(|(_, handle)| handle).collect();
        for handle in active_tools {
            tool_call_end(
                ToolCallEndParams::builder()
                    .handle(&handle)
                    .result(json!({ "status": "closed_by_agent_end" }))
                    .metadata(json!({ "status": "closed_by_agent_end" }))
                    .build(),
            )?;
        }
        Ok(())
    }

    // Pops active subagent scopes in stack order so nested subagents close from child to parent. The
    // map is cleared afterward to discard any out-of-order stale handles not present in the stack.
    fn close_active_subagents_for_agent_end(&mut self) -> Result<(), CliError> {
        while let Some(subagent_id) = self.subagent_stack.pop() {
            if let Some(handle) = self.subagents.remove(&subagent_id) {
                pop_scope(
                    PopScopeParams::builder()
                        .handle_uuid(&handle.uuid)
                        .output(json!({ "status": "closed_by_agent_end" }))
                        .build(),
                )?;
            }
        }
        self.subagents.clear();
        Ok(())
    }

    // Clears sticky LLM/tool ownership hints that should not survive an agent root shutdown.
    fn clear_correlation_state(&mut self) {
        self.pending_llm_hints.clear();
        self.pending_tool_hints.clear();
        self.last_llm_owner = None;
    }

    // Ends the root agent scope when present. Duplicate agent-end hooks can reach this path after the
    // scope is already gone, so absence is treated as a no-op.
    fn close_agent_scope(&mut self, payload: Value) -> Result<(), CliError> {
        let Some(scope) = self.agent_scope.take() else {
            return Ok(());
        };
        pop_scope(
            PopScopeParams::builder()
                .handle_uuid(&scope.uuid)
                .output(payload)
                .build(),
        )?;
        Ok(())
    }

    // Starts a subagent scope under the current session. Duplicate subagent starts are ignored so
    // integrations that retry or emit both "start" and "created" style hooks do not double-nest.
    fn start_subagent(&mut self, event: SubagentEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        if self.subagents.contains_key(&event.subagent_id) {
            return Ok(());
        }
        let scope = push_scope(
            PushScopeParams::builder()
                .name(format!("subagent:{}", event.subagent_id).as_str())
                .scope_type(ScopeType::Agent)
                .metadata(event.metadata)
                .input(event.payload)
                .build(),
        )?;
        self.subagent_stack.push(event.subagent_id.clone());
        self.subagents.insert(event.subagent_id, scope);
        Ok(())
    }

    // Ends a subagent only when it is the current top of the subagent stack. Unknown or out-of-order
    // endings become mark events instead of corrupting the scope stack, preserving evidence of the
    // mismatch for observability consumers.
    fn end_subagent(&mut self, event: SubagentEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        let Some(scope) = self.subagents.get(&event.subagent_id).cloned() else {
            eprintln!(
                "nemo-flow sidecar: received {} for subagent {} without a matching start",
                event.event_name, event.subagent_id
            );
            return self.mark(
                "subagent_end_without_start",
                SessionEvent {
                    session_id: event.session_id,
                    agent_kind: event.agent_kind,
                    event_name: event.event_name,
                    payload: event.payload,
                    metadata: event.metadata,
                },
            );
        };
        if self.subagent_stack.last() != Some(&event.subagent_id) {
            return emit_mark_event(
                EmitMarkEventParams::builder()
                    .name("subagent_end_not_top")
                    .data(event.payload)
                    .metadata(event.metadata)
                    .build(),
            )
            .map_err(CliError::from);
        }
        if pop_scope(
            PopScopeParams::builder()
                .handle_uuid(&scope.uuid)
                .output(event.payload.clone())
                .build(),
        )
        .is_err()
        {
            return emit_mark_event(
                EmitMarkEventParams::builder()
                    .name("subagent_end_not_top")
                    .data(event.payload)
                    .metadata(event.metadata)
                    .build(),
            )
            .map_err(CliError::from);
        }
        self.subagent_stack.pop();
        self.subagents.remove(&event.subagent_id);
        self.pending_tool_hints
            .retain(|pending| pending.hint.subagent_id.as_ref() != Some(&event.subagent_id));
        if self
            .last_llm_owner
            .as_ref()
            .is_some_and(|owner| owner.subagent_id.as_ref() == Some(&event.subagent_id))
        {
            self.last_llm_owner = None;
        }
        Ok(())
    }

    // Stores an LLM correlation hint from hook activity after pruning expired hints. Hints do not
    // emit runtime events themselves; they are consumed by the next matching gateway LLM call.
    fn add_llm_hint(&mut self, event: LlmHintEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        self.cleanup_correlation_state();
        let owner_subagent_id = event.subagent_id.clone().or_else(|| event.agent_id.clone());
        self.add_tool_hints_from_llm_response(event.payload.clone(), owner_subagent_id);
        self.pending_llm_hints.push(PendingLlmHint {
            hint: event,
            inserted_at: Instant::now(),
        });
        Ok(())
    }

    // Starts an LLM call from hook activity such as Hermes API request hooks. Duplicate call IDs are
    // ignored so repeated pre hooks do not create parallel handles for one provider call.
    fn start_hook_llm(&mut self, event: LlmEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        if self.llms.contains_key(&event.api_call_id) {
            return Ok(());
        }
        let handle = llm_call(
            LlmCallParams::builder()
                .name(event.provider.as_str())
                .request(&LlmRequest {
                    headers: Map::new(),
                    content: event.request,
                })
                .attributes(LlmAttributes::empty())
                .metadata(event.metadata)
                .model_name_opt(event.model_name)
                .build(),
        )?;
        self.llms.insert(event.api_call_id, handle);
        Ok(())
    }

    fn end_hook_llm(&mut self, event: LlmEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        let handle = match self.llms.remove(&event.api_call_id) {
            Some(handle) => handle,
            None => llm_call(
                LlmCallParams::builder()
                    .name(event.provider.as_str())
                    .request(&LlmRequest {
                        headers: Map::new(),
                        content: event.request,
                    })
                    .attributes(LlmAttributes::empty())
                    .metadata(event.metadata.clone())
                    .model_name_opt(event.model_name.clone())
                    .build(),
            )?,
        };
        llm_call_end(
            LlmCallEndParams::builder()
                .handle(&handle)
                .response(event.response)
                .metadata(event.metadata)
                .build(),
        )?;
        Ok(())
    }

    // Starts a tool call under an explicit subagent when available, otherwise under the agent
    // scope. Duplicate tool IDs are ignored so repeated pre-tool hooks do not create parallel
    // handles for one agent tool invocation.
    fn start_tool(&mut self, event: ToolEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        if self.tools.contains_key(&event.tool_call_id) {
            return Ok(());
        }
        let owner = self.resolve_tool_owner(&event);
        let arguments = if event.arguments.is_null() {
            owner
                .hint
                .as_ref()
                .map(|hint| hint.arguments.clone())
                .unwrap_or(event.arguments)
        } else {
            event.arguments
        };
        let metadata = tool_correlation_metadata(
            event.metadata,
            owner.status,
            owner.source.as_deref(),
            owner.subagent_id.as_deref(),
            owner.hint.as_ref(),
        );
        let handle = tool_call(
            ToolCallParams::builder()
                .name(event.tool_name.as_str())
                .args(arguments)
                .parent_opt(owner.parent.as_ref())
                .metadata(metadata)
                .tool_call_id(event.tool_call_id.clone())
                .build(),
        )?;
        self.tools.insert(event.tool_call_id, handle);
        Ok(())
    }

    // Ends a tool call, synthesizing a start if no matching handle exists. This keeps post-only
    // hooks observable and preserves the final result/status instead of dropping orphaned endings.
    fn end_tool(&mut self, event: ToolEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event.metadata.clone())?;
        let handle = match self.tools.remove(&event.tool_call_id) {
            Some(handle) => handle,
            None => {
                let owner = self.resolve_tool_owner(&event);
                let arguments = if event.arguments.is_null() {
                    owner
                        .hint
                        .as_ref()
                        .map(|hint| hint.arguments.clone())
                        .unwrap_or(event.arguments)
                } else {
                    event.arguments
                };
                let metadata = tool_correlation_metadata(
                    event.metadata.clone(),
                    owner.status,
                    owner.source.as_deref(),
                    owner.subagent_id.as_deref(),
                    owner.hint.as_ref(),
                );
                tool_call(
                    ToolCallParams::builder()
                        .name(event.tool_name.as_str())
                        .args(arguments)
                        .parent_opt(owner.parent.as_ref())
                        .metadata(metadata)
                        .tool_call_id(event.tool_call_id.clone())
                        .build(),
                )?
            }
        };
        tool_call_end(
            ToolCallEndParams::builder()
                .handle(&handle)
                .result(event.result)
                .metadata(merge_metadata(
                    event.metadata,
                    json!({ "status": event.status }),
                ))
                .build(),
        )?;
        Ok(())
    }

    // Emits a mark event after ensuring the agent scope exists. Generic and unknown hooks use this
    // path so unsupported agent events remain visible without changing scope structure.
    fn mark(&mut self, name: &str, event_payload: SessionEvent) -> Result<(), CliError> {
        self.ensure_agent_started(event_payload.metadata.clone())?;
        emit_mark_event(
            EmitMarkEventParams::builder()
                .name(name)
                .data(event_payload.payload)
                .metadata(event_payload.metadata)
                .build(),
        )?;
        Ok(())
    }

    // Prunes expired LLM hints and sticky owner state. The TTLs prevent old hook activity from
    // incorrectly capturing later gateway calls when agents reuse a process or session id.
    fn cleanup_correlation_state(&mut self) {
        let now = Instant::now();
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

    // Resolves the parent scope for a gateway LLM call. The precedence is explicit subagent header,
    // single pending hint, uniquely matched hint, sticky last owner, sole active subagent, then agent
    // fallback; ambiguous hints intentionally fall back to the agent and are reported in metadata.
    fn resolve_llm_owner(&mut self, start: &LlmGatewayStart) -> LlmOwnerResolution {
        self.cleanup_correlation_state();

        if let Some(resolution) = self.explicit_llm_owner(start) {
            return resolution;
        }
        if let Some(resolution) = self.single_hint_owner() {
            return resolution;
        }
        if let Some(resolution) = self.matched_hint_owner(start) {
            return resolution;
        }
        if let Some(resolution) = self.sticky_llm_owner() {
            return resolution;
        }
        if let Some(resolution) = self.sole_subagent_owner() {
            return resolution;
        }

        self.fallback_llm_owner()
    }

    // Uses an explicit gateway subagent id when it names an active subagent. Unknown ids do not
    // produce an explicit result because the caller should still have a chance to use hint-based
    // or fallback ownership.
    fn explicit_llm_owner(&mut self, start: &LlmGatewayStart) -> Option<LlmOwnerResolution> {
        if let Some(subagent_id) = &start.subagent_id
            && let Some(scope) = self.subagents.get(subagent_id).cloned()
        {
            self.set_last_llm_owner(Some(subagent_id.clone()));
            return Some(LlmOwnerResolution {
                parent: Some(scope),
                subagent_id: Some(subagent_id.clone()),
                status: "explicit",
                source: Some("gateway_header".to_string()),
                hint: None,
            });
        }
        None
    }

    // Consumes a sole pending hint without scoring. A single hint is unambiguous even when it only
    // contains model or event context, and retaining it would incorrectly affect later LLM calls.
    fn single_hint_owner(&mut self) -> Option<LlmOwnerResolution> {
        if self.pending_llm_hints.len() == 1 {
            let hint = self.pending_llm_hints.remove(0).hint;
            return Some(self.resolution_from_hint(hint, "single_hint"));
        }
        None
    }

    // Consumes the unique best-scoring hint for this gateway request. Tied scores are treated as
    // ambiguous by `matching_hint_index` so this helper only returns defensible correlations.
    fn matched_hint_owner(&mut self, start: &LlmGatewayStart) -> Option<LlmOwnerResolution> {
        if let Some(index) = self.matching_hint_index(start) {
            let hint = self.pending_llm_hints.remove(index).hint;
            return Some(self.resolution_from_hint(hint, "matched_hint"));
        }
        None
    }

    // Reuses the previous LLM owner while its TTL is valid and its scope can still be resolved.
    // This covers agents that emit one hint followed by a cluster of related provider calls.
    fn sticky_llm_owner(&self) -> Option<LlmOwnerResolution> {
        if let Some(owner) = self.last_llm_owner.as_ref()
            && let Some(parent) = self.scope_for_owner(owner.subagent_id.as_deref())
        {
            return Some(LlmOwnerResolution {
                parent: Some(parent),
                subagent_id: owner.subagent_id.clone(),
                status: "sticky_last_owner",
                source: None,
                hint: None,
            });
        }
        None
    }

    // Assigns an unhinted gateway call to the only active subagent. Multiple active subagents are
    // deliberately not guessed here; those cases fall back to the agent scope with ambiguity
    // metadata.
    fn sole_subagent_owner(&mut self) -> Option<LlmOwnerResolution> {
        if self.subagents.len() == 1
            && let Some((subagent_id, scope)) = self.subagents.iter().next()
        {
            let subagent_id = subagent_id.clone();
            let scope = scope.clone();
            self.set_last_llm_owner(Some(subagent_id.clone()));
            return Some(LlmOwnerResolution {
                parent: Some(scope),
                subagent_id: Some(subagent_id),
                status: "active_subagent",
                source: None,
                hint: None,
            });
        }
        None
    }

    // Final fallback for gateway calls that cannot be correlated to a subagent. Pending hints are
    // left intact in ambiguous cases so later calls with stronger identifiers can still match them.
    fn fallback_llm_owner(&self) -> LlmOwnerResolution {
        LlmOwnerResolution {
            parent: self.agent_scope.clone(),
            subagent_id: None,
            status: if self.pending_llm_hints.is_empty() {
                "agent_fallback"
            } else {
                "ambiguous_fallback"
            },
            source: None,
            hint: None,
        }
    }

    // Converts a consumed hint into an ownership resolution. If the hinted subagent is not currently
    // active, the LLM is attached to the agent scope but the hint metadata is still preserved for
    // correlation diagnostics.
    fn resolution_from_hint(
        &mut self,
        hint: LlmHintEvent,
        status: &'static str,
    ) -> LlmOwnerResolution {
        let hinted_subagent_id = hint.subagent_id.clone().or_else(|| hint.agent_id.clone());
        let (parent, subagent_id) = match hinted_subagent_id.as_deref() {
            Some(id) => match self.subagents.get(id).cloned() {
                Some(scope) => (Some(scope), Some(id.to_string())),
                None => (self.agent_scope.clone(), None),
            },
            None => (self.agent_scope.clone(), None),
        };
        if parent.is_some() {
            self.set_last_llm_owner(subagent_id.clone());
        }
        LlmOwnerResolution {
            parent,
            subagent_id,
            status,
            source: Some(hint.event_name.clone()),
            hint: Some(hint),
        }
    }

    // Finds a single best pending hint for a gateway call. Ties are treated as ambiguous and return
    // `None`, causing the caller to use fallback behavior rather than guessing between subagents.
    fn matching_hint_index(&self, start: &LlmGatewayStart) -> Option<usize> {
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

    // Resolves a stored owner back to a live scope, falling back to the agent scope when no subagent
    // id is present or the subagent has already ended.
    fn scope_for_owner(&self, subagent_id: Option<&str>) -> Option<ScopeHandle> {
        subagent_id
            .and_then(|id| self.subagents.get(id).cloned())
            .or_else(|| self.agent_scope.clone())
    }

    // Records the most recent LLM owner with a timestamp so nearby gateway calls can inherit the
    // same parent scope when explicit IDs and hints are absent.
    fn set_last_llm_owner(&mut self, subagent_id: Option<String>) {
        self.last_llm_owner = Some(LastLlmOwner {
            subagent_id,
            updated_at: Instant::now(),
        });
    }

    // Records tool-call suggestions from LLM responses as private correlation hints. These hints
    // are not emitted as events; they only help later tool hooks choose the same subagent scope as
    // the LLM that proposed the call.
    fn add_tool_hints_from_llm_response(
        &mut self,
        response: Value,
        owner_subagent_id: Option<String>,
    ) {
        self.cleanup_correlation_state();
        let hints = tool_hints_from_llm_response(&response, owner_subagent_id);
        self.pending_tool_hints
            .extend(hints.into_iter().map(|hint| PendingToolHint {
                hint,
                inserted_at: Instant::now(),
            }));
    }

    // Resolves tool hook ownership from explicit subagent data first, then private tool hints
    // extracted from LLM responses, and finally the agent scope.
    fn resolve_tool_owner(&mut self, event: &ToolEvent) -> ToolOwnerResolution {
        self.cleanup_correlation_state();

        if let Some(subagent_id) = &event.subagent_id
            && let Some(scope) = self.subagents.get(subagent_id).cloned()
        {
            self.consume_matching_tool_hint(event);
            return ToolOwnerResolution {
                parent: Some(scope),
                subagent_id: Some(subagent_id.clone()),
                status: "explicit",
                source: Some("hook_payload".to_string()),
                hint: None,
            };
        }

        if self.pending_tool_hints.len() == 1 {
            let hint = self.pending_tool_hints.remove(0).hint;
            return self.tool_resolution_from_hint(hint, "single_hint");
        }

        if let Some(index) = self.matching_tool_hint_index(event) {
            let hint = self.pending_tool_hints.remove(index).hint;
            return self.tool_resolution_from_hint(hint, "matched_hint");
        }

        ToolOwnerResolution {
            parent: self.agent_scope.clone(),
            subagent_id: None,
            status: if self.pending_tool_hints.is_empty() {
                "agent_fallback"
            } else {
                "ambiguous_fallback"
            },
            source: None,
            hint: None,
        }
    }

    // Converts a consumed tool hint into a live parent scope, falling back to the root agent if the
    // hinted subagent has already ended or never existed.
    fn tool_resolution_from_hint(
        &mut self,
        hint: ToolHint,
        status: &'static str,
    ) -> ToolOwnerResolution {
        let (parent, subagent_id) = match hint.subagent_id.as_deref() {
            Some(id) => match self.subagents.get(id).cloned() {
                Some(scope) => (Some(scope), Some(id.to_string())),
                None => (self.agent_scope.clone(), None),
            },
            None => (self.agent_scope.clone(), None),
        };
        ToolOwnerResolution {
            parent,
            subagent_id,
            status,
            source: Some(hint.source.clone()),
            hint: Some(hint),
        }
    }

    // Removes a stale matching hint when a hook already carried an explicit subagent owner.
    fn consume_matching_tool_hint(&mut self, event: &ToolEvent) {
        if let Some(index) = self.matching_tool_hint_index(event) {
            self.pending_tool_hints.remove(index);
        }
    }

    // Finds a unique best-scoring tool hint by call id, name, and argument equality. Ties remain
    // ambiguous and are not consumed.
    fn matching_tool_hint_index(&self, event: &ToolEvent) -> Option<usize> {
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
}

// Scores how strongly a pending hint matches a gateway LLM request. Subagent/agent identity is
// weighted highest, request/conversation/generation identifiers are equal, and model match is only
// a low-confidence tie breaker.
fn hint_match_score(hint: &LlmHintEvent, start: &LlmGatewayStart) -> u8 {
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
fn tool_hints_from_llm_response(
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
fn collect_openai_chat_tool_hints(
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
fn collect_openai_response_tool_hints(
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
fn collect_anthropic_tool_hints(
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

// Appends one provider tool hint when an object carries at least a tool-call id or tool name.
// Argument-only hints are intentionally skipped because they over-match across unrelated tools.
fn push_tool_hint(
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
    if tool_call_id.is_none() && tool_name.is_none() {
        return;
    }
    hints.push(ToolHint {
        tool_call_id,
        tool_name,
        subagent_id: owner_subagent_id.map(ToOwned::to_owned),
        arguments: json_value_at(object, argument_paths)
            .map(normalize_tool_arguments)
            .unwrap_or(Value::Null),
        source: source.to_string(),
    });
}

// Scores how strongly a pending provider tool hint matches an observed hook event. Tool-call id is
// strongest, tool name is secondary, and exact argument equality is only a tie breaker.
fn tool_hint_match_score(hint: &ToolHint, event: &ToolEvent) -> u8 {
    let mut score = 0;
    if same_optional(
        hint.tool_call_id.as_deref(),
        Some(event.tool_call_id.as_str()),
    ) {
        score += 12;
    }
    if same_optional(hint.tool_name.as_deref(), Some(event.tool_name.as_str())) {
        score += 4;
    }
    if !hint.arguments.is_null() && !event.arguments.is_null() && hint.arguments == event.arguments
    {
        score += 1;
    }
    score
}

fn same_optional(left: Option<&str>, right: Option<&str>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if left == right)
}

// Reads the first string-like value from any candidate JSON path. Scalar numbers and booleans are
// accepted for IDs because provider payloads are not always strict about identifier types.
fn json_string_at(payload: &Value, paths: &[&[&str]]) -> Option<String> {
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
// hint data must live independently of the response body stored on the LLM end event.
fn json_value_at(payload: &Value, paths: &[&[&str]]) -> Option<Value> {
    paths.iter().find_map(|path| {
        let mut current = payload;
        for key in *path {
            current = current.get(*key)?;
        }
        Some(current.clone())
    })
}

// Parses stringified tool arguments when providers encode them as JSON text. Non-JSON strings are
// preserved as strings so metadata still reflects what the provider actually returned.
fn normalize_tool_arguments(arguments: Value) -> Value {
    match arguments {
        Value::String(raw) => serde_json::from_str(&raw).unwrap_or(Value::String(raw)),
        value => value,
    }
}

// Adds correlation status and consumed-hint identifiers to the LLM event metadata. Caller metadata
// is merged first so correlation keys win when names collide.
fn llm_correlation_metadata(
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
fn tool_correlation_metadata(
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

// Inserts an optional string value into a JSON object while omitting absent fields entirely. This
// keeps correlation metadata compact and avoids serializing nulls as meaningful observations.
fn insert_optional(object: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_string(), json!(value));
    }
}

// Extracts the source agent kind from any normalized event variant so newly created sessions can
// inherit the correct agent identity before an explicit agent-start hook arrives.
fn event_agent_kind(event: &NormalizedEvent) -> AgentKind {
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
fn single_active_session_id(sessions: &HashMap<String, Session>) -> Option<String> {
    (sessions.len() == 1)
        .then(|| sessions.keys().next().cloned())
        .flatten()
}

// Infers the owning agent for a session created by a gateway request that beat its SessionStart
// hook. Mapping is by provider route name (set by `gateway::start_gateway_llm`):
// `anthropic.messages` → ClaudeCode, `openai.responses` → Codex. Unknown providers fall back to
// `Gateway` so synthetic sessions opened by pure proxy use still get the legacy label. This is
// the only chance to label the session correctly because observer identities (ATIF agent name,
// OpenInference scope name) are baked at scope-open time inside `ensure_agent_started`.
fn agent_kind_for_gateway_provider(provider: &str) -> AgentKind {
    match provider {
        "anthropic.messages" | "anthropic.count_tokens" => AgentKind::ClaudeCode,
        "openai.responses" => AgentKind::Codex,
        _ => AgentKind::Gateway,
    }
}

// Merges metadata objects with right-hand values taking precedence and null right-hand fields
// ignored. Non-object values are preserved under separate keys so callers do not lose unusual
// metadata shapes supplied by configuration or hooks.
fn merge_metadata(left: Value, right: Value) -> Value {
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
#[path = "../tests/coverage/session_tests.rs"]
mod tests;
