// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use nemo_relay::api::llm::{
    LlmAttributes, LlmCallEndParams, LlmCallParams, LlmHandle, LlmRequest, llm_call, llm_call_end,
};
use nemo_relay::api::runtime::{
    ScopeStackHandle, TASK_SCOPE_STACK, create_scope_stack, task_scope_push,
};
use nemo_relay::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeHandle, ScopeType,
    event as emit_mark_event, get_handle, pop_scope, push_scope,
};
use nemo_relay::api::tool::{
    ToolCallEndParams, ToolCallParams, ToolHandle, tool_call, tool_call_end,
    tool_conditional_execution,
};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::agents::shared::adapters::{SKILL_LOAD_SOURCE_KEY, SKILL_LOAD_SOURCE_PROMPT_EXPANSION};
use crate::agents::shared::alignment::{
    self, GatewayManagementPolicy, SessionAlias, SessionAlignmentState, insert_optional,
    json_string_at, json_value_at, merge_metadata,
};
use crate::configuration::{GatewayConfig, SessionConfig};
use crate::error::CliError;
mod correlation;
mod idle;
mod routing;
mod types;

use correlation::*;
use idle::*;
use routing::*;
pub(crate) use types::*;

use crate::events::{
    AgentKind, LlmEvent, LlmHintEvent, NormalizedEvent, SessionEvent, SubagentEvent, ToolEvent,
};

const LLM_HINT_TTL: Duration = Duration::from_secs(300);
const TOOL_HINT_TTL: Duration = Duration::from_secs(300);
const LAST_OWNER_TTL: Duration = Duration::from_secs(300);
const ROUTING_IDENTITY_HEADERS: &[&str] = &[
    "x-nemo-relay-session-id",
    "x-nemo-relay-agent-kind",
    "x-nemo-relay-turn-id",
    "x-nemo-relay-request-id",
    "x-nemo-relay-owner-id",
    "x-nemo-relay-subagent-id",
    "x-nemo-relay-parent-scope-id",
    "x-nemo-relay-root-scope-id",
    "x-nemo-relay-identity-quality",
    "x-nemo-relay-source",
];

#[derive(Clone)]
pub(crate) struct SessionManager {
    inner: Arc<Mutex<HashMap<String, Session>>>,
    // Cross-session alignment state owns child-session aliases and child-first SessionStart hooks.
    // Applies to Codex child threads today; the generic state lives in `alignment` so session code
    // only orchestrates when promotion is safe.
    alignment: Arc<Mutex<SessionAlignmentState>>,
    default_config: GatewayConfig,
}

struct RoutingIdentityHeaderContext<'a> {
    session_id: &'a str,
    agent_kind: AgentKind,
    turn_index: u64,
    request_id: Option<&'a str>,
    owner_id: Option<&'a str>,
    parent: Option<&'a ScopeHandle>,
    root: Option<&'a ScopeHandle>,
    metadata: &'a Value,
}

fn enrich_routing_identity_headers(
    request: &mut LlmRequest,
    context: RoutingIdentityHeaderContext<'_>,
) {
    request.headers.retain(|name, _| {
        !ROUTING_IDENTITY_HEADERS
            .iter()
            .any(|reserved| name.eq_ignore_ascii_case(reserved))
    });
    insert_routing_identity_header(
        &mut request.headers,
        "x-nemo-relay-session-id",
        context.session_id,
    );
    insert_routing_identity_header(
        &mut request.headers,
        "x-nemo-relay-agent-kind",
        context.agent_kind.as_str(),
    );
    insert_routing_identity_header(
        &mut request.headers,
        "x-nemo-relay-turn-id",
        &context.turn_index.to_string(),
    );
    let request_id = context
        .request_id
        .map(ToOwned::to_owned)
        .or_else(|| {
            json_string_at(
                context.metadata,
                &[
                    &["llm_correlation_request_id"][..],
                    &["request_id"][..],
                    &["requestId"][..],
                ],
            )
        })
        .unwrap_or_else(|| format!("relay-request-{}", uuid::Uuid::now_v7()));
    insert_routing_identity_header(&mut request.headers, "x-nemo-relay-request-id", &request_id);
    if let Some(owner_id) = context.owner_id {
        insert_routing_identity_header(&mut request.headers, "x-nemo-relay-owner-id", owner_id);
        insert_routing_identity_header(&mut request.headers, "x-nemo-relay-subagent-id", owner_id);
    }
    if let Some(parent) = context.parent {
        insert_routing_identity_header(
            &mut request.headers,
            "x-nemo-relay-parent-scope-id",
            &parent.uuid.to_string(),
        );
    }
    if let Some(root) = context.root {
        insert_routing_identity_header(
            &mut request.headers,
            "x-nemo-relay-root-scope-id",
            &root.uuid.to_string(),
        );
    }
    insert_routing_identity_header(
        &mut request.headers,
        "x-nemo-relay-identity-quality",
        "native",
    );
    insert_routing_identity_header(&mut request.headers, "x-nemo-relay-source", "gateway");
}

fn insert_routing_identity_header(headers: &mut Map<String, Value>, name: &str, value: &str) {
    headers.insert(name.to_string(), json!(value));
}

pub(super) struct Session {
    agent_kind: AgentKind,
    session_id: String,
    scope_stack: ScopeStackHandle,
    session_started: bool,
    session_metadata: Value,
    agent_scope: Option<ScopeHandle>,
    turn_scope: Option<ScopeHandle>,
    gateway_request_turn_open: bool,
    turn_index: u64,
    last_turn_llm_output: Option<Value>,
    subagents: HashMap<String, ScopeHandle>,
    // Each active subagent gets its own scope stack seeded with the parent agent handle. This lets
    // sibling workers close out of order without corrupting the task-local stack.
    subagent_stacks: HashMap<String, ScopeStackHandle>,
    subagent_stack: Vec<String>,
    // Tracks subagents closed by synthetic or provider-specific completion signals so a later
    // duplicate end hook does not reopen or mark an already-closed worker.
    completed_subagents: HashSet<String>,
    llms: HashMap<String, LlmHandle>,
    tools: HashMap<String, ActiveTool>,
    pending_llm_hints: Vec<PendingLlmHint>,
    pending_tool_hints: Vec<PendingToolHint>,
    // Maps stable user-task text from confidently owned LLM requests to the subagent that owns
    // that conversation. This gives parallel coding-agent workers a stronger provider-format
    // neutral fallback than a single global "last tool owner" when requests lack subagent headers.
    llm_request_affinity: HashMap<String, Option<String>>,
    last_llm_owner: Option<LastLlmOwner>,
    last_activity: Instant,
    active_gateway_calls: usize,
    config: SessionConfig,
}

#[derive(Debug, Clone)]
struct ActiveTool {
    handle: ToolHandle,
    name: String,
    arguments: Value,
    owner_subagent_id: Option<String>,
}

impl std::ops::Deref for ActiveTool {
    type Target = ToolHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
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
    subagent_id: String,
    updated_at: Instant,
    // The source is exported in correlation metadata, which makes sticky ownership easier to audit
    // in Phoenix when explicit gateway headers are absent.
    source: LastLlmOwnerSource,
}

#[derive(Debug, Clone, Copy)]
enum LastLlmOwnerSource {
    Llm,
    Tool,
    SubagentStart,
}

impl LastLlmOwnerSource {
    const fn status(self) -> &'static str {
        match self {
            Self::Llm => "sticky_last_owner",
            Self::Tool => "recent_tool_owner",
            Self::SubagentStart => "subagent_start",
        }
    }

    const fn metadata_source(self) -> Option<&'static str> {
        match self {
            Self::Llm => None,
            Self::Tool => Some("tool_owner"),
            Self::SubagentStart => Some("subagent_start"),
        }
    }
}

struct LlmOwnerResolution {
    parent: Option<ScopeHandle>,
    subagent_id: Option<String>,
    status: &'static str,
    source: Option<String>,
    hint: Option<LlmHintEvent>,
    metadata: Value,
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
            alignment: Arc::new(Mutex::new(SessionAlignmentState::default())),
            default_config,
        }
    }

    /// Starts the fail-safe idle closer used by the HTTP gateway.
    ///
    /// Some coding agents, notably Codex child threads, do not always emit native agent-end hooks.
    /// The sweeper is provider-neutral: it closes any open turn that has had no hook or gateway
    /// activity for a short interval, while leaving turns with active tools or managed LLM calls
    /// alone. Weak references keep the task from extending the manager lifetime in tests or
    /// shutdown paths.
    pub(crate) fn start_idle_sweeper(&self) {
        let inner = Arc::downgrade(&self.inner);
        let alignment = Arc::downgrade(&self.alignment);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(AGENT_IDLE_SWEEP_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let (Some(inner), Some(alignment)) = (inner.upgrade(), alignment.upgrade()) else {
                    break;
                };
                if let Err(error) = close_idle_sessions_from_parts(
                    &inner,
                    &alignment,
                    Instant::now(),
                    AGENT_IDLE_TIMEOUT,
                    "idle_timeout",
                )
                .await
                {
                    log::warn!(
                        target: "nemo_relay.session",
                        event = "session_cleanup_failed",
                        reason = "idle_timeout",
                        error_kind = error.log_kind();
                        "Idle session cleanup failed"
                    );
                }
            }
        });
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
        let mut alignment_state = self.alignment.lock().await;
        let mut sessions = self.inner.lock().await;
        for event in events {
            let mut event = event;
            let config = self.default_config.session_config_from_headers(headers);
            if queue_or_promote_child_start(
                &mut event,
                &mut sessions,
                &mut alignment_state,
                config.clone(),
            )
            .await?
            {
                continue;
            }

            let Some((event, session_id, is_agent_started)) =
                route_event_for_session(event, &mut sessions, &mut alignment_state)
            else {
                continue;
            };
            let event_kind = event_agent_kind(&event);
            let should_remove_session = apply_event_to_session(
                &mut sessions,
                &session_id,
                event,
                event_kind,
                config.clone(),
                is_agent_started,
            )
            .await?;
            if is_agent_started {
                // A just-opened parent may unlock one or more child SessionStart hooks that arrived
                // earlier in this batch or an earlier request.
                promote_pending_subagents_for_parent(
                    &mut sessions,
                    &mut alignment_state,
                    &session_id,
                    config.clone(),
                )
                .await?;
            }
            if should_remove_session {
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
        let mut start = start;
        let config = self.default_config.session_config_from_headers(headers);
        let alias = self.resolve_start_alias(&mut start, config.clone()).await?;
        let mut sessions = self.inner.lock().await;
        let session_id = start
            .session_id
            .clone()
            .or_else(|| single_active_session_id(&sessions))
            .unwrap_or_else(|| format!("{}-gateway", AgentKind::Gateway.as_str()));
        let inferred_agent_kind = alignment::agent_kind_for_gateway_provider(&start.provider);
        let session = sessions
            .entry(session_id.clone())
            .or_insert_with(|| Session::new(session_id, inferred_agent_kind, config));
        let mut active = session.start_llm(start).await?;
        if let Some(alias) = alias {
            active.session_id = alias.parent_session_id;
            active.owner_subagent_id = active.owner_subagent_id.or(Some(alias.subagent_id));
        }
        Ok(active)
    }

    /// Prepares a managed LLM execution against the right session and scope context.
    ///
    /// Resolves the session, opens the turn scope if needed, computes the parent scope and
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
        let mut start = start;
        let config = self.default_config.session_config_from_headers(headers);
        self.resolve_start_alias(&mut start, config.clone()).await?;
        let mut sessions = self.inner.lock().await;
        let (session_id, session_finish) = gateway_session_for_call(&start, &sessions);
        // Match `start_llm`: when this path creates a brand-new session (real agent's gateway
        // request beats its SessionStart hook), label the session by the provider so ATIF and
        // Phoenix scopes carry the agent identity instead of freezing on "gateway".
        let inferred_agent_kind = alignment::agent_kind_for_gateway_provider(&start.provider);
        let created_session = !sessions.contains_key(&session_id);
        let session = sessions
            .entry(session_id.clone())
            .or_insert_with(|| Session::new(session_id.clone(), inferred_agent_kind, config));
        let result = session.prepare_gateway_call(start).await;
        match result {
            Ok(mut prep) => {
                prep.session_finish = if prep.bypass_managed_pipeline
                    && sessions
                        .get(&session_id)
                        .is_some_and(|session| session.is_empty())
                {
                    GatewaySessionFinish::PruneIfEmpty
                } else {
                    session_finish
                };
                Ok(prep)
            }
            Err(error) => {
                if created_session
                    && sessions
                        .get(&session_id)
                        .is_some_and(|session| session.is_empty())
                {
                    sessions.remove(&session_id);
                }
                Err(error)
            }
        }
    }

    /// Marks a managed gateway LLM call as finished for idle-timeout purposes.
    ///
    /// Runtime-managed LLM spans are emitted outside the session lock, so the session keeps a small
    /// in-flight counter to prevent the idle sweeper from closing a turn while an upstream
    /// provider request or streaming response is still active.
    pub(crate) async fn finish_gateway_call(&self, session_id: &str, finish: GatewaySessionFinish) {
        let mut sessions = self.inner.lock().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.finish_gateway_call();
        }
        let completed = sessions.get(session_id).is_some_and(|session| {
            session.active_gateway_calls == 0
                && match finish {
                    GatewaySessionFinish::Retain => false,
                    GatewaySessionFinish::PruneIfEmpty => session.is_empty(),
                    GatewaySessionFinish::Close => true,
                }
        });
        let mut closing = completed.then(|| sessions.remove(session_id)).flatten();
        drop(sessions);

        if finish == GatewaySessionFinish::Close
            && let Some(session) = closing.as_mut()
        {
            match session
                .close_for_shutdown("uncorrelated_gateway_call_complete")
                .await
            {
                Ok(()) => log::info!(
                    target: "nemo_relay.session",
                    event = "session_closed",
                    session_id = session_id;
                    "Session closed"
                ),
                Err(error) => log::warn!(
                    target: "nemo_relay.session",
                    event = "session_cleanup_failed",
                    session_id = session_id,
                    reason = "isolated_session_close",
                    error_kind = error.log_kind();
                    "Session cleanup failed"
                ),
            }
        }
    }

    /// Returns true while any session still owns active observable work.
    ///
    /// Host sessions can remain durable after their current turn ends: Codex may omit `SessionEnd`,
    /// while Hermes keeps a session open for later resumption. A dormant agent scope must therefore
    /// not keep the MCP-managed sidecar alive forever. Active turns, subagents, tools, LLMs, and
    /// gateway calls still block idle shutdown; [`Self::close_all`] balances the dormant agent scope
    /// when the gateway exits.
    pub(crate) async fn has_open_sessions(&self) -> bool {
        self.inner
            .lock()
            .await
            .values()
            .any(Session::blocks_plugin_idle_shutdown)
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
            session.record_completed_llm_response(response_for_hints, owner_subagent_id);
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
        let alias = {
            let alignment_state = self.alignment.lock().await;
            alignment_state.alias_for_session(session_id)
        };
        let (session_id, owner_subagent_id) = match alias {
            Some(alias) => (
                alias.parent_session_id,
                owner_subagent_id.or(Some(alias.subagent_id)),
            ),
            None => (session_id.to_string(), owner_subagent_id),
        };
        let mut sessions = self.inner.lock().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.record_completed_llm_response(response, owner_subagent_id);
        }
    }

    /// Closes every still-open session before gateway teardown.
    ///
    /// Some harnesses can exit without a native `SessionEnd` hook. Gateway shutdown is the last
    /// deterministic lifecycle boundary for those sessions, so close open scopes while
    /// observability plugins are still active. Applies to Codex transparent runs today.
    pub(crate) async fn close_all(&self, reason: &str) -> Result<(), CliError> {
        self.alignment.lock().await.clear();
        let mut sessions = {
            let mut guard = self.inner.lock().await;
            guard
                .drain()
                .map(|(_, session)| session)
                .collect::<Vec<_>>()
        };
        close_sessions_for_shutdown(&mut sessions, reason).await
    }

    #[cfg(test)]
    pub(crate) async fn close_idle_sessions_at(
        &self,
        now: Instant,
        timeout: Duration,
        reason: &str,
    ) -> Result<usize, CliError> {
        close_idle_sessions_from_parts(&self.inner, &self.alignment, now, timeout, reason).await
    }

    // Applies known or pending child-session aliases before the gateway chooses a session. This is
    // deliberately before the `inner` lock in `start_llm`/`prepare_gateway_call` so a child-thread
    // gateway request can promote its pending parent/subagent relationship instead of creating a
    // synthetic child root.
    async fn resolve_start_alias(
        &self,
        start: &mut LlmGatewayStart,
        config: SessionConfig,
    ) -> Result<Option<SessionAlias>, CliError> {
        let Some(session_id) = start.session_id.clone() else {
            return Ok(None);
        };
        let mut alignment_state = self.alignment.lock().await;
        if let Some(alias) = alignment_state.alias_for_session(&session_id) {
            apply_start_alias(start, &alias);
            return Ok(Some(alias));
        }
        let Some(pending) = alignment_state.pending_for_session(&session_id) else {
            return Ok(None);
        };
        let mut sessions = self.inner.lock().await;
        let alias = promote_pending_subagent(
            &mut sessions,
            &mut alignment_state,
            session_id,
            pending,
            config,
        )
        .await?;
        if let Some(alias) = alias.as_ref() {
            apply_start_alias(start, alias);
        }
        Ok(alias)
    }
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
            session_started: false,
            session_metadata: Value::Null,
            agent_scope: None,
            turn_scope: None,
            gateway_request_turn_open: false,
            turn_index: 0,
            last_turn_llm_output: None,
            subagents: HashMap::new(),
            subagent_stacks: HashMap::new(),
            subagent_stack: Vec::new(),
            completed_subagents: HashSet::new(),
            llms: HashMap::new(),
            tools: HashMap::new(),
            pending_llm_hints: Vec::new(),
            pending_tool_hints: Vec::new(),
            llm_request_affinity: HashMap::new(),
            last_llm_owner: None,
            last_activity: Instant::now(),
            active_gateway_calls: 0,
            config,
        }
    }

    // A child session can only be converted into a subagent before any real scope, LLM, or tool
    // state has been opened for it. Once work exists under the child, reparenting would move only
    // future events and leave an inconsistent trace.
    fn can_reparent_as_subagent_alias(&self) -> bool {
        self.is_empty()
    }

    fn is_empty(&self) -> bool {
        !self.session_started
            && self.agent_scope.is_none()
            && self.turn_scope.is_none()
            && self.subagents.is_empty()
            && self.subagent_stacks.is_empty()
            && self.subagent_stack.is_empty()
            && self.llms.is_empty()
            && self.tools.is_empty()
    }

    fn blocks_plugin_idle_shutdown(&self) -> bool {
        self.turn_scope.is_some()
            || !self.subagents.is_empty()
            || !self.subagent_stacks.is_empty()
            || !self.subagent_stack.is_empty()
            || !self.llms.is_empty()
            || !self.tools.is_empty()
            || self.active_gateway_calls > 0
    }

    fn touch_activity(&mut self) {
        self.last_activity = Instant::now();
    }

    fn begin_gateway_call(&mut self) {
        self.touch_activity();
        self.active_gateway_calls += 1;
    }

    fn finish_gateway_call(&mut self) {
        self.touch_activity();
        self.active_gateway_calls = self.active_gateway_calls.saturating_sub(1);
    }

    fn is_idle_for(&self, now: Instant, timeout: Duration) -> bool {
        self.turn_scope.is_some()
            && self.active_gateway_calls == 0
            && self.llms.is_empty()
            && self.tools.is_empty()
            && now.duration_since(self.last_activity) >= timeout
    }

    fn is_active_or_recent(&self, now: Instant) -> bool {
        self.blocks_plugin_idle_shutdown()
            || now
                .checked_duration_since(self.last_activity)
                .is_none_or(|elapsed| elapsed < AGENT_IDLE_TIMEOUT)
    }

    // Runs one normalized hook event inside this session's scope stack. Dispatch stays synchronous
    // inside the scoped closure so lifecycle ordering from each hook request is preserved exactly.
    async fn apply(&mut self, event: NormalizedEvent) -> Result<(), CliError> {
        self.touch_activity();
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack, async move {
                match event {
                    NormalizedEvent::AgentStarted(event) => self.start_agent(event),
                    NormalizedEvent::AgentEnded(event) => self.end_agent(event).await,
                    NormalizedEvent::TurnEnded(event) => self.end_turn(event).await,
                    NormalizedEvent::SubagentStarted(event) => self.start_subagent(event).await,
                    NormalizedEvent::SubagentEnded(event) => self.end_subagent(event).await,
                    NormalizedEvent::LlmHint(event) => self.add_llm_hint(event),
                    NormalizedEvent::LlmStarted(event) => self.start_hook_llm(event),
                    NormalizedEvent::LlmEnded(event) => self.end_hook_llm(event),
                    NormalizedEvent::ToolStarted(event) => self.start_tool(event),
                    NormalizedEvent::ToolEnded(event) => self.end_tool(event).await,
                    NormalizedEvent::PromptSubmitted(event) => self.start_turn(event).await,
                    NormalizedEvent::Compaction(event) => self.mark("compaction", event),
                    NormalizedEvent::Notification(event) => self.mark("notification", event),
                    NormalizedEvent::HookMark(event) => {
                        let name = if event
                            .metadata
                            .get(SKILL_LOAD_SOURCE_KEY)
                            .and_then(Value::as_str)
                            == Some(SKILL_LOAD_SOURCE_PROMPT_EXPANSION)
                        {
                            "skill.load.inferred"
                        } else {
                            "hook_mark"
                        };
                        self.mark(name, event)
                    }
                }
            })
            .await
    }

    // Legacy manual-lifecycle gateway start used by tests. Production code uses
    // `prepare_gateway_call` + managed execution.
    #[cfg(test)]
    async fn start_llm(&mut self, start: LlmGatewayStart) -> Result<ActiveLlm, CliError> {
        self.touch_activity();
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack.clone(), async move {
                self.ensure_turn_started_for_gateway(&start)?;
                let mut attributes = LlmAttributes::empty();
                if start.streaming {
                    attributes |= LlmAttributes::STREAMING;
                }
                let owner = self.resolve_llm_owner(&start);
                self.record_llm_request_affinity(
                    &start.provider,
                    &start.request,
                    owner.subagent_id.as_deref(),
                    owner.status,
                );
                let metadata = merge_metadata(
                    llm_correlation_metadata(
                        start.metadata,
                        owner.status,
                        owner.source.as_deref(),
                        owner.subagent_id.as_deref(),
                        owner.hint.as_ref(),
                    ),
                    owner.metadata,
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
        self.begin_gateway_call();
        let stack = self.scope_stack.clone();
        let result = TASK_SCOPE_STACK
            .scope(stack.clone(), async {
                let policy = self.gateway_management_policy(&start);
                if !policy.bypasses_managed_pipeline() {
                    self.ensure_turn_started_for_gateway(&start)?;
                }
                let mut attributes = LlmAttributes::empty();
                if start.streaming {
                    attributes |= LlmAttributes::STREAMING;
                }
                let owner = if policy.bypasses_managed_pipeline() {
                    self.unmanaged_probe_owner(policy)
                } else {
                    self.resolve_llm_owner(&start)
                };
                self.record_llm_request_affinity(
                    &start.provider,
                    &start.request,
                    owner.subagent_id.as_deref(),
                    owner.status,
                );
                let metadata = merge_metadata(
                    llm_correlation_metadata(
                        start.metadata,
                        owner.status,
                        owner.source.as_deref(),
                        owner.subagent_id.as_deref(),
                        owner.hint.as_ref(),
                    ),
                    owner.metadata,
                );
                let mut request = start.request;
                enrich_routing_identity_headers(
                    &mut request,
                    RoutingIdentityHeaderContext {
                        session_id: &self.session_id,
                        agent_kind: self.agent_kind,
                        turn_index: self.turn_index,
                        request_id: start.request_id.as_deref(),
                        owner_id: owner.subagent_id.as_deref(),
                        parent: owner.parent.as_ref(),
                        root: self.agent_scope.as_ref().or(self.turn_scope.as_ref()),
                        metadata: &metadata,
                    },
                );
                Ok(GatewayCallPrep {
                    scope_stack: stack.clone(),
                    session_id: self.session_id.clone(),
                    provider_name: start.provider,
                    request,
                    parent: owner.parent,
                    attributes,
                    metadata,
                    model_name: start.model_name,
                    owner_subagent_id: owner.subagent_id,
                    bypass_managed_pipeline: policy.bypasses_managed_pipeline(),
                    session_finish: GatewaySessionFinish::Retain,
                })
            })
            .await;
        if result.is_err() {
            self.finish_gateway_call();
        }
        result
    }

    // Records a harness session start without assuming that every harness exposes a reliable
    // session-length span. Some session ids can outlive user-visible work, so those harnesses store
    // metadata here and wait for a bounded turn scope before emitting trace structure.
    fn start_agent(&mut self, event: SessionEvent) -> Result<(), CliError> {
        let emit_start_mark = !self.session_started;
        self.agent_kind = event.agent_kind;
        self.session_started = true;
        self.session_metadata =
            merge_metadata(self.session_metadata.clone(), event.metadata.clone());
        self.ensure_agent_started(event.metadata.clone())?;
        if emit_start_mark {
            emit_mark_event(
                EmitMarkEventParams::builder()
                    .name("session.start")
                    .parent_opt(self.agent_scope.as_ref())
                    .metadata(self.scope_metadata(event.metadata))
                    .build(),
            )?;
        }
        Ok(())
    }

    // Lazily opens the root agent scope for harnesses that have a meaningful session boundary.
    // Harnesses without a reliable session end deliberately skip this and use bounded Custom turn
    // scopes as the top-level observable unit.
    fn ensure_agent_started(&mut self, event_metadata: Value) -> Result<(), CliError> {
        if self.agent_scope.is_some()
            || !alignment::should_emit_session_agent_scope(self.agent_kind)
        {
            return Ok(());
        }
        let _root = get_handle()?;
        let metadata = merge_metadata(
            self.scope_metadata(event_metadata),
            json!({ "nemo_relay_scope_role": "session" }),
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

    // Opens a new Custom turn scope for a user prompt. If the previous turn never received a
    // terminal hook, close it first so each user input gets a bounded reviewable trace segment.
    async fn start_turn(&mut self, event: SessionEvent) -> Result<(), CliError> {
        if alignment::aliased_turn_subagent_id(&event).is_some() {
            self.ensure_turn_started(event.metadata.clone())?;
            return self.mark("prompt_submitted", event);
        }
        if self.turn_scope.is_some() {
            if self.gateway_request_turn_open {
                self.gateway_request_turn_open = false;
                return self.mark("prompt_submitted", event);
            }
            self.close_turn_for_reason("superseded_by_next_turn")
                .await?;
        }
        self.open_turn(event.metadata, event.payload, "user_prompt")
    }

    // Lazily creates an implicit turn when gateway/tool/LLM activity arrives before a prompt hook.
    // This keeps direct gateway traffic and sparse hook streams bounded by the same lifecycle as
    // prompt-driven turns.
    fn ensure_turn_started(&mut self, event_metadata: Value) -> Result<(), CliError> {
        if self.turn_scope.is_some() {
            return Ok(());
        }
        self.open_turn(event_metadata, Value::Null, "implicit")
    }

    fn ensure_turn_started_for_gateway(&mut self, start: &LlmGatewayStart) -> Result<(), CliError> {
        if self.turn_scope.is_some() {
            return Ok(());
        }
        if let Some(input) =
            alignment::gateway_turn_input(self.agent_kind, &start.provider, &start.request)
        {
            self.open_turn(start.metadata.clone(), input, "gateway_request")?;
            self.gateway_request_turn_open = true;
            return Ok(());
        }
        self.open_turn(Value::Null, Value::Null, "implicit")
    }

    fn gateway_management_policy(&self, start: &LlmGatewayStart) -> GatewayManagementPolicy {
        if self.turn_scope.is_some() {
            return GatewayManagementPolicy::Managed;
        }
        alignment::gateway_management_policy(
            self.agent_kind,
            &start.provider,
            start.model_name.as_deref(),
            &start.request,
        )
    }

    fn open_turn(
        &mut self,
        event_metadata: Value,
        input: Value,
        turn_source: &str,
    ) -> Result<(), CliError> {
        self.ensure_agent_started(event_metadata.clone())?;
        self.turn_index += 1;
        let metadata = merge_metadata(
            self.scope_metadata(event_metadata),
            json!({
                "nemo_relay_scope_role": "turn",
                "turn_index": self.turn_index,
                "turn_source": turn_source,
            }),
        );
        let turn_name = self.turn_scope_name();
        let scope = push_scope(
            PushScopeParams::builder()
                .name(turn_name.as_str())
                .scope_type(ScopeType::Custom)
                .parent_opt(self.agent_scope.as_ref())
                .metadata(metadata)
                .input(input)
                .build(),
        )?;
        self.turn_scope = Some(scope);
        self.gateway_request_turn_open = false;
        self.last_turn_llm_output = None;
        Ok(())
    }

    fn turn_scope_name(&self) -> String {
        format!("{}-turn", self.agent_kind.as_str())
    }

    fn scope_metadata(&self, event_metadata: Value) -> Value {
        let session_instance_id = self
            .scope_stack
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .root_uuid()
            .to_string();
        merge_metadata(
            merge_metadata(
                merge_metadata(
                    self.config.metadata.clone().unwrap_or(Value::Null),
                    self.session_metadata.clone(),
                ),
                event_metadata,
            ),
            json!({
                "session_id": self.session_id,
                "session_instance_id": session_instance_id,
                "agent_kind": self.agent_kind.as_str(),
                "gateway_config_profile": self.config.profile,
                "plugin_config": self.config.plugin_config,
                "gateway_mode": self.config.gateway_mode,
            }),
        )
    }

    // Tool hook payloads do not consistently repeat the harness session id.
    // Mirror the stable managed identity onto each tool event so external
    // consumers can correlate it without reconstructing the parent scope tree.
    fn event_identity_metadata(&self, event_metadata: Value) -> Value {
        merge_metadata(
            event_metadata,
            json!({
                "session_id": self.session_id,
                "turn_id": self.turn_index.to_string(),
                "harness": self.agent_kind.as_str(),
                "source": "hook",
                "identity_quality": "native",
            }),
        )
    }

    async fn end_turn(&mut self, event: SessionEvent) -> Result<(), CliError> {
        if let Some(subagent_id) = alignment::aliased_turn_subagent_id(&event) {
            self.close_subagent_scope(&subagent_id, event.payload)
                .await?;
            return Ok(());
        }
        self.close_turn(event.payload, Some(event.metadata), "closed_by_turn_end")
            .await?;
        Ok(())
    }

    async fn close_turn_for_reason(&mut self, reason: &str) -> Result<Vec<String>, CliError> {
        self.close_turn(json!({ "status": reason }), None, reason)
            .await
    }

    async fn close_turn(
        &mut self,
        output: Value,
        boundary_metadata: Option<Value>,
        reason: &str,
    ) -> Result<Vec<String>, CliError> {
        if self.turn_scope.is_none() {
            return Ok(Vec::new());
        }
        self.close_active_llms(reason)?;
        self.close_active_tools(reason)?;
        let closed_subagents = self.close_active_subagents(reason).await?;
        let output = self.last_turn_llm_output.take().unwrap_or(output);
        self.clear_correlation_state();
        self.close_turn_scope(output, boundary_metadata)?;
        Ok(closed_subagents)
    }

    // Closes the session in a fail-safe order: active turn first, then the root agent scope when
    // the harness has one. Duplicate terminal hooks must not reopen scopes.
    async fn end_agent(&mut self, event: SessionEvent) -> Result<(), CliError> {
        if !self.session_started && self.agent_scope.is_none() && self.turn_scope.is_none() {
            return Ok(());
        }
        self.close_turn_for_reason("closed_by_agent_end").await?;
        self.clear_correlation_state();
        self.close_agent_scope(event.payload)?;
        self.session_started = false;
        Ok(())
    }

    async fn close_for_shutdown(&mut self, reason: &str) -> Result<(), CliError> {
        let stack = self.scope_stack.clone();
        let payload = json!({ "status": reason });
        TASK_SCOPE_STACK
            .scope(stack, async move {
                if self.agent_scope.is_none() && self.turn_scope.is_none() {
                    return Ok(());
                }
                self.close_turn_for_reason(reason).await?;
                self.clear_correlation_state();
                self.close_agent_scope(payload)?;
                self.session_started = false;
                Ok(())
            })
            .await
    }

    // Ends all active hook-observed LLM calls before closing their containing scopes.
    fn close_active_llms(&mut self, reason: &str) -> Result<(), CliError> {
        let active_llms: Vec<_> = self.llms.drain().map(|(_, handle)| handle).collect();
        for handle in active_llms {
            llm_call_end(
                LlmCallEndParams::builder()
                    .handle(&handle)
                    .response(json!({ "status": reason }))
                    .metadata(json!({ "status": reason }))
                    .build(),
            )?;
        }
        Ok(())
    }

    // Ends all active tool calls with a synthetic close result before ending their containing scopes.
    // Draining first avoids holding mutable map state while the runtime emits lifecycle events.
    fn close_active_tools(&mut self, reason: &str) -> Result<(), CliError> {
        let active_tools: Vec<_> = self
            .tools
            .drain()
            .map(|(_, active)| active.handle)
            .collect();
        for handle in active_tools {
            tool_call_end(
                ToolCallEndParams::builder()
                    .handle(&handle)
                    .result(json!({ "status": reason }))
                    .metadata(json!({ "status": reason }))
                    .build(),
            )?;
        }
        Ok(())
    }

    // Pops active subagent scopes in reverse start order. Each subagent owns an independent runtime
    // stack so parallel harness workers can still close cleanly when their completion hooks arrive
    // out of order. Applies to Claude Code Agent workers and Codex child threads today.
    async fn close_active_subagents(&mut self, reason: &str) -> Result<Vec<String>, CliError> {
        let mut closed = Vec::new();
        while let Some(subagent_id) = self.subagent_stack.pop() {
            self.close_subagent_scope(&subagent_id, json!({ "status": reason }))
                .await?;
            closed.push(subagent_id);
        }
        self.subagents.clear();
        self.subagent_stacks.clear();
        Ok(closed)
    }

    // Clears sticky LLM/tool ownership hints that should not survive a turn boundary.
    fn clear_correlation_state(&mut self) {
        self.pending_llm_hints.clear();
        self.pending_tool_hints.clear();
        self.llm_request_affinity.clear();
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

    fn close_turn_scope(
        &mut self,
        output: Value,
        boundary_metadata: Option<Value>,
    ) -> Result<(), CliError> {
        let Some(scope) = self.turn_scope.take() else {
            return Ok(());
        };
        self.gateway_request_turn_open = false;
        pop_scope(
            PopScopeParams::builder()
                .handle_uuid(&scope.uuid)
                .output(output)
                .metadata_opt(boundary_metadata)
                .build(),
        )?;
        Ok(())
    }

    fn root_work_scope(&self) -> Option<ScopeHandle> {
        self.turn_scope.clone().or_else(|| self.agent_scope.clone())
    }

    // Starts an Agent subagent scope under the active Custom turn scope. Duplicate subagent starts
    // are ignored so integrations that retry or emit both "start" and "created" style hooks do
    // not double-nest.
    //
    // Subagents get their own runtime stack seeded with the turn parent. That keeps Phoenix
    // parentage sibling-shaped within a turn while still allowing parallel workers to end out of
    // order.
    async fn start_subagent(&mut self, event: SubagentEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event.metadata.clone())?;
        if self.subagents.contains_key(&event.subagent_id) {
            return Ok(());
        }
        let has_parallel_sibling = !self.subagents.is_empty();
        let parent_scope = self
            .turn_scope
            .clone()
            .expect("ensure_turn_started should initialize the turn scope");
        let agent_scope = self.agent_scope.clone();
        let subagent_id = event.subagent_id;
        let subagent_name = format!("subagent:{subagent_id}");
        let metadata = merge_metadata(
            event.metadata,
            json!({ "nemo_relay_scope_role": "subagent" }),
        );
        let subagent_stack = create_scope_stack();
        let scope = TASK_SCOPE_STACK
            .scope(subagent_stack.clone(), async {
                if let Some(agent_scope) = agent_scope {
                    task_scope_push(agent_scope);
                }
                task_scope_push(parent_scope.clone());
                push_scope(
                    PushScopeParams::builder()
                        .name(subagent_name.as_str())
                        .scope_type(ScopeType::Agent)
                        .parent(&parent_scope)
                        .metadata(metadata)
                        .input(event.payload)
                        .build(),
                )
                .map_err(CliError::from)
            })
            .await?;
        self.completed_subagents.remove(&subagent_id);
        if has_parallel_sibling {
            self.set_last_subagent_start_owner(Some(subagent_id.clone()));
        }
        self.subagent_stack.push(subagent_id.clone());
        self.subagent_stacks
            .insert(subagent_id.clone(), subagent_stack);
        self.subagents.insert(subagent_id, scope);
        Ok(())
    }

    // Ends a subagent by id. Unknown endings usually become mark events, while duplicate endings for
    // a subagent already closed by another provider-specific completion signal are ignored. Claude
    // Code can also report late orphan stops after a turn has closed; those are logged and ignored
    // when there is no active turn so they cannot create lifecycle-only traces.
    async fn end_subagent(&mut self, event: SubagentEvent) -> Result<(), CliError> {
        if self.completed_subagents.contains(&event.subagent_id) {
            return Ok(());
        }
        if !self.subagents.contains_key(&event.subagent_id) {
            log::warn!(
                target: "nemo_relay.session",
                event = "session_correlation_failed",
                session_id = event.session_id.as_str(),
                subagent_id = event.subagent_id.as_str(),
                lifecycle_event = event.event_name.as_str();
                "Subagent lifecycle event had no matching start"
            );
            if self.agent_kind == AgentKind::ClaudeCode && self.turn_scope.is_none() {
                return Ok(());
            }
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
        self.ensure_turn_started(event.metadata.clone())?;
        self.close_subagent_scope(&event.subagent_id, event.payload)
            .await?;
        Ok(())
    }

    // Closes one subagent using that subagent's own scope stack. This is shared by explicit end
    // hooks, provider-specific tool-completion signals, and agent shutdown so all paths clean up
    // ownership hints the same way. Applies to Claude Code Agent-tool completion today.
    async fn close_subagent_scope(
        &mut self,
        subagent_id: &str,
        output: Value,
    ) -> Result<bool, CliError> {
        let Some(scope) = self.subagents.remove(subagent_id) else {
            return Ok(false);
        };
        let stack = self
            .subagent_stacks
            .remove(subagent_id)
            .unwrap_or_else(|| self.scope_stack.clone());
        TASK_SCOPE_STACK
            .scope(stack, async {
                pop_scope(
                    PopScopeParams::builder()
                        .handle_uuid(&scope.uuid)
                        .output(output)
                        .build(),
                )
                .map_err(CliError::from)
            })
            .await?;
        self.subagent_stack.retain(|id| id != subagent_id);
        self.completed_subagents.insert(subagent_id.to_string());
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
        Ok(true)
    }

    // Stores an LLM correlation hint from hook activity after pruning expired hints. Hints do not
    // emit runtime events themselves; they are consumed by the next matching gateway LLM call.
    fn add_llm_hint(&mut self, event: LlmHintEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event.metadata.clone())?;
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
    // ignored so repeated pre hooks do not create parallel handles for one provider call. Aliased
    // child-session LLMs carry their subagent owner in metadata and are resolved by
    // `hook_llm_owner`.
    fn start_hook_llm(&mut self, event: LlmEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event.metadata.clone())?;
        if self.llms.contains_key(&event.api_call_id) {
            return Ok(());
        }
        let (parent, metadata) = self.hook_llm_owner(event.metadata);
        let handle = llm_call(
            LlmCallParams::builder()
                .name(event.provider.as_str())
                .request(&LlmRequest {
                    headers: Map::new(),
                    content: event.request,
                })
                .parent_opt(parent.as_ref())
                .attributes(LlmAttributes::empty())
                .metadata(metadata)
                .model_name_opt(event.model_name)
                .build(),
        )?;
        self.llms.insert(event.api_call_id, handle);
        Ok(())
    }

    // Ends a hook-observed LLM call, synthesizing a start if only the post hook arrives. The same
    // alias metadata recovery used by `start_hook_llm` keeps post-only aliased child LLMs under the
    // subagent instead of falling back to the root agent.
    fn end_hook_llm(&mut self, event: LlmEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event.metadata.clone())?;
        let (parent, metadata) = self.hook_llm_owner(event.metadata);
        let handle = match self.llms.remove(&event.api_call_id) {
            Some(handle) => handle,
            None => llm_call(
                LlmCallParams::builder()
                    .name(event.provider.as_str())
                    .request(&LlmRequest {
                        headers: Map::new(),
                        content: event.request,
                    })
                    .parent_opt(parent.as_ref())
                    .attributes(LlmAttributes::empty())
                    .metadata(metadata.clone())
                    .model_name_opt(event.model_name.clone())
                    .build(),
            )?,
        };
        let output = event.response;
        let root_owned =
            json_string_at(&metadata, &[&["llm_correlation_subagent_id"][..]]).is_none();
        if root_owned {
            self.record_turn_llm_output(output.clone());
        }
        llm_call_end(
            LlmCallEndParams::builder()
                .handle(&handle)
                .response(output)
                .metadata(metadata)
                .build(),
        )?;
        Ok(())
    }

    // Recovers owner information stamped by alignment when a hook-originated LLM event came from
    // an aliased child session. Gateway LLM calls have first-class owner resolution, but hook LLM
    // events only carry metadata, so this is the bridge that keeps aliased child LLMs under the
    // subagent instead of the root agent.
    fn hook_llm_owner(&mut self, metadata: Value) -> (Option<ScopeHandle>, Value) {
        let Some(subagent_id) = json_string_at(&metadata, &[&["llm_correlation_subagent_id"][..]])
        else {
            return (self.root_work_scope(), metadata);
        };
        let Some(scope) = self.subagents.get(&subagent_id).cloned() else {
            return (self.root_work_scope(), metadata);
        };
        self.set_last_llm_owner(Some(subagent_id.clone()));
        (
            Some(scope),
            merge_metadata(metadata, self.subagent_llm_metadata(&subagent_id)),
        )
    }

    // Starts a tool call under an explicit subagent when available, otherwise under the turn
    // scope. Duplicate tool IDs are ignored so repeated pre-tool hooks do not create parallel
    // handles for one agent tool invocation.
    fn start_tool(&mut self, event: ToolEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event.metadata.clone())?;
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
        let active_tool_arguments = arguments.clone();
        let active_tool_name = event.tool_name.clone();
        let active_tool_owner_subagent_id = owner.subagent_id.clone();
        tool_conditional_execution(event.tool_name.as_str(), &arguments)?;
        let metadata = tool_correlation_metadata(
            self.event_identity_metadata(event.metadata),
            owner.status,
            owner.source.as_deref(),
            owner.subagent_id.as_deref(),
            owner.hint.as_ref(),
        );
        self.set_last_tool_owner(owner.subagent_id.clone());
        let handle = tool_call(
            ToolCallParams::builder()
                .name(event.tool_name.as_str())
                .args(arguments)
                .parent_opt(owner.parent.as_ref())
                .metadata(metadata)
                .tool_call_id(event.tool_call_id.clone())
                .build(),
        )?;
        self.tools.insert(
            event.tool_call_id,
            ActiveTool {
                handle,
                name: active_tool_name,
                arguments: active_tool_arguments,
                owner_subagent_id: active_tool_owner_subagent_id,
            },
        );
        Ok(())
    }

    // Ends a tool call, synthesizing a start if no matching handle exists. This keeps post-only
    // hooks observable and preserves the final result/status instead of dropping orphaned endings.
    async fn end_tool(&mut self, event: ToolEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event.metadata.clone())?;
        let event_metadata = self.event_identity_metadata(event.metadata.clone());
        let completed_agent_subagent_id = alignment::completed_subagent_from_tool(&event);
        let explicit_subagent_id = event
            .subagent_id
            .clone()
            .filter(|subagent_id| self.subagents.contains_key(subagent_id));
        let handle = match self.remove_tool_handle_for_event(&event) {
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
                    event_metadata.clone(),
                    owner.status,
                    owner.source.as_deref(),
                    owner.subagent_id.as_deref(),
                    owner.hint.as_ref(),
                );
                self.set_last_tool_owner(owner.subagent_id.clone());
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
                .result(event.result.clone())
                .metadata(merge_metadata(
                    event_metadata,
                    json!({ "status": event.status }),
                ))
                .build(),
        )?;
        self.set_last_tool_owner(explicit_subagent_id);
        if let Some(subagent_id) = completed_agent_subagent_id {
            self.close_subagent_scope(&subagent_id, event.result)
                .await?;
        }
        Ok(())
    }

    // Hermes pre/post tool hooks can disagree on call IDs: pre hooks may omit the provider id
    // while post hooks carry the final chat-completions tool id. When the ID misses but exactly
    // one active tool owned by the same subagent/root scope has the same name and arguments, close
    // that start instead of synthesizing a second zero-duration span.
    fn remove_tool_handle_for_event(&mut self, event: &ToolEvent) -> Option<ToolHandle> {
        if let Some(active) = self.tools.remove(&event.tool_call_id) {
            return Some(active.handle);
        }
        let owner_subagent_id = self.tool_event_owner_subagent_id(event);
        let key = self.matching_active_tool_key(event, owner_subagent_id.as_deref())?;
        self.tools.remove(&key).map(|active| active.handle)
    }

    fn matching_active_tool_key(
        &self,
        event: &ToolEvent,
        owner_subagent_id: Option<&str>,
    ) -> Option<String> {
        if event.arguments.is_null() {
            return None;
        }
        let matches = self
            .tools
            .iter()
            .filter_map(|(key, active)| {
                (owner_subagent_id
                    .is_none_or(|owner| active.owner_subagent_id.as_deref() == Some(owner))
                    && active.name == event.tool_name
                    && active.arguments == event.arguments)
                    .then_some(key.clone())
            })
            .collect::<Vec<_>>();
        (matches.len() == 1).then(|| matches[0].clone())
    }

    fn tool_event_owner_subagent_id(&self, event: &ToolEvent) -> Option<String> {
        if let Some(subagent_id) = &event.subagent_id
            && self.subagents.contains_key(subagent_id)
        {
            return Some(subagent_id.clone());
        }
        self.matching_tool_hint_index(event)
            .and_then(|index| self.pending_tool_hints[index].hint.subagent_id.clone())
            .filter(|subagent_id| self.subagents.contains_key(subagent_id))
    }

    // Emits a mark event after ensuring the turn scope exists. Generic and unknown hooks use this
    // path so unsupported agent events remain visible without changing scope structure.
    fn mark(&mut self, name: &str, event_payload: SessionEvent) -> Result<(), CliError> {
        self.ensure_turn_started(event_payload.metadata.clone())?;
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
        if let Some(resolution) = self.request_affinity_owner(start) {
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
                metadata: self.subagent_llm_metadata(subagent_id),
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

    // Reuses a learned request affinity before falling back to the session-global sticky owner.
    // The key is derived from provider request payloads, not a harness-specific field, so it can
    // pair unhinted Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses calls with
    // the subagent that first owned the same coding task.
    fn request_affinity_owner(&mut self, start: &LlmGatewayStart) -> Option<LlmOwnerResolution> {
        let key = alignment::request_affinity_key(&start.provider, &start.request)?;
        let subagent_id = self.llm_request_affinity.get(&key).cloned().flatten()?;
        let parent = match self.subagents.get(&subagent_id).cloned() {
            Some(parent) => parent,
            None => {
                self.llm_request_affinity.remove(&key);
                return None;
            }
        };
        self.set_last_llm_owner(Some(subagent_id.clone()));
        Some(LlmOwnerResolution {
            parent: Some(parent),
            subagent_id: Some(subagent_id.clone()),
            status: "request_affinity",
            source: Some("request_payload".to_string()),
            hint: None,
            metadata: self.subagent_llm_metadata(&subagent_id),
        })
    }

    // Reuses the previous LLM owner while its TTL is valid and its scope can still be resolved.
    // This covers agents that emit one hint followed by a cluster of related provider calls.
    fn sticky_llm_owner(&self) -> Option<LlmOwnerResolution> {
        if let Some(owner) = self.last_llm_owner.as_ref()
            && let Some(parent) = self.subagents.get(&owner.subagent_id).cloned()
        {
            return Some(LlmOwnerResolution {
                parent: Some(parent),
                subagent_id: Some(owner.subagent_id.clone()),
                status: owner.source.status(),
                source: owner.source.metadata_source().map(ToOwned::to_owned),
                hint: None,
                metadata: self.subagent_llm_metadata(&owner.subagent_id),
            });
        }
        None
    }

    // Assigns an unhinted gateway call to the only active subagent. Multiple active subagents are
    // deliberately not guessed here; those cases fall back to the turn scope with ambiguity
    // metadata.
    fn sole_subagent_owner(&mut self) -> Option<LlmOwnerResolution> {
        if self.subagents.len() == 1
            && let Some((subagent_id, scope)) = self.subagents.iter().next()
        {
            let subagent_id = subagent_id.clone();
            let scope = scope.clone();
            let metadata = self.subagent_llm_metadata(&subagent_id);
            self.set_last_llm_owner(Some(subagent_id.clone()));
            return Some(LlmOwnerResolution {
                parent: Some(scope),
                subagent_id: Some(subagent_id),
                status: "active_subagent",
                source: None,
                hint: None,
                metadata,
            });
        }
        None
    }

    // Final fallback for gateway calls that cannot be correlated to a subagent. Pending hints are
    // left intact in ambiguous cases so later calls with stronger identifiers can still match them.
    fn fallback_llm_owner(&self) -> LlmOwnerResolution {
        LlmOwnerResolution {
            parent: self.root_work_scope(),
            subagent_id: None,
            status: if self.pending_llm_hints.is_empty() {
                "agent_fallback"
            } else {
                "ambiguous_fallback"
            },
            source: None,
            hint: None,
            metadata: Value::Null,
        }
    }

    fn unmanaged_probe_owner(&self, policy: GatewayManagementPolicy) -> LlmOwnerResolution {
        let (status, source) = policy
            .bypass_correlation()
            .expect("unmanaged probe owner requires unmanaged gateway policy");
        LlmOwnerResolution {
            parent: self.root_work_scope(),
            subagent_id: None,
            status,
            source: Some(source.to_string()),
            hint: None,
            metadata: Value::Null,
        }
    }

    // Converts a consumed hint into an ownership resolution. If the hinted subagent is not
    // currently active, the LLM is attached to the turn scope but the hint metadata is still
    // preserved for correlation diagnostics.
    fn resolution_from_hint(
        &mut self,
        hint: LlmHintEvent,
        status: &'static str,
    ) -> LlmOwnerResolution {
        let hinted_subagent_id = hint.subagent_id.clone().or_else(|| hint.agent_id.clone());
        let (parent, subagent_id, metadata) = match hinted_subagent_id.as_deref() {
            Some(id) => match self.subagents.get(id).cloned() {
                Some(scope) => (
                    Some(scope),
                    Some(id.to_string()),
                    self.subagent_llm_metadata(id),
                ),
                None => (self.root_work_scope(), None, Value::Null),
            },
            None => (self.root_work_scope(), None, Value::Null),
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
            metadata,
        }
    }

    fn subagent_llm_metadata(&self, subagent_id: &str) -> Value {
        let Some(scope) = self.subagents.get(subagent_id) else {
            return Value::Null;
        };
        alignment::llm_owner_metadata(scope.metadata.as_ref())
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

    // Records the most recent LLM owner with a timestamp so nearby gateway calls can inherit the
    // same parent scope when explicit IDs and hints are absent.
    fn set_last_llm_owner(&mut self, subagent_id: Option<String>) {
        self.last_llm_owner = subagent_id.map(|subagent_id| LastLlmOwner {
            subagent_id,
            updated_at: Instant::now(),
            source: LastLlmOwnerSource::Llm,
        });
    }

    // Records explicit or hint-resolved tool ownership as a short-lived cue for the next unhinted
    // LLM call. Coding-agent hooks often identify tool ownership more reliably than provider
    // requests, especially for subagents that do not propagate gateway headers.
    fn set_last_tool_owner(&mut self, subagent_id: Option<String>) {
        if let Some(subagent_id) = subagent_id {
            self.last_llm_owner = Some(LastLlmOwner {
                subagent_id,
                updated_at: Instant::now(),
                source: LastLlmOwnerSource::Tool,
            });
        }
    }

    // Parallel subagent starts are a weak but useful ownership signal: if a new worker starts while
    // siblings are active and the next LLM lacks headers/hints, prefer the newest worker over the
    // root agent. Single-subagent ownership is handled by `sole_subagent_owner`.
    fn set_last_subagent_start_owner(&mut self, subagent_id: Option<String>) {
        if let Some(subagent_id) = subagent_id {
            self.last_llm_owner = Some(LastLlmOwner {
                subagent_id,
                updated_at: Instant::now(),
                source: LastLlmOwnerSource::SubagentStart,
            });
        }
    }

    // Learns a subagent owner from high-confidence LLM resolutions only. Tool-owned and sticky
    // resolutions are intentionally excluded because they are the ambiguous path this affinity map
    // is meant to correct when multiple coding-agent workers share a root session.
    fn record_llm_request_affinity(
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

    // Remembers the latest completed LLM response owned by the turn or root Agent scope so the
    // enclosing Custom turn scope can export the final assistant output. Subagent-owned responses are
    // deliberately excluded; otherwise a worker's last local answer can overwrite the parent
    // agent's final synthesis.
    fn record_completed_llm_response(
        &mut self,
        response: Value,
        owner_subagent_id: Option<String>,
    ) {
        if owner_subagent_id.is_none() {
            self.record_turn_llm_output(response.clone());
        }
        self.add_tool_hints_from_llm_response(response, owner_subagent_id);
    }

    fn record_turn_llm_output(&mut self, response: Value) {
        if self.turn_scope.is_some() {
            self.last_turn_llm_output = Some(response);
        }
    }

    // Resolves tool hook ownership from explicit subagent data first, then private tool hints
    // extracted from LLM responses, and finally the turn scope.
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

        if let Some(index) = self.matching_tool_hint_index(event) {
            let status = if self.pending_tool_hints.len() == 1 {
                "single_hint"
            } else {
                "matched_hint"
            };
            let hint = self.pending_tool_hints.remove(index).hint;
            return self.tool_resolution_from_hint(hint, status);
        }

        ToolOwnerResolution {
            parent: self.root_work_scope(),
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

    // Converts a consumed tool hint into a live parent scope, falling back to the turn scope if the
    // hinted subagent has already ended or never existed.
    fn tool_resolution_from_hint(
        &mut self,
        hint: ToolHint,
        status: &'static str,
    ) -> ToolOwnerResolution {
        let (parent, subagent_id) = match hint.subagent_id.as_deref() {
            Some(id) => match self.subagents.get(id).cloned() {
                Some(scope) => (Some(scope), Some(id.to_string())),
                None => (self.root_work_scope(), None),
            },
            None => (self.root_work_scope(), None),
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

    // Finds a unique best-scoring tool hint by call id or name-plus-argument equality. Ties remain
    // ambiguous and are not consumed. Name-only matches are ignored because high-frequency
    // coding-agent tools repeat across parallel workers and are too weak to prove ownership.
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
#[cfg(test)]
#[path = "../../tests/coverage/shared/session_tests.rs"]
mod tests;
