// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::http::HeaderMap;
use nemo_flow::api::llm::{
    LlmAttributes, LlmCallEndParams, LlmCallParams, LlmHandle, LlmRequest, llm_call, llm_call_end,
};
use nemo_flow::api::runtime::{ScopeStackHandle, TASK_SCOPE_STACK, create_scope_stack};
use nemo_flow::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeHandle, ScopeType,
    event as emit_mark_event, get_handle, pop_scope, push_scope,
};
use nemo_flow::api::subscriber::scope_register_subscriber;
use nemo_flow::api::tool::{
    ToolCallEndParams, ToolCallParams, ToolHandle, tool_call, tool_call_end,
};
use nemo_flow::observability::atif::{AtifAgentInfo, AtifExporter};
use nemo_flow::observability::openinference::{OpenInferenceConfig, OpenInferenceSubscriber};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::config::{SessionConfig, SidecarConfig};
use crate::error::SidecarError;
use crate::model::{AgentKind, NormalizedEvent, SessionEvent, SubagentEvent, ToolEvent};

#[derive(Clone)]
pub(crate) struct SessionManager {
    inner: Arc<Mutex<HashMap<String, Session>>>,
    default_config: SidecarConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct LlmGatewayStart {
    pub(crate) session_id: Option<String>,
    pub(crate) provider: String,
    pub(crate) model_name: Option<String>,
    pub(crate) request: LlmRequest,
    pub(crate) streaming: bool,
    pub(crate) metadata: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct ActiveLlm {
    stack: ScopeStackHandle,
    handle: LlmHandle,
}

struct Session {
    agent_kind: AgentKind,
    session_id: String,
    scope_stack: ScopeStackHandle,
    agent_scope: Option<ScopeHandle>,
    subagents: HashMap<String, ScopeHandle>,
    tools: HashMap<String, ToolHandle>,
    config: SessionConfig,
    atif: Option<AtifExporter>,
    openinference: Option<OpenInferenceSubscriber>,
}

impl SessionManager {
    pub(crate) fn new(default_config: SidecarConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            default_config,
        }
    }

    pub(crate) async fn apply_events(
        &self,
        headers: &HeaderMap,
        events: Vec<NormalizedEvent>,
    ) -> Result<(), SidecarError> {
        let mut sessions = self.inner.lock().await;
        for event in events {
            let session_id = event.session_id().to_string();
            let config = self.default_config.session_config_from_headers(headers);
            let session = sessions.entry(session_id.clone()).or_insert_with(|| {
                Session::new(session_id.clone(), event_agent_kind(&event), config.clone())
            });
            session.apply(event).await?;
            if session.agent_scope.is_none()
                && session.subagents.is_empty()
                && session.tools.is_empty()
            {
                sessions.remove(&session_id);
            }
        }
        Ok(())
    }

    pub(crate) async fn start_llm(
        &self,
        headers: &HeaderMap,
        start: LlmGatewayStart,
    ) -> Result<ActiveLlm, SidecarError> {
        let mut sessions = self.inner.lock().await;
        let config = self.default_config.session_config_from_headers(headers);
        let session_id = start
            .session_id
            .clone()
            .or_else(|| single_active_session_id(&sessions))
            .unwrap_or_else(|| format!("{}-gateway", AgentKind::Gateway.as_str()));
        let session = sessions
            .entry(session_id.clone())
            .or_insert_with(|| Session::new(session_id, AgentKind::Gateway, config));
        session.start_llm(start).await
    }

    pub(crate) async fn end_llm(
        &self,
        active: ActiveLlm,
        response: Value,
        metadata: Value,
    ) -> Result<(), SidecarError> {
        TASK_SCOPE_STACK
            .scope(active.stack, async move {
                llm_call_end(
                    LlmCallEndParams::builder()
                        .handle(&active.handle)
                        .response(response)
                        .metadata(metadata)
                        .build(),
                )
                .map_err(SidecarError::from)
            })
            .await
    }
}

impl Session {
    fn new(session_id: String, agent_kind: AgentKind, config: SessionConfig) -> Self {
        Self {
            agent_kind,
            session_id,
            scope_stack: create_scope_stack(),
            agent_scope: None,
            subagents: HashMap::new(),
            tools: HashMap::new(),
            config,
            atif: None,
            openinference: None,
        }
    }

    async fn apply(&mut self, event: NormalizedEvent) -> Result<(), SidecarError> {
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack, async move {
                match event {
                    NormalizedEvent::AgentStarted(event) => self.start_agent(event),
                    NormalizedEvent::AgentEnded(event) => self.end_agent(event),
                    NormalizedEvent::SubagentStarted(event) => self.start_subagent(event),
                    NormalizedEvent::SubagentEnded(event) => self.end_subagent(event),
                    NormalizedEvent::ToolStarted(event) => self.start_tool(event),
                    NormalizedEvent::ToolEnded(event) => self.end_tool(event),
                    NormalizedEvent::PromptSubmitted(event) => self.mark("prompt_submitted", event),
                    NormalizedEvent::AgentResponse(event) => self.mark("agent_response", event),
                    NormalizedEvent::Compaction(event) => self.mark("compaction", event),
                    NormalizedEvent::Notification(event) => self.mark("notification", event),
                    NormalizedEvent::HookMark(event) => self.mark("hook_mark", event),
                }
            })
            .await
    }

    async fn start_llm(&mut self, start: LlmGatewayStart) -> Result<ActiveLlm, SidecarError> {
        let stack = self.scope_stack.clone();
        TASK_SCOPE_STACK
            .scope(stack.clone(), async move {
                self.ensure_agent_started(Value::Null)?;
                let mut attributes = LlmAttributes::empty();
                if start.streaming {
                    attributes |= LlmAttributes::STREAMING;
                }
                let handle = llm_call(
                    LlmCallParams::builder()
                        .name(start.provider.as_str())
                        .request(&start.request)
                        .attributes(attributes)
                        .metadata(start.metadata)
                        .model_name_opt(start.model_name)
                        .build(),
                )?;
                Ok(ActiveLlm { stack, handle })
            })
            .await
    }

    fn start_agent(&mut self, event: SessionEvent) -> Result<(), SidecarError> {
        self.agent_kind = event.agent_kind;
        self.ensure_agent_started(event.metadata)
    }

    fn ensure_agent_started(&mut self, event_metadata: Value) -> Result<(), SidecarError> {
        if self.agent_scope.is_some() {
            return Ok(());
        }
        let root = get_handle()?;
        self.install_observers(&root)?;
        let metadata = merge_metadata(
            merge_metadata(
                self.config.metadata.clone().unwrap_or(Value::Null),
                event_metadata,
            ),
            json!({
                "session_id": self.session_id,
                "sidecar_config_profile": self.config.profile,
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

    fn install_observers(&mut self, root: &ScopeHandle) -> Result<(), SidecarError> {
        if self.atif.is_none() && self.config.atif_dir.is_some() {
            let exporter = AtifExporter::new(
                self.session_id.clone(),
                AtifAgentInfo {
                    name: self.agent_kind.as_str().to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    model_name: None,
                    tool_definitions: None,
                    extra: self.config.metadata.clone(),
                },
            );
            scope_register_subscriber(&root.uuid, "sidecar-atif", exporter.subscriber())?;
            self.atif = Some(exporter);
        }
        if self.openinference.is_none()
            && let Some(endpoint) = &self.config.openinference_endpoint
        {
            let subscriber = OpenInferenceSubscriber::new(
                OpenInferenceConfig::new()
                    .with_endpoint(endpoint.clone())
                    .with_service_name("nemo-flow-sidecar"),
            )?;
            scope_register_subscriber(
                &root.uuid,
                "sidecar-openinference",
                subscriber.subscriber(),
            )?;
            self.openinference = Some(subscriber);
        }
        Ok(())
    }

    fn end_agent(&mut self, event: SessionEvent) -> Result<(), SidecarError> {
        self.ensure_agent_started(event.metadata.clone())?;
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
        let active_subagents: Vec<_> = self.subagents.drain().map(|(_, handle)| handle).collect();
        for handle in active_subagents.into_iter().rev() {
            let _ = pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&handle.uuid)
                    .output(json!({ "status": "closed_by_agent_end" }))
                    .build(),
            );
        }
        if let Some(scope) = self.agent_scope.take() {
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&scope.uuid)
                    .output(event.payload)
                    .build(),
            )?;
        }
        self.flush_observers()?;
        Ok(())
    }

    fn start_subagent(&mut self, event: SubagentEvent) -> Result<(), SidecarError> {
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
        self.subagents.insert(event.subagent_id, scope);
        Ok(())
    }

    fn end_subagent(&mut self, event: SubagentEvent) -> Result<(), SidecarError> {
        self.ensure_agent_started(event.metadata.clone())?;
        let Some(scope) = self.subagents.remove(&event.subagent_id) else {
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
        if pop_scope(
            PopScopeParams::builder()
                .handle_uuid(&scope.uuid)
                .output(event.payload.clone())
                .build(),
        )
        .is_err()
        {
            emit_mark_event(
                EmitMarkEventParams::builder()
                    .name("subagent_end_not_top")
                    .data(event.payload)
                    .metadata(event.metadata)
                    .build(),
            )?;
        }
        Ok(())
    }

    fn start_tool(&mut self, event: ToolEvent) -> Result<(), SidecarError> {
        self.ensure_agent_started(event.metadata.clone())?;
        if self.tools.contains_key(&event.tool_call_id) {
            return Ok(());
        }
        let parent = event
            .subagent_id
            .as_ref()
            .and_then(|id| self.subagents.get(id))
            .or(self.agent_scope.as_ref());
        let handle = tool_call(
            ToolCallParams::builder()
                .name(event.tool_name.as_str())
                .args(event.arguments)
                .parent_opt(parent)
                .metadata(event.metadata)
                .tool_call_id(event.tool_call_id.clone())
                .build(),
        )?;
        self.tools.insert(event.tool_call_id, handle);
        Ok(())
    }

    fn end_tool(&mut self, event: ToolEvent) -> Result<(), SidecarError> {
        self.ensure_agent_started(event.metadata.clone())?;
        let handle = match self.tools.remove(&event.tool_call_id) {
            Some(handle) => handle,
            None => {
                let parent = event
                    .subagent_id
                    .as_ref()
                    .and_then(|id| self.subagents.get(id))
                    .or(self.agent_scope.as_ref());
                tool_call(
                    ToolCallParams::builder()
                        .name(event.tool_name.as_str())
                        .args(event.arguments)
                        .parent_opt(parent)
                        .metadata(event.metadata.clone())
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

    fn mark(&mut self, name: &str, event_payload: SessionEvent) -> Result<(), SidecarError> {
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

    fn flush_observers(&mut self) -> Result<(), SidecarError> {
        if let Some(subscriber) = &self.openinference {
            subscriber.force_flush()?;
            subscriber.shutdown()?;
        }
        if let (Some(exporter), Some(directory)) = (&self.atif, &self.config.atif_dir) {
            write_atif(directory, &self.session_id, exporter)?;
        }
        Ok(())
    }
}

fn write_atif(
    directory: &PathBuf,
    session_id: &str,
    exporter: &AtifExporter,
) -> Result<(), SidecarError> {
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("{session_id}.atif.json"));
    let trajectory = exporter.export();
    let serialized = serde_json::to_vec_pretty(&trajectory)
        .map_err(|error| SidecarError::InvalidPayload(error.to_string()))?;
    std::fs::write(path, serialized)?;
    Ok(())
}

fn event_agent_kind(event: &NormalizedEvent) -> AgentKind {
    match event {
        NormalizedEvent::AgentStarted(event)
        | NormalizedEvent::AgentEnded(event)
        | NormalizedEvent::PromptSubmitted(event)
        | NormalizedEvent::AgentResponse(event)
        | NormalizedEvent::Compaction(event)
        | NormalizedEvent::Notification(event)
        | NormalizedEvent::HookMark(event) => event.agent_kind,
        NormalizedEvent::SubagentStarted(event) | NormalizedEvent::SubagentEnded(event) => {
            event.agent_kind
        }
        NormalizedEvent::ToolStarted(event) | NormalizedEvent::ToolEnded(event) => event.agent_kind,
    }
}

fn single_active_session_id(sessions: &HashMap<String, Session>) -> Option<String> {
    (sessions.len() == 1)
        .then(|| sessions.keys().next().cloned())
        .flatten()
}

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
