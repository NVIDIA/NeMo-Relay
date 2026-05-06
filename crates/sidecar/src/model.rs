// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AgentKind {
    Codex,
    ClaudeCode,
    Cursor,
    Hermes,
    Gateway,
}

impl AgentKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude-code",
            Self::Cursor => "cursor",
            Self::Hermes => "hermes",
            Self::Gateway => "gateway",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NormalizedEvent {
    AgentStarted(SessionEvent),
    AgentEnded(SessionEvent),
    SubagentStarted(SubagentEvent),
    SubagentEnded(SubagentEvent),
    LlmStarted(LlmEvent),
    LlmEnded(LlmEvent),
    ToolStarted(ToolEvent),
    ToolEnded(ToolEvent),
    PromptSubmitted(SessionEvent),
    AgentResponse(SessionEvent),
    Compaction(SessionEvent),
    Notification(SessionEvent),
    HookMark(SessionEvent),
}

impl NormalizedEvent {
    pub(crate) fn session_id(&self) -> &str {
        match self {
            Self::AgentStarted(event)
            | Self::AgentEnded(event)
            | Self::PromptSubmitted(event)
            | Self::AgentResponse(event)
            | Self::Compaction(event)
            | Self::Notification(event)
            | Self::HookMark(event) => &event.session_id,
            Self::LlmStarted(event) | Self::LlmEnded(event) => &event.session_id,
            Self::SubagentStarted(event) | Self::SubagentEnded(event) => &event.session_id,
            Self::ToolStarted(event) | Self::ToolEnded(event) => &event.session_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SessionEvent {
    pub(crate) session_id: String,
    pub(crate) agent_kind: AgentKind,
    pub(crate) event_name: String,
    pub(crate) payload: Value,
    pub(crate) metadata: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SubagentEvent {
    pub(crate) session_id: String,
    pub(crate) agent_kind: AgentKind,
    pub(crate) event_name: String,
    pub(crate) subagent_id: String,
    pub(crate) payload: Value,
    pub(crate) metadata: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LlmEvent {
    pub(crate) session_id: String,
    pub(crate) agent_kind: AgentKind,
    pub(crate) event_name: String,
    pub(crate) api_call_id: String,
    pub(crate) provider: String,
    pub(crate) model_name: Option<String>,
    pub(crate) request: Value,
    pub(crate) response: Value,
    pub(crate) metadata: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolEvent {
    pub(crate) session_id: String,
    pub(crate) agent_kind: AgentKind,
    pub(crate) event_name: String,
    pub(crate) tool_call_id: String,
    pub(crate) tool_name: String,
    pub(crate) subagent_id: Option<String>,
    pub(crate) arguments: Value,
    pub(crate) result: Value,
    pub(crate) status: Option<String>,
    pub(crate) payload: Value,
    pub(crate) metadata: Value,
}
