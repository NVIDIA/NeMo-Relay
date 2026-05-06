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
    // Returns the canonical metadata spelling for runtime events. These strings are consumed by
    // observability exporters and therefore avoid deriving from enum debug names.
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
    LlmHint(LlmHintEvent),
    ToolStarted(ToolEvent),
    ToolEnded(ToolEvent),
    #[allow(dead_code)]
    PromptSubmitted(SessionEvent),
    Compaction(SessionEvent),
    Notification(SessionEvent),
    HookMark(SessionEvent),
}

impl NormalizedEvent {
    // Extracts the routing session id regardless of normalized event kind. Keeping this on the
    // enum lets the session manager group events before it needs to inspect lifecycle semantics.
    pub(crate) fn session_id(&self) -> &str {
        match self {
            Self::AgentStarted(event)
            | Self::AgentEnded(event)
            | Self::PromptSubmitted(event)
            | Self::Compaction(event)
            | Self::Notification(event)
            | Self::HookMark(event) => &event.session_id,
            Self::LlmHint(event) => &event.session_id,
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
pub(crate) struct LlmHintEvent {
    pub(crate) session_id: String,
    pub(crate) agent_kind: AgentKind,
    pub(crate) event_name: String,
    pub(crate) subagent_id: Option<String>,
    pub(crate) agent_id: Option<String>,
    pub(crate) agent_type: Option<String>,
    pub(crate) conversation_id: Option<String>,
    pub(crate) generation_id: Option<String>,
    pub(crate) request_id: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) payload: Value,
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
