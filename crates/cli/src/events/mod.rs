// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

pub(crate) mod json_path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AgentKind {
    Codex,
    ClaudeCode,
    Pi,
    Gateway,
}

impl AgentKind {
    // Returns the canonical metadata spelling for runtime events. These strings are consumed by
    // observability exporters and therefore avoid deriving from enum debug names.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude-code",
            Self::Pi => "pi",
            Self::Gateway => "gateway",
        }
    }

    // Whether this harness reports the *opening* of a conversation turn, not just its close.
    //
    // Codex and Claude Code only signal a turn boundary on `Stop`, so the gateway has to open
    // turns lazily -- the first tool call, LLM call, or mark after a close starts the next one.
    // pi emits a native `turn_start`, which is classified into `NormalizedEvent::TurnStarted`
    // and opens the scope at pi's own boundary. For those harnesses a mark arriving *between*
    // turns must not manufacture one: doing so left an empty trailing turn holding nothing but
    // the `agent_end`/`agent_settled` marks at the end of every run.
    //
    // Kept next to `as_str` rather than on the agent descriptor because the session manager
    // works in terms of `AgentKind`; the matching hook names live in each adapter's
    // `ClassificationRules::turn_start`.
    pub(crate) const fn has_explicit_turn_start(self) -> bool {
        matches!(self, Self::Pi)
    }

    // Whether this harness can execute arguments the gateway rewrote.
    //
    // Only pi: its `tool_call` hook documents in-place mutation of `input` as the mechanism, and
    // the extension applies whatever the hook response carries. Codex and Claude Code have no
    // equivalent return path, so running the request-intercept chain for them would record
    // arguments on the tool span that never executed -- worse than not running it, because the
    // trace would then disagree with reality.
    pub(crate) const fn applies_tool_argument_transforms(self) -> bool {
        matches!(self, Self::Pi)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NormalizedEvent {
    AgentStarted(SessionEvent),
    AgentEnded(SessionEvent),
    /// Conversation-turn boundary that opens the turn scope at the harness's own signal.
    ///
    /// Only emitted for harnesses that report one (pi's `turn_start`). Without it the gateway
    /// opens turns lazily on the first tool, LLM, or mark event, which puts the boundary
    /// wherever traffic happens to arrive rather than where the harness says the turn begins.
    TurnStarted(SessionEvent),
    /// Conversation-turn boundary that the gateway uses to snapshot ATIF without closing the
    /// agent scope. Emitted alongside `LlmHint` for `Stop` hooks (Claude/Codex).
    /// Required for Codex transparent runs because Codex has no reliable `SessionEnd`-equivalent
    /// event — the last `Stop` of the session leaves an up-to-date ATIF on disk. Multi-turn
    /// sessions write progressively complete trajectories; the underlying `AtifExporter::export()`
    /// is non-destructive so each snapshot is a cumulative superset of prior writes.
    TurnEnded(SessionEvent),
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

#[cfg(test)]
#[path = "../../tests/coverage/shared/events_tests.rs"]
mod tests;

impl NormalizedEvent {
    // Extracts the routing session id regardless of normalized event kind. Keeping this on the
    // enum lets the session manager group events before it needs to inspect lifecycle semantics.
    pub(crate) fn session_id(&self) -> &str {
        match self {
            Self::AgentStarted(event)
            | Self::AgentEnded(event)
            | Self::TurnStarted(event)
            | Self::TurnEnded(event)
            | Self::PromptSubmitted(event)
            | Self::Compaction(event)
            | Self::Notification(event)
            | Self::HookMark(event) => &event.session_id,
            Self::LlmHint(event) => &event.session_id,
            Self::SubagentStarted(event) | Self::SubagentEnded(event) => &event.session_id,
            Self::ToolStarted(event) | Self::ToolEnded(event) => &event.session_id,
        }
    }

    pub(crate) fn is_terminal(&self) -> bool {
        // TurnStarted/TurnEnded are intentionally NOT terminal — the agent scope stays open
        // across turns.
        matches!(
            self,
            Self::AgentEnded(_) | Self::SubagentEnded(_) | Self::ToolEnded(_)
        )
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
