// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code-specific trace alignment.
//!
//! Claude Code already propagates a native session header and can report subagent completion via
//! the `Agent` tool result. These helpers keep those vendor-specific hints outside the generic
//! session state machine.

use crate::agents::shared::alignment::json_string_at;
use crate::configuration::header_string;
use crate::events::{AgentKind, ToolEvent};
use axum::http::HeaderMap;

/// Client-reported operation class, not evidence of human initiation or authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestClass {
    Main,
    Subagent,
    Workflow,
    Compaction,
    Auxiliary,
    Unknown,
}

impl RequestClass {
    pub(crate) fn is_internal(self) -> bool {
        matches!(self, Self::Compaction | Self::Auxiliary)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Subagent => "subagent",
            Self::Workflow => "workflow",
            Self::Compaction => "compaction",
            Self::Auxiliary => "auxiliary",
            Self::Unknown => "unknown",
        }
    }
}

/// Accept one exact documented value before header normalization loses duplicates.
/// Padded, combined, repeated, invalid, and future values remain unknown.
pub(crate) fn request_class(headers: &HeaderMap) -> RequestClass {
    let mut values = headers.get_all("x-claude-code-request-class").iter();
    let Some(value) = values.next() else {
        return RequestClass::Unknown;
    };
    if values.next().is_some() {
        return RequestClass::Unknown;
    }
    match value.to_str().ok() {
        Some("main") => RequestClass::Main,
        Some("subagent") => RequestClass::Subagent,
        Some("workflow") => RequestClass::Workflow,
        Some("compaction") => RequestClass::Compaction,
        Some("auxiliary") => RequestClass::Auxiliary,
        _ => RequestClass::Unknown,
    }
}

// Identifies gateway providers that should be labeled as Claude-owned when an Anthropic request
// arrives before a SessionStart hook. Other providers are left generic so mixed gateway traffic
// does not inherit Claude scope metadata by route alone.
pub(crate) fn owns_gateway_provider(provider: &str) -> bool {
    matches!(provider, "anthropic.messages" | "anthropic.count_tokens")
}

// Claude Code already has a stable session id header. Accept it after the explicit NeMo Relay
// header so existing Claude environments correlate without extra gateway-specific configuration.
pub(crate) fn session_id_from_headers(headers: &HeaderMap) -> Option<String> {
    header_string(headers, "x-claude-code-session-id")
}

// Claude's `Agent` tool can report either an asynchronous launch acknowledgement or a terminal
// worker result. Only the terminal result should close the subagent scope; otherwise parallel
// workers that launch in the background are closed before their later tool/LLM hooks arrive.
pub(crate) fn completed_subagent_from_agent_tool(event: &ToolEvent) -> Option<String> {
    if event.agent_kind != AgentKind::ClaudeCode || event.tool_name != "Agent" {
        return None;
    }
    if !is_terminal_agent_tool_result(&event.result) {
        return None;
    }
    json_string_at(
        &event.result,
        &[
            &["agentId"][..],
            &["agent_id"][..],
            &["subagentId"][..],
            &["subagent_id"][..],
        ],
    )
}

fn is_terminal_agent_tool_result(result: &serde_json::Value) -> bool {
    let status = json_string_at(result, &[&["status"][..]])
        .map(|status| status.trim().to_ascii_lowercase().replace(['-', ' '], "_"));
    match status.as_deref() {
        Some("async_launched" | "launched" | "started" | "running" | "pending" | "in_progress") => {
            false
        }
        Some(
            "completed" | "complete" | "success" | "succeeded" | "failed" | "error" | "errored"
            | "cancelled" | "canceled" | "timeout" | "timed_out",
        ) => true,
        Some(_) | None => has_terminal_agent_tool_evidence(result),
    }
}

fn has_terminal_agent_tool_evidence(result: &serde_json::Value) -> bool {
    [
        "content",
        "output",
        "totalDurationMs",
        "totalTokens",
        "totalToolUseCount",
        "durationMs",
        "usage",
    ]
    .into_iter()
    .any(|key| result.get(key).is_some_and(|value| !value.is_null()))
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/alignment_claude_code_tests.rs"]
mod tests;
