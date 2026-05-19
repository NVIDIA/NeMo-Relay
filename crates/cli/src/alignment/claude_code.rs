// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code-specific trace alignment.
//!
//! Claude Code already propagates a native session header and can report subagent completion via
//! the `Agent` tool result. These helpers keep those vendor-specific hints outside the generic
//! session state machine.

use axum::http::HeaderMap;

use crate::alignment::json_string_at;
use crate::config::header_string;
use crate::model::{AgentKind, ToolEvent};

// Identifies gateway providers that should be labeled as Claude-owned when an Anthropic request
// arrives before a SessionStart hook. Other providers are left generic so mixed gateway traffic
// does not inherit Claude scope metadata by route alone.
pub(crate) fn owns_gateway_provider(provider: &str) -> bool {
    matches!(provider, "anthropic.messages" | "anthropic.count_tokens")
}

// Claude Code already has a stable session id header. Accept it after the explicit NeMo Flow
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
#[path = "../../tests/coverage/alignment_claude_code_tests.rs"]
mod tests;
