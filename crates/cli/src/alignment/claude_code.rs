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

// Claude's `Agent` tool result names the spawned worker as `agentId`. Treating that as a
// completion signal gives the CLI a deterministic subagent end even when Claude does not emit a
// separate `SubagentStop` hook until session teardown.
pub(crate) fn completed_subagent_from_agent_tool(event: &ToolEvent) -> Option<String> {
    if event.agent_kind != AgentKind::ClaudeCode || event.tool_name != "Agent" {
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

#[cfg(test)]
#[path = "../../tests/coverage/alignment_claude_code_tests.rs"]
mod tests;
