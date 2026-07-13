// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code-specific trace alignment.
//!
//! Claude Code already propagates a native session header and can report subagent completion via
//! the `Agent` tool result. These helpers keep those vendor-specific hints outside the generic
//! session state machine.

use axum::http::HeaderMap;
use nemo_relay::api::llm::LlmRequest;
use serde_json::Value;

use crate::agents::alignment::json_string_at;
use crate::configuration::header_string;
use crate::events::{AgentKind, ToolEvent};

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

// Claude Code can issue a tiny pre-user startup probe through the Anthropic gateway before the
// first UserPromptSubmit hook. Treating it as normal LLM work pollutes traces with an unparented
// `user: test` span, so alignment classifies only this exact-body harness probe for suppression.
pub(crate) fn is_startup_probe(
    provider: &str,
    model_name: Option<&str>,
    request: &LlmRequest,
) -> bool {
    if provider != "anthropic.messages" {
        return false;
    }
    let model = model_name
        .or_else(|| request.content.get("model").and_then(Value::as_str))
        .unwrap_or_default();
    if !model.starts_with("claude-") {
        return false;
    }
    if request.content.get("max_tokens").and_then(Value::as_u64) != Some(1) {
        return false;
    }
    let Some(messages) = request.content.get("messages").and_then(Value::as_array) else {
        return false;
    };
    let [message] = messages.as_slice() else {
        return false;
    };
    message.get("role").and_then(Value::as_str) == Some("user")
        && message.get("content").and_then(Value::as_str) == Some("test")
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
