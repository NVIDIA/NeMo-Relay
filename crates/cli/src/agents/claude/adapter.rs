// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::agents::shared::adapters::{
    AdapterOutcome, CLAUDE_CODE_PAYLOAD_EXTRACTOR, ClassificationRules, classify,
};
use crate::events::{AgentKind, NormalizedEvent};

/// Normalizes Claude Code hook payloads and returns the hook response Claude expects.
///
/// Pre-tool events evaluate guardrails before the host runs the tool.
/// Note: Claude's hook output schema rejects `null` for optional string fields like `stopReason`;
/// omit them entirely instead.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let events = classify(
        &payload,
        headers,
        &CLAUDE_CODE_PAYLOAD_EXTRACTOR,
        &ClassificationRules {
            kind: AgentKind::ClaudeCode,
            agent_start: &["SessionStart", "sessionStart", "session_start"],
            agent_end: &["SessionEnd", "sessionEnd", "session_end"],
            subagent_start: &["SubagentStart", "subagentStart"],
            subagent_end: &["SubagentStop", "subagentStop", "SubagentEnd"],
            tool_start: &["PreToolUse", "preToolUse"],
            tool_end: &[
                "PostToolUse",
                "postToolUse",
                "PostToolUseFailure",
                "postToolUseFailure",
                "ToolUseFailed",
                "toolUseFailed",
                "PermissionDenied",
                "permissionDenied",
            ],
            // Claude Code reports only the close of a turn (`Stop`); its turns stay
            // lazily opened, and `PreCompact`/`PostCompact` are matched by the
            // shared fallback rather than by an adapter-specific rule.
            turn_start: &[],
            turn_end: &["Stop", "stop"],
            compaction: &[],
        },
    );
    let events = events
        .into_iter()
        .map(|event| match event {
            NormalizedEvent::HookMark(mut event)
                if event.event_name.eq_ignore_ascii_case("StopFailure") =>
            {
                event.payload["status"] = json!("error");
                event.metadata["otel.status_code"] = json!("ERROR");
                if let Some(error) = event.payload.get("error").and_then(Value::as_str) {
                    event.metadata["error.type"] = json!(error);
                    event.metadata["otel.status_description"] = json!(error);
                }
                NormalizedEvent::TurnEnded(event)
            }
            event => event,
        })
        .collect();
    let response = json!({ "continue": true });
    AdapterOutcome { events, response }
}
