// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::adapters::{AdapterOutcome, CURSOR_PAYLOAD_EXTRACTOR, ClassificationRules, classify};
use crate::model::{AgentKind, NormalizedEvent};

/// Normalizes Cursor hook payloads and returns Cursor-compatible continuation decisions.
///
/// Cursor has separate shell and MCP hook names, both of which are collapsed into normal tool
/// start/end events. Tool starts are fail-open with an explicit `allow` permission response so
/// the gateway records activity without becoming a policy engine for Cursor executions.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let events = classify(
        &payload,
        headers,
        &CURSOR_PAYLOAD_EXTRACTOR,
        &ClassificationRules {
            kind: AgentKind::Cursor,
            agent_start: &["sessionStart", "session_start"],
            agent_end: &["sessionEnd", "session_end", "stop"],
            subagent_start: &["subagentStart", "subagent_start"],
            subagent_end: &["subagentStop", "subagentEnd", "subagent_stop"],
            tool_start: &["preToolUse", "beforeShellExecution", "beforeMCPExecution"],
            tool_end: &[
                "postToolUse",
                "afterShellExecution",
                "afterMCPExecution",
                "postToolUseFailure",
            ],
        },
    );
    // Response shape is determined by the primary event (first in the vec).
    let response = match events.first() {
        Some(NormalizedEvent::ToolStarted(_)) => json!({
            "continue": true,
            "permission": "allow"
        }),
        Some(NormalizedEvent::AgentEnded(_)) => json!({ "continue": true }),
        _ => json!({ "continue": true }),
    };
    AdapterOutcome { events, response }
}
