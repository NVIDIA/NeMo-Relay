// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::agents::shared::adapters::{
    AdapterOutcome, CLAUDE_CODE_PAYLOAD_EXTRACTOR, ClassificationRules, classify,
    permission_request,
};
use crate::events::AgentKind;

/// Normalizes Claude Code hook payloads and returns the hook response Claude expects.
///
/// Claude Code uses permission-bearing tool hooks. Pre-tool events acknowledge guardrail success
/// without granting host permission; the later `PermissionRequest` receives the final decision.
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
    let response = json!({ "continue": true });
    AdapterOutcome {
        events,
        response,
        permission: permission_request(
            &payload,
            headers,
            AgentKind::ClaudeCode,
            &CLAUDE_CODE_PAYLOAD_EXTRACTOR,
        ),
    }
}
