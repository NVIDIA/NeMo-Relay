// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::agents::shared::adapters::{
    AdapterOutcome, CODEX_PAYLOAD_EXTRACTOR, ClassificationRules, classify, permission_request,
};
use crate::events::AgentKind;

/// Normalizes Codex hook payloads while leaving Codex hook control flow untouched.
///
/// Codex receives an empty response body from this adapter because the gateway currently records
/// hooks instead of making allow/deny decisions. Event spelling is accepted in both camelCase and
/// snake_case forms so installed hooks and inline `run` hook configuration share one path.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let events = classify(
        &payload,
        headers,
        &CODEX_PAYLOAD_EXTRACTOR,
        &ClassificationRules {
            kind: AgentKind::Codex,
            agent_start: &["sessionStart", "session_start", "agentStarted"],
            agent_end: &["sessionEnd", "session_end", "agentEnded"],
            subagent_start: &["subagentStart", "subagent_start"],
            subagent_end: &["subagentStop", "subagentEnd", "subagent_stop"],
            tool_start: &["preToolUse", "toolStarted", "tool_start"],
            tool_end: &["postToolUse", "toolEnded", "tool_end", "toolFailed"],
            // Codex reports only the close of a turn (`Stop`); its turns stay
            // lazily opened, and `PreCompact`/`PostCompact` are matched by the
            // shared fallback rather than by an adapter-specific rule.
            turn_start: &[],
            turn_end: &["Stop", "stop"],
            compaction: &[],
        },
    );
    AdapterOutcome {
        events,
        response: json!({}),
        permission: permission_request(
            &payload,
            headers,
            AgentKind::Codex,
            &CODEX_PAYLOAD_EXTRACTOR,
        ),
    }
}
