// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::agents::shared::adapters::{
    AdapterOutcome, CODEX_PAYLOAD_EXTRACTOR, ClassificationRules, classify,
};
use crate::events::{AgentKind, NormalizedEvent};

/// Normalizes Codex hook payloads and supplies the successful hook response.
///
/// This adapter supplies an empty response body. The gateway evaluates tool policy at `PreToolUse`
/// and rejects the hook if a conditional guardrail blocks it, before returning this response.
/// Event spelling is accepted in both camelCase and snake_case forms so installed hooks and
/// inline `run` hook configuration share one path.
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
    // An interrupted turn sends `Interrupt` instead of `Stop`; close it as a failed turn.
    let events = events
        .into_iter()
        .map(|event| match event {
            NormalizedEvent::HookMark(mut event)
                if event.event_name.eq_ignore_ascii_case("Interrupt") =>
            {
                event.payload["status"] = json!("error");
                event.metadata["otel.status_code"] = json!("ERROR");
                event.metadata["error.type"] = json!("interrupted");
                event.metadata["otel.status_description"] = json!("interrupted");
                NormalizedEvent::TurnEnded(event)
            }
            event => event,
        })
        .collect();
    AdapterOutcome {
        events,
        response: json!({}),
    }
}
