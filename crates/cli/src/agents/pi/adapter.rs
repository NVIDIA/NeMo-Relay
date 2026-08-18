// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::agents::shared::adapters::{
    AdapterOutcome, ClassificationRules, PI_PAYLOAD_EXTRACTOR, classify,
};
use crate::events::AgentKind;

/// Normalizes pi extension hook payloads and returns the response the extension expects.
///
/// The response body is deliberately minimal. A tool call is *allowed* by this
/// endpoint returning 2xx; it is *blocked* by `apply_events` failing the
/// conditional-execution guardrail chain, which surfaces as HTTP 403 with
/// `error.type = "nemo_relay_guardrail_rejected"` and the guardrail's own words
/// in `error.reason`. The extension turns that into pi's `{block, reason}`,
/// which pi passes verbatim to the model as an error tool result.
///
/// Mapping notes:
/// - pi's `session_start`/`session_shutdown` are the session boundary, not
///   `agent_start`/`agent_end`. One pi session can re-enter the agent run many
///   times (provider retry, compaction, queued follow-up), so treating
///   `agent_start` as a session start would open a session per retry.
/// - `agent_settled` is the only pi event that fires exactly once per logical
///   agent run, so it is the turn-boundary snapshot rather than `agent_end`.
/// - `tool_call` is the gating hook and maps to tool start. `tool_execution_start`
///   is deliberately NOT mapped: it fires before validation and before
///   `tool_call`, including for calls that never execute, so using it to open a
///   tool span would create spans for calls pi then discards.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let events = classify(
        &payload,
        headers,
        &PI_PAYLOAD_EXTRACTOR,
        &ClassificationRules {
            kind: AgentKind::Pi,
            agent_start: &["session_start", "sessionStart"],
            agent_end: &["session_shutdown", "sessionShutdown"],
            // pi ships no MCP client and has no nested-agent hook of its own;
            // subagents are an extension-level concept it does not surface.
            subagent_start: &[],
            subagent_end: &[],
            tool_start: &["tool_call", "toolCall"],
            tool_end: &["tool_execution_end", "toolExecutionEnd"],
        },
    );
    AdapterOutcome {
        events,
        response: json!({}),
    }
}
