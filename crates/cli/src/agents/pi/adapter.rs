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
///   agent run, which makes it the run boundary rather than `agent_end` -- but
///   deliberately *not* a turn boundary: a logical run spans several turns.
/// - pi reports both ends of a turn, so the turn scope opens at `turn_start`
///   instead of being opened implicitly by whichever event arrives first.
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
            // pi has an explicit turn boundary at both ends, unlike Codex and
            // Claude Code which only signal it through `Stop`. Classifying the
            // open as well as the close is what stops the gateway inventing a
            // turn at whichever event happens to arrive first.
            turn_start: &["turn_start", "turnStart"],
            // `agent_settled` is deliberately not here: it marks the end of a
            // logical agent run, which can span several turns, and closing the
            // turn there would merge every re-entry attempt into one.
            turn_end: &["turn_end", "turnEnd"],
            // Only the *completed* compaction. `session_before_compact` stays a
            // mark: it announces an intent that any later-loading extension can
            // still cancel, and the runtime treats a compaction event as proof
            // the context was actually rebuilt.
            compaction: &["session_compact", "sessionCompact"],
        },
    );
    AdapterOutcome {
        events,
        response: json!({}),
    }
}
