// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::adapters::{AdapterOutcome, ClassificationRules, classify};
use crate::model::AgentKind;

/// Normalizes Hermes shell hook payloads without emitting control directives.
///
/// Hermes hooks are installed as shell commands and may run outside `run`, so this adapter keeps
/// responses minimal and relies on the forwarder fail-open/fail-closed setting to decide whether
/// hook delivery problems affect the invoking agent.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let event = classify(
        &payload,
        headers,
        &ClassificationRules {
            kind: AgentKind::Hermes,
            agent_start: &["on_session_start", "sessionStart"],
            agent_end: &["on_session_finalize", "on_session_reset"],
            subagent_start: &["subagent_start", "subagentStart"],
            subagent_end: &["subagent_stop", "subagentStop"],
            tool_start: &["pre_tool_call", "preToolCall"],
            tool_end: &["post_tool_call", "postToolCall"],
        },
    );
    AdapterOutcome {
        events: vec![event],
        response: json!({}),
    }
}
