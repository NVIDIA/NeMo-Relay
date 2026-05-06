// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::adapters::{AdapterOutcome, ClassificationRules, classify};
use crate::model::AgentKind;

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
