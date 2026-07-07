// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::{Value, json};

use crate::adapters::{AdapterOutcome, ClassificationRules, OPENCLAW_PAYLOAD_EXTRACTOR, classify};
use crate::model::AgentKind;

/// Normalizes events emitted by the temporary OpenClaw CLI bridge.
///
/// The bridge is observational: model request mutation remains on the gateway's
/// managed LLM path, while these events provide session and tool correlation.
pub(crate) fn adapt(payload: Value, headers: &HeaderMap) -> AdapterOutcome {
    let events = classify(
        &payload,
        headers,
        &OPENCLAW_PAYLOAD_EXTRACTOR,
        &ClassificationRules {
            kind: AgentKind::OpenClaw,
            agent_start: &["session_start"],
            agent_end: &["session_end"],
            subagent_start: &["subagent_start"],
            subagent_end: &["subagent_end"],
            tool_start: &["tool_start"],
            tool_end: &["tool_end"],
        },
    );
    AdapterOutcome {
        events,
        response: json!({}),
    }
}
