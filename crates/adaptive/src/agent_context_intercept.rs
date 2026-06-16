// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in request intercept for copying scope-local agent context into LLM requests.

use std::sync::Arc;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::LlmRequestInterceptFn;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use serde_json::Value as Json;

use crate::config::AgentContextComponentConfig;
use crate::context_helpers::resolve_agent_context;

/// Opt-in LLM request intercept that injects canonical agent context into the request body.
pub struct AgentContextIntercept {
    inject_body_path: String,
}

impl AgentContextIntercept {
    /// Creates a new agent-context request intercept from component config.
    pub fn new(config: AgentContextComponentConfig) -> Self {
        Self {
            inject_body_path: config.inject_body_path,
        }
    }

    /// Converts this intercept into an [`LlmRequestInterceptFn`] suitable for registration.
    pub fn into_request_fn(self) -> LlmRequestInterceptFn {
        let inject_body_path = self.inject_body_path;
        Arc::new(
            move |_name: &str, mut request: LlmRequest, annotated: Option<AnnotatedLlmRequest>| {
                if let Some(agent_context) = resolve_agent_context() {
                    insert_json_path_if_absent(
                        &mut request.content,
                        &inject_body_path,
                        &agent_context,
                    );
                }
                Ok((request, annotated))
            },
        )
    }
}

fn insert_json_path_if_absent(root: &mut Json, path: &str, value: &Json) {
    let parts = path
        .split('.')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    insert_json_parts_if_absent(root, &parts, value);
}

fn insert_json_parts_if_absent(root: &mut Json, parts: &[&str], value: &Json) {
    let Some((head, tail)) = parts.split_first() else {
        return;
    };
    let Some(object) = root.as_object_mut() else {
        return;
    };
    if tail.is_empty() {
        object
            .entry((*head).to_string())
            .or_insert_with(|| value.clone());
        return;
    }
    let child = object
        .entry((*head).to_string())
        .or_insert_with(|| Json::Object(serde_json::Map::new()));
    insert_json_parts_if_absent(child, tail, value);
}

#[cfg(test)]
#[path = "../tests/unit/agent_context_intercept_tests.rs"]
mod tests;
