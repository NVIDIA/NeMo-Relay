// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for agent context request injection.

use super::*;

use nemo_relay::api::runtime::set_thread_scope_stack;
use nemo_relay::api::runtime::{LlmRequestInterceptFn, create_scope_stack};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use serde_json::json;

fn reset_scope_stack() {
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn agent_context_intercept_injects_nearest_scope_context() {
    reset_scope_stack();
    let parent_context = json!({
        "session_type_id": "codex",
        "session_id": "session-1",
        "trajectory_id": "session-1:turn:1"
    });
    let child_context = json!({
        "session_type_id": "codex",
        "session_id": "session-1",
        "trajectory_id": "worker-1",
        "parent_trajectory_id": "session-1:turn:1"
    });
    let parent = push_scope(
        PushScopeParams::builder()
            .name("turn")
            .scope_type(ScopeType::Agent)
            .metadata(json!({ "nemo_relay": { "agent_context": parent_context } }))
            .build(),
    )
    .unwrap();
    let child = push_scope(
        PushScopeParams::builder()
            .name("subagent:worker-1")
            .scope_type(ScopeType::Agent)
            .parent(&parent)
            .metadata(json!({ "nemo_relay": { "agent_context": child_context } }))
            .build(),
    )
    .unwrap();

    let request_fn =
        AgentContextIntercept::new(AgentContextComponentConfig::default()).into_request_fn();
    let (request, annotated) = request_fn(
        "openai.responses",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({ "model": "test", "nvext": { "agent_hints": { "priority": 1 } } }),
        },
        None,
    )
    .unwrap();

    assert_eq!(
        request.content["nvext"]["agent_context"],
        json!({
            "session_type_id": "codex",
            "session_id": "session-1",
            "trajectory_id": "worker-1",
            "parent_trajectory_id": "session-1:turn:1"
        })
    );
    assert_eq!(
        request.content["nvext"]["agent_hints"]["priority"],
        json!(1)
    );
    assert!(annotated.is_none());

    pop_scope(PopScopeParams::builder().handle_uuid(&child.uuid).build()).unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&parent.uuid).build()).unwrap();
    reset_scope_stack();
}

#[test]
fn agent_context_intercept_preserves_existing_request_context() {
    reset_scope_stack();
    let scope = push_scope(
        PushScopeParams::builder()
            .name("turn")
            .scope_type(ScopeType::Agent)
            .metadata(json!({
                "nemo_relay": {
                    "agent_context": {
                        "session_type_id": "codex",
                        "session_id": "session-1",
                        "trajectory_id": "session-1:turn:1"
                    }
                }
            }))
            .build(),
    )
    .unwrap();

    let existing = json!({
        "session_type_id": "client",
        "session_id": "client-session",
        "trajectory_id": "client-trajectory"
    });
    let request_fn =
        AgentContextIntercept::new(AgentContextComponentConfig::default()).into_request_fn();
    let (request, _) = request_fn(
        "openai.responses",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({ "nvext": { "agent_context": existing } }),
        },
        None,
    )
    .unwrap();

    assert_eq!(
        request.content["nvext"]["agent_context"],
        json!({
            "session_type_id": "client",
            "session_id": "client-session",
            "trajectory_id": "client-trajectory"
        })
    );

    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();
    reset_scope_stack();
}

#[test]
fn agent_context_intercept_noops_without_scope_context_or_object_body() {
    reset_scope_stack();
    let request_fn =
        AgentContextIntercept::new(AgentContextComponentConfig::default()).into_request_fn();

    let (request, _) = request_fn(
        "openai.responses",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({ "model": "test" }),
        },
        None,
    )
    .unwrap();
    assert_eq!(request.content, json!({ "model": "test" }));

    let scope = push_scope(
        PushScopeParams::builder()
            .name("turn")
            .scope_type(ScopeType::Agent)
            .metadata(json!({
                "nemo_relay": {
                    "agent_context": {
                        "session_type_id": "codex",
                        "session_id": "session-1",
                        "trajectory_id": "session-1:turn:1"
                    }
                }
            }))
            .build(),
    )
    .unwrap();
    let (scalar_request, _) = request_fn(
        "openai.responses",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!("scalar"),
        },
        None,
    )
    .unwrap();
    assert_eq!(scalar_request.content, json!("scalar"));

    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();
    reset_scope_stack();
}

#[test]
fn agent_context_intercept_request_fn_type_compiles() {
    let _request_fn: LlmRequestInterceptFn =
        AgentContextIntercept::new(AgentContextComponentConfig::default()).into_request_fn();
}
