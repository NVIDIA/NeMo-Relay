// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for context helpers in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::api::runtime::{create_scope_stack, set_thread_scope_stack};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};

#[test]
fn test_latency_sensitivity_pointer_is_valid_json_pointer() {
    // JSON pointer must start with /
    assert!(LATENCY_SENSITIVITY_POINTER.starts_with('/'));
}

#[test]
fn test_set_latency_sensitivity_basic() {
    // Sets value on the thread-local scope stack's root scope
    set_latency_sensitivity(3).unwrap();
    assert_eq!(read_manual_latency_sensitivity(), Some(3));

    // Clean up: reset root scope metadata
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_set_latency_sensitivity_max_merge_higher_wins() {
    set_latency_sensitivity(3).unwrap();
    set_latency_sensitivity(5).unwrap();
    assert_eq!(read_manual_latency_sensitivity(), Some(5));

    // Clean up
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_set_latency_sensitivity_max_merge_lower_noop() {
    set_latency_sensitivity(5).unwrap();
    set_latency_sensitivity(3).unwrap();
    // Lower value should not override
    assert_eq!(read_manual_latency_sensitivity(), Some(5));

    // Clean up
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_set_latency_sensitivity_read_roundtrip() {
    // Ensure read_manual_latency_sensitivity reads what set_latency_sensitivity writes
    set_latency_sensitivity(7).unwrap();
    assert_eq!(read_manual_latency_sensitivity(), Some(7));

    // Clean up
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_helpers_return_defaults_when_scope_stack_lock_is_poisoned() {
    let poisoned = create_scope_stack();
    let poisoned_for_panic = poisoned.clone();
    let _ = std::panic::catch_unwind(move || {
        let _guard = poisoned_for_panic.write().unwrap();
        panic!("poison scope stack");
    });

    set_thread_scope_stack(poisoned);
    assert!(extract_scope_path().is_empty());
    assert_eq!(read_manual_latency_sensitivity(), None);
    assert_eq!(resolve_agent_id(), None);
    assert_eq!(resolve_agent_context(), None);

    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_resolve_agent_context_uses_nearest_scope_metadata() {
    set_thread_scope_stack(create_scope_stack());
    let parent = push_scope(
        PushScopeParams::builder()
            .name("parent")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({
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
    let child = push_scope(
        PushScopeParams::builder()
            .name("child")
            .scope_type(ScopeType::Agent)
            .parent(&parent)
            .metadata(serde_json::json!({
                "nemo_relay": {
                    "agent_context": {
                        "session_type_id": "codex",
                        "session_id": "session-1",
                        "trajectory_id": "worker-1",
                        "parent_trajectory_id": "session-1:turn:1"
                    }
                }
            }))
            .build(),
    )
    .unwrap();

    assert_eq!(
        resolve_agent_context(),
        Some(serde_json::json!({
            "session_type_id": "codex",
            "session_id": "session-1",
            "trajectory_id": "worker-1",
            "parent_trajectory_id": "session-1:turn:1"
        }))
    );

    pop_scope(PopScopeParams::builder().handle_uuid(&child.uuid).build()).unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&parent.uuid).build()).unwrap();
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_set_latency_sensitivity_ignores_non_object_metadata() {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = Some(serde_json::json!("metadata"));
    drop(stack);

    set_latency_sensitivity(9).unwrap();

    let mut stack = stack_handle.write().unwrap();
    assert_eq!(
        stack.top_mut().metadata,
        Some(serde_json::json!("metadata"))
    );
    stack.top_mut().metadata = None;
}
