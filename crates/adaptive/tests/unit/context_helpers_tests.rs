// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for context helpers in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::api::runtime::{create_scope_stack, set_thread_scope_stack};

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
