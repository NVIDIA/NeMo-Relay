// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
