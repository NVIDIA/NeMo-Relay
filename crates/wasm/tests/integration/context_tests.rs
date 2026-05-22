// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for context in the NeMo Relay WebAssembly crate.

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use nemo_relay_wasm::api::*;
use nemo_relay_wasm::types::*;

fn parent_handle(handle: Option<ScopeHandle>) -> JsValue {
    handle.map(JsValue::from).unwrap_or(JsValue::NULL)
}

fn push_scope(
    name: &str,
    scope_type: ScopeType,
    handle: Option<ScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
) -> Result<ScopeHandle, JsValue> {
    nemo_relay_wasm::api::push_scope(
        name,
        scope_type,
        parent_handle(handle),
        attributes,
        data,
        metadata,
        JsValue::NULL,
        None,
    )
}

fn pop_scope(handle: &ScopeHandle) -> Result<(), JsValue> {
    nemo_relay_wasm::api::pop_scope(handle, JsValue::NULL, None)
}

// ===========================================================================
// Context isolation
// ===========================================================================

#[wasm_bindgen_test]
fn test_create_scope_stack_returns_wasm_scope_stack() {
    let stack = create_scope_stack();
    // Just verify it's a valid object we can use (no panic)
    let _ = &stack;
}

#[wasm_bindgen_test]
fn test_current_scope_stack_returns_stack() {
    let s1 = current_scope_stack();
    let s2 = current_scope_stack();
    // Both should succeed without panic
    let _ = (&s1, &s2);
}

#[wasm_bindgen_test]
fn test_set_thread_scope_stack_isolates_scopes() {
    let original = current_scope_stack();
    let new_stack = create_scope_stack();

    // Switch to new stack and push a scope on it
    set_thread_scope_stack(&new_stack);
    let scope = push_scope(
        "isolated_scope",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let handle = get_handle().unwrap();
    assert_eq!(handle.name(), "isolated_scope");
    pop_scope(&scope).unwrap();

    // Restore original stack — the isolated scope should not be visible
    set_thread_scope_stack(&original);
    let restored = get_handle().unwrap();
    assert_ne!(restored.name(), "isolated_scope");
}

#[wasm_bindgen_test]
fn test_scope_stack_active_true_after_set() {
    let stack = create_scope_stack();
    set_thread_scope_stack(&stack);
    assert!(scope_stack_active());
}

#[wasm_bindgen_test]
fn test_two_scope_stacks_are_independent() {
    let original = current_scope_stack();
    let stack1 = create_scope_stack();
    let stack2 = create_scope_stack();

    // Push a scope on stack1
    set_thread_scope_stack(&stack1);
    let s1 = push_scope(
        "stack1_scope",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();

    // Switch to stack2 and push a different scope
    set_thread_scope_stack(&stack2);
    let s2 = push_scope(
        "stack2_scope",
        ScopeType::Tool,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();

    // Verify stack2 sees its own scope
    let handle2 = get_handle().unwrap();
    assert_eq!(handle2.name(), "stack2_scope");

    // Switch back to stack1 — should see stack1's scope
    set_thread_scope_stack(&stack1);
    let handle1 = get_handle().unwrap();
    assert_eq!(handle1.name(), "stack1_scope");

    // Clean up
    pop_scope(&s1).unwrap();
    set_thread_scope_stack(&stack2);
    pop_scope(&s2).unwrap();

    // Restore original
    set_thread_scope_stack(&original);
}
