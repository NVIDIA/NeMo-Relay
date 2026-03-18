// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use wasm_bindgen_test::*;

use nvidia_nat_nexus_wasm::api::*;
use nvidia_nat_nexus_wasm::types::*;

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
    let scope = nat_nexus_push_scope("isolated_scope", SCOPE_TYPE_AGENT, None, None).unwrap();
    let handle = nat_nexus_get_handle().unwrap();
    assert_eq!(handle.name(), "isolated_scope");
    nat_nexus_pop_scope(&scope).unwrap();

    // Restore original stack — the isolated scope should not be visible
    set_thread_scope_stack(&original);
    let restored = nat_nexus_get_handle().unwrap();
    assert_ne!(restored.name(), "isolated_scope");
}

#[wasm_bindgen_test]
fn test_two_scope_stacks_are_independent() {
    let original = current_scope_stack();
    let stack1 = create_scope_stack();
    let stack2 = create_scope_stack();

    // Push a scope on stack1
    set_thread_scope_stack(&stack1);
    let s1 = nat_nexus_push_scope("stack1_scope", SCOPE_TYPE_AGENT, None, None).unwrap();

    // Switch to stack2 and push a different scope
    set_thread_scope_stack(&stack2);
    let s2 = nat_nexus_push_scope("stack2_scope", SCOPE_TYPE_TOOL, None, None).unwrap();

    // Verify stack2 sees its own scope
    let handle2 = nat_nexus_get_handle().unwrap();
    assert_eq!(handle2.name(), "stack2_scope");

    // Switch back to stack1 — should see stack1's scope
    set_thread_scope_stack(&stack1);
    let handle1 = nat_nexus_get_handle().unwrap();
    assert_eq!(handle1.name(), "stack1_scope");

    // Clean up
    nat_nexus_pop_scope(&s1).unwrap();
    set_thread_scope_stack(&stack2);
    nat_nexus_pop_scope(&s2).unwrap();

    // Restore original
    set_thread_scope_stack(&original);
}
