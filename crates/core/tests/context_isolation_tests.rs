// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context isolation tests for per-request scope stack isolation.

use std::sync::Arc;

use nemo_flow_core::context::{
    TASK_SCOPE_STACK, create_scope_stack, current_scope_stack, propagate_scope_to_thread,
    scope_stack_active, set_thread_scope_stack,
};
use nemo_flow_core::types::*;
use nemo_flow_core::{task_scope_push, task_scope_top};

/// Two ScopeStackHandles push different scopes → verify independent.
#[test]
fn test_two_scope_stacks_are_independent() {
    let stack_a = create_scope_stack();
    let stack_b = create_scope_stack();

    // Push a scope on stack_a
    {
        let mut guard = stack_a.write().unwrap();
        let handle = ScopeHandle::new(
            "scope_a".into(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        guard.push(handle);
    }

    // Push a different scope on stack_b
    {
        let mut guard = stack_b.write().unwrap();
        let handle = ScopeHandle::new(
            "scope_b".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        guard.push(handle);
    }

    // Verify independence
    let top_a = stack_a.read().unwrap().top().clone();
    let top_b = stack_b.read().unwrap().top().clone();
    assert_eq!(top_a.name, "scope_a");
    assert_eq!(top_b.name, "scope_b");

    // Root scopes have different UUIDs
    let root_a_uuid = stack_a.read().unwrap().top().uuid; // after removing scope_a, would be root
    let root_b_uuid = stack_b.read().unwrap().top().uuid;
    // They each have their own root
    assert_ne!(root_a_uuid, root_b_uuid); // scope_a != scope_b
}

/// Two tokio tasks with TASK_SCOPE_STACK.scope() → verify isolated.
#[tokio::test]
async fn test_tokio_tasks_isolated() {
    let stack_a = create_scope_stack();
    let stack_b = create_scope_stack();

    let stack_a_clone = stack_a.clone();
    let stack_b_clone = stack_b.clone();

    let handle_a = tokio::spawn(async move {
        TASK_SCOPE_STACK
            .scope(stack_a_clone, async {
                let h = ScopeHandle::new(
                    "task_a_scope".into(),
                    ScopeType::Agent,
                    ScopeAttributes::empty(),
                    None,
                    None,
                    None,
                );
                task_scope_push(h);
                // Yield to let other task run
                tokio::task::yield_now().await;
                let top = task_scope_top();
                assert_eq!(top.name, "task_a_scope");
                top.name.clone()
            })
            .await
    });

    let handle_b = tokio::spawn(async move {
        TASK_SCOPE_STACK
            .scope(stack_b_clone, async {
                let h = ScopeHandle::new(
                    "task_b_scope".into(),
                    ScopeType::Function,
                    ScopeAttributes::empty(),
                    None,
                    None,
                    None,
                );
                task_scope_push(h);
                tokio::task::yield_now().await;
                let top = task_scope_top();
                assert_eq!(top.name, "task_b_scope");
                top.name.clone()
            })
            .await
    });

    let (result_a, result_b) = tokio::join!(handle_a, handle_b);
    assert_eq!(result_a.unwrap(), "task_a_scope");
    assert_eq!(result_b.unwrap(), "task_b_scope");
}

/// Thread-local fallback creates independent stacks per thread.
#[test]
fn test_thread_local_independent_stacks() {
    use std::sync::{Arc, Barrier};

    let barrier = Arc::new(Barrier::new(2));

    let b1 = barrier.clone();
    let t1 = std::thread::spawn(move || {
        let h = ScopeHandle::new(
            "thread1_scope".into(),
            ScopeType::Agent,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        task_scope_push(h);
        b1.wait(); // sync with thread 2
        let top = task_scope_top();
        assert_eq!(top.name, "thread1_scope");
        top.name.clone()
    });

    let b2 = barrier.clone();
    let t2 = std::thread::spawn(move || {
        let h = ScopeHandle::new(
            "thread2_scope".into(),
            ScopeType::Function,
            ScopeAttributes::empty(),
            None,
            None,
            None,
        );
        task_scope_push(h);
        b2.wait(); // sync with thread 1
        let top = task_scope_top();
        assert_eq!(top.name, "thread2_scope");
        top.name.clone()
    });

    assert_eq!(t1.join().unwrap(), "thread1_scope");
    assert_eq!(t2.join().unwrap(), "thread2_scope");
}

/// set_thread_scope_stack binds a specific stack to the current thread.
#[test]
fn test_set_thread_scope_stack() {
    // This test runs on its own thread to avoid polluting other tests
    let result = std::thread::spawn(|| {
        let custom_stack = create_scope_stack();
        {
            let mut guard = custom_stack.write().unwrap();
            let h = ScopeHandle::new(
                "custom".into(),
                ScopeType::Agent,
                ScopeAttributes::empty(),
                None,
                None,
                None,
            );
            guard.push(h);
        }

        // Before binding, thread has its default stack with just root
        assert_eq!(task_scope_top().name, "root");

        // Bind the custom stack
        set_thread_scope_stack(custom_stack);

        // Now task_scope_top should see "custom"
        assert_eq!(task_scope_top().name, "custom");
    })
    .join();

    result.unwrap();
}

/// scope_stack_active returns false on a fresh thread (auto-created default).
#[test]
fn test_scope_stack_active_false_by_default() {
    let result = std::thread::spawn(scope_stack_active).join();
    assert!(
        !result.unwrap(),
        "scope_stack_active should be false on a fresh thread"
    );
}

/// scope_stack_active returns true after set_thread_scope_stack is called.
#[test]
fn test_scope_stack_active_true_after_explicit_set() {
    let result = std::thread::spawn(|| {
        assert!(!scope_stack_active());
        let custom = create_scope_stack();
        set_thread_scope_stack(custom);
        scope_stack_active()
    })
    .join();
    assert!(
        result.unwrap(),
        "scope_stack_active should be true after set_thread_scope_stack"
    );
}

/// scope_stack_active returns true inside a TASK_SCOPE_STACK.scope() block.
#[tokio::test]
async fn test_scope_stack_active_in_task_local() {
    let stack = create_scope_stack();
    let active = TASK_SCOPE_STACK
        .scope(stack, async { scope_stack_active() })
        .await;
    assert!(
        active,
        "scope_stack_active should be true inside task-local scope"
    );
}

/// propagate_scope_to_thread fails when no scope is active.
#[test]
fn test_propagate_scope_to_thread_fails_when_inactive() {
    let result = std::thread::spawn(propagate_scope_to_thread).join();
    assert!(
        result.unwrap().is_err(),
        "propagate_scope_to_thread should fail on a fresh thread"
    );
}

/// propagate_scope_to_thread returns the current scope stack handle.
#[test]
fn test_propagate_scope_to_thread_returns_correct_stack() {
    let result = std::thread::spawn(|| {
        let custom = create_scope_stack();
        // Push a scope so we can identify the stack
        {
            let mut guard = custom.write().unwrap();
            let h = ScopeHandle::new(
                "propagated".into(),
                ScopeType::Agent,
                ScopeAttributes::empty(),
                None,
                None,
                None,
            );
            guard.push(h);
        }
        set_thread_scope_stack(custom);
        let propagated = propagate_scope_to_thread().expect("should succeed");
        let top = propagated.read().unwrap().top().clone();
        top.name.clone()
    })
    .join();
    assert_eq!(result.unwrap(), "propagated");
}

/// propagate_scope_to_thread handle can be used on another thread via set_thread_scope_stack.
#[test]
fn test_propagate_scope_to_thread_cross_thread() {
    // Parent thread: create and set a scope stack, return the propagated handle
    let parent_handle = std::thread::spawn(|| {
        let custom = create_scope_stack();
        {
            let mut guard = custom.write().unwrap();
            let h = ScopeHandle::new(
                "parent_scope".into(),
                ScopeType::Agent,
                ScopeAttributes::empty(),
                None,
                None,
                None,
            );
            guard.push(h);
        }
        set_thread_scope_stack(custom);
        propagate_scope_to_thread().expect("should succeed")
    })
    .join()
    .unwrap();

    // Child thread: receive and bind the propagated handle
    let child_result = std::thread::spawn(move || {
        assert!(!scope_stack_active());
        set_thread_scope_stack(parent_handle);
        assert!(scope_stack_active());
        task_scope_top().name.clone()
    })
    .join();
    assert_eq!(child_result.unwrap(), "parent_scope");
}

/// current_scope_stack returns different handles for different tokio tasks.
#[tokio::test]
async fn test_current_scope_stack_differs_across_tasks() {
    let stack_a = create_scope_stack();
    let stack_b = create_scope_stack();

    let sa = stack_a.clone();
    let sb = stack_b.clone();

    let ptr_a = tokio::spawn(async move {
        TASK_SCOPE_STACK
            .scope(sa, async {
                let s = current_scope_stack();
                Arc::as_ptr(&s) as usize
            })
            .await
    });

    let ptr_b = tokio::spawn(async move {
        TASK_SCOPE_STACK
            .scope(sb, async {
                let s = current_scope_stack();
                Arc::as_ptr(&s) as usize
            })
            .await
    });

    let (a, b) = tokio::join!(ptr_a, ptr_b);
    // Different Arc pointers = different stacks
    assert_ne!(a.unwrap(), b.unwrap());
}
