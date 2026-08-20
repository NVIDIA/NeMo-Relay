// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context isolation tests for per-request scope stack isolation.

use std::sync::Arc;

use nemo_relay::api::runtime::{
    PropagationContext, ScopeStack, TASK_SCOPE_STACK, capture_traceparent, create_scope_stack,
    create_scope_stack_from_propagation, current_scope_stack, fork_scope_stack,
    propagate_scope_to_thread, scope_stack_active, set_thread_scope_stack, sync_thread_scope_stack,
    task_scope_push, task_scope_remove, task_scope_top, with_scope_stack,
};
use nemo_relay::api::scope::{
    PopScopeParams, PushScopeParams, ScopeHandle, ScopeType, pop_scope, push_scope,
};
use nemo_relay::error::FlowError;
use uuid::Uuid;

/// Two ScopeStackHandles push different scopes → verify independent.
#[test]
fn test_two_scope_stacks_are_independent() {
    let stack_a = create_scope_stack();
    let stack_b = create_scope_stack();

    // Push a scope on stack_a
    {
        let mut guard = stack_a.write().unwrap();
        let handle = ScopeHandle::builder()
            .name("scope_a")
            .scope_type(ScopeType::Agent)
            .build();
        guard.push(handle);
    }

    // Push a different scope on stack_b
    {
        let mut guard = stack_b.write().unwrap();
        let handle = ScopeHandle::builder()
            .name("scope_b")
            .scope_type(ScopeType::Function)
            .build();
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

#[test]
fn test_propagation_context_seeds_a_synthetic_root_and_parent() {
    let root_uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();
    let stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid,
    })
    .unwrap();
    let stack = stack.read().unwrap();
    assert_eq!(stack.root_uuid(), root_uuid);
    assert_eq!(stack.top().uuid, parent_uuid);
    assert_eq!(stack.scopes().len(), 2);
}

#[test]
fn test_rootless_propagation_context_uses_the_parent_as_root() {
    let parent_uuid = Uuid::now_v7();
    let stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: None,
        parent_uuid,
    })
    .unwrap();
    let stack = stack.read().unwrap();
    assert_eq!(stack.root_uuid(), parent_uuid);
    assert_eq!(stack.top().uuid, parent_uuid);
    assert_eq!(stack.scopes().len(), 1);
}

#[test]
fn test_propagation_context_with_root_as_parent_uses_one_synthetic_root() {
    let root_uuid = Uuid::now_v7();
    let stack = create_scope_stack_from_propagation(&PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid: root_uuid,
    })
    .unwrap();
    let stack = stack.read().unwrap();
    assert_eq!(stack.root_uuid(), root_uuid);
    assert_eq!(stack.top().uuid, root_uuid);
    assert_eq!(stack.scopes().len(), 1);
    assert!(stack.is_propagated_parent(root_uuid));
}

#[test]
fn test_propagation_context_rejects_invalid_wire_values() {
    for context in [
        PropagationContext {
            version: PropagationContext::VERSION + 1,
            root_uuid: None,
            parent_uuid: Uuid::now_v7(),
        },
        PropagationContext {
            version: PropagationContext::VERSION,
            root_uuid: None,
            parent_uuid: Uuid::nil(),
        },
        PropagationContext {
            version: PropagationContext::VERSION,
            root_uuid: Some(Uuid::from_u128(1_u128 << 64)),
            parent_uuid: Uuid::now_v7(),
        },
    ] {
        assert!(create_scope_stack_from_propagation(&context).is_err());
    }
}

#[test]
fn test_propagation_context_json_round_trips_and_validates_input() {
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(Uuid::now_v7()),
        parent_uuid: Uuid::now_v7(),
    };

    let json = context.to_json().unwrap();
    assert_eq!(PropagationContext::from_json(&json).unwrap(), context);
    assert!(PropagationContext::from_json("not JSON").is_err());
    assert!(
        PropagationContext::from_json(
            r#"{"version":2,"parent_uuid":"018f13f0-7c1a-7a80-8000-000000000002"}"#,
        )
        .is_err()
    );
}

#[test]
fn test_rooted_propagation_context_formats_traceparent() {
    let root_uuid = Uuid::from_u128(0x00112233445566778899aabbccddeeff);
    let parent_uuid = Uuid::from_u128(0xffeeddccbbaa99887766554433221100);
    let context = PropagationContext {
        version: PropagationContext::VERSION,
        root_uuid: Some(root_uuid),
        parent_uuid,
    };
    assert_eq!(
        context.to_traceparent().unwrap(),
        "00-00112233445566778899aabbccddeeff-7766554433221100-01"
    );
    assert!(
        PropagationContext {
            root_uuid: None,
            ..context
        }
        .to_traceparent()
        .is_err()
    );
}

#[test]
fn test_capture_traceparent_requires_an_emitted_scope() {
    set_thread_scope_stack(create_scope_stack());
    assert!(capture_traceparent().is_err());
}

#[test]
fn test_pop_scope_rejects_non_top_and_unknown_handles() {
    set_thread_scope_stack(create_scope_stack());

    let outer = push_scope(
        PushScopeParams::builder()
            .name("outer")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let inner = push_scope(
        PushScopeParams::builder()
            .name("inner")
            .scope_type(ScopeType::Function)
            .build(),
    )
    .unwrap();

    let non_top = pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build());
    assert!(matches!(non_top, Err(FlowError::InvalidArgument(_))));

    let unknown = Uuid::now_v7();
    let missing = pop_scope(PopScopeParams::builder().handle_uuid(&unknown).build());
    assert!(matches!(missing, Err(FlowError::NotFound(_))));

    pop_scope(PopScopeParams::builder().handle_uuid(&inner.uuid).build()).unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build()).unwrap();
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
                let h = ScopeHandle::builder()
                    .name("task_a_scope")
                    .scope_type(ScopeType::Agent)
                    .build();
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
                let h = ScopeHandle::builder()
                    .name("task_b_scope")
                    .scope_type(ScopeType::Function)
                    .build();
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

#[tokio::test]
async fn test_fork_scope_stack_isolates_child_tasks_and_preserves_parentage() {
    let parent_stack = create_scope_stack();
    TASK_SCOPE_STACK
        .scope(parent_stack, async {
            let parent = ScopeHandle::builder()
                .name("fork-parent")
                .scope_type(ScopeType::Agent)
                .build();
            task_scope_push(parent.clone());

            let first_stack = fork_scope_stack().unwrap();
            let second_stack = fork_scope_stack().unwrap();
            assert!(!Arc::ptr_eq(&first_stack, &second_stack));
            assert_eq!(first_stack.read().unwrap().top().uuid, parent.uuid);
            assert_eq!(second_stack.read().unwrap().top().uuid, parent.uuid);
            let parent_uuid = parent.uuid;
            let (second_pushed_tx, second_pushed_rx) = tokio::sync::oneshot::channel();
            let (first_popped_tx, first_popped_rx) = tokio::sync::oneshot::channel();

            let first = tokio::spawn(TASK_SCOPE_STACK.scope(first_stack, async move {
                let child = ScopeHandle::builder()
                    .name("first-child")
                    .scope_type(ScopeType::Function)
                    .parent_uuid(parent_uuid)
                    .build();
                task_scope_push(child.clone());
                second_pushed_rx.await.unwrap();
                task_scope_remove(&child.uuid).unwrap();
                first_popped_tx.send(()).unwrap();
                child
            }));
            let second = tokio::spawn(TASK_SCOPE_STACK.scope(second_stack, async move {
                let child = ScopeHandle::builder()
                    .name("second-child")
                    .scope_type(ScopeType::Function)
                    .parent_uuid(parent_uuid)
                    .build();
                task_scope_push(child.clone());
                second_pushed_tx.send(()).unwrap();
                first_popped_rx.await.unwrap();
                task_scope_remove(&child.uuid).unwrap();
                child
            }));

            let (first, second) = tokio::join!(first, second);
            assert_eq!(first.unwrap().parent_uuid, Some(parent_uuid));
            assert_eq!(second.unwrap().parent_uuid, Some(parent_uuid));
            assert_eq!(task_scope_top().uuid, parent.uuid);
        })
        .await;
}

#[test]
fn test_fork_scope_stack_uses_current_thread_parent() {
    let stack = create_scope_stack();
    with_scope_stack(stack.clone(), || {
        let parent = ScopeHandle::builder()
            .name("thread-parent")
            .scope_type(ScopeType::Agent)
            .build();
        task_scope_push(parent.clone());

        let fork = fork_scope_stack().unwrap();
        assert_eq!(fork.read().unwrap().top().uuid, parent.uuid);
        assert!(!Arc::ptr_eq(&fork, &stack));
    });
}

/// Thread-local fallback creates independent stacks per thread.
#[test]
fn test_thread_local_independent_stacks() {
    use std::sync::{Arc, Barrier};

    let barrier = Arc::new(Barrier::new(2));

    let b1 = barrier.clone();
    let t1 = std::thread::spawn(move || {
        let h = ScopeHandle::builder()
            .name("thread1_scope")
            .scope_type(ScopeType::Agent)
            .build();
        task_scope_push(h);
        b1.wait(); // sync with thread 2
        let top = task_scope_top();
        assert_eq!(top.name, "thread1_scope");
        top.name.clone()
    });

    let b2 = barrier.clone();
    let t2 = std::thread::spawn(move || {
        let h = ScopeHandle::builder()
            .name("thread2_scope")
            .scope_type(ScopeType::Function)
            .build();
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
            let h = ScopeHandle::builder()
                .name("custom")
                .scope_type(ScopeType::Agent)
                .build();
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
            let h = ScopeHandle::builder()
                .name("propagated")
                .scope_type(ScopeType::Agent)
                .build();
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
            let h = ScopeHandle::builder()
                .name("parent_scope")
                .scope_type(ScopeType::Agent)
                .build();
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

#[test]
fn test_scope_stack_helpers_cover_lookup_mutation_and_remove_paths() {
    let mut stack = ScopeStack::default();
    let root_uuid = stack.root_uuid();

    assert_eq!(stack.scopes().len(), 1);
    assert_eq!(stack.find(&root_uuid).unwrap().name, "root");

    stack.top_mut().name = "root-renamed".into();
    assert_eq!(stack.top().name, "root-renamed");

    let child = ScopeHandle::builder()
        .name("child")
        .scope_type(ScopeType::Function)
        .build();
    let child_uuid = child.uuid;
    stack.push(child);

    assert_eq!(stack.scopes().len(), 2);
    assert_eq!(stack.find(&child_uuid).unwrap().name, "child");

    match stack.remove(&root_uuid) {
        Err(FlowError::InvalidArgument(message)) => {
            assert!(message.contains("not at the top of the stack"));
        }
        other => panic!("unexpected root removal error while child is active: {other:?}"),
    }

    let removed = stack.remove(&child_uuid).unwrap();
    assert_eq!(removed.name, "child");
    assert!(stack.find(&child_uuid).is_none());

    match stack.remove(&root_uuid) {
        Err(FlowError::InvalidArgument(message)) => {
            assert!(message.contains("root scope cannot be removed"));
        }
        other => panic!("unexpected root removal error: {other:?}"),
    }

    match stack.remove(&Uuid::now_v7()) {
        Err(FlowError::NotFound(message)) => {
            assert!(message.contains("scope handle not found"));
        }
        other => panic!("unexpected missing-scope removal result: {other:?}"),
    }

    let debug = format!("{stack:?}");
    assert!(debug.contains("ScopeStack"));
    assert!(debug.contains("scope_registries_count"));
}

#[test]
fn test_sync_thread_scope_stack_and_task_scope_remove_use_bound_handle() {
    std::thread::spawn(|| {
        let initial = create_scope_stack();
        set_thread_scope_stack(initial);

        let replacement = create_scope_stack();
        {
            let mut guard = replacement.write().unwrap();
            guard.push(
                ScopeHandle::builder()
                    .name("replacement")
                    .scope_type(ScopeType::Agent)
                    .build(),
            );
        }
        sync_thread_scope_stack(replacement);
        assert_eq!(task_scope_top().name, "replacement");

        let nested = ScopeHandle::builder()
            .name("nested")
            .scope_type(ScopeType::Function)
            .build();
        let nested_uuid = nested.uuid;
        task_scope_push(nested);
        assert_eq!(task_scope_top().name, "nested");

        let removed = task_scope_remove(&nested_uuid).unwrap();
        assert_eq!(removed.name, "nested");
        assert_eq!(task_scope_top().name, "replacement");
    })
    .join()
    .unwrap();
}
