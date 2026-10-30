// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::optimization::{
    LlmOptimizationRecorder, record_llm_optimization_contribution, scope_llm_optimization_recorder,
};
use crate::api::runtime::scope_stack::{
    TASK_SCOPE_STACK, active_event_uuid, capture_thread_scope_stack, create_scope_stack,
    current_scope_stack, restore_thread_scope_stack, snapshot_scope_stack,
    sync_thread_active_event_for_stack, task_scope_push, task_scope_remove, with_active_event_uuid,
    with_scope_stack,
};
use crate::api::scope::{ScopeHandle, ScopeType};
use crate::codec::optimization::LlmOptimizationContribution;
use crate::error::FlowError;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn continuation_context_restores_all_managed_execution_state() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let scope_stack = create_scope_stack();
        let event_uuid = uuid::Uuid::now_v7();
        let recorder = LlmOptimizationRecorder::default();
        let context = TASK_SCOPE_STACK
            .scope(
                scope_stack.clone(),
                with_active_event_uuid(
                    event_uuid,
                    scope_llm_optimization_recorder(recorder, async {
                        MiddlewareContinuationContext::capture()
                    }),
                ),
            )
            .await;

        let observed = tokio::spawn(async move {
            context
                .invoke(move || {
                    let prelude_event_uuid = active_event_uuid();
                    let prelude_scope_stack = current_scope_stack();
                    async move {
                        let recorded = record_llm_optimization_contribution(
                            LlmOptimizationContribution::new("test.continuation", "context"),
                        );
                        (
                            prelude_event_uuid,
                            prelude_scope_stack,
                            active_event_uuid(),
                            recorded,
                            current_scope_stack(),
                        )
                    }
                })
                .await
        })
        .await
        .unwrap();

        assert_eq!(observed.0, Some(event_uuid));
        assert!(Arc::ptr_eq(&observed.1, &scope_stack));
        assert_eq!(observed.2, Some(event_uuid));
        assert!(observed.3);
        assert!(Arc::ptr_eq(&observed.4, &scope_stack));
    });
}

#[test]
fn continuation_context_isolates_each_scope_stack_snapshot() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let scope_stack = create_scope_stack();
        let context = TASK_SCOPE_STACK
            .scope(scope_stack.clone(), async {
                MiddlewareContinuationContext::capture()
            })
            .await;
        let first = context.isolated().unwrap();
        let second = context.isolated().unwrap();

        assert!(!Arc::ptr_eq(&first.scope_stack, &scope_stack));
        assert!(!Arc::ptr_eq(&second.scope_stack, &scope_stack));
        assert!(!Arc::ptr_eq(&first.scope_stack, &second.scope_stack));
    });
}

#[test]
fn continuation_context_preserves_the_managed_event_anchor_across_nested_scopes() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let scope_stack = create_scope_stack();
        let managed_event_uuid = uuid::Uuid::now_v7();
        TASK_SCOPE_STACK
            .scope(
                scope_stack,
                with_active_event_uuid(managed_event_uuid, async {
                    let context = MiddlewareContinuationContext::capture();
                    let nested = ScopeHandle::builder()
                        .name("plugin-a")
                        .scope_type(ScopeType::Custom)
                        .parent_uuid(managed_event_uuid)
                        .build();
                    let nested_uuid = nested.uuid;
                    task_scope_push(nested);
                    let invocation = context.isolated().unwrap();

                    let observed_parent = invocation
                        .run(async {
                            let callback_stack =
                                snapshot_scope_stack(&current_scope_stack()).unwrap();
                            let previous_thread_binding = capture_thread_scope_stack();
                            sync_thread_active_event_for_stack(&callback_stack);
                            let parent = with_scope_stack(callback_stack, || {
                                crate::api::shared::resolve_parent_uuid(None)
                            });
                            restore_thread_scope_stack(previous_thread_binding);
                            parent
                        })
                        .await;

                    task_scope_remove(&nested_uuid).unwrap();
                    assert_eq!(observed_parent, Some(nested_uuid));
                }),
            )
            .await;
    });
}

#[test]
fn continuation_lease_honors_an_explicit_scope_stack_in_a_spawned_task() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let captured_stack = create_scope_stack();
        let (lease, guard) = TASK_SCOPE_STACK
            .scope(captured_stack, async {
                MiddlewareContinuationLease::capture()
            })
            .await;
        let alternate_stack = create_scope_stack();
        let expected_scope_uuid = alternate_stack
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .top()
            .uuid;

        let invocation =
            tokio::spawn(TASK_SCOPE_STACK.scope(alternate_stack, async move { lease.begin() }))
                .await
                .unwrap()
                .unwrap();
        let actual_scope_uuid = invocation
            .context
            .scope_stack
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .top()
            .uuid;

        assert_eq!(actual_scope_uuid, expected_scope_uuid);
        drop(guard);
    });
}

#[test]
fn continuation_lease_rejects_late_calls_and_cancels_in_flight_work() {
    struct DropSignal(Arc<AtomicBool>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let (lease, guard) = MiddlewareContinuationLease::capture();
        let invocation = lease.begin().unwrap();
        let entered = Arc::new(tokio::sync::Notify::new());
        let entered_for_task = Arc::clone(&entered);
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_for_task = Arc::clone(&dropped);
        let pending = tokio::spawn(async move {
            invocation
                .invoke(move || async move {
                    let _drop_signal = DropSignal(dropped_for_task);
                    entered_for_task.notify_one();
                    std::future::pending::<Result<()>>().await
                })
                .await
        });

        entered.notified().await;
        drop(guard);

        let error = pending.await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            FlowError::InvalidArgument(message)
                if message == "execution continuation is no longer active"
        ));
        assert!(dropped.load(Ordering::Acquire));
        assert!(matches!(
            lease.begin(),
            Err(FlowError::InvalidArgument(message))
                if message == "execution continuation is no longer active"
        ));
    });
}
