// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::optimization::{
    LlmOptimizationRecorder, record_llm_optimization_contribution, scope_llm_optimization_recorder,
};
use crate::api::runtime::scope_stack::{
    TASK_SCOPE_STACK, active_event_uuid, create_scope_stack, current_scope_stack,
    with_active_event_uuid,
};
use crate::codec::optimization::LlmOptimizationContribution;
use std::sync::Arc;

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
