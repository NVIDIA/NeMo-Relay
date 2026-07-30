// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn execution_next_context_restores_scope() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let scope_stack = nemo_relay::api::runtime::create_scope_stack();
        let context = nemo_relay::api::runtime::with_scope_stack(scope_stack.clone(), || {
            MiddlewareContinuationContext::capture()
        });

        let observed =
            tokio::spawn(async move { context.run(async move { current_scope_stack() }).await })
                .await
                .unwrap();

        assert!(Arc::ptr_eq(&observed, &scope_stack));
    });
}
