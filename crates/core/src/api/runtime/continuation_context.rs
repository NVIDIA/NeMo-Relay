// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal task context captured for middleware continuations.

use std::future::Future;

use crate::api::optimization::{
    LlmOptimizationRecorder, current_llm_optimization_recorder, scope_llm_optimization_recorder,
};
use crate::api::runtime::scope_stack::{
    ScopeStackHandle, TASK_SCOPE_STACK, active_event_uuid, current_scope_stack,
    with_active_event_uuid,
};
use crate::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, PublicationContext, capture_nested_publication_buffer,
    capture_publication_context, with_task_nested_publication_buffer,
    with_task_publication_context,
};

/// Opaque Relay task context captured for a middleware `next` continuation.
///
/// This is an internal cross-crate bridge for Relay's language bindings and
/// dynamic-plugin adapters. Its fields intentionally remain private.
#[doc(hidden)]
#[derive(Clone)]
pub struct MiddlewareContinuationContext {
    scope_stack: ScopeStackHandle,
    active_event_uuid: Option<uuid::Uuid>,
    publication_context: Option<PublicationContext>,
    publication_buffer: Option<PublicationBuffer>,
    optimization_recorder: Option<LlmOptimizationRecorder>,
}

impl MiddlewareContinuationContext {
    /// Capture the Relay task context visible to the current middleware call.
    #[doc(hidden)]
    #[must_use]
    pub fn capture() -> Self {
        Self {
            scope_stack: current_scope_stack(),
            active_event_uuid: active_event_uuid(),
            publication_context: capture_publication_context(),
            publication_buffer: capture_nested_publication_buffer(),
            optimization_recorder: current_llm_optimization_recorder(),
        }
    }

    /// Poll `future` with the captured Relay task context restored.
    #[doc(hidden)]
    pub async fn run<F: Future>(&self, future: F) -> F::Output {
        let scoped = TASK_SCOPE_STACK.scope(self.scope_stack.clone(), future);
        let published = with_task_publication_context(self.publication_context.clone(), scoped);
        let published =
            with_task_nested_publication_buffer(self.publication_buffer.clone(), published);
        let active = async {
            match self.active_event_uuid {
                Some(uuid) => with_active_event_uuid(uuid, published).await,
                None => published.await,
            }
        };
        match &self.optimization_recorder {
            Some(recorder) => scope_llm_optimization_recorder(recorder.clone(), active).await,
            None => active.await,
        }
    }

    /// Invoke a callback and poll its future with the captured Relay context.
    ///
    /// The callback itself can inspect Relay task state before constructing its
    /// future, so it must be invoked only after the context is restored.
    #[doc(hidden)]
    pub async fn invoke<C, F>(&self, callback: C) -> F::Output
    where
        C: FnOnce() -> F,
        F: Future,
    {
        self.run(async move { callback().await }).await
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/continuation_context_tests.rs"]
mod tests;
