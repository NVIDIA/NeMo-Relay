// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal task context captured for middleware continuations.

use std::future::Future;

use crate::api::optimization::{
    LlmOptimizationRecorder, current_llm_optimization_recorder, scope_llm_optimization_recorder,
};
use crate::api::runtime::scope_stack::{
    ScopeStackHandle, TASK_SCOPE_STACK, active_event_uuid, current_context_scope_stack,
    current_scope_stack, scope_stack_active, snapshot_scope_stack, with_active_event_uuid,
};
use crate::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, PublicationContext, capture_nested_publication_buffer,
    capture_publication_context, with_task_nested_publication_buffer,
    with_task_publication_context,
};
use crate::error::{FlowError, Result};

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

    /// Clone this context with an isolated snapshot of its visible scope stack.
    ///
    /// Execution-intercept continuations use one isolated context per `next`
    /// invocation so concurrent retry or fan-out branches cannot mutate the
    /// same scope stack.
    #[doc(hidden)]
    pub fn isolated(&self) -> Result<Self> {
        self.isolated_with_scope_stack(&self.scope_stack)
    }

    /// Clone this context with an isolated snapshot of `scope_stack`.
    ///
    /// Bindings use this when their own async-local scope selection can differ
    /// from the stack captured when the middleware callback began.
    #[doc(hidden)]
    pub fn isolated_with_scope_stack(&self, scope_stack: &ScopeStackHandle) -> Result<Self> {
        Ok(Self {
            scope_stack: snapshot_scope_stack(scope_stack)?,
            active_event_uuid: self.active_event_uuid,
            publication_context: self.publication_context.clone(),
            publication_buffer: self.publication_buffer.clone(),
            optimization_recorder: self.optimization_recorder.clone(),
        })
    }

    /// Clone this context with the scope selection visible to a continuation call.
    pub(crate) fn isolated_for_current_invocation(&self) -> Result<Self> {
        let visible_scope_stack = current_context_scope_stack().unwrap_or_else(|| {
            if tokio::task::try_id().is_none() && scope_stack_active() {
                current_scope_stack()
            } else {
                self.scope_stack.clone()
            }
        });
        self.isolated_with_scope_stack(&visible_scope_stack)
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

/// A continuation that remains valid while its owning interceptor is running.
pub(crate) struct MiddlewareContinuationLease {
    context: MiddlewareContinuationContext,
    revoked: tokio::sync::watch::Receiver<bool>,
}

impl Clone for MiddlewareContinuationLease {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            revoked: self.revoked.clone(),
        }
    }
}

/// Revokes a middleware continuation when its interceptor settles or is dropped.
pub(crate) struct MiddlewareContinuationGuard {
    revoke: tokio::sync::watch::Sender<bool>,
}

impl Drop for MiddlewareContinuationGuard {
    fn drop(&mut self) {
        let _ = self.revoke.send(true);
    }
}

/// One isolated invocation of a middleware continuation.
pub(crate) struct MiddlewareContinuationInvocation {
    context: MiddlewareContinuationContext,
    revoked: tokio::sync::watch::Receiver<bool>,
}

impl MiddlewareContinuationLease {
    /// Capture the current Relay context and create its callback-lifetime guard.
    pub(crate) fn capture() -> (Self, MiddlewareContinuationGuard) {
        let (revoke, revoked) = tokio::sync::watch::channel(false);
        (
            Self {
                context: MiddlewareContinuationContext::capture(),
                revoked,
            },
            MiddlewareContinuationGuard { revoke },
        )
    }

    /// Begin a `next` call with an invocation-time scope-stack snapshot.
    pub(crate) fn begin(&self) -> Result<MiddlewareContinuationInvocation> {
        if *self.revoked.borrow() {
            return Err(inactive_continuation_error());
        }
        let context = self.context.isolated_for_current_invocation()?;
        Ok(MiddlewareContinuationInvocation {
            context,
            revoked: self.revoked.clone(),
        })
    }
}

impl MiddlewareContinuationInvocation {
    /// Return the isolated Relay context for work that outlives construction.
    pub(crate) fn context(&self) -> &MiddlewareContinuationContext {
        &self.context
    }

    /// Run downstream work until it completes or the owning interceptor settles.
    pub(crate) async fn invoke<C, F, T>(mut self, callback: C) -> Result<T>
    where
        C: FnOnce() -> F,
        F: Future<Output = Result<T>>,
    {
        if *self.revoked.borrow() {
            return Err(inactive_continuation_error());
        }
        let future = self.context.invoke(callback);
        tokio::pin!(future);
        tokio::select! {
            biased;
            _ = self.revoked.changed() => Err(inactive_continuation_error()),
            result = &mut future => result,
        }
    }
}

fn inactive_continuation_error() -> FlowError {
    FlowError::InvalidArgument("execution continuation is no longer active".into())
}

#[cfg(test)]
#[path = "../../../tests/unit/continuation_context_tests.rs"]
mod tests;
