// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque scope ownership and propagation across tasks and threads.

use nemo_relay_runtime::api::runtime;
use std::future::Future;

pub use nemo_relay_types::api::scope::{ScopeStackEntry, ScopeStackSnapshot};
pub use runtime::{
    PropagationContext, capture_propagation_context, capture_propagation_context_with_root,
    capture_rootless_propagation_context, capture_traceparent, current_scope_stack_snapshot,
};

/// A shared scope-stack identity, without access to the runtime's locks or registries.
///
/// Cloning shares the stack. Use [`fork_scope_stack`] for independent child work
/// and [`Self::snapshot`] for read-only inspection.
///
/// ```compile_fail,E0599
/// let stack = nemo_relay::context::create_scope_stack();
/// let _ = stack.write();
/// ```
///
/// ```compile_fail,E0616
/// let stack = nemo_relay::context::create_scope_stack();
/// let _ = stack.0;
/// ```
#[derive(Clone)]
pub struct ScopeStackHandle(runtime::ScopeStackHandle);

impl ScopeStackHandle {
    /// Copy the visible scope hierarchy without exposing mutable runtime state.
    pub fn snapshot(&self) -> crate::Result<ScopeStackSnapshot> {
        runtime::scope_stack::scope_stack_snapshot(&self.0)
    }
}

/// A captured thread binding, including its active managed-event context.
#[derive(Clone)]
pub struct ThreadScopeStackBinding(runtime::ThreadScopeStackBinding);

impl ThreadScopeStackBinding {
    /// Clone the captured stack identity without exposing its implementation.
    pub fn stack(&self) -> ScopeStackHandle {
        ScopeStackHandle(self.0.stack())
    }
}

/// Create an independent stack containing one implicit root scope.
pub fn create_scope_stack() -> ScopeStackHandle {
    ScopeStackHandle(runtime::create_scope_stack())
}

/// Copy a stack into an isolated emission-time snapshot.
pub fn snapshot_scope_stack(stack: &ScopeStackHandle) -> crate::Result<ScopeStackHandle> {
    runtime::scope_stack::snapshot_scope_stack(&stack.0).map(ScopeStackHandle)
}

/// Create an independent stack below a transported causal parent.
pub fn create_scope_stack_from_propagation(
    context: &PropagationContext,
) -> crate::Result<ScopeStackHandle> {
    runtime::create_scope_stack_from_propagation(context).map(ScopeStackHandle)
}

/// Create an independent child stack below the current causal parent.
pub fn fork_scope_stack() -> crate::Result<ScopeStackHandle> {
    runtime::fork_scope_stack().map(ScopeStackHandle)
}

/// Run a future with a stack bound across executor thread switches.
pub async fn with_task_scope_stack<T>(
    stack: ScopeStackHandle,
    future: impl Future<Output = T>,
) -> T {
    runtime::with_task_scope_stack(stack.0, future).await
}

/// Run synchronous work with a stack, restoring the previous binding afterward.
pub fn with_scope_stack<T>(stack: ScopeStackHandle, work: impl FnOnce() -> T) -> T {
    runtime::with_scope_stack(stack.0, work)
}

/// Install a stack on the current thread.
pub fn set_thread_scope_stack(stack: ScopeStackHandle) {
    runtime::set_thread_scope_stack(stack.0);
}

/// Capture the current thread binding for later restoration.
pub fn capture_thread_scope_stack() -> ThreadScopeStackBinding {
    ThreadScopeStackBinding(runtime::capture_thread_scope_stack())
}

/// Restore a captured thread binding and its managed-event context.
pub fn restore_thread_scope_stack(binding: ThreadScopeStackBinding) {
    runtime::restore_thread_scope_stack(binding.0);
}

/// Capture the active stack for propagation to another thread.
pub fn propagate_scope_to_thread() -> crate::Result<ScopeStackHandle> {
    runtime::propagate_scope_to_thread().map(ScopeStackHandle)
}
