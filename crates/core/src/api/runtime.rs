// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advanced runtime state, callbacks, and scope-stack helpers.

pub mod callbacks;
mod continuation_context;
pub mod global;
pub mod scope_stack;
pub mod state;
pub mod subscriber_dispatcher;

pub use callbacks::{
    BuiltinLlmCodec, EventMetadataInjectorFn, EventSanitizeFn, EventSubscriberFn, LlmCodecIdentity,
    LlmCollectorFn, LlmConditionalFn, LlmExecutionFn, LlmExecutionNextFn, LlmFinalizerFn,
    LlmJsonStream, LlmRequestInterceptFn, LlmSanitizeRequestContext, LlmSanitizeRequestFn,
    LlmSanitizeResponseContext, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, LlmStreamInner, ToolConditionalFn, ToolExecutionFn,
    ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
#[doc(hidden)]
pub use continuation_context::MiddlewareContinuationContext;
#[cfg(test)]
pub(crate) use continuation_context::MiddlewareContinuationLease;
pub use global::global_context;
pub use scope_stack::{
    PropagationContext, ScopeStack, ScopeStackHandle, TASK_SCOPE_STACK, ThreadScopeStackBinding,
    capture_propagation_context, capture_propagation_context_with_root, capture_thread_scope_stack,
    capture_traceparent, create_scope_stack, create_scope_stack_from_propagation,
    current_scope_stack, fork_scope_stack, propagate_scope_to_thread, restore_thread_scope_stack,
    scope_stack_active, set_thread_scope_stack, sync_thread_scope_stack, task_scope_push,
    task_scope_remove, task_scope_top, with_active_event_uuid, with_scope_stack,
};
pub use state::NemoRelayContextState;
#[doc(hidden)]
pub use subscriber_dispatcher::SubscriberDelivery;
pub use subscriber_dispatcher::flush_subscribers;
