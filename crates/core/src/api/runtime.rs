// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advanced runtime state, callbacks, and scope-stack helpers.

pub mod callbacks;
pub mod global;
pub mod scope_stack;
pub mod state;

pub use callbacks::{
    EventSubscriberFn, LlmCollectorFn, LlmConditionalFn, LlmConditionalSharedFn, LlmExecutionFn,
    LlmExecutionNextFn, LlmFinalizerFn, LlmJsonStream, LlmRequestInterceptFn, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, LlmStreamExecutionFn, LlmStreamExecutionNextFn,
    LlmStreamExecutionRegistryRef, LlmStreamExecutionRegistryRefs, ToolConditionalFn,
    ToolExecutionFn, ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
pub use global::global_context;
pub use scope_stack::{
    ScopeStack, ScopeStackHandle, TASK_SCOPE_STACK, ThreadScopeStackBinding,
    capture_thread_scope_stack, create_scope_stack, current_scope_stack, propagate_scope_to_thread,
    restore_thread_scope_stack, scope_stack_active, set_thread_scope_stack,
    sync_thread_scope_stack, task_scope_push, task_scope_remove, task_scope_top,
};
pub use state::NemoFlowContextState;
