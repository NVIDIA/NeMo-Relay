// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Each scope stack is independent: scopes pushed on one do not appear on another.
/// Use `nemo_flow_scope_stack_set_thread` to bind a stack to the current thread
/// before making other NeMo Flow API calls.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeStack` that must be
///   freed with `nemo_flow_scope_stack_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_stack_create(
    out: *mut *mut FfiScopeStack,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let handle = create_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiScopeStack(handle))) };
    NemoFlowStatus::Ok
}

/// Bind an isolated scope stack to the current OS thread.
///
/// After this call, all NeMo Flow scope operations on the current thread
/// (e.g. `nemo_flow_push_scope`, `nemo_flow_get_handle`) will use the
/// given scope stack. This is typically used from Go goroutines that have
/// called `runtime.LockOSThread()`.
///
/// The `FfiScopeStack` is **not** consumed — the caller retains ownership
/// and must still free it when done.
///
/// # Safety
/// `stack` must be a valid, non-null `FfiScopeStack` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_stack_set_thread(
    stack: *const FfiScopeStack,
) -> NemoFlowStatus {
    clear_last_error();
    if stack.is_null() {
        set_last_error("stack pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let handle = unsafe { &*stack }.0.clone();
    set_thread_scope_stack(handle);
    NemoFlowStatus::Ok
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `nemo_flow_scope_stack_set_thread` has been called on the
/// current OS thread (or the caller is inside a tokio task-local scope).
/// Returns `false` when only the auto-created default is present.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_scope_stack_active() -> bool {
    scope_stack_active()
}
