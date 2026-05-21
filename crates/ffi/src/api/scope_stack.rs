// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    FfiScopeStack, FfiThreadScopeStackBinding, NemoRelayStatus, capture_thread_scope_stack,
    clear_last_error, create_scope_stack, restore_thread_scope_stack, scope_stack_active,
    set_last_error, set_thread_scope_stack,
};

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Each scope stack is independent: scopes pushed on one do not appear on another.
/// Use `nemo_relay_scope_stack_set_thread` to bind a stack to the current thread
/// before making other NeMo Relay API calls.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeStack` that must be
///   freed with `nemo_relay_scope_stack_free`.
///
/// # Returns
/// - Returns [`NemoRelayStatus::Ok`] on success and writes the new scope stack
///   to `out`.
/// - Returns [`NemoRelayStatus::NullPointer`] when `out` is null.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_stack_create(
    out: *mut *mut FfiScopeStack,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let handle = create_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiScopeStack(handle))) };
    NemoRelayStatus::Ok
}

/// Bind an isolated scope stack to the current OS thread.
///
/// After this call, all NeMo Relay scope operations on the current thread
/// (e.g. `nemo_relay_push_scope`, `nemo_relay_get_handle`) will use the
/// given scope stack. This is typically used from Go goroutines that have
/// called `runtime.LockOSThread()`.
///
/// The `FfiScopeStack` is **not** consumed — the caller retains ownership
/// and must still free it when done.
///
/// # Parameters
/// - `stack`: Scope stack to bind to the current OS thread.
///
/// # Returns
/// - Returns [`NemoRelayStatus::Ok`] when the thread-local scope stack was
///   updated successfully.
/// - Returns [`NemoRelayStatus::NullPointer`] when `stack` is null.
///
/// # Safety
/// `stack` must be a valid, non-null `FfiScopeStack` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_stack_set_thread(
    stack: *const FfiScopeStack,
) -> NemoRelayStatus {
    clear_last_error();
    if stack.is_null() {
        set_last_error("stack pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let handle = unsafe { &*stack }.0.clone();
    set_thread_scope_stack(handle);
    NemoRelayStatus::Ok
}

/// Capture the current thread-local scope stack binding.
///
/// The returned binding must be restored with
/// `nemo_relay_scope_stack_restore_thread`.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated binding handle.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_stack_capture_thread(
    out: *mut *mut FfiThreadScopeStackBinding,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let binding = capture_thread_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiThreadScopeStackBinding(binding))) };
    NemoRelayStatus::Ok
}

/// Restore and free a captured thread-local scope stack binding.
///
/// # Safety
/// `binding` must be a valid pointer returned by
/// `nemo_relay_scope_stack_capture_thread`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_stack_restore_thread(
    binding: *mut FfiThreadScopeStackBinding,
) -> NemoRelayStatus {
    clear_last_error();
    if binding.is_null() {
        set_last_error("binding pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let binding = unsafe { Box::from_raw(binding) };
    restore_thread_scope_stack(binding.0);
    NemoRelayStatus::Ok
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `nemo_relay_scope_stack_set_thread` has been called on the
/// current OS thread (or the caller is inside a tokio task-local scope).
/// Returns `false` when only the auto-created default is present.
///
/// # Notes
/// This helper does not allocate or install a scope stack. It only reports
/// whether one is already explicit in the current execution context.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_scope_stack_active() -> bool {
    scope_stack_active()
}
