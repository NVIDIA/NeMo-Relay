// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closeable event subscription FFI functions.

use libc::c_char;
use nemo_relay::api::subscriber as core_subscriber_api;

use crate::callable::{
    NemoRelayEventSubscriberCb, NemoRelayFreeFn, wrap_event_subscriber_deferred,
};
use crate::convert::c_str_to_string;
use crate::error::{NemoRelayStatus, clear_last_error, status_from_error};
use crate::types::{FfiSubscription, FfiSubscriptionHandle};

/// Register an anonymous global event subscriber and return a closeable handle.
///
/// # Parameters
/// - `cb`: Event callback. The `FfiEvent` is valid only during the call.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data` after registration ownership
///   has been accepted.
/// - `out`: Receives the subscription handle on success.
///
/// # Safety
/// `cb` must be a valid function pointer. `out` must be non-null and valid for
/// writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_subscribe(
    cb: NemoRelayEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
    out: *mut *mut FfiSubscriptionHandle,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = std::ptr::null_mut();
    }
    let (wrapped, ownership) = wrap_event_subscriber_deferred(cb, user_data, free_fn);
    match core_subscriber_api::subscribe(wrapped) {
        Ok(handle) => {
            ownership.accept();
            unsafe {
                *out = Box::into_raw(Box::new(FfiSubscriptionHandle(FfiSubscription::Global(
                    handle,
                ))));
            }
            NemoRelayStatus::Ok
        }
        Err(error) => status_from_error(&error),
    }
}

/// Register an anonymous scope-local event subscriber and return a closeable handle.
///
/// # Parameters
/// - `scope_uuid`: UUID of the active scope that owns the subscriber.
/// - `cb`: Event callback. The `FfiEvent` is valid only during the call.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data` after registration ownership
///   has been accepted.
/// - `out`: Receives the subscription handle on success.
///
/// # Safety
/// `scope_uuid` must be a valid C string. `cb` must be a valid function
/// pointer. `out` must be non-null and valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_subscribe(
    scope_uuid: *const c_char,
    cb: NemoRelayEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
    out: *mut *mut FfiSubscriptionHandle,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out = std::ptr::null_mut();
    }
    let scope_uuid = match c_str_to_string(scope_uuid) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let uuid = match uuid::Uuid::parse_str(&scope_uuid) {
        Ok(uuid) => uuid,
        Err(_) => return NemoRelayStatus::InvalidArg,
    };
    let (wrapped, ownership) = wrap_event_subscriber_deferred(cb, user_data, free_fn);
    match core_subscriber_api::scope_subscribe(&uuid, wrapped) {
        Ok(handle) => {
            ownership.accept();
            unsafe {
                *out = Box::into_raw(Box::new(FfiSubscriptionHandle(FfiSubscription::Scope(
                    handle,
                ))));
            }
            NemoRelayStatus::Ok
        }
        Err(error) => status_from_error(&error),
    }
}

/// Close a subscription handle.
///
/// This function is idempotent. Closing after automatic scope cleanup succeeds.
/// `removed` receives true only when this call removed a live subscriber.
///
/// # Safety
/// `handle` must be a valid pointer returned by `nemo_relay_subscribe` or
/// `nemo_relay_scope_subscribe`. `removed` must be non-null and valid for
/// writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_subscription_close(
    handle: *mut FfiSubscriptionHandle,
    removed: *mut bool,
) -> NemoRelayStatus {
    clear_last_error();
    if removed.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *removed = false;
    }
    if handle.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let result = match unsafe { &*handle } {
        FfiSubscriptionHandle(FfiSubscription::Global(handle)) => handle.close(),
        FfiSubscriptionHandle(FfiSubscription::Scope(handle)) => handle.close(),
    };
    match result {
        Ok(value) => {
            unsafe {
                *removed = value;
            }
            NemoRelayStatus::Ok
        }
        Err(error) => status_from_error(&error),
    }
}
