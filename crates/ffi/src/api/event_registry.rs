// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    NemoRelayEventSanitizeCb, NemoRelayFreeFn, NemoRelayStatus, c_char, c_str_to_string,
    clear_last_error, core_registry_api, set_last_error, status_from_error, wrap_event_sanitize_fn,
};

#[derive(Clone, Copy)]
enum Surface {
    Mark,
    Start,
    End,
}

unsafe fn register_global(
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
    surface: Surface,
) -> NemoRelayStatus {
    clear_last_error();
    let callback = wrap_event_sanitize_fn(cb, user_data, free_fn);
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let result = match surface {
        Surface::Mark => {
            core_registry_api::register_mark_sanitize_guardrail(&name, priority, callback)
        }
        Surface::Start => {
            core_registry_api::register_scope_sanitize_start_guardrail(&name, priority, callback)
        }
        Surface::End => {
            core_registry_api::register_scope_sanitize_end_guardrail(&name, priority, callback)
        }
    };
    result
        .map(|()| NemoRelayStatus::Ok)
        .unwrap_or_else(|error| status_from_error(&error))
}

unsafe fn deregister_global(name: *const c_char, surface: Surface) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let result = match surface {
        Surface::Mark => core_registry_api::deregister_mark_sanitize_guardrail(&name),
        Surface::Start => core_registry_api::deregister_scope_sanitize_start_guardrail(&name),
        Surface::End => core_registry_api::deregister_scope_sanitize_end_guardrail(&name),
    };
    result
        .map(|_| NemoRelayStatus::Ok)
        .unwrap_or_else(|error| status_from_error(&error))
}

fn parse_scope_uuid(value: *const c_char) -> Result<uuid::Uuid, NemoRelayStatus> {
    let value = c_str_to_string(value)?;
    uuid::Uuid::parse_str(&value).map_err(|error| {
        set_last_error(&format!("invalid scope UUID: {error}"));
        NemoRelayStatus::InvalidArg
    })
}

unsafe fn register_scope(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
    surface: Surface,
) -> NemoRelayStatus {
    clear_last_error();
    let callback = wrap_event_sanitize_fn(cb, user_data, free_fn);
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let result = match surface {
        Surface::Mark => core_registry_api::scope_register_mark_sanitize_guardrail(
            &uuid, &name, priority, callback,
        ),
        Surface::Start => core_registry_api::scope_register_scope_sanitize_start_guardrail(
            &uuid, &name, priority, callback,
        ),
        Surface::End => core_registry_api::scope_register_scope_sanitize_end_guardrail(
            &uuid, &name, priority, callback,
        ),
    };
    result
        .map(|()| NemoRelayStatus::Ok)
        .unwrap_or_else(|error| status_from_error(&error))
}

unsafe fn deregister_scope(
    scope_uuid: *const c_char,
    name: *const c_char,
    surface: Surface,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let result = match surface {
        Surface::Mark => core_registry_api::scope_deregister_mark_sanitize_guardrail(&uuid, &name),
        Surface::Start => {
            core_registry_api::scope_deregister_scope_sanitize_start_guardrail(&uuid, &name)
        }
        Surface::End => {
            core_registry_api::scope_deregister_scope_sanitize_end_guardrail(&uuid, &name)
        }
    };
    result
        .map(|_| NemoRelayStatus::Ok)
        .unwrap_or_else(|error| status_from_error(&error))
}

/// Register a global mark event sanitizer.
/// # Safety
/// Pointers must be valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_mark_sanitize_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe { register_global(name, priority, cb, user_data, free_fn, Surface::Mark) }
}
/// Deregister a global mark event sanitizer.
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_mark_sanitize_guardrail(
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { deregister_global(name, Surface::Mark) }
}
/// Register a global scope-start event sanitizer.
/// # Safety
/// Pointers must be valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_scope_sanitize_start_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe { register_global(name, priority, cb, user_data, free_fn, Surface::Start) }
}
/// Deregister a global scope-start event sanitizer.
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_scope_sanitize_start_guardrail(
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { deregister_global(name, Surface::Start) }
}
/// Register a global scope-end event sanitizer.
/// # Safety
/// Pointers must be valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_scope_sanitize_end_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe { register_global(name, priority, cb, user_data, free_fn, Surface::End) }
}
/// Deregister a global scope-end event sanitizer.
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_scope_sanitize_end_guardrail(
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { deregister_global(name, Surface::End) }
}

/// Register a scope-local mark event sanitizer.
/// # Safety
/// Pointers must be valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_mark_sanitize_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe {
        register_scope(
            scope_uuid,
            name,
            priority,
            cb,
            user_data,
            free_fn,
            Surface::Mark,
        )
    }
}
/// Deregister a scope-local mark event sanitizer.
/// # Safety
/// String pointers must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_mark_sanitize_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { deregister_scope(scope_uuid, name, Surface::Mark) }
}
/// Register a scope-local scope-start event sanitizer.
/// # Safety
/// Pointers must be valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_scope_sanitize_start_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe {
        register_scope(
            scope_uuid,
            name,
            priority,
            cb,
            user_data,
            free_fn,
            Surface::Start,
        )
    }
}
/// Deregister a scope-local scope-start event sanitizer.
/// # Safety
/// String pointers must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_scope_sanitize_start_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { deregister_scope(scope_uuid, name, Surface::Start) }
}
/// Register a scope-local scope-end event sanitizer.
/// # Safety
/// Pointers must be valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_scope_sanitize_end_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe {
        register_scope(
            scope_uuid,
            name,
            priority,
            cb,
            user_data,
            free_fn,
            Surface::End,
        )
    }
}
/// Deregister a scope-local scope-end event sanitizer.
/// # Safety
/// String pointers must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_scope_sanitize_end_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { deregister_scope(scope_uuid, name, Surface::End) }
}
