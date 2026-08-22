// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use super::{
    NemoRelayConditionalMiddlewareGuardrailCb, NemoRelayFreeFn, NemoRelayStatus, c_char,
    c_str_to_opt_json, c_str_to_string, clear_last_error, core_registry_api, json_to_c_string,
    set_last_error, status_from_error, wrap_conditional_middleware_guardrail_fn,
};

fn parse_kinds(
    kinds_json: *const c_char,
) -> Result<BTreeSet<core_registry_api::RuntimeRegistrationKind>, NemoRelayStatus> {
    let value = c_str_to_opt_json(kinds_json)
        .ok_or(NemoRelayStatus::InvalidJson)?
        .unwrap_or_else(|| serde_json::json!([]));
    serde_json::from_value(value).map_err(|error| {
        set_last_error(&format!("invalid runtime registration kinds: {error}"));
        NemoRelayStatus::InvalidArg
    })
}

/// Register a global conditional middleware guardrail.
///
/// # Safety
/// String pointers must be valid for the call. Callback input strings are
/// borrowed for each invocation only. A non-null callback result transfers
/// ownership to Relay and must be allocated compatibly with
/// `nemo_relay_string_free`. The callback and user data remain owned by Relay
/// until deregistration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_conditional_middleware_guardrail(
    name: *const c_char,
    kinds_json: *const c_char,
    registration_name: *const c_char,
    cb: NemoRelayConditionalMiddlewareGuardrailCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let Some(cb) = cb else {
        set_last_error("conditional middleware guardrail callback is null");
        return NemoRelayStatus::NullPointer;
    };
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let kinds = match parse_kinds(kinds_json) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let registration_name = match c_str_to_string(registration_name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    core_registry_api::register_conditional_middleware_guardrail(
        &name,
        kinds,
        &registration_name,
        wrap_conditional_middleware_guardrail_fn(cb, user_data, free_fn),
    )
    .map(|()| NemoRelayStatus::Ok)
    .unwrap_or_else(|error| status_from_error(&error))
}

/// Deregister a global conditional middleware guardrail.
///
/// # Safety
/// `name` must point to a valid C string for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_conditional_middleware_guardrail(
    name: *const c_char,
    out_removed: *mut bool,
) -> NemoRelayStatus {
    clear_last_error();
    if out_removed.is_null() {
        set_last_error("out_removed pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    match core_registry_api::deregister_conditional_middleware_guardrail(&name) {
        Ok(removed) => {
            unsafe { *out_removed = removed };
            NemoRelayStatus::Ok
        }
        Err(error) => status_from_error(&error),
    }
}

/// List global gateable runtime registrations as JSON.
///
/// On success, `out_json` receives a Relay-allocated string. The caller owns
/// that string and must release it exactly once with `nemo_relay_string_free`.
///
/// # Safety
/// `out_json` must be a valid writable pointer. `kinds_json` may be null or a
/// valid JSON array of registration kind strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_list_runtime_registrations(
    kinds_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let kinds = if kinds_json.is_null() {
        None
    } else {
        match parse_kinds(kinds_json) {
            Ok(value) => Some(value),
            Err(status) => return status,
        }
    };
    match core_registry_api::list_runtime_registrations(kinds.as_ref()) {
        Ok(registrations) => match serde_json::to_value(registrations) {
            Ok(value) => {
                unsafe { *out_json = json_to_c_string(&value) };
                NemoRelayStatus::Ok
            }
            Err(error) => {
                set_last_error(&error.to_string());
                NemoRelayStatus::Internal
            }
        },
        Err(error) => status_from_error(&error),
    }
}
