// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    NemoRelayEventSubscriberCb, NemoRelayFreeFn, NemoRelayJsonCb, NemoRelayLlmConditionalCb,
    NemoRelayLlmExecInterceptCb, NemoRelayLlmRequestCb, NemoRelayLlmRequestInterceptCb,
    NemoRelayStatus, NemoRelayToolConditionalCb, NemoRelayToolExecInterceptCb,
    NemoRelayToolSanitizeCb, c_char, c_str_to_string, clear_last_error, core_registry_api,
    core_subscriber_api, set_last_error, status_from_error, wrap_event_subscriber,
    wrap_llm_conditional_fn, wrap_llm_exec_intercept_fn, wrap_llm_request_intercept_fn,
    wrap_llm_response_fn, wrap_llm_sanitize_request_fn, wrap_llm_stream_exec_intercept_fn,
    wrap_tool_conditional_fn, wrap_tool_exec_intercept_fn, wrap_tool_request_intercept_fn,
    wrap_tool_sanitize_fn,
};

// ---------------------------------------------------------------------------
// Scope-local tool guardrail registrations
// ---------------------------------------------------------------------------

/// Helper to parse a scope UUID from a C string.
fn parse_scope_uuid(scope_uuid: *const c_char) -> Result<uuid::Uuid, NemoRelayStatus> {
    let uuid_str = c_str_to_string(scope_uuid)?;
    uuid::Uuid::parse_str(&uuid_str).map_err(|e| {
        set_last_error(&format!("invalid scope UUID: {e}"));
        NemoRelayStatus::InvalidArg
    })
}

macro_rules! ffi_scope_guardrail_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $register_name(
            scope_uuid: *const c_char,
            name: *const c_char,
            priority: i32,
            cb: NemoRelayToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoRelayFreeFn,
        ) -> NemoRelayStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&uuid, &name, priority, wrapped) {
                Ok(()) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            scope_uuid: *const c_char,
            name: *const c_char,
        ) -> NemoRelayStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&uuid, &name) {
                Ok(_) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_scope_guardrail_tool_api!(
    /// Register a scope-local tool request sanitization guardrail.
    ///
    /// # Parameters
    /// - `scope_uuid`: UUID of the target scope (null-terminated C string).
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nemo_relay_scope_register_tool_sanitize_request_guardrail,
    /// Deregister a scope-local tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nemo_relay_scope_deregister_tool_sanitize_request_guardrail,
    core_registry_api::scope_register_tool_sanitize_request_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_request_guardrail,
    wrap_tool_sanitize_fn
);

ffi_scope_guardrail_tool_api!(
    /// Register a scope-local tool response sanitization guardrail.
    ///
    /// # Parameters
    /// - `scope_uuid`: UUID of the target scope (null-terminated C string).
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nemo_relay_scope_register_tool_sanitize_response_guardrail,
    /// Deregister a scope-local tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nemo_relay_scope_deregister_tool_sanitize_response_guardrail,
    core_registry_api::scope_register_tool_sanitize_response_guardrail,
    core_registry_api::scope_deregister_tool_sanitize_response_guardrail,
    wrap_tool_sanitize_fn
);

/// Register a scope-local tool conditional execution guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// The callback is fallible. To signal an internal callback failure instead of
/// allow/reject, call [`crate::error::nemo_relay_set_last_error_message`] from C
/// and return null.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_tool_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core_registry_api::scope_register_tool_conditional_execution_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local tool conditional execution guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_tool_conditional_execution_guardrail(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_scope_intercept_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $register_name(
            scope_uuid: *const c_char,
            name: *const c_char,
            priority: i32,
            break_chain: bool,
            cb: NemoRelayToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoRelayFreeFn,
        ) -> NemoRelayStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&uuid, &name, priority, break_chain, wrapped) {
                Ok(()) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            scope_uuid: *const c_char,
            name: *const c_char,
        ) -> NemoRelayStatus {
            clear_last_error();
            let uuid = match parse_scope_uuid(scope_uuid) {
                Ok(u) => u,
                Err(status) => return status,
            };
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&uuid, &name) {
                Ok(_) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_scope_intercept_tool_api!(
    /// Register a scope-local tool request intercept.
    ///
    /// # Parameters
    /// - `scope_uuid`: UUID of the target scope (null-terminated C string).
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// The callback is fallible. To signal failure, call
    /// [`crate::error::nemo_relay_set_last_error_message`] from C and return null.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nemo_relay_scope_register_tool_request_intercept,
    /// Deregister a scope-local tool request intercept by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nemo_relay_scope_deregister_tool_request_intercept,
    core_registry_api::scope_register_tool_request_intercept,
    core_registry_api::scope_deregister_tool_request_intercept,
    wrap_tool_request_intercept_fn
);

/// Register a scope-local tool execution intercept following the middleware
/// chain pattern.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving args and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. Callback pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_tool_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NemoRelayToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_tool_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::scope_register_tool_execution_intercept(&uuid, &name, priority, exec) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local tool execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_tool_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_tool_execution_intercept(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register a scope-local LLM request sanitization guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Request sanitize callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_llm_sanitize_request_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core_registry_api::scope_register_llm_sanitize_request_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM request sanitization guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_llm_sanitize_request_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_llm_sanitize_request_guardrail(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM response sanitization guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: JSON-to-JSON callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_llm_sanitize_response_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match core_registry_api::scope_register_llm_sanitize_response_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM response sanitization guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_llm_sanitize_response_guardrail(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM conditional execution guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// The callback is fallible. To signal an internal callback failure instead of
/// allow/reject, call [`crate::error::nemo_relay_set_last_error_message`] from C
/// and return null.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_llm_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match core_registry_api::scope_register_llm_conditional_execution_guardrail(
        &uuid, &name, priority, wrapped,
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM conditional execution guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_llm_conditional_execution_guardrail(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register a scope-local LLM request intercept.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: LLM request transform callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// The callback is fallible. To signal failure, call
/// [`crate::error::nemo_relay_set_last_error_message`] from C and return null.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_llm_request_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_request_intercept_fn(cb, user_data, free_fn);
    match core_registry_api::scope_register_llm_request_intercept(
        &uuid,
        &name,
        priority,
        break_chain,
        wrapped,
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM request intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_llm_request_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_llm_request_intercept(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM execution intercept following the middleware
/// chain pattern.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. Callback pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_llm_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NemoRelayLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::scope_register_llm_execution_intercept(&uuid, &name, priority, exec) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_llm_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_llm_execution_intercept(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register a scope-local LLM streaming execution intercept following the
/// middleware chain pattern.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. Callback pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_llm_stream_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NemoRelayLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_stream_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::scope_register_llm_stream_execution_intercept(
        &uuid, &name, priority, exec,
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM streaming execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_llm_stream_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::scope_deregister_llm_stream_execution_intercept(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope-local subscriber registrations
// ---------------------------------------------------------------------------

/// Register a scope-local event subscriber.
///
/// # Parameters
/// - `scope_uuid`: UUID of the target scope (null-terminated C string).
/// - `name`: Unique subscriber name.
/// - `cb`: Event callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_register_subscriber(
    scope_uuid: *const c_char,
    name: *const c_char,
    cb: NemoRelayEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match core_subscriber_api::scope_register_subscriber(&uuid, &name, wrapped) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local event subscriber by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_deregister_subscriber(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let uuid = match parse_scope_uuid(scope_uuid) {
        Ok(u) => u,
        Err(status) => return status,
    };
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_subscriber_api::scope_deregister_subscriber(&uuid, &name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}
