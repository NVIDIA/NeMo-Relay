// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    NemoRelayFreeFn, NemoRelayStatus, NemoRelayToolConditionalCb, NemoRelayToolExecInterceptCb,
    NemoRelayToolSanitizeCb, c_char, c_str_to_string, clear_last_error, core_registry_api,
    status_from_error, wrap_tool_conditional_fn, wrap_tool_exec_intercept_fn,
    wrap_tool_request_intercept_fn, wrap_tool_sanitize_fn,
};

// ---------------------------------------------------------------------------
// Tool guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_guardrail_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            cb: NemoRelayToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoRelayFreeFn,
        ) -> NemoRelayStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, wrapped) {
                Ok(()) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NemoRelayStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_guardrail_tool_api!(
    /// Register a tool request sanitization guardrail. The callback can inspect
    /// and modify tool arguments before the tool executes.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback that receives tool name and args JSON, returns sanitized args JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nemo_relay_register_tool_sanitize_request_guardrail,
    /// Deregister a tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nemo_relay_deregister_tool_sanitize_request_guardrail,
    core_registry_api::register_tool_sanitize_request_guardrail,
    core_registry_api::deregister_tool_sanitize_request_guardrail,
    wrap_tool_sanitize_fn
);

ffi_guardrail_tool_api!(
    /// Register a tool response sanitization guardrail. The callback can inspect
    /// and modify tool results after the tool executes.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback that receives tool name and result JSON, returns sanitized result JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nemo_relay_register_tool_sanitize_response_guardrail,
    /// Deregister a tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nemo_relay_deregister_tool_sanitize_response_guardrail,
    core_registry_api::register_tool_sanitize_response_guardrail,
    core_registry_api::deregister_tool_sanitize_response_guardrail,
    wrap_tool_sanitize_fn
);

/// Register a tool conditional execution guardrail. The callback decides whether
/// a tool call should proceed. Returns an error message to reject, or null to allow.
///
/// # Parameters
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
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_tool_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoRelayToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core_registry_api::register_tool_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_tool_conditional_execution_guardrail(
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_tool_conditional_execution_guardrail(&name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_intercept_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            break_chain: bool,
            cb: NemoRelayToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoRelayFreeFn,
        ) -> NemoRelayStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, break_chain, wrapped) {
                Ok(()) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NemoRelayStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NemoRelayStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_intercept_tool_api!(
    /// Register a tool request intercept. The callback can transform tool
    /// arguments before execution. Runs after request guardrails in the
    /// middleware pipeline.
    ///
    /// # Parameters
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback that receives tool name and args JSON, returns modified args JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// The callback is fallible. To signal failure, call
    /// [`crate::error::nemo_relay_set_last_error_message`] from C and return null.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nemo_relay_register_tool_request_intercept,
    /// Deregister a tool request intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nemo_relay_deregister_tool_request_intercept,
    core_registry_api::register_tool_request_intercept,
    core_registry_api::deregister_tool_request_intercept,
    wrap_tool_request_intercept_fn
);

/// Register a tool execution intercept following the middleware chain pattern.
/// The callback receives `(args, next_fn, next_ctx)` — call
/// `next_fn(args, next_ctx)` to invoke the next intercept or the original
/// tool function, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving args and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_tool_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NemoRelayToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_tool_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::register_tool_execution_intercept(&name, priority, exec) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_tool_execution_intercept(
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_tool_execution_intercept(&name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}
