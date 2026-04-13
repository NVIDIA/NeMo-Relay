// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register an LLM request sanitization guardrail. The callback can modify or
/// replace the LLM request before it is sent.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Request sanitize callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_llm_sanitize_request_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core_registry_api::register_llm_sanitize_request_guardrail(&name, priority, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_llm_sanitize_request_guardrail(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_llm_sanitize_request_guardrail(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM response sanitization guardrail. The callback can inspect
/// and modify the LLM response after it is received.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: JSON-to-JSON callback that receives the response JSON and returns sanitized JSON.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_llm_sanitize_response_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoFlowJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match core_registry_api::register_llm_sanitize_response_guardrail(&name, priority, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM response sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_llm_sanitize_response_guardrail(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_llm_sanitize_response_guardrail(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM conditional execution guardrail. The callback decides
/// whether an LLM call should proceed.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback. Returns null to allow, or error message to reject.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// The callback is fallible. To signal an internal callback failure instead of
/// allow/reject, call [`crate::error::nemo_flow_set_last_error_message`] from C
/// and return null.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_llm_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match core_registry_api::register_llm_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_llm_conditional_execution_guardrail(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_llm_conditional_execution_guardrail(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept. The callback can transform the
/// `LLMRequest` before it reaches the LLM provider.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: LLM request transform callback (receives/returns `FfiLLMRequest`).
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// The callback is fallible. To signal failure, call
/// [`crate::error::nemo_flow_set_last_error_message`] from C and return null.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_llm_request_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoFlowLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_request_intercept_fn(cb, user_data, free_fn);
    match core_registry_api::register_llm_request_intercept(&name, priority, break_chain, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_llm_request_intercept(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_llm_request_intercept(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM execution intercept following the middleware chain pattern.
/// The callback receives `(request, next_fn, next_ctx)` — call
/// `next_fn(request, next_ctx)` to invoke the next intercept or the original
/// LLM call, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_llm_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NemoFlowLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::register_llm_execution_intercept(&name, priority, exec) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_llm_execution_intercept(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_llm_execution_intercept(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM streaming execution intercept following the middleware chain
/// pattern. The callback receives `(request, next_fn, next_ctx)` — call
/// `next_fn(request, next_ctx)` to invoke the next intercept or the original
/// streaming LLM call, or skip calling it to short-circuit.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `exec_cb`: Middleware callback receiving request and a next function.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_llm_stream_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NemoFlowLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_llm_stream_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::register_llm_stream_execution_intercept(&name, priority, exec) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM streaming execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_llm_stream_execution_intercept(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_llm_stream_execution_intercept(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register an event subscriber. The callback is invoked for every lifecycle
/// event emitted by the runtime.
///
/// # Parameters
/// - `name`: Unique subscriber name.
/// - `cb`: Event callback. The `FfiEvent` is valid only during the call.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_subscriber(
    name: *const c_char,
    cb: NemoFlowEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match core_subscriber_api::register_subscriber(&name, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an event subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_subscriber(name: *const c_char) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}
