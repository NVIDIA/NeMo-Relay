// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level FFI API functions exported as `extern "C"`.
//!
//! Each function clears the thread-local error before executing and returns an
//! [`NemoRelayStatus`]. On failure, call [`nemo_relay_last_error`] to retrieve
//! the error message.

use std::ffi::CStr;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::callable::{
    NemoRelayCodecDecodeFn, NemoRelayCodecEncodeFn, NemoRelayCollectorCb, NemoRelayEventSanitizeCb,
    NemoRelayEventSubscriberCb, NemoRelayFinalizerCb, NemoRelayFreeFn, NemoRelayLlmConditionalCb,
    NemoRelayLlmExecCb, NemoRelayLlmExecInterceptCb, NemoRelayLlmRequestInterceptCb,
    NemoRelayLlmSanitizeRequestCb, NemoRelayLlmSanitizeResponseCb, NemoRelayPluginRegisterCb,
    NemoRelayPluginValidateCb, NemoRelayToolConditionalCb, NemoRelayToolExecCb,
    NemoRelayToolExecInterceptCb, NemoRelayToolSanitizeCb, wrap_codec_fn, wrap_collector_fn,
    wrap_event_sanitize_fn, wrap_event_subscriber, wrap_finalizer_fn, wrap_llm_conditional_fn,
    wrap_llm_exec_fn, wrap_llm_exec_intercept_fn, wrap_llm_request_intercept_fn,
    wrap_llm_sanitize_request_fn, wrap_llm_sanitize_response_fn, wrap_llm_stream_exec_fn,
    wrap_llm_stream_exec_intercept_fn, wrap_tool_conditional_fn, wrap_tool_exec_fn,
    wrap_tool_exec_intercept_fn, wrap_tool_request_intercept_fn, wrap_tool_sanitize_fn,
};
use crate::convert::{
    c_str_to_json, c_str_to_opt_json, c_str_to_string, json_to_c_string, nemo_relay_string_free,
    str_to_c_string, unix_micros_to_opt_timestamp,
};
use crate::error::{
    NemoRelayStatus, clear_last_error, last_error_message, set_last_error, status_from_error,
    status_from_plugin_error,
};
use crate::types::{
    FfiAtifExporter, FfiAtofExporter, FfiCodecHandle, FfiLLMHandle, FfiLLMRequest,
    FfiOpenInferenceSubscriber, FfiOpenTelemetrySubscriber, FfiPluginActivation, FfiPluginContext,
    FfiScopeHandle, FfiScopeStack, FfiThreadScopeStackBinding, FfiToolHandle, NemoRelayScopeType,
};
pub use crate::types::{nemo_relay_openinference_subscriber_free, nemo_relay_otel_subscriber_free};
use libc::c_char;
use nemo_relay::api::llm as core_llm_api;
use nemo_relay::api::llm::{LlmAttributes, LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::registry as core_registry_api;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::runtime::{
    TASK_SCOPE_STACK, capture_thread_scope_stack, create_scope_stack, current_scope_stack,
    restore_thread_scope_stack, scope_stack_active, set_thread_scope_stack,
};
use nemo_relay::api::scope as core_scope_api;
use nemo_relay::api::scope::ScopeAttributes;
use nemo_relay::api::subscriber as core_subscriber_api;
use nemo_relay::api::tool as core_tool_api;
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::error::Result as FlowResult;
use nemo_relay::plugin::dynamic::{DynamicPluginActivationSpec, PluginHostActivation};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError,
    PluginRegistrationContext, active_plugin_report, clear_plugin_configuration, deregister_plugin,
    initialize_plugins, list_plugin_kinds, register_plugin, validate_plugin_config,
};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use tokio::runtime::Runtime;

mod adaptive;
mod event_registry;
mod llm;
mod llm_registry;
mod observability;
mod plugin;
mod scope;
mod scope_registry;
mod scope_stack;
mod tool_lifecycle;
mod tool_registry;

pub use adaptive::*;
pub use event_registry::*;
pub use llm::*;
pub use llm_registry::*;
pub use observability::*;
pub use plugin::*;
pub use scope::*;
pub use scope_registry::*;
pub use scope_stack::*;
pub use tool_lifecycle::*;
pub use tool_registry::*;

fn tokio_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

// ---------------------------------------------------------------------------
// Standalone middleware chains
// ---------------------------------------------------------------------------

/// Run the registered tool request intercept chain on the given arguments.
///
/// This helper applies only the request-intercept middleware and does not emit
/// lifecycle events or execute the tool callback.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nemo_relay_string_free`).
///
/// # Returns
/// Returns [`NemoRelayStatus::Ok`] on success and writes the transformed JSON
/// string to `out`.
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_request_intercepts(
    name: *const c_char,
    args_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = std::ptr::null_mut() };

    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoRelayStatus::InvalidJson,
    };
    match core_tool_api::tool_request_intercepts(&name, args) {
        Ok(result) => {
            unsafe { *out = json_to_c_string(&result) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered tool conditional execution guardrail chain.
///
/// Returns `NemoRelayStatus::Ok` if all guardrails pass, or
/// `NemoRelayStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
///
/// # Returns
/// Returns [`NemoRelayStatus::Ok`] when execution is allowed and
/// [`NemoRelayStatus::GuardrailRejected`] when a guardrail blocks the call.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_conditional_execution(
    name: *const c_char,
    args_json: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoRelayStatus::InvalidJson,
    };
    match core_tool_api::tool_conditional_execution(&name, &args) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM request intercept chain on the given request.
///
/// This helper applies only the request-intercept middleware and does not emit
/// lifecycle events or execute the provider callback.
///
/// # Parameters
/// - `name`: Optional provider name as a null-terminated C string. Pass null to
///   use an empty logical name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LlmRequest` (`{"headers": {...}, "content": {...}}`).
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nemo_relay_string_free`). The output is a serialized `LlmRequest`.
///
/// # Returns
/// Returns [`NemoRelayStatus::Ok`] on success and writes the transformed
/// serialized request to `out`.
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_intercepts(
    name: *const c_char,
    native_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = std::ptr::null_mut() };

    let name_str = if name.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default()
    };
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NemoRelayStatus::InvalidJson,
    };
    let request: LlmRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LlmRequest");
            return NemoRelayStatus::InvalidJson;
        }
    };
    match core_llm_api::llm_request_intercepts(name_str, request) {
        Ok(transformed) => {
            let result_json = serde_json::to_value(&transformed).unwrap_or(serde_json::Value::Null);
            unsafe { *out = json_to_c_string(&result_json) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Allocate canonical JSON for a C LLM request-intercept callback result.
///
/// `annotated_json` may be null. `pending_marks_json` may be null, in which
/// case an empty list is serialized. When used by a
/// `NemoRelayLlmRequestInterceptCb`, assign the successful output to the
/// callback's `out_outcome_json`; ownership transfers to Relay when the
/// callback returns, so the callback must not free or reuse it. Outside a
/// callback, the caller owns the returned string and must release it with
/// `nemo_relay_string_free`.
///
/// # Safety
///
/// `request` must point to a live `FfiLLMRequest`, optional JSON inputs must
/// be valid null-terminated strings when non-null, and `out_outcome_json` must
/// be writable. A successful output must either be transferred through a
/// callback's `out_outcome_json` or freed by its caller with
/// `nemo_relay_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_intercept_outcome_json_new(
    request: *const FfiLLMRequest,
    annotated_json: *const c_char,
    pending_marks_json: *const c_char,
    out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_llm_request_intercept_outcome_json_new_v2(
            request,
            annotated_json,
            pending_marks_json,
            std::ptr::null(),
            out_outcome_json,
        )
    }
}

/// Allocate canonical JSON for a C LLM request-intercept callback result,
/// including optional plugin-neutral optimization contributions.
///
/// `annotated_json`, `pending_marks_json`, and
/// `optimization_contributions_json` may be null. Null list pointers serialize
/// as empty lists. Contributions use the canonical
/// `LlmOptimizationContribution` JSON shape; custom `kind` strings and unknown
/// top-level fields are preserved. The existing unversioned helper remains
/// ABI-compatible and behaves as though this function received a null
/// `optimization_contributions_json` pointer.
///
/// # Safety
///
/// `request` must point to a live `FfiLLMRequest`, optional JSON inputs must
/// be valid null-terminated strings when non-null, and `out_outcome_json` must
/// be writable. A successful output must either be transferred through a
/// callback's `out_outcome_json` or freed by its caller with
/// `nemo_relay_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_intercept_outcome_json_new_v2(
    request: *const FfiLLMRequest,
    annotated_json: *const c_char,
    pending_marks_json: *const c_char,
    optimization_contributions_json: *const c_char,
    out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_outcome_json.is_null() {
        set_last_error("out_outcome_json must be non-null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_outcome_json = std::ptr::null_mut() };
    if request.is_null() {
        set_last_error("request must be non-null");
        return NemoRelayStatus::NullPointer;
    }
    let annotated_request =
        match unsafe { parse_optional_intercept_json(annotated_json, "annotated request") } {
            Ok(value) => value,
            Err(status) => return status,
        };
    let pending_marks =
        match unsafe { parse_optional_intercept_json(pending_marks_json, "pending marks") } {
            Ok(value) => value.unwrap_or_default(),
            Err(status) => return status,
        };
    let optimization_contributions = match unsafe {
        parse_optional_intercept_json(
            optimization_contributions_json,
            "optimization contributions",
        )
    } {
        Ok(value) => value.unwrap_or_default(),
        Err(status) => return status,
    };
    let outcome = LlmRequestInterceptOutcome {
        request: unsafe { &*request }.0.clone(),
        annotated_request,
        pending_marks,
        optimization_contributions,
    };
    match serde_json::to_value(outcome) {
        Ok(value) => {
            unsafe { *out_outcome_json = json_to_c_string(&value) };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&format!("failed to serialize intercept outcome: {error}"));
            NemoRelayStatus::Internal
        }
    }
}

unsafe fn parse_optional_intercept_json<T: serde::de::DeserializeOwned>(
    input: *const c_char,
    description: &str,
) -> std::result::Result<Option<T>, NemoRelayStatus> {
    if input.is_null() {
        return Ok(None);
    }
    let value = c_str_to_json(input).ok_or(NemoRelayStatus::InvalidJson)?;
    serde_json::from_value(value).map(Some).map_err(|error| {
        set_last_error(&format!("invalid {description} JSON: {error}"));
        NemoRelayStatus::InvalidJson
    })
}

/// Run the registered LLM conditional execution guardrail chain.
///
/// Returns `NemoRelayStatus::Ok` if all guardrails pass, or
/// `NemoRelayStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `native_json`: The request payload as a JSON C string representing an
///   `LlmRequest` (`{"headers": {...}, "content": {...}}`).
///
/// # Returns
/// Returns [`NemoRelayStatus::Ok`] when execution is allowed and
/// [`NemoRelayStatus::GuardrailRejected`] when a guardrail blocks the call.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_conditional_execution(
    native_json: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NemoRelayStatus::InvalidJson,
    };
    let request: LlmRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LlmRequest");
            return NemoRelayStatus::InvalidJson;
        }
    };
    match core_llm_api::llm_conditional_execution(&request) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/api_tests.rs"]
mod tests;
