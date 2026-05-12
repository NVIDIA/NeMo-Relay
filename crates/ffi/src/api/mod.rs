// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level FFI API functions exported as `extern "C"`.
//!
//! Each function clears the thread-local error before executing and returns an
//! [`NemoFlowStatus`]. On failure, call [`nemo_flow_last_error`] to retrieve
//! the error message.

use std::ffi::CStr;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::callable::{
    NemoFlowCodecDecodeFn, NemoFlowCodecEncodeFn, NemoFlowCollectorCb, NemoFlowEventSubscriberCb,
    NemoFlowFinalizerCb, NemoFlowFreeFn, NemoFlowJsonCb, NemoFlowLlmConditionalCb,
    NemoFlowLlmExecCb, NemoFlowLlmExecInterceptCb, NemoFlowLlmRequestCb,
    NemoFlowLlmRequestInterceptCb, NemoFlowPluginRegisterCb, NemoFlowPluginValidateCb,
    NemoFlowToolConditionalCb, NemoFlowToolExecCb, NemoFlowToolExecInterceptCb,
    NemoFlowToolSanitizeCb, wrap_codec_fn, wrap_collector_fn, wrap_event_subscriber,
    wrap_finalizer_fn, wrap_llm_conditional_fn, wrap_llm_exec_fn, wrap_llm_exec_intercept_fn,
    wrap_llm_request_intercept_fn, wrap_llm_response_fn, wrap_llm_sanitize_request_fn,
    wrap_llm_stream_exec_fn, wrap_llm_stream_exec_intercept_fn, wrap_tool_conditional_fn,
    wrap_tool_exec_fn, wrap_tool_exec_intercept_fn, wrap_tool_request_intercept_fn,
    wrap_tool_sanitize_fn,
};
use crate::convert::{
    c_str_to_json, c_str_to_opt_json, c_str_to_string, json_to_c_string, nemo_flow_string_free,
    str_to_c_string, unix_micros_to_opt_timestamp,
};
use crate::error::{
    NemoFlowStatus, clear_last_error, last_error_message, set_last_error, status_from_error,
    status_from_plugin_error,
};
use crate::types::{
    FfiAtifExporter, FfiAtofExporter, FfiCodecHandle, FfiLLMHandle, FfiOpenInferenceSubscriber,
    FfiOpenTelemetrySubscriber, FfiPluginContext, FfiScopeHandle, FfiScopeStack, FfiToolHandle,
    NemoFlowScopeType,
};
pub use crate::types::{nemo_flow_openinference_subscriber_free, nemo_flow_otel_subscriber_free};
use libc::c_char;
use nemo_flow::api::llm as core_llm_api;
use nemo_flow::api::llm::{LlmAttributes, LlmRequest};
use nemo_flow::api::registry as core_registry_api;
use nemo_flow::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_flow::api::runtime::{
    TASK_SCOPE_STACK, create_scope_stack, current_scope_stack, scope_stack_active,
    set_thread_scope_stack,
};
use nemo_flow::api::scope as core_scope_api;
use nemo_flow::api::scope::ScopeAttributes;
use nemo_flow::api::subscriber as core_subscriber_api;
use nemo_flow::api::tool as core_tool_api;
use nemo_flow::api::tool::ToolAttributes;
use nemo_flow::error::Result as FlowResult;
use nemo_flow::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError,
    PluginRegistrationContext, active_plugin_report, clear_plugin_configuration, deregister_plugin,
    initialize_plugins, list_plugin_kinds, register_plugin, validate_plugin_config,
};
use nemo_flow_adaptive::plugin_component::register_adaptive_component;
use tokio::runtime::Runtime;

mod llm;
mod llm_registry;
mod observability;
mod plugin;
mod scope;
mod scope_registry;
mod scope_stack;
mod tool_lifecycle;
mod tool_registry;

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
///   with `nemo_flow_string_free`).
///
/// # Returns
/// Returns [`NemoFlowStatus::Ok`] on success and writes the transformed JSON
/// string to `out`.
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_request_intercepts(
    name: *const c_char,
    args_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoFlowStatus::InvalidJson,
    };
    match core_tool_api::tool_request_intercepts(&name, args) {
        Ok(result) => {
            unsafe { *out = json_to_c_string(&result) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered tool conditional execution guardrail chain.
///
/// Returns `NemoFlowStatus::Ok` if all guardrails pass, or
/// `NemoFlowStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
///
/// # Returns
/// Returns [`NemoFlowStatus::Ok`] when execution is allowed and
/// [`NemoFlowStatus::GuardrailRejected`] when a guardrail blocks the call.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_conditional_execution(
    name: *const c_char,
    args_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoFlowStatus::InvalidJson,
    };
    match core_tool_api::tool_conditional_execution(&name, &args) {
        Ok(()) => NemoFlowStatus::Ok,
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
///   with `nemo_flow_string_free`). The output is a serialized `LlmRequest`.
///
/// # Returns
/// Returns [`NemoFlowStatus::Ok`] on success and writes the transformed
/// serialized request to `out`.
///
/// # Safety
/// All pointers must be valid. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_request_intercepts(
    name: *const c_char,
    native_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name_str = if name.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default()
    };
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LlmRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LlmRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    match core_llm_api::llm_request_intercepts(name_str, request) {
        Ok(transformed) => {
            let result_json = serde_json::to_value(&transformed).unwrap_or(serde_json::Value::Null);
            unsafe { *out = json_to_c_string(&result_json) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Run the registered LLM conditional execution guardrail chain.
///
/// Returns `NemoFlowStatus::Ok` if all guardrails pass, or
/// `NemoFlowStatus::GuardrailRejected` if blocked.
///
/// # Parameters
/// - `native_json`: The request payload as a JSON C string representing an
///   `LlmRequest` (`{"headers": {...}, "content": {...}}`).
///
/// # Returns
/// Returns [`NemoFlowStatus::Ok`] when execution is allowed and
/// [`NemoFlowStatus::GuardrailRejected`] when a guardrail blocks the call.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_conditional_execution(
    native_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let native = match c_str_to_json(native_json) {
        Some(j) => j,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LlmRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LlmRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    match core_llm_api::llm_conditional_execution(&request) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/api_tests.rs"]
mod tests;
