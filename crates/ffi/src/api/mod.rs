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

use libc::c_char;
use nemo_flow::api::llm as core_llm_api;
use nemo_flow::api::registry as core_registry_api;
use nemo_flow::api::scope as core_scope_api;
use nemo_flow::api::subscriber as core_subscriber_api;
use nemo_flow::api::tool as core_tool_api;
use nemo_flow::context::callbacks::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn,
};
use nemo_flow::context::scope_stack::{
    TASK_SCOPE_STACK, create_scope_stack, current_scope_stack, scope_stack_active,
    set_thread_scope_stack,
};
use nemo_flow::error::Result as FlowResult;
use nemo_flow::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginConfig, PluginError,
    PluginRegistrationContext, active_plugin_report, clear_plugin_configuration, deregister_plugin,
    initialize_plugins, list_plugin_kinds, register_plugin, validate_plugin_config,
};
use nemo_flow::types::llm::{LLMAttributes, LLMRequest};
use nemo_flow::types::scope::ScopeAttributes;
use nemo_flow::types::tool::ToolAttributes;
use nemo_flow_adaptive::plugin_component::register_adaptive_component;
use tokio::runtime::Runtime;
use tokio_stream::StreamExt;

use crate::callable::*;
use crate::convert::*;
use crate::error::*;
use crate::types::*;

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
/// # Parameters
/// - `name`: Tool name (null-terminated C string).
/// - `args_json`: Tool arguments as a JSON C string.
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nemo_flow_string_free`).
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
/// # Parameters
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `out`: On success, receives the transformed JSON string (caller must free
///   with `nemo_flow_string_free`). The output is a serialized `LLMRequest`.
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
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
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
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
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
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
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
