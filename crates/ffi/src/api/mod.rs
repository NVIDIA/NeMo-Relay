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

// ---------------------------------------------------------------------------
// Tokio runtime singleton (for blocking on async functions)
// ---------------------------------------------------------------------------

fn tokio_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

struct FfiHostedPluginUserData {
    ptr: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
}

unsafe impl Send for FfiHostedPluginUserData {}
unsafe impl Sync for FfiHostedPluginUserData {}

impl Drop for FfiHostedPluginUserData {
    fn drop(&mut self) {
        if let Some(free_fn) = self.free_fn {
            unsafe { free_fn(self.ptr) };
        }
    }
}

struct FfiHostedPluginAdapter {
    plugin_kind: String,
    validate_cb: Option<NemoFlowPluginValidateCb>,
    register_cb: NemoFlowPluginRegisterCb,
    user_data: Arc<FfiHostedPluginUserData>,
}

impl Plugin for FfiHostedPluginAdapter {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn validate(
        &self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
    ) -> Vec<ConfigDiagnostic> {
        let Some(validate_cb) = self.validate_cb else {
            return vec![];
        };

        clear_last_error();
        let plugin_config_json =
            json_to_c_string(&serde_json::Value::Object(plugin_config.clone()));
        let result_ptr = unsafe { validate_cb(self.user_data.ptr, plugin_config_json) };
        unsafe { nemo_flow_string_free(plugin_config_json) };

        if result_ptr.is_null() {
            let message = last_error_message().unwrap_or_else(|| {
                format!(
                    "hosted plugin '{}' validate callback returned null",
                    self.plugin_kind
                )
            });
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".to_string(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message,
            }];
        }

        let diagnostics = unsafe { CStr::from_ptr(result_ptr) }
            .to_str()
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<ConfigDiagnostic>>(text).ok());
        unsafe { nemo_flow_string_free(result_ptr) };
        diagnostics.unwrap_or_else(|| {
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".to_string(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message: format!(
                    "hosted plugin '{}' validate callback returned invalid diagnostics JSON",
                    self.plugin_kind
                ),
            }]
        })
    }

    fn register<'a>(
        &'a self,
        plugin_config: &serde_json::Map<String, serde_json::Value>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), PluginError>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            clear_last_error();
            let plugin_config_json = json_to_c_string(&serde_json::Value::Object(plugin_config));
            let mut ffi_ctx = FfiPluginContext(ctx as *mut _);
            let status =
                unsafe { (self.register_cb)(self.user_data.ptr, plugin_config_json, &mut ffi_ctx) };
            unsafe { nemo_flow_string_free(plugin_config_json) };
            if status == NemoFlowStatus::Ok {
                Ok(())
            } else if let Some(message) = last_error_message() {
                Err(PluginError::RegistrationFailed(message))
            } else {
                Err(PluginError::RegistrationFailed(format!(
                    "hosted plugin '{}' register callback failed with status {:?}",
                    self.plugin_kind, status
                )))
            }
        })
    }
}

fn ensure_adaptive_component_registered() -> std::result::Result<(), NemoFlowStatus> {
    register_adaptive_component().map_err(|err| status_from_plugin_error(&err))
}

/// Validate a generic plugin config document and return the diagnostics report as JSON.
///
/// # Safety
/// `config_json` must be a valid C string and `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_validate_plugin_config(
    config_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    let config_value = match c_str_to_json(config_json) {
        Some(value) => value,
        None => return NemoFlowStatus::InvalidJson,
    };
    let config: PluginConfig = match serde_json::from_value(config_value) {
        Ok(config) => config,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoFlowStatus::InvalidJson;
        }
    };
    let report_json = match serde_json::to_value(validate_plugin_config(&config)) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoFlowStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&report_json) };
    NemoFlowStatus::Ok
}

/// Initialize the active global plugin components and return the resulting diagnostics report.
///
/// # Safety
/// `config_json` must be a valid C string and `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_initialize_plugins(
    config_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    let config_value = match c_str_to_json(config_json) {
        Some(value) => value,
        None => return NemoFlowStatus::InvalidJson,
    };
    let config: PluginConfig = match serde_json::from_value(config_value) {
        Ok(config) => config,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoFlowStatus::InvalidJson;
        }
    };
    let report = match tokio_runtime().block_on(initialize_plugins(config)) {
        Ok(report) => report,
        Err(err) => return status_from_plugin_error(&err),
    };
    let report_json = match serde_json::to_value(report) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoFlowStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&report_json) };
    NemoFlowStatus::Ok
}

/// Clear the active global plugin configuration.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_clear_plugin_configuration() -> NemoFlowStatus {
    clear_last_error();
    match clear_plugin_configuration() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Return the last successfully configured plugin report as JSON.
///
/// # Safety
/// `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_active_plugin_report_json(
    out_json: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let report_json = match serde_json::to_value(active_plugin_report()) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoFlowStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&report_json) };
    NemoFlowStatus::Ok
}

/// Return the registered plugin kinds as JSON.
///
/// # Safety
/// `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_list_plugin_kinds_json(
    out_json: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    let kinds_json = match serde_json::to_value(list_plugin_kinds()) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoFlowStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&kinds_json) };
    NemoFlowStatus::Ok
}

/// Register a plugin backed by foreign callbacks.
///
/// # Safety
/// `plugin_kind` must be a valid C string and `register_cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_plugin(
    plugin_kind: *const c_char,
    validate_cb: Option<NemoFlowPluginValidateCb>,
    register_cb: NemoFlowPluginRegisterCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let plugin_kind = match c_str_to_string(plugin_kind) {
        Ok(value) => value,
        Err(status) => return status,
    };

    let plugin = Arc::new(FfiHostedPluginAdapter {
        plugin_kind: plugin_kind.clone(),
        validate_cb,
        register_cb,
        user_data: Arc::new(FfiHostedPluginUserData {
            ptr: user_data,
            free_fn,
        }),
    });
    match register_plugin(plugin) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Deregister a plugin by kind.
///
/// # Safety
/// `plugin_kind` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_plugin(plugin_kind: *const c_char) -> NemoFlowStatus {
    clear_last_error();
    let plugin_kind = match c_str_to_string(plugin_kind) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if deregister_plugin(&plugin_kind) {
        NemoFlowStatus::Ok
    } else {
        set_last_error(&format!("not found: plugin '{plugin_kind}'"));
        NemoFlowStatus::NotFound
    }
}

/// Register an event subscriber into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_subscriber(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    cb: NemoFlowEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_subscriber(&name, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool sanitize-request guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_tool_sanitize_request_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_sanitize_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_tool_sanitize_request_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool sanitize-response guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_tool_sanitize_response_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_sanitize_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_tool_sanitize_response_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool conditional-execution guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_tool_conditional_execution_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_tool_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM sanitize-request guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_llm_sanitize_request_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_sanitize_request_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM sanitize-response guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_llm_sanitize_response_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_sanitize_response_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM conditional-execution guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_llm_conditional_execution_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM request intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_llm_request_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoFlowLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_request_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_llm_request_intercept(
        &name,
        priority,
        break_chain,
        wrapped,
    ) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool request intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_tool_request_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoFlowToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_request_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_tool_request_intercept(
        &name,
        priority,
        break_chain,
        wrapped,
    ) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM execution intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_llm_execution_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_exec_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_llm_execution_intercept(&name, priority, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM stream execution intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_llm_stream_execution_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_stream_exec_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_stream_execution_intercept(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool execution intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the hosted plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_plugin_context_register_tool_execution_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowToolExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_exec_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_tool_execution_intercept(&name, priority, wrapped) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Retrieve the current scope handle from the thread-local scope stack.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must be
///   freed with `nemo_flow_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_get_handle(out: *mut *mut FfiScopeHandle) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    match core_scope_api::get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Push a new scope onto the scope stack.
///
/// # Parameters
/// - `name`: Null-terminated scope name.
/// - `scope_type`: The type of scope to create.
/// - `parent`: Optional parent scope handle, or null for auto-parenting.
/// - `attributes`: Bitfield of scope attributes.
/// - `data_json`: Optional null-terminated JSON string for scope data, or null.
/// - `metadata_json`: Optional null-terminated JSON string for scope metadata, or null.
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle`.
///
/// # Safety
/// `name` must be a valid C string. `out` must be non-null. `parent`,
/// `data_json`, and `metadata_json` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_push_scope(
    name: *const c_char,
    scope_type: NemoFlowScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut FfiScopeHandle,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ScopeAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_scope_api::push_scope(&name, scope_type.into(), parent_ref, attrs, data, metadata) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Pop a scope from the scope stack by its handle.
///
/// # Parameters
/// - `handle`: The scope handle to pop.
///
/// # Safety
/// `handle` must be a valid, non-null `FfiScopeHandle` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_pop_scope(handle: *const FfiScopeHandle) -> NemoFlowStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoFlowStatus::NullPointer;
    }
    match core_scope_api::pop_scope(&unsafe { &*handle }.0.uuid) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Emit a named lifecycle event.
///
/// # Parameters
/// - `name`: Null-terminated event name.
/// - `parent`: Optional parent scope handle, or null.
/// - `data_json`: Optional JSON data payload, or null.
/// - `metadata_json`: Optional JSON metadata payload, or null.
///
/// # Safety
/// `name` must be a valid C string. Other pointer args may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_scope_api::event(&name, parent_ref, data, metadata) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a tool call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a JSON C string.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `tool_call_id`: Optional external correlation ID for the tool call, or null.
/// - `out`: On success, receives a heap-allocated `FfiToolHandle`.
///
/// # Safety
/// `name` and `args_json` must be valid C strings. `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    tool_call_id: *const c_char,
    out: *mut *mut FfiToolHandle,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoFlowStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let tool_call_id_opt = if tool_call_id.is_null() {
        None
    } else {
        match c_str_to_string(tool_call_id) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core_tool_api::tool_call(
        &name,
        args,
        parent_ref,
        attrs,
        data,
        metadata,
        tool_call_id_opt,
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiToolHandle(h))) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a tool call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The tool handle from `nemo_flow_tool_call`.
/// - `result_json`: Tool result as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `result_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoFlowStatus::NullPointer;
    }
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NemoFlowStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_tool_api::tool_call_end(&unsafe { &*handle }.0, result, data, metadata) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute a tool call end-to-end: run conditional-execution guardrails (on raw
/// args), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, and sanitize-response
/// guardrails. On rejection, only a standalone Mark event is emitted (no
/// Start/End pair) and `GuardrailRejected` is returned. Blocks the calling
/// thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a JSON C string.
/// - `func`: C callback that performs the actual tool execution.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives the result as a JSON C string. Caller must free
///   with `nemo_flow_string_free`.
///
/// # Safety
/// `name`, `args_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_tool_call_execute(
    name: *const c_char,
    args_json: *const c_char,
    func: NemoFlowToolExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoFlowFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NemoFlowStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    let exec_fn = wrap_tool_exec_fn(func, func_user_data, func_free);
    let default_fn: ToolExecutionNextFn = Arc::new(move |args| exec_fn(args));

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_tool_api::tool_call_execute(
            &name,
            args,
            default_fn,
            parent_handle,
            attrs,
            data,
            metadata,
        )
        .await
    }));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives a heap-allocated `FfiLLMHandle`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_call(
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiLLMHandle,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core_llm_api::llm_call(
        &name,
        &request,
        parent_ref,
        attrs,
        data,
        metadata,
        model_name_opt,
        None,
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiLLMHandle(h))) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End an LLM call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The LLM handle from `nemo_flow_llm_call`.
/// - `response_json`: LLM response as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `response_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoFlowStatus::NullPointer;
    }
    let response = match c_str_to_json(response_json) {
        Some(r) => r,
        None => return NemoFlowStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_llm_api::llm_call_end(&unsafe { &*handle }.0, response, data, metadata, None) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Built-in codec constructors
// ---------------------------------------------------------------------------

/// Create a new OpenAI Chat Completions codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_flow_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_flow_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_openai_chat_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
        response_codec: Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
    }))
}

/// Create a new OpenAI Responses API codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_flow_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_flow_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_openai_responses_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_flow::codec::openai_responses::OpenAIResponsesCodec),
        response_codec: Arc::new(nemo_flow::codec::openai_responses::OpenAIResponsesCodec),
    }))
}

/// Create a new Anthropic Messages API codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_flow_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_flow_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_anthropic_messages_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
        response_codec: Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
    }))
}

/// Execute an LLM call end-to-end: run conditional-execution guardrails (on raw
/// request), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, and sanitize-response
/// guardrails. On rejection, only a standalone Mark event is emitted (no
/// Start/End pair) and `GuardrailRejected` is returned. Blocks the calling
/// thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives the response as a JSON C string. Caller must
///   free with `nemo_flow_string_free`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NemoFlowLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoFlowFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoFlowCodecDecodeFn,
    codec_encode: NemoFlowCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoFlowFreeFn,
    response_codec: *const FfiCodecHandle,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let codec = match (codec_decode, codec_encode) {
        (Some(decode_cb), Some(encode_cb)) => Some(wrap_codec_fn(
            decode_cb,
            encode_cb,
            codec_user_data,
            codec_free_fn,
        )),
        _ => None,
    };
    let response_codec = if response_codec.is_null() {
        None
    } else {
        Some(unsafe { &*response_codec }.response_codec.clone())
    };

    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);
    let default_fn: LlmExecutionNextFn = Arc::new(move |request| exec_fn(request));

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_llm_api::llm_call_execute(
            &name,
            request,
            default_fn,
            parent_handle,
            attrs,
            data,
            metadata,
            model_name_opt,
            codec,
            response_codec,
        )
        .await
    }));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nemo_flow_stream_next` to poll and `nemo_flow_stream_free` to release.
pub struct FfiStream {
    receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>>,
}

/// Execute a streaming LLM call end-to-end. Conditional-execution guardrails
/// run first on the raw request. Returns a stream handle that can be polled
/// with `nemo_flow_stream_next`. Blocks until the stream is set up.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `collector`: Callback invoked with each intercepted chunk as a JSON string.
///   May be null, in which case chunks are not collected.
/// - `finalizer`: Callback invoked once when the stream is exhausted to produce
///   the aggregated response as a JSON C string. May be null, in which case the
///   finalizer returns `Json::Null`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives a heap-allocated `FfiStream`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers. `collector`
/// and `finalizer` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_stream_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NemoFlowLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoFlowFreeFn,
    collector: Option<NemoFlowCollectorCb>,
    finalizer: Option<NemoFlowFinalizerCb>,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoFlowCodecDecodeFn,
    codec_encode: NemoFlowCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoFlowFreeFn,
    response_codec: *const FfiCodecHandle,
    out: *mut *mut FfiStream,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let codec = match (codec_decode, codec_encode) {
        (Some(decode_cb), Some(encode_cb)) => Some(wrap_codec_fn(
            decode_cb,
            encode_cb,
            codec_user_data,
            codec_free_fn,
        )),
        _ => None,
    };
    let response_codec = if response_codec.is_null() {
        None
    } else {
        Some(unsafe { &*response_codec }.response_codec.clone())
    };

    let exec_fn = wrap_llm_stream_exec_fn(func, func_user_data, func_free);
    let default_fn: LlmStreamExecutionNextFn = Arc::new(move |request| exec_fn(request));

    let wrapped_collector: Box<dyn FnMut(serde_json::Value) -> FlowResult<()> + Send> =
        match collector {
            Some(cb) => wrap_collector_fn(cb),
            None => Box::new(|_: serde_json::Value| Ok(())),
        };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => wrap_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_llm_api::llm_stream_call_execute(
            &name,
            request,
            default_fn,
            wrapped_collector,
            wrapped_finalizer,
            parent_handle,
            attrs,
            data,
            metadata,
            model_name_opt,
            codec,
            response_codec,
        )
        .await
    }));

    match result {
        Ok(rust_stream) => {
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            tokio_runtime().spawn(async move {
                let mut stream = rust_stream;
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            });
            let ffi_stream = Box::new(FfiStream {
                receiver: tokio::sync::Mutex::new(rx),
            });
            unsafe { *out = Box::into_raw(ffi_stream) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Poll the next chunk from a streaming LLM response. Blocks until a chunk is
/// available.
///
/// # Returns
/// - `1`: A chunk was written to `*out_chunk`. Caller must free with
///   `nemo_flow_string_free`.
/// - `0`: The stream is complete (no more chunks).
/// - `-1`: An error occurred. Call `nemo_flow_last_error` for details.
///
/// # Safety
/// `stream` and `out_chunk` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_stream_next(
    stream: *mut FfiStream,
    out_chunk: *mut *mut c_char,
) -> i32 {
    if stream.is_null() || out_chunk.is_null() {
        return -1;
    }
    let stream = unsafe { &*stream };
    let result = tokio_runtime().block_on(async {
        let mut guard = stream.receiver.lock().await;
        guard.recv().await
    });
    match result {
        None => 0, // stream done
        Some(Ok(chunk)) => {
            unsafe { *out_chunk = json_to_c_string(&chunk) };
            1
        }
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free a stream handle and release its resources.
///
/// # Safety
/// `stream` must be a valid `FfiStream` pointer returned by
/// `nemo_flow_llm_stream_call_execute`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_stream_free(stream: *mut FfiStream) {
    if !stream.is_null() {
        drop(unsafe { Box::from_raw(stream) });
    }
}

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
            cb: NemoFlowToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoFlowFreeFn,
        ) -> NemoFlowStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, wrapped) {
                Ok(()) => NemoFlowStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NemoFlowStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NemoFlowStatus::Ok,
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
    nemo_flow_register_tool_sanitize_request_guardrail,
    /// Deregister a tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nemo_flow_deregister_tool_sanitize_request_guardrail,
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
    nemo_flow_register_tool_sanitize_response_guardrail,
    /// Deregister a tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nemo_flow_deregister_tool_sanitize_response_guardrail,
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
/// allow/reject, call [`crate::error::nemo_flow_set_last_error_message`] from C
/// and return null.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_register_tool_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NemoFlowToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core_registry_api::register_tool_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_tool_conditional_execution_guardrail(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_tool_conditional_execution_guardrail(&name) {
        Ok(_) => NemoFlowStatus::Ok,
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
            cb: NemoFlowToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoFlowFreeFn,
        ) -> NemoFlowStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, break_chain, wrapped) {
                Ok(()) => NemoFlowStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NemoFlowStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NemoFlowStatus::Ok,
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
    /// [`crate::error::nemo_flow_set_last_error_message`] from C and return null.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nemo_flow_register_tool_request_intercept,
    /// Deregister a tool request intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nemo_flow_deregister_tool_request_intercept,
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
pub unsafe extern "C" fn nemo_flow_register_tool_execution_intercept(
    name: *const c_char,
    priority: i32,
    exec_cb: NemoFlowToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoFlowFreeFn,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let exec = wrap_tool_exec_intercept_fn(exec_cb, exec_user_data, exec_free);
    match core_registry_api::register_tool_execution_intercept(&name, priority, exec) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_deregister_tool_execution_intercept(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_registry_api::deregister_tool_execution_intercept(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

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

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Each scope stack is independent: scopes pushed on one do not appear on another.
/// Use `nemo_flow_scope_stack_set_thread` to bind a stack to the current thread
/// before making other NeMo Flow API calls.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeStack` that must be
///   freed with `nemo_flow_scope_stack_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_stack_create(
    out: *mut *mut FfiScopeStack,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let handle = create_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiScopeStack(handle))) };
    NemoFlowStatus::Ok
}

/// Bind an isolated scope stack to the current OS thread.
///
/// After this call, all NeMo Flow scope operations on the current thread
/// (e.g. `nemo_flow_push_scope`, `nemo_flow_get_handle`) will use the
/// given scope stack. This is typically used from Go goroutines that have
/// called `runtime.LockOSThread()`.
///
/// The `FfiScopeStack` is **not** consumed — the caller retains ownership
/// and must still free it when done.
///
/// # Safety
/// `stack` must be a valid, non-null `FfiScopeStack` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_stack_set_thread(
    stack: *const FfiScopeStack,
) -> NemoFlowStatus {
    clear_last_error();
    if stack.is_null() {
        set_last_error("stack pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let handle = unsafe { &*stack }.0.clone();
    set_thread_scope_stack(handle);
    NemoFlowStatus::Ok
}

/// Returns whether the current execution context has an explicitly-initialized
/// scope stack.
///
/// Returns `true` if `nemo_flow_scope_stack_set_thread` has been called on the
/// current OS thread (or the caller is inside a tokio task-local scope).
/// Returns `false` when only the auto-created default is present.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_scope_stack_active() -> bool {
    scope_stack_active()
}

// ---------------------------------------------------------------------------
// ATIF exporter
// ---------------------------------------------------------------------------

/// Creates a new ATIF exporter.
///
/// # Parameters
/// - `session_id`: Session identifier string (required, non-null).
/// - `agent_name`: Agent name string (required, non-null).
/// - `agent_version`: Agent version string (required, non-null).
/// - `model_name`: Default model name (nullable).
/// - `out`: On success, receives a heap-allocated `FfiAtifExporter`.
///
/// # Safety
/// All non-null string pointers must be valid C strings. `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_create(
    session_id: *const c_char,
    agent_name: *const c_char,
    agent_version: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiAtifExporter,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let session_id = match c_str_to_string(session_id) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let agent_name = match c_str_to_string(agent_name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let agent_version = match c_str_to_string(agent_version) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    let agent_info = nemo_flow::atif::AtifAgentInfo {
        name: agent_name,
        version: agent_version,
        model_name: model_name_opt,
        tool_definitions: None,
        extra: None,
    };

    let exporter = nemo_flow::atif::AtifExporter::new(session_id, agent_info);
    unsafe { *out = Box::into_raw(Box::new(FfiAtifExporter(exporter))) };
    NemoFlowStatus::Ok
}

/// Registers the exporter as an event subscriber.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `name`: Subscriber name (required, non-null).
///
/// # Safety
/// `exporter` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_register(
    exporter: *const FfiAtifExporter,
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let subscriber = unsafe { &*exporter }.0.subscriber();
    match core_subscriber_api::register_subscriber(&name, subscriber) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregisters the exporter subscriber.
///
/// # Parameters
/// - `name`: Subscriber name (required, non-null).
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_deregister(name: *const c_char) -> NemoFlowStatus {
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

/// Exports collected events as an ATIF trajectory JSON string.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `out`: On success, receives a JSON string (caller must free with
///   `nemo_flow_string_free`).
///
/// # Safety
/// `exporter` and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_export(
    exporter: *const FfiAtifExporter,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let trajectory = unsafe { &*exporter }.0.export();
    match serde_json::to_string(&trajectory) {
        Ok(json_str) => {
            unsafe { *out = str_to_c_string(&json_str) };
            NemoFlowStatus::Ok
        }
        Err(e) => {
            set_last_error(&format!("failed to serialize trajectory: {e}"));
            NemoFlowStatus::Internal
        }
    }
}

/// Clears all collected events from the exporter.
///
/// # Parameters
/// - `exporter`: The exporter handle.
///
/// # Safety
/// `exporter` must be a valid, non-null `FfiAtifExporter` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_clear(
    exporter: *const FfiAtifExporter,
) -> NemoFlowStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    unsafe { &*exporter }.0.clear();
    NemoFlowStatus::Ok
}

// ---------------------------------------------------------------------------
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

fn parse_string_map_json(
    json_ptr: *const c_char,
    field_name: &str,
) -> Result<std::collections::HashMap<String, String>, NemoFlowStatus> {
    if json_ptr.is_null() {
        return Ok(std::collections::HashMap::new());
    }

    let json_string = c_str_to_string(json_ptr)?;
    let value: serde_json::Value = serde_json::from_str(&json_string).map_err(|e| {
        set_last_error(&format!("invalid {field_name} JSON: {e}"));
        NemoFlowStatus::InvalidJson
    })?;

    let serde_json::Value::Object(map) = value else {
        set_last_error(&format!(
            "{field_name} must be a JSON object of string values"
        ));
        return Err(NemoFlowStatus::InvalidArg);
    };

    let mut out = std::collections::HashMap::with_capacity(map.len());
    for (key, value) in map {
        let serde_json::Value::String(value) = value else {
            set_last_error(&format!(
                "{field_name} must be a JSON object of string values"
            ));
            return Err(NemoFlowStatus::InvalidArg);
        };
        out.insert(key, value);
    }
    Ok(out)
}

/// Creates a new OpenTelemetry subscriber.
///
/// Nullable strings use crate defaults when omitted. `headers_json` and
/// `resource_attributes_json` must be JSON objects of string values when
/// provided.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_create(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    out: *mut *mut FfiOpenTelemetrySubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    let transport = if transport.is_null() {
        "http_binary".to_string()
    } else {
        match c_str_to_string(transport) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };

    let service_name = if service_name.is_null() {
        "nemo-flow".to_string()
    } else {
        match c_str_to_string(service_name) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };

    let mut config = match transport.as_str() {
        "http_binary" => {
            nemo_flow::observability::otel::OpenTelemetryConfig::http_binary(service_name)
        }
        "grpc" => nemo_flow::observability::otel::OpenTelemetryConfig::grpc(service_name),
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            return NemoFlowStatus::InvalidArg;
        }
    };

    if !endpoint.is_null() {
        let endpoint = match c_str_to_string(endpoint) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_endpoint(endpoint);
    }
    if !service_namespace.is_null() {
        let namespace = match c_str_to_string(service_namespace) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_namespace(namespace);
    }
    if !service_version.is_null() {
        let version = match c_str_to_string(service_version) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_version(version);
    }
    if !instrumentation_scope.is_null() {
        let scope = match c_str_to_string(instrumentation_scope) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_instrumentation_scope(scope);
    }
    if timeout_millis != 0 {
        config = config.with_timeout(Duration::from_millis(timeout_millis));
    }

    for (key, value) in match parse_string_map_json(headers_json, "headers") {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_header(key, value);
    }
    for (key, value) in match parse_string_map_json(resource_attributes_json, "resource_attributes")
    {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_resource_attribute(key, value);
    }

    let _runtime_guard = tokio_runtime().enter();
    let subscriber_result = nemo_flow::observability::otel::OpenTelemetrySubscriber::new(config);
    match subscriber_result {
        Ok(subscriber) => {
            unsafe { *out = Box::into_raw(Box::new(FfiOpenTelemetrySubscriber(subscriber))) };
            NemoFlowStatus::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Registers the OpenTelemetry subscriber as an event subscriber.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_register(
    subscriber: *const FfiOpenTelemetrySubscriber,
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Deregisters the OpenTelemetry subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_deregister(
    name: *const c_char,
) -> NemoFlowStatus {
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

/// Forces a flush of finished spans through the exporter.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_force_flush(
    subscriber: *const FfiOpenTelemetrySubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Shuts down the underlying tracer provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_shutdown(
    subscriber: *const FfiOpenTelemetrySubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Creates a new OpenInference subscriber.
///
/// Nullable strings use crate defaults when omitted. `headers_json` and
/// `resource_attributes_json` must be JSON objects of string values when
/// provided.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_create(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    out: *mut *mut FfiOpenInferenceSubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    let transport = if transport.is_null() {
        "http_binary".to_string()
    } else {
        match c_str_to_string(transport) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };

    let mut config = nemo_flow::observability::openinference::OpenInferenceConfig::new();
    config = match transport.as_str() {
        "http_binary" => config
            .with_transport(nemo_flow::observability::openinference::OtlpTransport::HttpBinary),
        "grpc" => {
            config.with_transport(nemo_flow::observability::openinference::OtlpTransport::Grpc)
        }
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            return NemoFlowStatus::InvalidArg;
        }
    };

    if !service_name.is_null() {
        let value = match c_str_to_string(service_name) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_name(value);
    }
    if !endpoint.is_null() {
        let endpoint = match c_str_to_string(endpoint) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_endpoint(endpoint);
    }
    if !service_namespace.is_null() {
        let namespace = match c_str_to_string(service_namespace) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_namespace(namespace);
    }
    if !service_version.is_null() {
        let version = match c_str_to_string(service_version) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_version(version);
    }
    if !instrumentation_scope.is_null() {
        let scope = match c_str_to_string(instrumentation_scope) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_instrumentation_scope(scope);
    }
    if timeout_millis != 0 {
        config = config.with_timeout(Duration::from_millis(timeout_millis));
    }

    for (key, value) in match parse_string_map_json(headers_json, "headers") {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_header(key, value);
    }
    for (key, value) in match parse_string_map_json(resource_attributes_json, "resource_attributes")
    {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_resource_attribute(key, value);
    }

    let _runtime_guard = tokio_runtime().enter();
    let subscriber_result =
        nemo_flow::observability::openinference::OpenInferenceSubscriber::new(config);
    match subscriber_result {
        Ok(subscriber) => {
            unsafe { *out = Box::into_raw(Box::new(FfiOpenInferenceSubscriber(subscriber))) };
            NemoFlowStatus::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Registers the OpenInference subscriber as an event subscriber.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_register(
    subscriber: *const FfiOpenInferenceSubscriber,
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Deregisters the OpenInference subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_deregister(
    name: *const c_char,
) -> NemoFlowStatus {
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

/// Forces a flush of finished spans through the exporter.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_force_flush(
    subscriber: *const FfiOpenInferenceSubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Shuts down the underlying tracer provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_shutdown(
    subscriber: *const FfiOpenInferenceSubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

// ---------------------------------------------------------------------------
// Scope-local tool guardrail registrations
// ---------------------------------------------------------------------------

/// Helper to parse a scope UUID from a C string.
fn parse_scope_uuid(scope_uuid: *const c_char) -> Result<uuid::Uuid, NemoFlowStatus> {
    let uuid_str = c_str_to_string(scope_uuid)?;
    uuid::Uuid::parse_str(&uuid_str).map_err(|e| {
        set_last_error(&format!("invalid scope UUID: {e}"));
        NemoFlowStatus::InvalidArg
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
            cb: NemoFlowToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoFlowFreeFn,
        ) -> NemoFlowStatus {
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
                Ok(()) => NemoFlowStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            scope_uuid: *const c_char,
            name: *const c_char,
        ) -> NemoFlowStatus {
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
                Ok(_) => NemoFlowStatus::Ok,
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
    nemo_flow_scope_register_tool_sanitize_request_guardrail,
    /// Deregister a scope-local tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nemo_flow_scope_deregister_tool_sanitize_request_guardrail,
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
    nemo_flow_scope_register_tool_sanitize_response_guardrail,
    /// Deregister a scope-local tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nemo_flow_scope_deregister_tool_sanitize_response_guardrail,
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
/// allow/reject, call [`crate::error::nemo_flow_set_last_error_message`] from C
/// and return null.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_register_tool_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local tool conditional execution guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
            cb: NemoFlowToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NemoFlowFreeFn,
        ) -> NemoFlowStatus {
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
                Ok(()) => NemoFlowStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $deregister_name(
            scope_uuid: *const c_char,
            name: *const c_char,
        ) -> NemoFlowStatus {
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
                Ok(_) => NemoFlowStatus::Ok,
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
    /// [`crate::error::nemo_flow_set_last_error_message`] from C and return null.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
    nemo_flow_scope_register_tool_request_intercept,
    /// Deregister a scope-local tool request intercept by name.
    ///
    /// # Safety
    /// `scope_uuid` and `name` must be valid C strings.
    nemo_flow_scope_deregister_tool_request_intercept,
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
pub unsafe extern "C" fn nemo_flow_scope_register_tool_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NemoFlowToolExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local tool execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_tool_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
pub unsafe extern "C" fn nemo_flow_scope_register_llm_sanitize_request_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM request sanitization guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_llm_sanitize_request_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
pub unsafe extern "C" fn nemo_flow_scope_register_llm_sanitize_response_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM response sanitization guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_llm_sanitize_response_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
/// allow/reject, call [`crate::error::nemo_flow_set_last_error_message`] from C
/// and return null.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_register_llm_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    cb: NemoFlowLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM conditional execution guardrail by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
/// [`crate::error::nemo_flow_set_last_error_message`] from C and return null.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings. `cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_register_llm_request_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoFlowLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM request intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_llm_request_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
pub unsafe extern "C" fn nemo_flow_scope_register_llm_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NemoFlowLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_llm_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
pub unsafe extern "C" fn nemo_flow_scope_register_llm_stream_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
    priority: i32,
    exec_cb: NemoFlowLlmExecInterceptCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local LLM streaming execution intercept by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_llm_stream_execution_intercept(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
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
pub unsafe extern "C" fn nemo_flow_scope_register_subscriber(
    scope_uuid: *const c_char,
    name: *const c_char,
    cb: NemoFlowEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoFlowFreeFn,
) -> NemoFlowStatus {
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
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a scope-local event subscriber by name.
///
/// # Safety
/// `scope_uuid` and `name` must be valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_scope_deregister_subscriber(
    scope_uuid: *const c_char,
    name: *const c_char,
) -> NemoFlowStatus {
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
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
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
        unsafe { std::ffi::CStr::from_ptr(name) }
            .to_str()
            .unwrap_or_default()
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
