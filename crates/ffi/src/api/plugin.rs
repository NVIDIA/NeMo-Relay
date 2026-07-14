// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Arc, CStr, ConfigDiagnostic, DiagnosticLevel, DynamicPluginActivationSpec, FfiPluginActivation,
    FfiPluginContext, Future, NemoRelayEventSanitizeCb, NemoRelayEventSubscriberCb,
    NemoRelayFreeFn, NemoRelayJsonCb, NemoRelayLlmConditionalCb, NemoRelayLlmExecInterceptCb,
    NemoRelayLlmRequestCb, NemoRelayLlmRequestInterceptCb, NemoRelayPluginRegisterCb,
    NemoRelayPluginValidateCb, NemoRelayStatus, NemoRelayToolConditionalCb,
    NemoRelayToolExecInterceptCb, NemoRelayToolSanitizeCb, Pin, Plugin, PluginConfig, PluginError,
    PluginHostActivation, PluginRegistrationContext, active_plugin_report, c_char, c_str_to_json,
    c_str_to_string, clear_last_error, clear_plugin_configuration, deregister_plugin,
    initialize_plugins, json_to_c_string, last_error_message, list_plugin_kinds,
    nemo_relay_string_free, register_adaptive_component, register_plugin, set_last_error,
    status_from_plugin_error, tokio_runtime, validate_plugin_config, wrap_event_sanitize_fn,
    wrap_event_subscriber, wrap_llm_conditional_fn, wrap_llm_exec_intercept_fn,
    wrap_llm_request_intercept_fn, wrap_llm_response_fn, wrap_llm_sanitize_request_fn,
    wrap_llm_stream_exec_intercept_fn, wrap_tool_conditional_fn, wrap_tool_exec_intercept_fn,
    wrap_tool_request_intercept_fn, wrap_tool_sanitize_fn,
};
use crate::api::event_registry::Surface;
use nemo_relay_pii_redaction::component::register_pii_redaction_component;

struct FfiHostedPluginUserData {
    ptr: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
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
    validate_cb: Option<NemoRelayPluginValidateCb>,
    register_cb: NemoRelayPluginRegisterCb,
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
        unsafe { nemo_relay_string_free(plugin_config_json) };

        if result_ptr.is_null() {
            let message = last_error_message().unwrap_or_else(|| {
                format!(
                    "plugin '{}' validate callback returned null",
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
        unsafe { nemo_relay_string_free(result_ptr) };
        diagnostics.unwrap_or_else(|| {
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.validate_failed".to_string(),
                component: Some(self.plugin_kind.clone()),
                field: None,
                message: format!(
                    "plugin '{}' validate callback returned invalid diagnostics JSON",
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
            unsafe { nemo_relay_string_free(plugin_config_json) };
            if status == NemoRelayStatus::Ok {
                Ok(())
            } else if let Some(message) = last_error_message() {
                Err(PluginError::RegistrationFailed(message))
            } else {
                Err(PluginError::RegistrationFailed(format!(
                    "plugin '{}' register callback failed with status {:?}",
                    self.plugin_kind, status
                )))
            }
        })
    }
}

fn ensure_adaptive_component_registered() -> std::result::Result<(), NemoRelayStatus> {
    register_adaptive_component().map_err(|err| status_from_plugin_error(&err))
}

fn ensure_pii_redaction_component_registered() -> std::result::Result<(), NemoRelayStatus> {
    register_pii_redaction_component().map_err(|err| status_from_plugin_error(&err))
}

fn parse_plugin_config(
    config_json: *const c_char,
) -> std::result::Result<PluginConfig, NemoRelayStatus> {
    let value = c_str_to_json(config_json).ok_or(NemoRelayStatus::InvalidJson)?;
    serde_json::from_value(value).map_err(|error| {
        set_last_error(&format!("invalid plugin config: {error}"));
        NemoRelayStatus::InvalidJson
    })
}

fn parse_dynamic_plugin_specs(
    dynamic_plugins_json: *const c_char,
) -> std::result::Result<Vec<DynamicPluginActivationSpec>, NemoRelayStatus> {
    let value = c_str_to_json(dynamic_plugins_json).ok_or(NemoRelayStatus::InvalidJson)?;
    serde_json::from_value(value).map_err(|error| {
        set_last_error(&format!("invalid dynamic plugin specifications: {error}"));
        NemoRelayStatus::InvalidJson
    })
}

fn lock_plugin_activation(
    activation: &FfiPluginActivation,
) -> std::result::Result<std::sync::MutexGuard<'_, Option<PluginHostActivation>>, NemoRelayStatus> {
    activation.0.lock().map_err(|error| {
        set_last_error(&format!("plugin activation lock poisoned: {error}"));
        NemoRelayStatus::Internal
    })
}

/// Load and activate dynamic plugins as one owned transaction.
///
/// **Experimental:** this API needs a production consumer before its lifecycle
/// contract is considered stable.
///
/// Relay discovers `plugins.toml` once during this startup call and layers
/// `config_json` over the discovered configuration. Explicit values take
/// precedence where both sources configure the same setting. Static components
/// from the resolved configuration initialize before components appended by
/// the dynamic plugins. Configuration files are not watched or reloaded.
/// `dynamic_plugins_json` must contain at least one explicit dynamic-plugin
/// activation specification; use `nemo_relay_initialize_plugins` for a
/// static-only configuration.
///
/// The explicit configuration uses this JSON shape:
///
/// ```text
/// {"version":1,"components":[{"kind":"static.kind","enabled":true,"config":{}}]}
/// ```
///
/// Dynamic plugin specifications use this JSON shape:
///
/// ```text
/// [{"plugin_id":"example","kind":"rust_dynamic","manifest_ref":"/absolute/path/relay-plugin.toml","config":{}}]
/// ```
///
/// `kind` must be `rust_dynamic` or `worker`. `environment_ref` is optional
/// and applies only to worker plugins. `manifest_ref` is resolved by the
/// embedding application; this API does not discover installed plugins.
///
/// On success, the caller owns `out_activation` and
/// must clear and free it with `nemo_relay_plugin_activation_clear` and
/// `nemo_relay_plugin_activation_free`. `out_report_json` is a library-owned C
/// string and must be released with `nemo_relay_string_free`.
///
/// # Safety
/// Both input pointers must reference valid, null-terminated C strings.
/// `out_activation` and `out_report_json` must be valid, non-null, non-overlapping
/// output pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_initialize_with_dynamic_plugins(
    config_json: *const c_char,
    dynamic_plugins_json: *const c_char,
    out_activation: *mut *mut FfiPluginActivation,
    out_report_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_activation.is_null() {
        if !out_report_json.is_null() {
            unsafe { *out_report_json = std::ptr::null_mut() };
        }
        set_last_error("out_activation pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if out_report_json.is_null() {
        unsafe { *out_activation = std::ptr::null_mut() };
        set_last_error("out_report_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if out_activation.cast::<std::ffi::c_void>() == out_report_json.cast::<std::ffi::c_void>() {
        unsafe { *out_activation = std::ptr::null_mut() };
        set_last_error("out_activation and out_report_json must not overlap");
        return NemoRelayStatus::InvalidArg;
    }
    unsafe {
        *out_activation = std::ptr::null_mut();
        *out_report_json = std::ptr::null_mut();
    }

    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    if let Err(status) = ensure_pii_redaction_component_registered() {
        return status;
    }
    let config = match parse_plugin_config(config_json) {
        Ok(config) => config,
        Err(status) => return status,
    };
    let dynamic_plugins = match parse_dynamic_plugin_specs(dynamic_plugins_json) {
        Ok(dynamic_plugins) => dynamic_plugins,
        Err(status) => return status,
    };
    let (activation, report) = match tokio_runtime().block_on(
        PluginHostActivation::activate_with_discovered_config(config, dynamic_plugins),
    ) {
        Ok(result) => result,
        Err(error) => return status_from_plugin_error(&error),
    };
    let report_json = match serde_json::to_value(report) {
        Ok(value) => value,
        Err(error) => {
            let _ = activation.clear();
            set_last_error(&error.to_string());
            return NemoRelayStatus::Internal;
        }
    };

    unsafe {
        *out_activation = Box::into_raw(Box::new(FfiPluginActivation(std::sync::Mutex::new(
            Some(activation),
        ))));
        *out_report_json = json_to_c_string(&report_json);
    }
    NemoRelayStatus::Ok
}

/// Clear one owned dynamic plugin activation.
///
/// This operation is idempotent. A null handle is treated as already cleared.
/// If teardown fails, the error is reported only by the call that performs the
/// teardown. The activation is consumed regardless of the outcome, so a later
/// clear returns success and does not report the earlier error again.
/// Concurrent clear calls for the same handle are serialized, but they must not
/// overlap with `nemo_relay_plugin_activation_free`.
/// The handle allocation remains owned by the caller and must still be passed
/// to `nemo_relay_plugin_activation_free`.
///
/// # Safety
/// `activation` must be a valid activation handle returned by
/// `nemo_relay_initialize_with_dynamic_plugins`, or null. The caller must ensure the
/// handle remains allocated for this call and that
/// `nemo_relay_plugin_activation_free` does not run concurrently with it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_activation_clear(
    activation: *mut FfiPluginActivation,
) -> NemoRelayStatus {
    clear_last_error();
    if activation.is_null() {
        return NemoRelayStatus::Ok;
    }
    let activation = unsafe { &*activation };
    let mut guard = match lock_plugin_activation(activation) {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(activation) = guard.take() else {
        return NemoRelayStatus::Ok;
    };
    match activation.clear() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_plugin_error(&error),
    }
}

/// Validate a generic plugin config document and return the diagnostics report as JSON.
///
/// # Safety
/// `config_json` must be a valid C string and `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_validate_plugin_config(
    config_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    if let Err(status) = ensure_pii_redaction_component_registered() {
        return status;
    }
    let config_value = match c_str_to_json(config_json) {
        Some(value) => value,
        None => return NemoRelayStatus::InvalidJson,
    };
    let config: PluginConfig = match serde_json::from_value(config_value) {
        Ok(config) => config,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoRelayStatus::InvalidJson;
        }
    };
    let report_json = match serde_json::to_value(validate_plugin_config(&config)) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoRelayStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&report_json) };
    NemoRelayStatus::Ok
}

/// Initialize the active global plugin components and return the resulting diagnostics report.
///
/// # Safety
/// `config_json` must be a valid C string and `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_initialize_plugins(
    config_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    if let Err(status) = ensure_pii_redaction_component_registered() {
        return status;
    }
    let config_value = match c_str_to_json(config_json) {
        Some(value) => value,
        None => return NemoRelayStatus::InvalidJson,
    };
    let config: PluginConfig = match serde_json::from_value(config_value) {
        Ok(config) => config,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoRelayStatus::InvalidJson;
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
            return NemoRelayStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&report_json) };
    NemoRelayStatus::Ok
}

/// Clear the active global plugin configuration.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_clear_plugin_configuration() -> NemoRelayStatus {
    clear_last_error();
    match clear_plugin_configuration() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Return the last successfully configured plugin report as JSON.
///
/// # Safety
/// `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_active_plugin_report_json(
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let report_json = match serde_json::to_value(active_plugin_report()) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoRelayStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&report_json) };
    NemoRelayStatus::Ok
}

/// Return the registered plugin kinds as JSON.
///
/// # Safety
/// `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_list_plugin_kinds_json(
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if let Err(status) = ensure_adaptive_component_registered() {
        return status;
    }
    if let Err(status) = ensure_pii_redaction_component_registered() {
        return status;
    }
    let kinds_json = match serde_json::to_value(list_plugin_kinds()) {
        Ok(value) => value,
        Err(err) => {
            set_last_error(&err.to_string());
            return NemoRelayStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&kinds_json) };
    NemoRelayStatus::Ok
}

/// Register a plugin backed by foreign callbacks.
///
/// # Safety
/// `plugin_kind` must be a valid C string and `register_cb` must be a valid function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_register_plugin(
    plugin_kind: *const c_char,
    validate_cb: Option<NemoRelayPluginValidateCb>,
    register_cb: NemoRelayPluginRegisterCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
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
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Deregister a plugin by kind.
///
/// # Safety
/// `plugin_kind` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_deregister_plugin(
    plugin_kind: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let plugin_kind = match c_str_to_string(plugin_kind) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if deregister_plugin(&plugin_kind) {
        NemoRelayStatus::Ok
    } else {
        set_last_error(&format!("not found: plugin '{plugin_kind}'"));
        NemoRelayStatus::NotFound
    }
}

/// Register an event subscriber into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_subscriber(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    cb: NemoRelayEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_subscriber(&name, wrapped) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

unsafe fn plugin_register_event_sanitizer(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
    surface: Surface,
) -> NemoRelayStatus {
    clear_last_error();
    let callback = wrap_event_sanitize_fn(cb, user_data, free_fn);
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let context = unsafe { &mut *((*ctx).0) };
    let result = match surface {
        Surface::Mark => context.register_mark_sanitize_guardrail(&name, priority, callback),
        Surface::Start => {
            context.register_scope_sanitize_start_guardrail(&name, priority, callback)
        }
        Surface::End => context.register_scope_sanitize_end_guardrail(&name, priority, callback),
    };
    result
        .map(|()| NemoRelayStatus::Ok)
        .unwrap_or_else(|error| status_from_plugin_error(&error))
}

/// Register a mark event sanitizer into a plugin context.
/// # Safety
/// Pointers must remain valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_mark_sanitize_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe {
        plugin_register_event_sanitizer(ctx, name, priority, cb, user_data, free_fn, Surface::Mark)
    }
}

/// Register a scope-start event sanitizer into a plugin context.
/// # Safety
/// Pointers must remain valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_scope_sanitize_start_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe {
        plugin_register_event_sanitizer(ctx, name, priority, cb, user_data, free_fn, Surface::Start)
    }
}

/// Register a scope-end event sanitizer into a plugin context.
/// # Safety
/// Pointers must remain valid for the documented call lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_scope_sanitize_end_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayEventSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    unsafe {
        plugin_register_event_sanitizer(ctx, name, priority, cb, user_data, free_fn, Surface::End)
    }
}

/// Register a tool sanitize-request guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_tool_sanitize_request_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_sanitize_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_tool_sanitize_request_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool sanitize-response guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_tool_sanitize_response_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_sanitize_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_tool_sanitize_response_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool conditional-execution guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_tool_conditional_execution_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_tool_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM sanitize-request guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_llm_sanitize_request_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_sanitize_request_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM sanitize-response guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_llm_sanitize_response_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_response_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_sanitize_response_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM conditional-execution guardrail into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_llm_conditional_execution_guardrail(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_conditional_execution_guardrail(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM request intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_llm_request_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayLlmRequestInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
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
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool request intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_tool_request_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NemoRelayToolSanitizeCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
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
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM execution intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_llm_execution_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_exec_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_llm_execution_intercept(&name, priority, wrapped) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register an LLM stream execution intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_llm_stream_execution_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayLlmExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_stream_exec_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }
        .register_llm_stream_execution_intercept(&name, priority, wrapped)
    {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}

/// Register a tool execution intercept into the plugin registration context.
///
/// # Safety
/// `ctx` and `name` must be valid pointers and the callback must remain valid for the duration
/// of the plugin registration lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_plugin_context_register_tool_execution_intercept(
    ctx: *mut FfiPluginContext,
    name: *const c_char,
    priority: i32,
    cb: NemoRelayToolExecInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NemoRelayFreeFn,
) -> NemoRelayStatus {
    clear_last_error();
    if ctx.is_null() {
        set_last_error("plugin context is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_exec_intercept_fn(cb, user_data, free_fn);
    match unsafe { &mut *((*ctx).0) }.register_tool_execution_intercept(&name, priority, wrapped) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(err) => status_from_plugin_error(&err),
    }
}
