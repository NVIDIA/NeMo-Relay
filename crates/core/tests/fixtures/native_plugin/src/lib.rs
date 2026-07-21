// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::ptr;

use nemo_relay_plugin::{
    CategoryProfile, ConfigDiagnostic, DiagnosticLevel, Event, EventCategory, EventSanitizeFields,
    Json, LlmJsonStream, LlmRequest, LlmRequestInterceptOutcome, NemoRelayNativeHostApiV1,
    NemoRelayNativePluginContext, NemoRelayNativePluginV1, NemoRelayNativeString, NemoRelayStatus,
    NemoRelayNativeToolNextFn, NativePlugin, PendingMarkSpec, PluginContext, PluginRuntime,
    ScopeCategory, ScopeType, ToolExecutionInterceptOutcome,
};
use serde_json::{Map, json};

struct FixtureNativePlugin;

impl NativePlugin for FixtureNativePlugin {
    fn plugin_kind(&self) -> &str {
        "fixture_native"
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        if plugin_config
            .get("reject")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "fixture.rejected".into(),
                component: Some("fixture_native".into()),
                field: Some("reject".into()),
                message: "fixture rejection requested".into(),
            }];
        }
        vec![]
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let runtime = ctx.runtime();
        ctx.register_subscriber("fixture_subscriber", {
            let runtime = runtime.clone();
            move |event| subscriber_mark(&runtime, event)
        })?;
        ctx.register_mark_sanitize_guardrail("fixture_mark_sanitize", 0, |_, fields| {
            mark_event_fields(fields, "native_plugin_mark")
        })?;
        ctx.register_scope_sanitize_start_guardrail(
            "fixture_scope_start_sanitize",
            0,
            |_, fields| mark_event_fields(fields, "native_plugin_scope_start"),
        )?;
        ctx.register_scope_sanitize_end_guardrail(
            "fixture_scope_end_sanitize",
            0,
            |_, fields| mark_event_fields(fields, "native_plugin_scope_end"),
        )?;

        ctx.register_tool_sanitize_request_guardrail(
            "fixture_tool_sanitize_request",
            0,
            |_name, args| mark_json(args, "native_plugin_tool_sanitize_request"),
        )?;
        ctx.register_tool_sanitize_response_guardrail(
            "fixture_tool_sanitize_response",
            0,
            |_name, result| mark_json(result, "native_plugin_tool_sanitize_response"),
        )?;
        ctx.register_tool_conditional_execution_guardrail(
            "fixture_tool_conditional",
            0,
            |_name, _args| Ok(None),
        )?;
        ctx.register_tool_request_intercept("fixture_rewrite_args", 0, false, {
            let runtime = runtime.clone();
            move |_name, args| {
                emit_runtime_events(&runtime)?;
                Ok(mark_json(args, "native_plugin"))
            }
        })?;
        ctx.register_tool_execution_intercept("fixture_tool_execution", 0, {
            let runtime = runtime.clone();
            move |_name, args, next| {
                let args = mark_json(args, "native_plugin_tool_execution_request");
                let result = if args
                    .get("use_isolated_next")
                    .and_then(Json::as_bool)
                    .unwrap_or(false)
                {
                    let isolated = runtime.create_scope_stack()?;
                    let mut result = None;
                    isolated.with_current(|| {
                        let mut scope = runtime.scope(
                            "fixture.native.isolated.next",
                            ScopeType::Custom,
                            None,
                            None,
                            Some(&Json::String("isolated-next-input".into())),
                        )?;
                        result = Some(next.call(args)?);
                        scope.close(Some(&Json::String("isolated-next-output".into())), None)
                    })?;
                    result.ok_or_else(|| "isolated next did not produce a result".to_string())?
                } else {
                    next.call(args)?
                };
                let result = mark_json(result, "native_plugin_tool_execution");
                Ok(
                    ToolExecutionInterceptOutcome::new(result).with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("fixture.native.tool_execution.mark")
                            .category(EventCategory::custom())
                            .category_profile(CategoryProfile {
                                subtype: Some("fixture.native.tool_execution".into()),
                                ..CategoryProfile::default()
                            })
                            .data(json!({ "source": "native_tool_execution" }))
                            .metadata(json!({ "fixture": true }))
                            .build(),
                    ),
                )
            }
        })?;

        ctx.register_llm_sanitize_request_guardrail(
            "fixture_llm_sanitize_request",
            0,
            |request| mark_llm_request(request, "native_plugin_llm_sanitize_request"),
        )?;
        ctx.register_llm_sanitize_response_guardrail(
            "fixture_llm_sanitize_response",
            0,
            |response| mark_json(response, "native_plugin_llm_sanitize_response"),
        )?;
        ctx.register_llm_conditional_execution_guardrail(
            "fixture_llm_conditional",
            0,
            |_request| Ok(None),
        )?;
        ctx.register_llm_request_intercept(
            "fixture_llm_request_intercept",
            0,
            false,
            |_name, request, annotated| {
                Ok(LlmRequestInterceptOutcome::new(
                    mark_llm_request(request, "native_plugin_llm_request_intercept"),
                    annotated,
                )
                .with_pending_mark(
                    PendingMarkSpec::builder()
                        .name("fixture.native.llm_request.mark")
                        .category(EventCategory::custom())
                        .category_profile(CategoryProfile {
                            subtype: Some("fixture.native.pending".into()),
                            ..CategoryProfile::default()
                        })
                        .data(json!({ "source": "native_request_intercept" }))
                        .metadata(json!({ "fixture": true }))
                        .build(),
                ))
            },
        )?;
        ctx.register_llm_execution_intercept("fixture_llm_execution", 0, |_name, request, next| {
            let response = next.call(mark_llm_request(
                request,
                "native_plugin_llm_execution_request",
            ))?;
            Ok(mark_json(response, "native_plugin_llm_execution"))
        })?;
        ctx.register_llm_stream_execution_intercept(
            "fixture_llm_stream_execution",
            0,
            |_name, request, next| {
                let stream = next.call(mark_llm_request(
                    request,
                    "native_plugin_llm_stream_execution_request",
                ))?;
                let stream: LlmJsonStream = Box::new(stream.map(|chunk| {
                    chunk.map(|chunk| mark_json(chunk, "native_plugin_llm_stream_execution"))
                }));
                Ok(stream)
            },
        )?;

        Ok(())
    }
}

fn mark_event_fields(mut fields: EventSanitizeFields, marker: &str) -> EventSanitizeFields {
    let mut metadata = fields.metadata.unwrap_or_else(|| json!({}));
    metadata[marker] = json!(true);
    fields.metadata = Some(metadata);
    fields
}

fn subscriber_mark(runtime: &PluginRuntime, event: &Event) {
    if event.name() == "native-plugin-test-outer"
        && event.scope_category() == Some(ScopeCategory::Start)
    {
        let _ = runtime.emit_mark(
            "fixture.native.subscriber.mark",
            Some(&Json::String("subscriber".into())),
            None,
        );
    }
}

fn emit_runtime_events(runtime: &PluginRuntime) -> nemo_relay_plugin::Result<()> {
    runtime.emit_mark(
        "fixture.native.mark",
        Some(&Json::String("current".into())),
        None,
    )?;
    let scope = runtime.push_scope(
        "fixture.native.scope",
        ScopeType::Custom,
        None,
        None,
        Some(&Json::String("current-scope-input".into())),
    )?;
    runtime.emit_mark(
        "fixture.native.scope.mark",
        Some(&Json::String("inside-current-scope".into())),
        None,
    )?;
    runtime.pop_scope(
        &scope,
        Some(&Json::String("current-scope-output".into())),
        None,
    )?;

    let thread_stack = runtime.create_scope_stack()?;
    {
        let _thread_guard = runtime.bind_scope_stack_thread(&thread_stack)?;
        runtime.emit_mark(
            "fixture.native.thread_stack.mark",
            Some(&Json::String("thread-stack".into())),
            None,
        )?;
    }

    let isolated = runtime.create_scope_stack()?;
    isolated.with_current(|| {
        runtime.emit_mark(
            "fixture.native.isolated.mark",
            Some(&Json::String("isolated".into())),
            None,
        )?;
        let scope = runtime.push_scope(
            "fixture.native.isolated.scope",
            ScopeType::Custom,
            None,
            None,
            Some(&Json::String("isolated-scope-input".into())),
        )?;
        runtime.pop_scope(
            &scope,
            Some(&Json::String("isolated-scope-output".into())),
            None,
        )
    })
}

fn mark_llm_request(mut request: LlmRequest, key: &str) -> LlmRequest {
    request.content = mark_json(request.content, key);
    request
}

fn mark_json(mut value: Json, key: &str) -> Json {
    if let Json::Object(object) = &mut value {
        object.insert(key.into(), json!(true));
    }
    value
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_fixture_native_plugin, || FixtureNativePlugin);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_observability_collision(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "observability",
            None,
            None,
            Some(raw_noop_register),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_entry_error(
    host: *const NemoRelayNativeHostApiV1,
    _out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe { set_raw_last_error(host, "fixture entry failed") };
    NemoRelayStatus::Internal
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_small_descriptor(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            Some(0),
            None,
            Some(raw_noop_register),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_null_kind(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    let status =
        unsafe { write_raw_descriptor(host, out, "", None, None, Some(raw_noop_register)) };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    unsafe {
        if !(*out).plugin_kind.is_null() {
            let host = &*host;
            (host.string_free)((*out).plugin_kind);
        }
        (*out).plugin_kind = ptr::null_mut();
    }
    NemoRelayStatus::Ok
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_no_register(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe { write_raw_descriptor(host, out, "fixture_native", None, None, None) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_validate_error(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            None,
            Some(raw_validate_error),
            Some(raw_noop_register),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_invalid_diagnostics(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            None,
            Some(raw_invalid_diagnostics_validate),
            Some(raw_noop_register),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_register_error(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            None,
            None,
            Some(raw_register_error),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_tool_outcome_errors(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            None,
            None,
            Some(raw_register_tool_outcome_errors),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_event_sanitize_errors(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            None,
            None,
            Some(raw_register_event_sanitize_errors),
        )
    }
}

type RawValidate = unsafe extern "C" fn(
    *mut c_void,
    *const NemoRelayNativeString,
    *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

type RawRegister = unsafe extern "C" fn(
    *mut c_void,
    *const NemoRelayNativeString,
    *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus;

unsafe fn write_raw_descriptor(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
    kind: &str,
    struct_size: Option<usize>,
    validate: Option<RawValidate>,
    register: Option<RawRegister>,
) -> NemoRelayStatus {
    if host.is_null() || out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let host = unsafe { *host };
    let mut plugin = NemoRelayNativePluginV1::default();
    plugin.struct_size = struct_size.unwrap_or(std::mem::size_of::<NemoRelayNativePluginV1>());
    plugin.plugin_kind = unsafe { raw_host_string(&host, kind) };
    if plugin.plugin_kind.is_null() && !kind.is_empty() {
        return NemoRelayStatus::Internal;
    }
    plugin.user_data = Box::into_raw(Box::new(host)).cast();
    plugin.validate = validate;
    plugin.register = register;
    plugin.drop = Some(raw_drop_host);
    unsafe { *out = plugin };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn raw_validate_error(
    user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    _out_diagnostics_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    unsafe { set_raw_last_error_from_user_data(user_data, "fixture validate failed") };
    NemoRelayStatus::InvalidArg
}

unsafe extern "C" fn raw_invalid_diagnostics_validate(
    user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    out_diagnostics_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_diagnostics_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let host = unsafe { raw_host_from_user_data(user_data) };
    let Some(host) = host else {
        return NemoRelayStatus::NullPointer;
    };
    unsafe {
        *out_diagnostics_json = raw_host_string(host, "not-json");
    }
    NemoRelayStatus::Ok
}

unsafe extern "C" fn raw_noop_register(
    _user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    _ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn raw_register_error(
    user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    _ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    unsafe { set_raw_last_error_from_user_data(user_data, "fixture register failed") };
    NemoRelayStatus::Internal
}

unsafe extern "C" fn raw_register_tool_outcome_errors(
    user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    let Some(host) = (unsafe { raw_host_from_user_data(user_data) }) else {
        return NemoRelayStatus::NullPointer;
    };
    let name = unsafe { raw_host_string(host, "fixture_raw_tool_outcome") };
    if name.is_null() {
        return NemoRelayStatus::Internal;
    }
    let status = unsafe {
        (host.plugin_context_register_tool_execution_intercept)(
            ctx,
            name,
            0,
            raw_tool_outcome_callback,
            user_data,
            None,
        )
    };
    unsafe { (host.string_free)(name) };
    status
}

unsafe extern "C" fn raw_register_event_sanitize_errors(
    user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    let Some(host) = (unsafe { raw_host_from_user_data(user_data) }) else {
        return NemoRelayStatus::NullPointer;
    };
    let name = unsafe { raw_host_string(host, "fixture_raw_event_sanitize") };
    if name.is_null() {
        return NemoRelayStatus::Internal;
    }
    let status = unsafe {
        (host.plugin_context_register_mark_sanitize_guardrail)(
            ctx,
            name,
            0,
            raw_event_sanitize_error_callback,
            user_data,
            None,
        )
    };
    unsafe { (host.string_free)(name) };
    status
}

unsafe extern "C" fn raw_tool_outcome_callback(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    _args_json: *const NemoRelayNativeString,
    _next_fn: NemoRelayNativeToolNextFn,
    _next_ctx: *mut c_void,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_outcome_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_outcome_json = ptr::null_mut() };
    let Some(host) = (unsafe { raw_host_from_user_data(user_data) }) else {
        return NemoRelayStatus::NullPointer;
    };
    let Some(name) = (unsafe { raw_host_string_value(host, name) }) else {
        return NemoRelayStatus::InvalidUtf8;
    };
    match name.as_str() {
        "fixture-null-outcome" => NemoRelayStatus::Ok,
        "fixture-malformed-outcome" => {
            unsafe {
                *out_outcome_json = raw_host_string(host, r#"{"pending_marks":[]}"#);
            }
            NemoRelayStatus::Ok
        }
        "fixture-status-error-outcome" => {
            unsafe {
                *out_outcome_json = raw_host_string(
                    host,
                    r#"{"result":{"stale":true},"pending_marks":[]}"#,
                );
                set_raw_last_error_from_user_data(user_data, "fixture tool execution failed");
            }
            NemoRelayStatus::Internal
        }
        _ => {
            unsafe {
                *out_outcome_json = raw_host_string(
                    host,
                    r#"{"result":{"raw_tool_outcome":true},"pending_marks":[]}"#,
                );
            }
            NemoRelayStatus::Ok
        }
    }
}

unsafe extern "C" fn raw_event_sanitize_error_callback(
    user_data: *mut c_void,
    _event_json: *const NemoRelayNativeString,
    _fields_json: *const NemoRelayNativeString,
    out_fields_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_fields_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_fields_json = ptr::null_mut() };
    unsafe { set_raw_last_error_from_user_data(user_data, "fixture event sanitizer failed") };
    NemoRelayStatus::Internal
}

unsafe extern "C" fn raw_drop_host(user_data: *mut c_void) {
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data as *mut NemoRelayNativeHostApiV1) });
    }
}

unsafe fn raw_host_from_user_data<'a>(
    user_data: *mut c_void,
) -> Option<&'a NemoRelayNativeHostApiV1> {
    if user_data.is_null() {
        None
    } else {
        Some(unsafe { &*(user_data as *const NemoRelayNativeHostApiV1) })
    }
}

unsafe fn set_raw_last_error_from_user_data(user_data: *mut c_void, message: &str) {
    if let Some(host) = unsafe { raw_host_from_user_data(user_data) } {
        unsafe { set_raw_last_error(host as *const _, message) };
    }
}

unsafe fn set_raw_last_error(host: *const NemoRelayNativeHostApiV1, message: &str) {
    if host.is_null() {
        return;
    }
    let host = unsafe { &*host };
    let message = unsafe { raw_host_string(host, message) };
    if !message.is_null() {
        unsafe {
            (host.last_error_set)(message);
            (host.string_free)(message);
        }
    }
}

unsafe fn raw_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: &str,
) -> *mut NemoRelayNativeString {
    let mut out = ptr::null_mut();
    let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut out) };
    if status == NemoRelayStatus::Ok {
        out
    } else {
        ptr::null_mut()
    }
}

unsafe fn raw_host_string_value(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let len = unsafe { (host.string_len)(value) };
    let data = unsafe { (host.string_data)(value) };
    if data.is_null() && len > 0 {
        return None;
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    std::str::from_utf8(bytes).ok().map(str::to_owned)
}
