// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use nemo_relay_plugin::{
    CategoryProfile, ConfigDiagnostic, DiagnosticLevel, Event, EventCategory, EventSanitizeFields,
    Json, LlmJsonAsyncStream, LlmRequest, LlmRequestInterceptOutcome, MetricKind,
    MetricMeasurement, MetricValueType, NEMO_RELAY_NATIVE_ABI_VERSION,
    NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE, NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY,
    NativeExecutorConfig, NativePlugin, NemoRelayNativeAsyncCallbackState,
    NemoRelayNativeAsyncMiddlewareCb, NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncStream, NemoRelayNativeHostApiV1, NemoRelayNativeHostApiV3,
    NemoRelayNativeHostApiV4, NemoRelayNativePluginContext, NemoRelayNativePluginV1,
    NemoRelayNativeString, NemoRelayNativeToolNextFn, NemoRelayStatus, PendingMarkSpec,
    PluginContext, PluginRuntime, RuntimeRegistrationKind, ScopeCategory, ScopeType,
    ToolExecutionInterceptOutcome,
};
use serde_json::{Map, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct FixtureNativePlugin;

static ASYNC_PENDING_ENTERED: AtomicBool = AtomicBool::new(false);

#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_fixture_async_pending_entered() -> bool {
    ASYNC_PENDING_ENTERED.swap(false, Ordering::AcqRel)
}

impl NativePlugin for FixtureNativePlugin {
    fn plugin_kind(&self) -> &str {
        "fixture_native"
    }

    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig { worker_threads: 3 }
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
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let event_metadata_injector_error = plugin_config
            .get("event_metadata_injector_error")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        register_fixture_event_metadata_injector(ctx, event_metadata_injector_error)?;
        if plugin_config
            .get("event_metadata_injector_only")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            return Ok(());
        }
        let runtime = ctx.runtime();
        ctx.register_subscriber("fixture_subscriber", {
            let runtime = runtime.clone();
            move |event| subscriber_mark(&runtime, event)
        })?;
        let subscriber_kinds = BTreeSet::from([RuntimeRegistrationKind::Subscriber]);
        if !runtime
            .list_runtime_registrations(Some(&subscriber_kinds))?
            .iter()
            .any(|registration| registration.local_name == "fixture_subscriber")
        {
            return Err("native runtime did not discover its subscriber".into());
        }
        ctx.register_conditional_middleware_guardrail(
            "fixture_initial_gate",
            &subscriber_kinds,
            "missing-initial-target",
            |_, _| Some("fixture initial gate".into()),
        )?;
        if plugin_config
            .get("conditional_gate_test")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            ctx.register_conditional_middleware_guardrail(
                "fixture_blocking_callback_gate",
                &subscriber_kinds,
                "fixture-native-blocked-subscriber",
                |_, _| Some("fixture callback blocked target".into()),
            )?;
            ctx.register_conditional_middleware_guardrail(
                "fixture_allowing_callback_gate",
                &subscriber_kinds,
                "fixture-native-allowed-subscriber",
                |_, _| None,
            )?;
            ctx.register_conditional_middleware_guardrail(
                "fixture_failing_callback_gate",
                &subscriber_kinds,
                "fixture-native-failed-open-subscriber",
                |_, _| panic!("fixture callback failure"),
            )?;
        }
        let dynamic_gate = runtime.register_conditional_middleware_guardrail(
            "fixture_dynamic_gate",
            &subscriber_kinds,
            "missing-dynamic-target",
            |_, _| Some("fixture dynamic gate".into()),
        )?;
        if !runtime.deregister_conditional_middleware_guardrail(&dynamic_gate)? {
            return Err("native runtime did not remove its dynamic gate".into());
        }
        ctx.register_mark_sanitize_guardrail("fixture_mark_sanitize", 0, |_, fields| async move {
            Ok(mark_event_fields(fields, "native_plugin_mark"))
        })?;
        ctx.register_scope_sanitize_start_guardrail(
            "fixture_scope_start_sanitize",
            0,
            |_, fields| async move { Ok(mark_event_fields(fields, "native_plugin_scope_start")) },
        )?;
        ctx.register_scope_sanitize_end_guardrail(
            "fixture_scope_end_sanitize",
            0,
            |_, fields| async move { Ok(mark_event_fields(fields, "native_plugin_scope_end")) },
        )?;

        ctx.register_tool_sanitize_request_guardrail(
            "fixture_tool_sanitize_request",
            0,
            |_name, args| async move { Ok(mark_json(args, "native_plugin_tool_sanitize_request")) },
        )?;
        ctx.register_tool_sanitize_response_guardrail(
            "fixture_tool_sanitize_response",
            0,
            |_name, result| async move {
                Ok(mark_json(result, "native_plugin_tool_sanitize_response"))
            },
        )?;
        ctx.register_tool_conditional_execution_guardrail(
            "fixture_tool_conditional",
            0,
            |_name, _args| async move { Ok(None) },
        )?;
        ctx.register_tool_request_intercept("fixture_rewrite_args", 0, false, {
            let runtime = runtime.clone();
            move |_name, args| {
                let runtime = runtime.clone();
                async move {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    let (mut writer, mut reader) = tokio::io::duplex(8);
                    writer
                        .write_all(b"async")
                        .await
                        .map_err(|error| error.to_string())?;
                    let mut readiness = [0_u8; 5];
                    reader
                        .read_exact(&mut readiness)
                        .await
                        .map_err(|error| error.to_string())?;
                    emit_runtime_events(&runtime)?;
                    Ok(mark_json(args, "native_plugin"))
                }
            }
        })?;
        ctx.register_tool_execution_intercept("fixture_tool_execution", 0, {
            let runtime = runtime.clone();
            move |_name, args, next| {
                let runtime = runtime.clone();
                async move {
                    let args = mark_json(args, "native_plugin_tool_execution_request");
                    let result = if args
                        .get("use_isolated_next")
                        .and_then(Json::as_bool)
                        .unwrap_or(false)
                    {
                        let isolated = runtime.create_scope_stack()?;
                        let previous = runtime.capture_scope_stack_thread()?;
                        if isolated.set_thread() != NemoRelayStatus::Ok {
                            return Err("failed to install isolated scope stack".into());
                        }
                        let mut scope = runtime.scope(
                            "fixture.native.isolated.next",
                            ScopeType::Custom,
                            None,
                            None,
                            Some(&Json::String("isolated-next-input".into())),
                        )?;
                        let call_result = next.call(args).await;
                        let close_result =
                            scope.close(Some(&Json::String("isolated-next-output".into())), None);
                        if previous.restore() != NemoRelayStatus::Ok {
                            return Err("failed to restore callback scope stack".into());
                        }
                        close_result?;
                        call_result?
                    } else if args
                        .get("use_concurrent_next")
                        .and_then(Json::as_bool)
                        .unwrap_or(false)
                    {
                        let first_next = next.clone();
                        let (first, second) =
                            tokio::join!(first_next.call(args.clone()), next.call(args),);
                        let result = first?;
                        second?;
                        result
                    } else {
                        next.call(args).await?
                    };
                    let mut result = result;
                    result.result = mark_json(result.result, "native_plugin_tool_execution");
                    Ok(
                        ToolExecutionInterceptOutcome::from(result).with_pending_mark(
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
            }
        })?;

        ctx.register_llm_sanitize_request_guardrail(
            "fixture_llm_sanitize_request",
            0,
            |request, context| async move {
                let request = if let Some(codec) = context.resolve_codec() {
                    let annotated = codec.decode(&request)?;
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    codec.encode(&annotated, &request)?
                } else {
                    request
                };
                Ok(Some(mark_llm_request(
                    request,
                    "native_plugin_llm_sanitize_request",
                )))
            },
        )?;
        ctx.register_llm_sanitize_response_guardrail(
            "fixture_llm_sanitize_response",
            0,
            |response, context| async move {
                if let Some(codec) = context.resolve_codec() {
                    codec.decode(&response)?;
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    codec.decode(&response)?;
                }
                Ok(Some(mark_json(
                    response,
                    "native_plugin_llm_sanitize_response",
                )))
            },
        )?;
        ctx.register_llm_conditional_execution_guardrail(
            "fixture_llm_conditional",
            0,
            |_request| async move { Ok(None) },
        )?;
        ctx.register_llm_request_intercept(
            "fixture_llm_request_intercept",
            0,
            false,
            |_name, request, annotated| async move {
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
        ctx.register_llm_execution_intercept(
            "fixture_llm_execution",
            0,
            |_name, request, next| async move {
                let response = next
                    .call(mark_llm_request(
                        request,
                        "native_plugin_llm_execution_request",
                    ))
                    .await?;
                Ok(mark_json(response, "native_plugin_llm_execution"))
            },
        )?;
        ctx.register_llm_stream_execution_intercept(
            "fixture_llm_stream_execution",
            0,
            |_name, request, next| async move {
                let stream = next
                    .call(mark_llm_request(
                        request,
                        "native_plugin_llm_stream_execution_request",
                    ))
                    .await?;
                let stream: LlmJsonAsyncStream = Box::pin(stream.map(|chunk| {
                    chunk.map(|chunk| mark_json(chunk, "native_plugin_llm_stream_execution"))
                }));
                Ok(stream)
            },
        )?;

        Ok(())
    }
}

fn register_fixture_event_metadata_injector(
    ctx: &mut PluginContext<'_>,
    return_error: bool,
) -> nemo_relay_plugin::Result<()> {
    ctx.register_event_metadata_injector(
        "fixture_event_metadata_injector",
        0,
        move |event| async move {
            if return_error {
                return Err("fixture Event metadata injector error requested".into());
            }
            let mut additions = BTreeMap::new();
            if event
                .name()
                .starts_with("external-plugin-event-metadata-injection")
            {
                additions.insert(
                    "external.injector.transport".into(),
                    json!("native_dynamic_rust"),
                );
            }
            Ok(additions)
        },
    )
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
    runtime.emit_metric(
        "fixture.native.metric",
        vec![
            MetricMeasurement::builder()
                .name("fixture.native.count")
                .kind(MetricKind::Counter)
                .value_type(MetricValueType::U64)
                .value(json!(1))
                .build(),
        ],
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

unsafe extern "C" fn fixture_compat_register(
    _user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    _ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

/// Raw ABI-v3 entry used to verify host fallback for already-built plugins.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_native_plugin_v3(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        fixture_compat_entry(
            host,
            out,
            NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE,
            std::mem::size_of::<NemoRelayNativeHostApiV3>(),
            b"fixture_native_v3",
        )
    }
}

/// Raw ABI-v2 entry used to verify host fallback for already-built plugins.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_native_plugin_v2(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        fixture_compat_entry(
            host,
            out,
            NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY,
            std::mem::size_of::<NemoRelayNativeHostApiV1>(),
            b"fixture_native_v2",
        )
    }
}

/// Raw ABI-v4 entry used to verify the final current table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_native_plugin_v4(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe {
        fixture_compat_entry(
            host,
            out,
            NEMO_RELAY_NATIVE_ABI_VERSION,
            std::mem::size_of::<NemoRelayNativeHostApiV4>(),
            b"fixture_native_v4",
        )
    }
}

unsafe fn fixture_compat_entry(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
    expected_version: u32,
    minimum_size: usize,
    kind: &[u8],
) -> NemoRelayStatus {
    if host.is_null() || out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let host = unsafe { &*host };
    if host.abi_version != expected_version || host.struct_size < minimum_size {
        return NemoRelayStatus::InvalidArg;
    }
    let mut plugin_kind = ptr::null_mut();
    let status = unsafe { (host.string_new)(kind.as_ptr(), kind.len(), &mut plugin_kind) };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    unsafe {
        *out = NemoRelayNativePluginV1 {
            struct_size: std::mem::size_of::<NemoRelayNativePluginV1>(),
            plugin_kind,
            allows_multiple_components: false,
            user_data: ptr::null_mut(),
            validate: None,
            register: Some(fixture_compat_register),
            drop: None,
        };
    }
    NemoRelayStatus::Ok
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_fixture_async_entry(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    unsafe { nemo_relay_plugin::export_plugin(host, out, FixtureAsyncPlugin { host: None }) }
}

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
pub unsafe extern "C" fn nemo_relay_fixture_abi_v2_api1(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus {
    if host.is_null() || out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    if unsafe { (*host).abi_version } != NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY {
        return NemoRelayStatus::InvalidArg;
    }
    unsafe {
        write_raw_descriptor(
            host,
            out,
            "fixture_native",
            None,
            None,
            Some(raw_register_canonical_tool_outcome),
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

unsafe extern "C" fn raw_register_canonical_tool_outcome(
    user_data: *mut c_void,
    _plugin_config_json: *const NemoRelayNativeString,
    ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    let Some(host) = (unsafe { raw_host_from_user_data(user_data) }) else {
        return NemoRelayStatus::NullPointer;
    };
    let name = unsafe { raw_host_string(host, "fixture_abi_v2_api1") };
    if name.is_null() {
        return NemoRelayStatus::Internal;
    }
    let status = unsafe {
        (host.plugin_context_register_tool_execution_intercept)(
            ctx,
            name,
            0,
            raw_canonical_tool_outcome_callback,
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

struct FixtureAsyncPlugin {
    host: Option<Box<NemoRelayNativeHostApiV3>>,
}

impl NativePlugin for FixtureAsyncPlugin {
    fn plugin_kind(&self) -> &str {
        "fixture_async"
    }

    fn register(
        &mut self,
        _plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let host = ctx.host_api();
        if host.abi_version < 3
            || host.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV3>()
        {
            return Err("fixture async plugin requires ABI v3".into());
        }
        self.host = Some(Box::new(unsafe {
            *(host as *const _ as *const NemoRelayNativeHostApiV3)
        }));
        let user_data = self
            .host
            .as_deref()
            .map(|host| (host as *const NemoRelayNativeHostApiV3).cast_mut().cast())
            .expect("fixture async host was initialized");

        let registrations: [(
            NemoRelayNativeAsyncMiddlewareKind,
            &str,
            NemoRelayNativeAsyncMiddlewareCb,
        ); 13] = [
            (
                NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest,
                "fixture_async_tool_sanitize_request",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse,
                "fixture_async_tool_sanitize_response",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::ToolConditionalExecution,
                "fixture_async_tool_conditional",
                raw_async_allow_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept,
                "fixture_async_request",
                raw_async_tool_request_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept,
                "fixture_async_execution",
                raw_async_tool_execution_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest,
                "fixture_async_llm_sanitize_request",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse,
                "fixture_async_llm_sanitize_response",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::LlmConditionalExecution,
                "fixture_async_llm_conditional",
                raw_async_allow_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept,
                "fixture_async_llm_request",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept,
                "fixture_async_llm_execution",
                raw_async_tool_execution_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::MarkSanitize,
                "fixture_async_mark",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart,
                "fixture_async_scope_start",
                raw_async_passthrough_callback,
            ),
            (
                NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd,
                "fixture_async_scope_end",
                raw_async_passthrough_callback,
            ),
        ];
        for (kind, registration_name, callback) in registrations {
            let status = unsafe {
                ctx.register_async_middleware_raw(
                    kind,
                    registration_name,
                    0,
                    false,
                    callback,
                    user_data,
                    None,
                )
            };
            if status != NemoRelayStatus::Ok {
                return Err(format!("async registration failed: {status:?}"));
            }
        }
        let status = unsafe {
            ctx.register_async_stream_middleware_raw(
                "fixture_async_llm_stream",
                0,
                raw_async_stream_callback,
                user_data,
                None,
            )
        };
        if status != NemoRelayStatus::Ok {
            return Err(format!("async stream registration failed: {status:?}"));
        }
        Ok(())
    }
}

struct AsyncStreamForward {
    host: NemoRelayNativeHostApiV3,
    stream: *const NemoRelayNativeAsyncStream,
}

unsafe extern "C" fn raw_async_stream_forward(
    user_data: *mut c_void,
    chunk: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) -> bool {
    let state = unsafe { &*(user_data as *const AsyncStreamForward) };
    if !chunk.is_null() {
        if unsafe { (state.host.async_stream_push_json)(state.stream, chunk) }
            == NemoRelayStatus::Ok
        {
            return true;
        }
        unsafe {
            (state.host.async_stream_release)(state.stream);
            drop(Box::from_raw(user_data as *mut AsyncStreamForward));
        }
        return false;
    }
    if !error.is_null() {
        unsafe { (state.host.async_stream_reject)(state.stream, error) };
    } else if done {
        unsafe { (state.host.async_stream_finish)(state.stream) };
    } else {
        return true;
    }
    unsafe {
        (state.host.async_stream_release)(state.stream);
        drop(Box::from_raw(user_data as *mut AsyncStreamForward));
    }
    false
}

unsafe extern "C" fn raw_async_stream_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    stream: *const NemoRelayNativeAsyncStream,
) -> u32 {
    let Some(host) = (unsafe { (user_data as *const NemoRelayNativeHostApiV3).as_ref() }) else {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let request = unsafe { raw_host_string_value(&host.v1, invocation_json) }
        .and_then(|json| serde_json::from_str::<Json>(&json).ok())
        .and_then(|invocation| invocation.get("request").cloned())
        .and_then(|request| serde_json::to_string(&request).ok())
        .map(|request| unsafe { raw_host_string(&host.v1, &request) });
    let Some(request) = request.filter(|request| !request.is_null()) else {
        unsafe {
            (host.async_next_release)(next);
            (host.async_stream_release)(stream);
        }
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let state = Box::into_raw(Box::new(AsyncStreamForward {
        host: *host,
        stream,
    }));
    let status = unsafe {
        (host.async_next_invoke_stream)(
            next,
            request,
            stream,
            raw_async_stream_forward,
            state.cast(),
        )
    };
    unsafe {
        (host.v1.string_free)(request);
        (host.async_next_release)(next);
    }
    if status == NemoRelayStatus::Ok {
        NemoRelayNativeAsyncCallbackState::Pending as u32
    } else {
        unsafe {
            drop(Box::from_raw(state));
            (host.async_stream_release)(stream);
        }
        NemoRelayNativeAsyncCallbackState::Complete as u32
    }
}

unsafe extern "C" fn raw_async_allow_callback(
    user_data: *mut c_void,
    _invocation_json: *const NemoRelayNativeString,
    _next: *const nemo_relay_plugin::NemoRelayNativeAsyncNext,
    completion: *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
) -> u32 {
    let Some(host) = (unsafe { (user_data as *const NemoRelayNativeHostApiV3).as_ref() }) else {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let result = unsafe { raw_host_string(&host.v1, "null") };
    if result.is_null() {
        unsafe {
            reject_async_completion(host, completion, "failed to allocate async allow result")
        };
    } else {
        unsafe {
            (host.async_completion_resolve_json)(completion, result);
            (host.v1.string_free)(result);
        }
    }
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

unsafe extern "C" fn raw_async_passthrough_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    _next: *const nemo_relay_plugin::NemoRelayNativeAsyncNext,
    completion: *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
) -> u32 {
    let Some(host) = (unsafe { (user_data as *const NemoRelayNativeHostApiV3).as_ref() }) else {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let result = unsafe { raw_host_string_value(&host.v1, invocation_json) }
        .and_then(|value| serde_json::from_str::<Json>(&value).ok())
        .and_then(|invocation| {
            invocation
                .get("annotated")
                .map(|annotated| {
                    json!({
                        "request": invocation["request"],
                        "annotated_request": annotated,
                        "pending_marks": [],
                        "optimization_contributions": [],
                    })
                })
                .or_else(|| {
                    ["value", "request", "response", "fields"]
                        .into_iter()
                        .find_map(|key| invocation.get(key).cloned())
                })
        })
        .and_then(|value| serde_json::to_string(&value).ok());
    let Some(result) = result else {
        unsafe {
            reject_async_completion(host, completion, "invalid async passthrough invocation")
        };
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let result = unsafe { raw_host_string(&host.v1, &result) };
    if result.is_null() {
        unsafe {
            reject_async_completion(
                host,
                completion,
                "failed to allocate async passthrough result",
            )
        };
    } else {
        unsafe {
            (host.async_completion_resolve_json)(completion, result);
            (host.v1.string_free)(result);
        }
    }
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

unsafe extern "C" fn raw_async_tool_request_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    _next: *const nemo_relay_plugin::NemoRelayNativeAsyncNext,
    completion: *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
) -> u32 {
    let Some(host) = (unsafe { (user_data as *const NemoRelayNativeHostApiV3).as_ref() }) else {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let invocation = unsafe { raw_host_string_value(&host.v1, invocation_json) }
        .and_then(|json| serde_json::from_str::<Json>(&json).ok())
        .and_then(|mut invocation| {
            let pending = invocation["name"].as_str() == Some("async-pending");
            let duplicate = invocation["name"].as_str() == Some("async-double");
            invocation
                .get_mut("value")
                .and_then(Json::as_object_mut)
                .map(|value| {
                    value.insert("native_async".into(), json!(true));
                    (Json::Object(value.clone()), pending, duplicate)
                })
        })
        .and_then(|(value, pending, duplicate)| {
            serde_json::to_string(&value)
                .ok()
                .map(|value| (value, pending, duplicate))
        });
    let Some((result, pending, duplicate)) = invocation else {
        unsafe {
            reject_async_completion(host, completion, "invalid async tool request invocation")
        };
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    if pending {
        ASYNC_PENDING_ENTERED.store(true, Ordering::Release);
        let host = *host;
        let completion = completion as usize;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let result = unsafe { raw_host_string(&host.v1, &result) };
            if !result.is_null() {
                unsafe {
                    (host.async_completion_resolve_json)(
                        completion as *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
                        result,
                    );
                    (host.v1.string_free)(result);
                    (host.async_completion_release)(
                        completion as *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
                    );
                }
            } else {
                unsafe {
                    reject_async_completion(
                        &host,
                        completion as *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
                        "failed to allocate async tool request result",
                    );
                    (host.async_completion_release)(
                        completion as *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
                    );
                }
            }
        });
        return NemoRelayNativeAsyncCallbackState::Pending as u32;
    }
    let result = unsafe { raw_host_string(&host.v1, &result) };
    if !result.is_null() {
        unsafe {
            (host.async_completion_resolve_json)(completion, result);
            if duplicate {
                let _ = (host.async_completion_resolve_json)(completion, result);
            }
            (host.v1.string_free)(result);
        }
    } else {
        unsafe {
            reject_async_completion(
                host,
                completion,
                "failed to allocate async tool request result",
            )
        };
    }
    NemoRelayNativeAsyncCallbackState::Complete as u32
}

unsafe extern "C" fn raw_async_tool_execution_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const nemo_relay_plugin::NemoRelayNativeAsyncNext,
    completion: *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
) -> u32 {
    let Some(host) = (unsafe { (user_data as *const NemoRelayNativeHostApiV3).as_ref() }) else {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    if next.is_null() || completion.is_null() {
        unsafe {
            reject_async_completion(
                host,
                completion,
                "async tool execution requires next and completion",
            )
        };
        if !next.is_null() {
            unsafe { (host.async_next_release)(next) };
        }
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let value = unsafe { raw_host_string_value(&host.v1, invocation_json) }
        .and_then(|json| serde_json::from_str::<Json>(&json).ok())
        .and_then(|mut invocation| {
            let cancellation_probe = invocation["name"].as_str() == Some("async-cancel-next");
            let value =
                if let Some(value) = invocation.get_mut("value").and_then(Json::as_object_mut) {
                    value.insert("native_async_execution".into(), json!(true));
                    Some(Json::Object(value.clone()))
                } else {
                    invocation.get("request").cloned()
                };
            value.map(|value| (value, cancellation_probe))
        })
        .and_then(|(value, cancellation_probe)| {
            serde_json::to_string(&value)
                .ok()
                .map(|value| (value, cancellation_probe))
        });
    let Some((value, cancellation_probe)) = value else {
        unsafe {
            reject_async_completion(host, completion, "invalid async tool execution invocation")
        };
        unsafe { (host.async_next_release)(next) };
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    let value = unsafe { raw_host_string(&host.v1, &value) };
    if value.is_null() {
        unsafe {
            reject_async_completion(
                host,
                completion,
                "failed to allocate async tool execution invocation",
            )
        };
        unsafe { (host.async_next_release)(next) };
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let status = unsafe { (host.async_next_invoke)(next, value, completion) };
    unsafe {
        (host.v1.string_free)(value);
    }
    if status == NemoRelayStatus::Ok {
        if cancellation_probe {
            ASYNC_PENDING_ENTERED.store(true, Ordering::Release);
        }
        unsafe {
            (host.async_next_release)(next);
            (host.async_completion_release)(completion);
        }
        NemoRelayNativeAsyncCallbackState::Pending as u32
    } else {
        unsafe {
            reject_async_completion(
                host,
                completion,
                "failed to invoke async tool execution next",
            )
        };
        unsafe { (host.async_next_release)(next) };
        NemoRelayNativeAsyncCallbackState::Complete as u32
    }
}

unsafe fn reject_async_completion(
    host: &NemoRelayNativeHostApiV3,
    completion: *const nemo_relay_plugin::NemoRelayNativeAsyncCompletion,
    message: &str,
) {
    if completion.is_null() {
        return;
    }
    let message = unsafe { raw_host_string(&host.v1, message) };
    if message.is_null() {
        return;
    }
    unsafe {
        let _ = (host.async_completion_reject)(completion, message);
        (host.v1.string_free)(message);
    }
}

unsafe extern "C" fn raw_tool_outcome_callback(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeToolNextFn,
    next_ctx: *mut c_void,
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
                *out_outcome_json =
                    raw_host_string(host, r#"{"result":{"stale":true},"pending_marks":[]}"#);
                set_raw_last_error_from_user_data(user_data, "fixture tool execution failed");
            }
            NemoRelayStatus::Internal
        }
        _ => {
            let mut next_result = ptr::null_mut();
            let status = unsafe { next_fn(args_json, next_ctx, &mut next_result) };
            if status != NemoRelayStatus::Ok {
                return status;
            }
            let encoded = unsafe { raw_host_string_value(host, next_result) };
            unsafe { (host.string_free)(next_result) };
            let Some(encoded) = encoded else {
                return NemoRelayStatus::InvalidUtf8;
            };
            let Ok(downstream) = serde_json::from_str::<Json>(&encoded) else {
                return NemoRelayStatus::InvalidArg;
            };
            let Ok(encoded) = serde_json::to_string(&json!({
                "result": {
                    "raw_tool_outcome": true,
                    "downstream": downstream,
                },
                "pending_marks": [],
            })) else {
                return NemoRelayStatus::Internal;
            };
            let output = unsafe { raw_host_string(host, &encoded) };
            if output.is_null() {
                return NemoRelayStatus::Internal;
            }
            unsafe { *out_outcome_json = output };
            NemoRelayStatus::Ok
        }
    }
}

unsafe extern "C" fn raw_canonical_tool_outcome_callback(
    user_data: *mut c_void,
    _name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeToolNextFn,
    next_ctx: *mut c_void,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out_outcome_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_outcome_json = ptr::null_mut() };
    let Some(host) = (unsafe { raw_host_from_user_data(user_data) }) else {
        return NemoRelayStatus::NullPointer;
    };
    let mut next_result = ptr::null_mut();
    let status = unsafe { next_fn(args_json, next_ctx, &mut next_result) };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let encoded = unsafe { raw_host_string_value(host, next_result) };
    unsafe { (host.string_free)(next_result) };
    let Some(encoded) = encoded else {
        return NemoRelayStatus::InvalidUtf8;
    };
    let Ok(mut outcome) = serde_json::from_str::<Json>(&encoded) else {
        return NemoRelayStatus::InvalidArg;
    };
    let Some(outcome) = outcome.as_object_mut() else {
        return NemoRelayStatus::InvalidArg;
    };
    outcome.insert("pending_marks".into(), Json::Array(Vec::new()));
    let Ok(encoded) = serde_json::to_string(&outcome) else {
        return NemoRelayStatus::Internal;
    };
    let output = unsafe { raw_host_string(host, &encoded) };
    if output.is_null() {
        return NemoRelayStatus::Internal;
    }
    unsafe { *out_outcome_json = output };
    NemoRelayStatus::Ok
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
