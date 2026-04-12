// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use nemo_flow::plugin::PluginRegistrationContext;
use serde_json::{Value as Json, json};
use uuid::Uuid;

use crate::convert::nemo_flow_string_free;
use crate::error::{NemoFlowStatus, nemo_flow_last_error};
use crate::types::{
    FfiAtifExporter, FfiEvent, FfiLLMHandle, FfiLLMRequest, FfiOpenTelemetrySubscriber,
    FfiScopeStack, FfiToolHandle, nemo_flow_atif_exporter_free, nemo_flow_event_data,
    nemo_flow_event_input, nemo_flow_event_metadata, nemo_flow_event_model_name,
    nemo_flow_event_name, nemo_flow_event_output, nemo_flow_event_parent_uuid,
    nemo_flow_event_scope_type, nemo_flow_event_timestamp, nemo_flow_event_tool_call_id,
    nemo_flow_event_uuid, nemo_flow_llm_handle_attributes, nemo_flow_llm_handle_free,
    nemo_flow_llm_handle_name, nemo_flow_llm_handle_parent_uuid, nemo_flow_llm_handle_uuid,
    nemo_flow_llm_request_content, nemo_flow_llm_request_free, nemo_flow_llm_request_headers,
    nemo_flow_llm_request_new, nemo_flow_otel_subscriber_free, nemo_flow_scope_handle_attributes,
    nemo_flow_scope_handle_data, nemo_flow_scope_handle_free, nemo_flow_scope_handle_metadata,
    nemo_flow_scope_handle_name, nemo_flow_scope_handle_parent_uuid,
    nemo_flow_scope_handle_scope_type, nemo_flow_scope_handle_uuid, nemo_flow_scope_stack_free,
    nemo_flow_tool_handle_attributes, nemo_flow_tool_handle_free, nemo_flow_tool_handle_name,
    nemo_flow_tool_handle_parent_uuid, nemo_flow_tool_handle_uuid,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());
static EVENT_LOG: OnceLock<Mutex<Vec<Json>>> = OnceLock::new();
static COLLECTED_CHUNKS: OnceLock<Mutex<Vec<Json>>> = OnceLock::new();
static FINALIZER_CALLS: OnceLock<Mutex<usize>> = OnceLock::new();
static HOSTED_PLUGIN_FREES: OnceLock<Mutex<usize>> = OnceLock::new();

fn event_log() -> &'static Mutex<Vec<Json>> {
    EVENT_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn collected_chunks() -> &'static Mutex<Vec<Json>> {
    COLLECTED_CHUNKS.get_or_init(|| Mutex::new(Vec::new()))
}

fn finalizer_calls() -> &'static Mutex<usize> {
    FINALIZER_CALLS.get_or_init(|| Mutex::new(0))
}

fn hosted_plugin_frees() -> &'static Mutex<usize> {
    HOSTED_PLUGIN_FREES.get_or_init(|| Mutex::new(0))
}

fn unique_name(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::now_v7().simple())
}

fn lock_unpoisoned<T>(mutex: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn take_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { nemo_flow_string_free(ptr) };
    Some(s)
}

unsafe fn read_last_error() -> Option<String> {
    let ptr = nemo_flow_last_error();
    if ptr.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

unsafe fn returned_json(ptr: *mut c_char) -> Json {
    serde_json::from_str(&unsafe { take_string(ptr) }.unwrap()).unwrap()
}

unsafe fn fresh_scope_stack() -> *mut FfiScopeStack {
    let mut stack = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_scope_stack_create(&mut stack) },
        NemoFlowStatus::Ok
    );
    assert!(!stack.is_null());
    assert_eq!(
        unsafe { nemo_flow_scope_stack_set_thread(stack) },
        NemoFlowStatus::Ok
    );
    stack
}

fn reset_globals() {
    lock_unpoisoned(event_log()).clear();
    lock_unpoisoned(collected_chunks()).clear();
    *lock_unpoisoned(finalizer_calls()) = 0;
    *lock_unpoisoned(hosted_plugin_frees()) = 0;
}

unsafe extern "C" fn subscriber_cb(_user_data: *mut libc::c_void, event: *const FfiEvent) {
    let payload = json!({
        "uuid": unsafe { take_string(nemo_flow_event_uuid(event)) }.unwrap_or_default(),
        "name": unsafe { take_string(nemo_flow_event_name(event)) }.unwrap_or_default(),
        "kind": unsafe { take_string(crate::types::nemo_flow_event_kind(event)) }.unwrap_or_default(),
        "data": unsafe { take_string(nemo_flow_event_data(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "metadata": unsafe { take_string(nemo_flow_event_metadata(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "timestamp": unsafe { take_string(nemo_flow_event_timestamp(event)) }.unwrap_or_default(),
        "input": unsafe { take_string(nemo_flow_event_input(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "output": unsafe { take_string(nemo_flow_event_output(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "model_name": unsafe { take_string(nemo_flow_event_model_name(event)) },
        "tool_call_id": unsafe { take_string(nemo_flow_event_tool_call_id(event)) },
        "parent_uuid": unsafe { take_string(nemo_flow_event_parent_uuid(event)) },
        "scope_type": unsafe { take_string(nemo_flow_event_scope_type(event)) },
    });
    lock_unpoisoned(event_log()).push(payload);
}

unsafe extern "C" fn tool_request_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    let mut args: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(args_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    args["intercepted"] = json!(true);
    CString::new(args.to_string()).unwrap().into_raw()
}

#[test]
fn test_ffi_plugin_config_validate_initialize_and_clear() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    let config = cstring(
        &json!({
            "version": 1,
            "components": [
                {
                    "kind": "adaptive",
                    "enabled": true,
                    "config": {
                        "version": 1,
                        "state": {
                            "backend": {
                                "kind": "in_memory",
                                "config": {}
                            }
                        },
                        "telemetry": {
                            "learners": ["latency_sensitivity"]
                        },
                        "adaptive_hints": {},
                        "tool_parallelism": {}
                    }
                }
            ]
        })
        .to_string(),
    );

    let mut report_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_validate_plugin_config(config.as_ptr(), &mut report_json) },
        NemoFlowStatus::Ok
    );
    let report = unsafe { returned_json(report_json) };
    assert_eq!(report["diagnostics"], json!([]));

    let mut kinds_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_list_plugin_kinds_json(&mut kinds_json) },
        NemoFlowStatus::Ok
    );
    let kinds = unsafe { returned_json(kinds_json) };
    assert!(
        kinds
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "adaptive"))
    );

    let mut configured_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_initialize_plugins(config.as_ptr(), &mut configured_json) },
        NemoFlowStatus::Ok
    );
    let configured_report = unsafe { returned_json(configured_json) };
    assert_eq!(configured_report["diagnostics"], json!([]));

    let mut active_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_active_plugin_report_json(&mut active_json) },
        NemoFlowStatus::Ok
    );
    let active_report = unsafe { returned_json(active_json) };
    assert_eq!(active_report["diagnostics"], json!([]));

    assert_eq!(nemo_flow_clear_plugin_configuration(), NemoFlowStatus::Ok);

    let mut cleared_json = ptr::null_mut();
    assert_eq!(
        unsafe { nemo_flow_active_plugin_report_json(&mut cleared_json) },
        NemoFlowStatus::Ok
    );
    assert_eq!(unsafe { returned_json(cleared_json) }, Json::Null);
}

unsafe extern "C" fn tool_allow_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _args_json: *const c_char,
) -> *mut c_char {
    ptr::null_mut()
}

unsafe extern "C" fn tool_reject_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _args_json: *const c_char,
) -> *mut c_char {
    CString::new("blocked tool").unwrap().into_raw()
}

unsafe extern "C" fn tool_exec_cb(
    _user_data: *mut libc::c_void,
    args_json: *const c_char,
) -> *mut c_char {
    let mut args: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(args_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    args["executed"] = json!(true);
    CString::new(args.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn tool_exec_intercept_cb(
    _user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: crate::callable::NemoFlowToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    unsafe { next_fn(args_json, next_ctx) }
}

unsafe extern "C" fn llm_request_cb(
    _user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest {
    let headers = unsafe { take_string(nemo_flow_llm_request_headers(request)) }
        .unwrap_or_else(|| "{}".to_string());
    let content = unsafe { take_string(nemo_flow_llm_request_content(request)) }
        .unwrap_or_else(|| "null".to_string());
    let mut content_json: Json = serde_json::from_str(&content).unwrap();
    content_json["intercepted"] = json!(true);
    let headers_c = CString::new(headers).unwrap();
    let content_c = CString::new(content_json.to_string()).unwrap();
    unsafe { nemo_flow_llm_request_new(headers_c.as_ptr(), content_c.as_ptr()) }
}

/// Intercept-specific callback with the unified annotated-aware signature.
/// Modifies the request content (sets `intercepted: true`) and passes through
/// any annotated JSON unchanged.
unsafe extern "C" fn llm_request_intercept_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    request: *const FfiLLMRequest,
    annotated_json: *const c_char,
    out_request: *mut *mut FfiLLMRequest,
    out_annotated_json: *mut *mut c_char,
) -> NemoFlowStatus {
    let headers = unsafe { take_string(nemo_flow_llm_request_headers(request)) }
        .unwrap_or_else(|| "{}".to_string());
    let content = unsafe { take_string(nemo_flow_llm_request_content(request)) }
        .unwrap_or_else(|| "null".to_string());
    let mut content_json: Json = serde_json::from_str(&content).unwrap();
    content_json["intercepted"] = json!(true);
    let headers_c = CString::new(headers).unwrap();
    let content_c = CString::new(content_json.to_string()).unwrap();
    unsafe { *out_request = nemo_flow_llm_request_new(headers_c.as_ptr(), content_c.as_ptr()) };
    // Pass through annotated JSON if present
    if annotated_json.is_null() {
        unsafe { *out_annotated_json = ptr::null_mut() };
    } else {
        let s = unsafe { CStr::from_ptr(annotated_json) }
            .to_string_lossy()
            .into_owned();
        unsafe { *out_annotated_json = CString::new(s).unwrap().into_raw() };
    }
    NemoFlowStatus::Ok
}

unsafe extern "C" fn llm_allow_cb(
    _user_data: *mut libc::c_void,
    _request: *const FfiLLMRequest,
) -> *mut c_char {
    ptr::null_mut()
}

unsafe extern "C" fn llm_reject_cb(
    _user_data: *mut libc::c_void,
    _request: *const FfiLLMRequest,
) -> *mut c_char {
    CString::new("blocked llm").unwrap().into_raw()
}

unsafe extern "C" fn llm_response_cb(
    _user_data: *mut libc::c_void,
    response_json: *const c_char,
) -> *mut c_char {
    let mut response: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(response_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    response["sanitized"] = json!(true);
    CString::new(response.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn llm_exec_cb(
    _user_data: *mut libc::c_void,
    native_json: *const c_char,
) -> *mut c_char {
    let request: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(native_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    let model = request
        .get("content")
        .and_then(|v| v.get("model"))
        .cloned()
        .unwrap_or(Json::Null);
    let response = json!({
        "content": "hello from ffi",
        "role": "assistant",
        "tool_calls": [],
        "model_seen": model,
    });
    CString::new(response.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn llm_exec_intercept_cb(
    _user_data: *mut libc::c_void,
    native_json: *const c_char,
    next_fn: crate::callable::NemoFlowLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    unsafe { next_fn(native_json, next_ctx) }
}

unsafe extern "C" fn collector_cb(chunk: *const c_char) {
    let chunk: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(chunk) }.to_str().unwrap_or("null")).unwrap();
    lock_unpoisoned(collected_chunks()).push(chunk);
}

unsafe extern "C" fn finalizer_cb() -> *mut c_char {
    *lock_unpoisoned(finalizer_calls()) += 1;
    CString::new(json!({"finalized": true}).to_string())
        .unwrap()
        .into_raw()
}

unsafe extern "C" fn hosted_plugin_free(user_data: *mut libc::c_void) {
    *lock_unpoisoned(hosted_plugin_frees()) += 1;
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data as *mut usize) });
    }
}

unsafe extern "C" fn hosted_plugin_validate_warn(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
) -> *mut c_char {
    CString::new(
        json!([{
            "level": "warning",
            "code": "hosted.warning",
            "component": "ffi.hosted",
            "message": "hosted validation ran"
        }])
        .to_string(),
    )
    .unwrap()
    .into_raw()
}

unsafe extern "C" fn hosted_plugin_validate_invalid(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
) -> *mut c_char {
    CString::new("not-json").unwrap().into_raw()
}

unsafe extern "C" fn hosted_plugin_validate_null(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
) -> *mut c_char {
    ptr::null_mut()
}

unsafe extern "C" fn hosted_plugin_register_subscriber(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
    ctx: *mut FfiPluginContext,
) -> NemoFlowStatus {
    let name = CString::new("subscriber").unwrap();
    unsafe {
        nemo_flow_plugin_context_register_subscriber(
            ctx,
            name.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        )
    }
}

unsafe extern "C" fn hosted_plugin_register_fail(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
    _ctx: *mut FfiPluginContext,
) -> NemoFlowStatus {
    NemoFlowStatus::Internal
}

#[test]
fn test_ffi_plugin_top_level_null_and_invalid_paths() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    let valid_config = cstring(
        &json!({
            "version": 1,
            "components": []
        })
        .to_string(),
    );
    let invalid_json = cstring("{");
    let invalid_shape = cstring(r#"{"version":"bad","components":"nope"}"#);

    unsafe {
        assert_eq!(
            nemo_flow_validate_plugin_config(valid_config.as_ptr(), ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("out_json pointer is null")
        );

        let mut out_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_validate_plugin_config(invalid_json.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_validate_plugin_config(invalid_shape.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("invalid type")
        );

        assert_eq!(
            nemo_flow_initialize_plugins(valid_config.as_ptr(), ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_initialize_plugins(invalid_json.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_initialize_plugins(invalid_shape.as_ptr(), &mut out_json),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_active_plugin_report_json(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_list_plugin_kinds_json(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_register_plugin(
                ptr::null(),
                None,
                hosted_plugin_register_fail,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_deregister_plugin(ptr::null()),
            NemoFlowStatus::NullPointer
        );
    }
}

#[test]
fn test_ffi_error_paths_and_scope_stack() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        assert_eq!(
            nemo_flow_get_handle(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert!(read_last_error().unwrap().contains("out pointer is null"));

        let name = cstring("ffi_invalid_scope");
        let invalid_json = cstring("{");
        let mut handle = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                name.as_ptr(),
                NemoFlowScopeType::Agent,
                ptr::null(),
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::InvalidJson
        );

        let stack = fresh_scope_stack();
        assert!(nemo_flow_scope_stack_active());

        let mut root = ptr::null_mut();
        assert_eq!(nemo_flow_get_handle(&mut root), NemoFlowStatus::Ok);
        let root_uuid = take_string(nemo_flow_scope_handle_uuid(root)).unwrap();
        assert!(!root_uuid.is_empty());
        assert_eq!(
            nemo_flow_scope_handle_scope_type(root) as i32,
            NemoFlowScopeType::Agent as i32
        );
        assert_eq!(nemo_flow_scope_handle_attributes(root), 0);
        nemo_flow_scope_handle_free(root);

        let scope_name = cstring("ffi_scope");
        let scope_data = cstring(r#"{"scope":true}"#);
        let scope_metadata = cstring(r#"{"meta":"ok"}"#);
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                scope_name.as_ptr(),
                NemoFlowScopeType::Function,
                ptr::null(),
                1,
                scope_data.as_ptr(),
                scope_metadata.as_ptr(),
                &mut scope,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            take_string(nemo_flow_scope_handle_name(scope)).unwrap(),
            "ffi_scope"
        );
        assert_eq!(
            nemo_flow_scope_handle_scope_type(scope) as i32,
            NemoFlowScopeType::Function as i32
        );
        assert_eq!(nemo_flow_scope_handle_attributes(scope), 1);
        assert!(take_string(nemo_flow_scope_handle_parent_uuid(scope)).is_some());
        assert_eq!(
            serde_json::from_str::<Json>(&take_string(nemo_flow_scope_handle_data(scope)).unwrap())
                .unwrap(),
            json!({"scope": true})
        );
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_flow_scope_handle_metadata(scope)).unwrap()
            )
            .unwrap(),
            json!({"meta": "ok"})
        );
        assert_eq!(nemo_flow_pop_scope(scope), NemoFlowStatus::Ok);
        nemo_flow_scope_handle_free(scope);

        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_tool_lifecycle_execute_and_helpers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let subscriber_name = unique_name("ffi_subscriber");
        let subscriber_name_c = cstring(&subscriber_name);
        assert_eq!(
            nemo_flow_register_subscriber(
                subscriber_name_c.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let intercept_name = unique_name("ffi_tool_intercept");
        let intercept_name_c = cstring(&intercept_name);
        assert_eq!(
            nemo_flow_register_tool_request_intercept(
                intercept_name_c.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let conditional_name = unique_name("ffi_tool_conditional");
        let conditional_name_c = cstring(&conditional_name);
        assert_eq!(
            nemo_flow_register_tool_conditional_execution_guardrail(
                conditional_name_c.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let tool_name = cstring("ffi_tool");
        let args = cstring(r#"{"value": 1}"#);
        let mut intercepted_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_tool_request_intercepts(
                tool_name.as_ptr(),
                args.as_ptr(),
                &mut intercepted_out
            ),
            NemoFlowStatus::Ok
        );
        let intercepted_json = returned_json(intercepted_out);
        assert_eq!(intercepted_json["intercepted"], json!(true));

        assert_eq!(
            nemo_flow_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
            NemoFlowStatus::Ok
        );

        let tool_call_id = cstring("call_ffi_123");
        let metadata = cstring(r#"{"source":"ffi-test"}"#);
        let mut handle: *mut FfiToolHandle = ptr::null_mut();
        assert_eq!(
            nemo_flow_tool_call(
                tool_name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                1,
                ptr::null(),
                metadata.as_ptr(),
                tool_call_id.as_ptr(),
                &mut handle,
            ),
            NemoFlowStatus::Ok
        );
        assert!(take_string(nemo_flow_tool_handle_uuid(handle)).is_some());
        assert_eq!(
            take_string(nemo_flow_tool_handle_name(handle)).unwrap(),
            "ffi_tool"
        );
        assert_eq!(nemo_flow_tool_handle_attributes(handle), 1);
        assert!(take_string(nemo_flow_tool_handle_parent_uuid(handle)).is_some());

        let result = cstring(r#"{"ok": true}"#);
        assert_eq!(
            nemo_flow_tool_call_end(handle, result.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_tool_handle_free(handle);

        let mut execute_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_tool_call_execute(
                tool_name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut execute_out,
            ),
            NemoFlowStatus::Ok
        );
        let executed_json = returned_json(execute_out);
        assert_eq!(executed_json["intercepted"], json!(true));
        assert_eq!(executed_json["executed"], json!(true));

        let events = lock_unpoisoned(event_log()).clone();
        assert!(events.iter().any(|event| event["name"] == "ffi_tool"));
        assert!(
            events
                .iter()
                .any(|event| event["tool_call_id"] == "call_ffi_123")
        );
        assert!(
            events
                .iter()
                .any(|event| event["timestamp"].as_str().is_some_and(|s| !s.is_empty()))
        );

        let mark_name = cstring("ffi_mark");
        let mark_data = cstring(r#"{"mark":true}"#);
        let mark_metadata = cstring(r#"{"origin":"ffi"}"#);
        assert_eq!(
            nemo_flow_event(
                mark_name.as_ptr(),
                ptr::null(),
                mark_data.as_ptr(),
                mark_metadata.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );
        let events = lock_unpoisoned(event_log()).clone();
        assert!(events.iter().any(|event| {
            event["name"] == "ffi_mark"
                && event["kind"] == json!("Mark")
                && event["data"] == json!({"mark": true})
                && event["metadata"] == json!({"origin": "ffi"})
        }));

        assert_eq!(
            nemo_flow_deregister_tool_request_intercept(intercept_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_tool_conditional_execution_guardrail(conditional_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_additional_null_and_invalid_json_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let name = cstring("ffi_edge_paths");
        let args = cstring(r#"{"value": 1}"#);
        let invalid_json = cstring("{");
        let invalid_request_shape = cstring(r#"{"headers":[],"content":"bad"}"#);
        let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model"}}"#);
        let mut handle: *mut FfiToolHandle = ptr::null_mut();
        let mut llm_handle: *mut FfiLLMHandle = ptr::null_mut();
        let mut out_json: *mut c_char = ptr::null_mut();
        let mut stream: *mut FfiStream = ptr::null_mut();

        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                invalid_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_tool_call(
                name.as_ptr(),
                args.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut handle,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_tool_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_call_end(handle, invalid_json.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_call_end(handle, args.as_ptr(), invalid_json.as_ptr(), ptr::null(),),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_call_end(handle, args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_tool_handle_free(handle);

        assert_eq!(
            nemo_flow_tool_call_execute(
                name.as_ptr(),
                args.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_call_execute(
                name.as_ptr(),
                invalid_json.as_ptr(),
                tool_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut out_json,
            ),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoFlowStatus::InvalidJson
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("failed to parse native_json as LLMRequest")
        );

        assert_eq!(
            nemo_flow_llm_call(
                name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut llm_handle,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_llm_call_end(ptr::null(), args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_call_end(llm_handle, invalid_json.as_ptr(), ptr::null(), ptr::null(),),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_llm_call_end(llm_handle, args.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_llm_handle_free(llm_handle);

        assert_eq!(
            nemo_flow_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_call_execute(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut out_json,
            ),
            NemoFlowStatus::InvalidJson
        );

        assert_eq!(
            nemo_flow_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_llm_stream_call_execute(
                name.as_ptr(),
                invalid_request_shape.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                None,
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut stream,
            ),
            NemoFlowStatus::InvalidJson
        );

        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_registration_and_exporter_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        assert_eq!(
            nemo_flow_scope_stack_create(ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_scope_stack_set_thread(ptr::null()),
            NemoFlowStatus::NullPointer
        );

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_scope_local");
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                scope_name.as_ptr(),
                NemoFlowScopeType::Function,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut scope,
            ),
            NemoFlowStatus::Ok
        );
        let scope_uuid = cstring(&take_string(nemo_flow_scope_handle_uuid(scope)).unwrap());
        let invalid_uuid = cstring("not-a-uuid");

        let global_tool_san_req = cstring(&unique_name("ffi_tool_san_req"));
        assert_eq!(
            nemo_flow_register_tool_sanitize_request_guardrail(
                global_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_register_tool_sanitize_request_guardrail(
                global_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::AlreadyExists
        );
        assert_eq!(
            nemo_flow_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_tool_sanitize_request_guardrail(global_tool_san_req.as_ptr()),
            NemoFlowStatus::Ok
        );

        let global_tool_san_resp = cstring(&unique_name("ffi_tool_san_resp"));
        assert_eq!(
            nemo_flow_register_tool_sanitize_response_guardrail(
                global_tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_tool_sanitize_response_guardrail(global_tool_san_resp.as_ptr()),
            NemoFlowStatus::Ok
        );

        let global_tool_exec = cstring(&unique_name("ffi_tool_exec"));
        assert_eq!(
            nemo_flow_register_tool_execution_intercept(
                global_tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_tool_execution_intercept(global_tool_exec.as_ptr()),
            NemoFlowStatus::Ok
        );

        let global_llm_san_req = cstring(&unique_name("ffi_llm_san_req"));
        assert_eq!(
            nemo_flow_register_llm_sanitize_request_guardrail(
                global_llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_llm_sanitize_request_guardrail(global_llm_san_req.as_ptr()),
            NemoFlowStatus::Ok
        );

        let global_llm_exec = cstring(&unique_name("ffi_llm_exec"));
        assert_eq!(
            nemo_flow_register_llm_execution_intercept(
                global_llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_llm_execution_intercept(global_llm_exec.as_ptr()),
            NemoFlowStatus::Ok
        );

        let global_llm_stream_exec = cstring(&unique_name("ffi_llm_stream_exec"));
        assert_eq!(
            nemo_flow_register_llm_stream_execution_intercept(
                global_llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_llm_stream_execution_intercept(global_llm_stream_exec.as_ptr()),
            NemoFlowStatus::Ok
        );

        let scope_tool_san_req = cstring(&unique_name("scope_tool_san_req"));
        assert_eq!(
            nemo_flow_scope_register_tool_sanitize_request_guardrail(
                invalid_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::InvalidArg
        );
        assert_eq!(
            nemo_flow_scope_register_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_tool_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_req.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_tool_san_resp = cstring(&unique_name("scope_tool_san_resp"));
        assert_eq!(
            nemo_flow_scope_register_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_resp.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_tool_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_san_resp.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_tool_cond = cstring(&unique_name("scope_tool_cond"));
        assert_eq!(
            nemo_flow_scope_register_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_tool_req = cstring(&unique_name("scope_tool_req"));
        assert_eq!(
            nemo_flow_scope_register_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_tool_exec = cstring(&unique_name("scope_tool_exec"));
        assert_eq!(
            nemo_flow_scope_register_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_tool_exec.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_tool_execution_intercept(
                scope_uuid.as_ptr(),
                scope_tool_exec.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_san_req = cstring(&unique_name("scope_llm_san_req"));
        assert_eq!(
            nemo_flow_scope_register_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_req.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_sanitize_request_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_req.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_san_resp = cstring(&unique_name("scope_llm_san_resp"));
        assert_eq!(
            nemo_flow_scope_register_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_sanitize_response_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_san_resp.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_cond = cstring(&unique_name("scope_llm_cond"));
        assert_eq!(
            nemo_flow_scope_register_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_req = cstring(&unique_name("scope_llm_req"));
        assert_eq!(
            nemo_flow_scope_register_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_exec = cstring(&unique_name("scope_llm_exec"));
        assert_eq!(
            nemo_flow_scope_register_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_exec.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_stream_exec = cstring(&unique_name("scope_llm_stream_exec"));
        assert_eq!(
            nemo_flow_scope_register_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_stream_exec.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_stream_execution_intercept(
                scope_uuid.as_ptr(),
                scope_llm_stream_exec.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_subscriber = cstring(&unique_name("scope_subscriber"));
        assert_eq!(
            nemo_flow_scope_register_subscriber(
                scope_uuid.as_ptr(),
                scope_subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoFlowStatus::Ok
        );

        let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
        let session = cstring("ffi-session");
        let agent = cstring("ffi-agent");
        let version = cstring("1.0.0");
        assert_eq!(
            nemo_flow_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_atif_exporter_register(ptr::null(), scope_subscriber.as_ptr()),
            NemoFlowStatus::NullPointer
        );
        let mut null_export = ptr::null_mut();
        assert_eq!(
            nemo_flow_atif_exporter_export(ptr::null(), &mut null_export),
            NemoFlowStatus::NullPointer
        );
        let exporter_name = cstring(&unique_name("ffi_exporter_sub"));
        assert_eq!(
            nemo_flow_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoFlowStatus::AlreadyExists
        );
        assert_eq!(
            nemo_flow_atif_exporter_export(exporter, ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_atif_exporter_clear(ptr::null()),
            NemoFlowStatus::NullPointer
        );
        let missing_exporter = cstring("missing_exporter");
        assert_eq!(
            nemo_flow_atif_exporter_deregister(missing_exporter.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoFlowStatus::Ok
        );
        nemo_flow_atif_exporter_free(exporter);

        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_flow_stream_next(ptr::null_mut(), &mut chunk), -1);
        assert_eq!(nemo_flow_stream_next(ptr::null_mut(), ptr::null_mut()), -1);

        assert_eq!(nemo_flow_pop_scope(scope), NemoFlowStatus::Ok);
        nemo_flow_scope_handle_free(scope);
        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_open_telemetry_subscriber_lifecycle_and_errors() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut subscriber: *mut FfiOpenTelemetrySubscriber = ptr::null_mut();
        let endpoint = cstring("http://localhost:4318/v1/traces");
        let headers = cstring(r#"{"authorization":"Bearer token"}"#);
        let resource_attributes = cstring(r#"{"deployment.environment":"test"}"#);
        let service_name = cstring("ffi-agent");
        let service_namespace = cstring("agents");
        let service_version = cstring("1.0.0");
        let instrumentation_scope = cstring("ffi-tests");
        let invalid_transport = cstring("invalid");
        let invalid_headers = cstring(r#"{"authorization":1}"#);

        assert_eq!(
            nemo_flow_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_otel_subscriber_create(
                invalid_transport.as_ptr(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoFlowStatus::InvalidArg
        );
        assert_eq!(
            nemo_flow_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                invalid_headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoFlowStatus::InvalidArg
        );
        assert_eq!(
            nemo_flow_otel_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoFlowStatus::Ok
        );
        assert!(!subscriber.is_null());

        let name = cstring(&unique_name("ffi_otel"));
        assert_eq!(
            nemo_flow_otel_subscriber_register(ptr::null(), name.as_ptr()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_otel_subscriber_force_flush(ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_otel_subscriber_shutdown(ptr::null()),
            NemoFlowStatus::NullPointer
        );

        assert_eq!(
            nemo_flow_otel_subscriber_register(subscriber, name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_otel_subscriber_deregister(name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_otel_subscriber_deregister(name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_otel_subscriber_force_flush(subscriber),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_otel_subscriber_shutdown(subscriber),
            NemoFlowStatus::Ok
        );
        nemo_flow_otel_subscriber_free(subscriber);
    }
}

#[test]
fn test_ffi_open_inference_subscriber_lifecycle_and_errors() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let mut subscriber: *mut FfiOpenInferenceSubscriber = ptr::null_mut();
        let endpoint = cstring("http://localhost:4318/v1/traces");
        let headers = cstring(r#"{"authorization":"Bearer token"}"#);
        let resource_attributes = cstring(r#"{"deployment.environment":"test"}"#);
        let service_name = cstring("ffi-agent");
        let service_namespace = cstring("agents");
        let service_version = cstring("1.0.0");
        let instrumentation_scope = cstring("ffi-tests");
        let invalid_transport = cstring("invalid");
        let invalid_headers = cstring(r#"{"authorization":1}"#);

        assert_eq!(
            nemo_flow_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                ptr::null_mut(),
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_create(
                invalid_transport.as_ptr(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoFlowStatus::InvalidArg
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                invalid_headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoFlowStatus::InvalidArg
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_create(
                ptr::null(),
                endpoint.as_ptr(),
                headers.as_ptr(),
                resource_attributes.as_ptr(),
                service_name.as_ptr(),
                service_namespace.as_ptr(),
                service_version.as_ptr(),
                instrumentation_scope.as_ptr(),
                1250,
                &mut subscriber,
            ),
            NemoFlowStatus::Ok
        );
        assert!(!subscriber.is_null());

        let name = cstring(&unique_name("ffi_openinference"));
        assert_eq!(
            nemo_flow_openinference_subscriber_register(ptr::null(), name.as_ptr()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_force_flush(ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_shutdown(ptr::null()),
            NemoFlowStatus::NullPointer
        );

        assert_eq!(
            nemo_flow_openinference_subscriber_register(subscriber, name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_deregister(name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_deregister(name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_force_flush(subscriber),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_openinference_subscriber_shutdown(subscriber),
            NemoFlowStatus::Ok
        );
        nemo_flow_openinference_subscriber_free(subscriber);
    }
}

#[test]
fn test_ffi_helper_rejection_and_null_name_paths() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();
        let args = cstring(r#"{"value": 7}"#);
        let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let invalid_json = cstring("{");
        let tool_name = cstring("tool");
        let llm_name = cstring("llm");
        let mut null_llm_out = ptr::null_mut();

        assert_eq!(
            nemo_flow_tool_request_intercepts(ptr::null(), args.as_ptr(), ptr::null_mut()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_request_intercepts(
                tool_name.as_ptr(),
                invalid_json.as_ptr(),
                ptr::null_mut()
            ),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_tool_conditional_execution(ptr::null(), args.as_ptr()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_tool_conditional_execution(tool_name.as_ptr(), invalid_json.as_ptr()),
            NemoFlowStatus::InvalidJson
        );

        let tool_guard = cstring(&unique_name("ffi_tool_reject"));
        assert_eq!(
            nemo_flow_register_tool_conditional_execution_guardrail(
                tool_guard.as_ptr(),
                1,
                tool_reject_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_tool_conditional_execution(tool_name.as_ptr(), args.as_ptr()),
            NemoFlowStatus::GuardrailRejected
        );
        assert_eq!(
            nemo_flow_deregister_tool_conditional_execution_guardrail(tool_guard.as_ptr()),
            NemoFlowStatus::Ok
        );

        let mut llm_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_llm_request_intercepts(ptr::null(), request.as_ptr(), &mut llm_out),
            NemoFlowStatus::Ok
        );
        let llm_json = returned_json(llm_out);
        assert_eq!(llm_json["content"]["model"], json!("ffi-model"));

        assert_eq!(
            nemo_flow_llm_request_intercepts(
                llm_name.as_ptr(),
                invalid_json.as_ptr(),
                &mut null_llm_out
            ),
            NemoFlowStatus::InvalidJson
        );
        assert_eq!(
            nemo_flow_llm_conditional_execution(invalid_json.as_ptr()),
            NemoFlowStatus::InvalidJson
        );

        let llm_guard = cstring(&unique_name("ffi_llm_reject"));
        assert_eq!(
            nemo_flow_register_llm_conditional_execution_guardrail(
                llm_guard.as_ptr(),
                1,
                llm_reject_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_llm_conditional_execution(request.as_ptr()),
            NemoFlowStatus::GuardrailRejected
        );
        assert_eq!(
            nemo_flow_deregister_llm_conditional_execution_guardrail(llm_guard.as_ptr()),
            NemoFlowStatus::Ok
        );

        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_registration_name_and_uuid_error_sweep() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_invalid_arg {
        ($expr:expr_2021) => {
            assert_eq!($expr, NemoFlowStatus::InvalidArg);
        };
    }
    macro_rules! assert_null_pointer {
        ($expr:expr_2021) => {
            assert_eq!($expr, NemoFlowStatus::NullPointer);
        };
    }

    unsafe {
        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_error_sweep_scope");
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                scope_name.as_ptr(),
                NemoFlowScopeType::Function,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut scope,
            ),
            NemoFlowStatus::Ok
        );

        let valid_scope_uuid = cstring(&take_string(nemo_flow_scope_handle_uuid(scope)).unwrap());
        let invalid_scope_uuid = cstring("not-a-uuid");

        assert_null_pointer!(nemo_flow_register_tool_sanitize_request_guardrail(
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_tool_sanitize_request_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_tool_sanitize_response_guardrail(
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_tool_sanitize_response_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_tool_conditional_execution_guardrail(
            ptr::null(),
            1,
            tool_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_tool_conditional_execution_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_tool_request_intercept(
            ptr::null(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_tool_request_intercept(ptr::null()));
        assert_null_pointer!(nemo_flow_register_tool_execution_intercept(
            ptr::null(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_tool_execution_intercept(ptr::null()));
        assert_null_pointer!(nemo_flow_register_llm_sanitize_request_guardrail(
            ptr::null(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_llm_sanitize_request_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_llm_sanitize_response_guardrail(
            ptr::null(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_llm_sanitize_response_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_llm_conditional_execution_guardrail(
            ptr::null(),
            1,
            llm_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_llm_conditional_execution_guardrail(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_llm_request_intercept(
            ptr::null(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_llm_request_intercept(ptr::null()));
        assert_null_pointer!(nemo_flow_register_llm_execution_intercept(
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_llm_execution_intercept(ptr::null()));
        assert_null_pointer!(nemo_flow_register_llm_stream_execution_intercept(
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_llm_stream_execution_intercept(
            ptr::null()
        ));
        assert_null_pointer!(nemo_flow_register_subscriber(
            ptr::null(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_deregister_subscriber(ptr::null()));

        assert_invalid_arg!(nemo_flow_scope_register_tool_sanitize_request_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_flow_scope_deregister_tool_sanitize_request_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_flow_scope_register_tool_sanitize_response_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_scope_deregister_tool_sanitize_response_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(
            nemo_flow_scope_register_tool_conditional_execution_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_invalid_arg!(
            nemo_flow_scope_deregister_tool_conditional_execution_guardrail(
                invalid_scope_uuid.as_ptr(),
                ptr::null(),
            )
        );
        assert_null_pointer!(nemo_flow_scope_register_tool_request_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_scope_deregister_tool_request_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(nemo_flow_scope_register_tool_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            tool_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_flow_scope_deregister_tool_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_flow_scope_register_llm_sanitize_request_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_scope_deregister_llm_sanitize_request_guardrail(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(nemo_flow_scope_register_llm_sanitize_response_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_flow_scope_deregister_llm_sanitize_response_guardrail(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(
            nemo_flow_scope_register_llm_conditional_execution_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_null_pointer!(
            nemo_flow_scope_deregister_llm_conditional_execution_guardrail(
                valid_scope_uuid.as_ptr(),
                ptr::null(),
            )
        );
        assert_invalid_arg!(nemo_flow_scope_register_llm_request_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_flow_scope_deregister_llm_request_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_flow_scope_register_llm_execution_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_scope_deregister_llm_execution_intercept(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_invalid_arg!(nemo_flow_scope_register_llm_stream_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
            1,
            llm_exec_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_invalid_arg!(nemo_flow_scope_deregister_llm_stream_execution_intercept(
            invalid_scope_uuid.as_ptr(),
            ptr::null(),
        ));
        assert_null_pointer!(nemo_flow_scope_register_subscriber(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_null_pointer!(nemo_flow_scope_deregister_subscriber(
            valid_scope_uuid.as_ptr(),
            ptr::null(),
        ));

        assert_eq!(nemo_flow_pop_scope(scope), NemoFlowStatus::Ok);
        nemo_flow_scope_handle_free(scope);
        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_duplicate_registration_sweep_and_helper_callbacks() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    macro_rules! assert_already_exists {
        ($expr:expr_2021) => {
            assert_eq!($expr, NemoFlowStatus::AlreadyExists);
        };
    }

    unsafe extern "C" fn tool_next_passthrough(
        _args_json: *const c_char,
        _next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        CString::new(r#"{"next":true}"#).unwrap().into_raw()
    }

    unsafe extern "C" fn llm_next_passthrough(
        _native_json: *const c_char,
        _next_ctx: *mut libc::c_void,
    ) -> *mut c_char {
        CString::new(r#"{"role":"assistant","content":"next","tool_calls":[]}"#)
            .unwrap()
            .into_raw()
    }

    unsafe {
        clear_last_error();
        assert!(read_last_error().is_none());

        let stack = fresh_scope_stack();
        let scope_name = cstring("ffi_duplicate_scope");
        let mut scope = ptr::null_mut();
        assert_eq!(
            nemo_flow_push_scope(
                scope_name.as_ptr(),
                NemoFlowScopeType::Function,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut scope,
            ),
            NemoFlowStatus::Ok
        );
        let scope_uuid = cstring(&take_string(nemo_flow_scope_handle_uuid(scope)).unwrap());

        let tool_cond = cstring(&unique_name("dup_tool_cond"));
        assert_eq!(
            nemo_flow_register_tool_conditional_execution_guardrail(
                tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_register_tool_conditional_execution_guardrail(
            tool_cond.as_ptr(),
            1,
            tool_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_deregister_tool_conditional_execution_guardrail(tool_cond.as_ptr()),
            NemoFlowStatus::Ok
        );

        let tool_req = cstring(&unique_name("dup_tool_req"));
        assert_eq!(
            nemo_flow_register_tool_request_intercept(
                tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_register_tool_request_intercept(
            tool_req.as_ptr(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_deregister_tool_request_intercept(tool_req.as_ptr()),
            NemoFlowStatus::Ok
        );

        let llm_san_resp = cstring(&unique_name("dup_llm_san_resp"));
        assert_eq!(
            nemo_flow_register_llm_sanitize_response_guardrail(
                llm_san_resp.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_register_llm_sanitize_response_guardrail(
            llm_san_resp.as_ptr(),
            1,
            llm_response_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_deregister_llm_sanitize_response_guardrail(llm_san_resp.as_ptr()),
            NemoFlowStatus::Ok
        );

        let llm_cond = cstring(&unique_name("dup_llm_cond"));
        assert_eq!(
            nemo_flow_register_llm_conditional_execution_guardrail(
                llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_register_llm_conditional_execution_guardrail(
            llm_cond.as_ptr(),
            1,
            llm_allow_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_deregister_llm_conditional_execution_guardrail(llm_cond.as_ptr()),
            NemoFlowStatus::Ok
        );

        let llm_req = cstring(&unique_name("dup_llm_req"));
        assert_eq!(
            nemo_flow_register_llm_request_intercept(
                llm_req.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_register_llm_request_intercept(
            llm_req.as_ptr(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_deregister_llm_request_intercept(llm_req.as_ptr()),
            NemoFlowStatus::Ok
        );

        let subscriber = cstring(&unique_name("dup_subscriber"));
        assert_eq!(
            nemo_flow_register_subscriber(
                subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_register_subscriber(
            subscriber.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_deregister_subscriber(subscriber.as_ptr()),
            NemoFlowStatus::Ok
        );

        let scope_tool_cond = cstring(&unique_name("dup_scope_tool_cond"));
        assert_eq!(
            nemo_flow_scope_register_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(
            nemo_flow_scope_register_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_eq!(
            nemo_flow_scope_deregister_tool_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_tool_cond.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_tool_req = cstring(&unique_name("dup_scope_tool_req"));
        assert_eq!(
            nemo_flow_scope_register_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_scope_register_tool_request_intercept(
            scope_uuid.as_ptr(),
            scope_tool_req.as_ptr(),
            1,
            false,
            tool_request_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_scope_deregister_tool_request_intercept(
                scope_uuid.as_ptr(),
                scope_tool_req.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_cond = cstring(&unique_name("dup_scope_llm_cond"));
        assert_eq!(
            nemo_flow_scope_register_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(
            nemo_flow_scope_register_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            )
        );
        assert_eq!(
            nemo_flow_scope_deregister_llm_conditional_execution_guardrail(
                scope_uuid.as_ptr(),
                scope_llm_cond.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_llm_req = cstring(&unique_name("dup_scope_llm_req"));
        assert_eq!(
            nemo_flow_scope_register_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_scope_register_llm_request_intercept(
            scope_uuid.as_ptr(),
            scope_llm_req.as_ptr(),
            1,
            false,
            llm_request_intercept_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_scope_deregister_llm_request_intercept(
                scope_uuid.as_ptr(),
                scope_llm_req.as_ptr(),
            ),
            NemoFlowStatus::Ok
        );

        let scope_subscriber = cstring(&unique_name("dup_scope_subscriber"));
        assert_eq!(
            nemo_flow_scope_register_subscriber(
                scope_uuid.as_ptr(),
                scope_subscriber.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_scope_register_subscriber(
            scope_uuid.as_ptr(),
            scope_subscriber.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        ));
        assert_eq!(
            nemo_flow_scope_deregister_subscriber(scope_uuid.as_ptr(), scope_subscriber.as_ptr(),),
            NemoFlowStatus::Ok
        );

        let session = cstring("dup-session");
        let agent = cstring("dup-agent");
        let version = cstring("1.0.0");
        let mut exporter = ptr::null_mut();
        assert_eq!(
            nemo_flow_atif_exporter_create(
                ptr::null(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_atif_exporter_create(
                session.as_ptr(),
                ptr::null(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut exporter,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                ptr::null(),
                &mut exporter,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_atif_exporter_register(exporter, ptr::null()),
            NemoFlowStatus::NullPointer
        );
        let exporter_name = cstring(&unique_name("dup_exporter_subscriber"));
        assert_eq!(
            nemo_flow_atif_exporter_register(exporter, exporter_name.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_already_exists!(nemo_flow_atif_exporter_register(
            exporter,
            exporter_name.as_ptr(),
        ));
        assert_eq!(
            nemo_flow_atif_exporter_deregister(ptr::null()),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_atif_exporter_deregister(exporter_name.as_ptr()),
            NemoFlowStatus::Ok
        );
        nemo_flow_atif_exporter_free(exporter);

        let args = cstring(r#"{"value":1}"#);
        let tool_intercept_json = take_string(tool_exec_intercept_cb(
            ptr::null_mut(),
            args.as_ptr(),
            tool_next_passthrough,
            ptr::null_mut(),
        ))
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(&tool_intercept_json).unwrap(),
            json!({"next": true})
        );

        let request = cstring(r#"{"headers":{},"content":{"model":"ffi-model","messages":[]}}"#);
        let llm_intercept_json = take_string(llm_exec_intercept_cb(
            ptr::null_mut(),
            request.as_ptr(),
            llm_next_passthrough,
            ptr::null_mut(),
        ))
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(&llm_intercept_json).unwrap(),
            json!({"role":"assistant","content":"next","tool_calls":[]})
        );

        assert_eq!(nemo_flow_pop_scope(scope), NemoFlowStatus::Ok);
        nemo_flow_scope_handle_free(scope);
        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_llm_execute_stream_and_atif_exporter() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    unsafe {
        let stack = fresh_scope_stack();

        let subscriber_name = unique_name("ffi_llm_subscriber");
        let subscriber_name_c = cstring(&subscriber_name);
        assert_eq!(
            nemo_flow_register_subscriber(
                subscriber_name_c.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let mut root = ptr::null_mut();
        assert_eq!(nemo_flow_get_handle(&mut root), NemoFlowStatus::Ok);
        nemo_flow_scope_handle_free(root);

        let intercept_name = unique_name("ffi_llm_intercept");
        let intercept_name_c = cstring(&intercept_name);
        assert_eq!(
            nemo_flow_register_llm_request_intercept(
                intercept_name_c.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let conditional_name = unique_name("ffi_llm_conditional");
        let conditional_name_c = cstring(&conditional_name);
        assert_eq!(
            nemo_flow_register_llm_conditional_execution_guardrail(
                conditional_name_c.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let sanitize_name = unique_name("ffi_llm_sanitize");
        let sanitize_name_c = cstring(&sanitize_name);
        assert_eq!(
            nemo_flow_register_llm_sanitize_response_guardrail(
                sanitize_name_c.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );

        let mut exporter: *mut FfiAtifExporter = ptr::null_mut();
        let session = cstring("ffi-session");
        let agent = cstring("ffi-agent");
        let version = cstring("1.0.0");
        let model_name = cstring("ffi-model");
        assert_eq!(
            nemo_flow_atif_exporter_create(
                session.as_ptr(),
                agent.as_ptr(),
                version.as_ptr(),
                model_name.as_ptr(),
                &mut exporter,
            ),
            NemoFlowStatus::Ok
        );

        let exporter_sub = unique_name("ffi_exporter");
        let exporter_sub_c = cstring(&exporter_sub);
        assert_eq!(
            nemo_flow_atif_exporter_register(exporter, exporter_sub_c.as_ptr()),
            NemoFlowStatus::Ok
        );

        let llm_name = cstring("ffi_llm");
        let request = cstring(
            r#"{"headers":{},"content":{"messages":[{"role":"user","content":"hi"}],"model":"ffi-model"}}"#,
        );
        let headers = cstring(r#"{"Authorization":"Bearer token"}"#);
        let content = cstring(r#"{"messages":[],"model":"ffi-model"}"#);
        let llm_request = nemo_flow_llm_request_new(headers.as_ptr(), content.as_ptr());
        assert!(!llm_request.is_null());
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_flow_llm_request_headers(llm_request)).unwrap()
            )
            .unwrap(),
            json!({"Authorization": "Bearer token"})
        );
        assert_eq!(
            serde_json::from_str::<Json>(
                &take_string(nemo_flow_llm_request_content(llm_request)).unwrap()
            )
            .unwrap(),
            json!({"messages": [], "model": "ffi-model"})
        );
        nemo_flow_llm_request_free(llm_request);

        let mut helper_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_llm_request_intercepts(llm_name.as_ptr(), request.as_ptr(), &mut helper_out),
            NemoFlowStatus::Ok
        );
        let helper_json = returned_json(helper_out);
        assert_eq!(helper_json["content"]["intercepted"], json!(true));

        assert_eq!(
            nemo_flow_llm_conditional_execution(request.as_ptr()),
            NemoFlowStatus::Ok
        );

        let mut handle: *mut FfiLLMHandle = ptr::null_mut();
        assert_eq!(
            nemo_flow_llm_call(
                llm_name.as_ptr(),
                request.as_ptr(),
                ptr::null(),
                2,
                ptr::null(),
                ptr::null(),
                model_name.as_ptr(),
                &mut handle,
            ),
            NemoFlowStatus::Ok
        );
        assert!(take_string(nemo_flow_llm_handle_uuid(handle)).is_some());
        assert_eq!(
            take_string(nemo_flow_llm_handle_name(handle)).unwrap(),
            "ffi_llm"
        );
        assert_eq!(nemo_flow_llm_handle_attributes(handle), 2);
        assert!(take_string(nemo_flow_llm_handle_parent_uuid(handle)).is_some());

        let response = cstring(r#"{"content":"manual end","role":"assistant","tool_calls":[]}"#);
        assert_eq!(
            nemo_flow_llm_call_end(handle, response.as_ptr(), ptr::null(), ptr::null()),
            NemoFlowStatus::Ok
        );
        nemo_flow_llm_handle_free(handle);

        let mut execute_out = ptr::null_mut();
        assert_eq!(
            nemo_flow_llm_call_execute(
                llm_name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                model_name.as_ptr(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut execute_out,
            ),
            NemoFlowStatus::Ok
        );
        let execute_json = returned_json(execute_out);
        assert_eq!(execute_json["content"], json!("hello from ffi"));
        assert_eq!(execute_json["model_seen"], json!("ffi-model"));
        let events = lock_unpoisoned(event_log()).clone();
        assert!(
            events
                .iter()
                .any(|event| event["output"]["sanitized"] == json!(true))
        );
        assert!(
            events
                .iter()
                .any(|event| event["model_name"] == "ffi-model")
        );

        let mut stream = ptr::null_mut();
        assert_eq!(
            nemo_flow_llm_stream_call_execute(
                llm_name.as_ptr(),
                request.as_ptr(),
                llm_exec_cb,
                ptr::null_mut(),
                None,
                Some(collector_cb),
                Some(finalizer_cb),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                model_name.as_ptr(),
                None,
                None,
                ptr::null_mut(),
                None,
                ptr::null(),
                &mut stream,
            ),
            NemoFlowStatus::Ok
        );
        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_flow_stream_next(stream, &mut chunk), 1);
        let chunk_json = returned_json(chunk);
        assert_eq!(chunk_json["content"], json!("hello from ffi"));
        assert_eq!(nemo_flow_stream_next(stream, &mut chunk), 0);
        nemo_flow_stream_free(stream);

        assert_eq!(lock_unpoisoned(collected_chunks()).len(), 1);
        assert_eq!(*lock_unpoisoned(finalizer_calls()), 1);

        let mut exported = ptr::null_mut();
        assert_eq!(
            nemo_flow_atif_exporter_export(exporter, &mut exported),
            NemoFlowStatus::Ok
        );
        let trajectory = returned_json(exported);
        assert_eq!(trajectory["schema_version"], json!("ATIF-v1.6"));
        assert!(trajectory["steps"].as_array().unwrap().len() >= 4);

        assert_eq!(nemo_flow_atif_exporter_clear(exporter), NemoFlowStatus::Ok);
        let mut cleared = ptr::null_mut();
        assert_eq!(
            nemo_flow_atif_exporter_export(exporter, &mut cleared),
            NemoFlowStatus::Ok
        );
        let cleared_json = returned_json(cleared);
        assert_eq!(cleared_json["steps"].as_array().unwrap().len(), 0);

        assert_eq!(
            nemo_flow_atif_exporter_deregister(exporter_sub_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        nemo_flow_atif_exporter_free(exporter);
        assert_eq!(
            nemo_flow_deregister_subscriber(subscriber_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );

        assert_eq!(
            nemo_flow_deregister_llm_request_intercept(intercept_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_llm_conditional_execution_guardrail(conditional_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_llm_sanitize_response_guardrail(sanitize_name_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        nemo_flow_scope_stack_free(stack);
    }
}

#[test]
fn test_ffi_hosted_plugin_registration_validation_and_cleanup() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    let plugin_kind = unique_name("ffi_hosted_plugin");
    let plugin_kind_c = cstring(&plugin_kind);
    let config = cstring(
        &json!({
            "version": 1,
            "components": [{
                "kind": plugin_kind,
                "enabled": true,
                "config": {}
            }]
        })
        .to_string(),
    );
    let user_data = Box::into_raw(Box::new(7usize)) as *mut libc::c_void;

    unsafe {
        assert_eq!(
            nemo_flow_register_plugin(
                plugin_kind_c.as_ptr(),
                Some(hosted_plugin_validate_warn),
                hosted_plugin_register_subscriber,
                user_data,
                Some(hosted_plugin_free),
            ),
            NemoFlowStatus::Ok
        );

        let mut report_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_validate_plugin_config(config.as_ptr(), &mut report_json),
            NemoFlowStatus::Ok
        );
        let report = returned_json(report_json);
        assert!(
            report["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "hosted.warning")
        );

        let mut init_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_initialize_plugins(config.as_ptr(), &mut init_json),
            NemoFlowStatus::Ok
        );
        let initialized = returned_json(init_json);
        assert!(
            initialized["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diag| diag["code"] == "hosted.warning")
        );

        let mut active_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_active_plugin_report_json(&mut active_json),
            NemoFlowStatus::Ok
        );
        let active = returned_json(active_json);
        assert_eq!(active["diagnostics"], initialized["diagnostics"]);

        assert_eq!(nemo_flow_clear_plugin_configuration(), NemoFlowStatus::Ok);
        assert_eq!(
            nemo_flow_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoFlowStatus::NotFound
        );
    }

    assert_eq!(*lock_unpoisoned(hosted_plugin_frees()), 1);
}

#[test]
fn test_ffi_hosted_plugin_validation_failure_modes_are_reported() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    for (suffix, validate_cb, expected_fragment) in [
        (
            "invalid",
            Some(hosted_plugin_validate_invalid as crate::callable::NemoFlowPluginValidateCb),
            "invalid diagnostics JSON",
        ),
        (
            "null",
            Some(hosted_plugin_validate_null as crate::callable::NemoFlowPluginValidateCb),
            "returned null",
        ),
    ] {
        let plugin_kind = unique_name(&format!("ffi_hosted_{suffix}"));
        let plugin_kind_c = cstring(&plugin_kind);
        let config = cstring(
            &json!({
                "version": 1,
                "components": [{
                    "kind": plugin_kind,
                    "enabled": true,
                    "config": {}
                }]
            })
            .to_string(),
        );
        let user_data = Box::into_raw(Box::new(9usize)) as *mut libc::c_void;

        unsafe {
            assert_eq!(
                nemo_flow_register_plugin(
                    plugin_kind_c.as_ptr(),
                    validate_cb,
                    hosted_plugin_register_fail,
                    user_data,
                    Some(hosted_plugin_free),
                ),
                NemoFlowStatus::Ok
            );

            let mut report_json = ptr::null_mut();
            assert_eq!(
                nemo_flow_validate_plugin_config(config.as_ptr(), &mut report_json),
                NemoFlowStatus::Ok
            );
            let report = returned_json(report_json);
            let diag = report["diagnostics"].as_array().unwrap();
            assert!(
                diag.iter().any(|value| {
                    value["code"] == "plugin.validate_failed"
                        && value["message"]
                            .as_str()
                            .is_some_and(|message| message.contains(expected_fragment))
                }),
                "missing expected hosted-plugin validation diagnostic: {expected_fragment}"
            );

            assert_eq!(
                nemo_flow_deregister_plugin(plugin_kind_c.as_ptr()),
                NemoFlowStatus::Ok
            );
        }
    }

    assert_eq!(*lock_unpoisoned(hosted_plugin_frees()), 2);
}

#[test]
fn test_ffi_hosted_plugin_without_validate_callback_uses_registration_fallback_error() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();
    let _ = nemo_flow_clear_plugin_configuration();

    let plugin_kind = unique_name("ffi_hosted_no_validate");
    let plugin_kind_c = cstring(&plugin_kind);
    let config = cstring(
        &json!({
            "version": 1,
            "components": [{
                "kind": plugin_kind,
                "enabled": true,
                "config": {}
            }]
        })
        .to_string(),
    );
    let user_data = Box::into_raw(Box::new(11usize)) as *mut libc::c_void;

    unsafe {
        assert_eq!(
            nemo_flow_register_plugin(
                plugin_kind_c.as_ptr(),
                None,
                hosted_plugin_register_fail,
                user_data,
                Some(hosted_plugin_free),
            ),
            NemoFlowStatus::Ok
        );

        let mut report_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_validate_plugin_config(config.as_ptr(), &mut report_json),
            NemoFlowStatus::Ok
        );
        let report = returned_json(report_json);
        assert_eq!(report["diagnostics"], json!([]));

        let mut init_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_initialize_plugins(config.as_ptr(), &mut init_json),
            NemoFlowStatus::Internal
        );
        let err = read_last_error().expect("expected hosted registration failure message");
        assert!(err.contains("register callback failed with status Internal"));

        let mut active_json = ptr::null_mut();
        assert_eq!(
            nemo_flow_active_plugin_report_json(&mut active_json),
            NemoFlowStatus::Ok
        );
        assert_eq!(returned_json(active_json), Json::Null);

        assert_eq!(
            nemo_flow_deregister_plugin(plugin_kind_c.as_ptr()),
            NemoFlowStatus::Ok
        );
    }

    assert_eq!(*lock_unpoisoned(hosted_plugin_frees()), 1);
}

#[test]
fn test_ffi_plugin_context_helpers_cover_null_and_success_paths() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_globals();

    let name = cstring("registered");
    let llm_name = cstring("llm");
    let tool_name = cstring("tool");

    unsafe {
        assert_eq!(
            nemo_flow_plugin_context_register_subscriber(
                ptr::null_mut(),
                name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_sanitize_request_guardrail(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_sanitize_response_guardrail(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_conditional_execution_guardrail(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_sanitize_request_guardrail(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_sanitize_response_guardrail(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_conditional_execution_guardrail(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_request_intercept(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_request_intercept(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_execution_intercept(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_stream_execution_intercept(
                ptr::null_mut(),
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_execution_intercept(
                ptr::null_mut(),
                tool_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::NullPointer
        );
    }

    let mut inner = PluginRegistrationContext::with_namespace("ffi::");
    let mut ctx = FfiPluginContext(&mut inner as *mut _);

    unsafe {
        assert_eq!(
            nemo_flow_plugin_context_register_subscriber(
                &mut ctx,
                name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_sanitize_response_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_conditional_execution_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_sanitize_request_guardrail(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_sanitize_response_guardrail(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_response_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_conditional_execution_guardrail(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_allow_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_request_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                false,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_execution_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_stream_execution_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                llm_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
    }

    let mut registrations = inner.into_registrations();
    let registered_names = registrations
        .iter()
        .map(|registration| registration.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        registered_names,
        vec![
            "ffi::registered",
            "ffi::tool",
            "ffi::tool",
            "ffi::tool",
            "ffi::llm",
            "ffi::llm",
            "ffi::llm",
            "ffi::llm",
            "ffi::tool",
            "ffi::llm",
            "ffi::llm",
            "ffi::tool",
        ]
    );
    nemo_flow::plugin::rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
}

#[test]
fn test_ffi_plugin_context_helpers_reject_duplicate_names() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let subscriber_name = cstring("duplicate-subscriber");
    let llm_name = cstring("duplicate-llm");
    let tool_name = cstring("duplicate-tool");

    let mut inner = PluginRegistrationContext::with_namespace("ffi::");
    let mut ctx = FfiPluginContext(&mut inner as *mut _);

    unsafe {
        assert_eq!(
            nemo_flow_plugin_context_register_subscriber(
                &mut ctx,
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_subscriber(
                &mut ctx,
                subscriber_name.as_ptr(),
                subscriber_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );

        assert_eq!(
            nemo_flow_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_sanitize_request_guardrail(
                &mut ctx,
                tool_name.as_ptr(),
                2,
                tool_request_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );

        assert_eq!(
            nemo_flow_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                1,
                false,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_llm_request_intercept(
                &mut ctx,
                llm_name.as_ptr(),
                2,
                true,
                llm_request_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );

        assert_eq!(
            nemo_flow_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                1,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Ok
        );
        assert_eq!(
            nemo_flow_plugin_context_register_tool_execution_intercept(
                &mut ctx,
                tool_name.as_ptr(),
                2,
                tool_exec_intercept_cb,
                ptr::null_mut(),
                None,
            ),
            NemoFlowStatus::Internal
        );
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("already exists")
        );
    }
}

#[test]
fn test_ffi_stream_next_reports_error_items() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.blocking_send(Err(nemo_flow::error::FlowError::Internal(
        "ffi stream failed".to_string(),
    )))
    .expect("expected error payload to be queued");
    drop(tx);

    let stream = Box::into_raw(Box::new(FfiStream {
        receiver: tokio::sync::Mutex::new(rx),
    }));

    unsafe {
        let mut chunk = ptr::null_mut();
        assert_eq!(nemo_flow_stream_next(stream, &mut chunk), -1);
        assert!(chunk.is_null());
        assert!(
            read_last_error()
                .unwrap_or_default()
                .contains("ffi stream failed")
        );
        nemo_flow_stream_free(stream);
    }
}
