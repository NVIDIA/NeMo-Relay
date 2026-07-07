// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for api in the NeMo Relay FFI crate.

use super::*;
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use nemo_relay::plugin::PluginRegistrationContext;
use serde_json::{Value as Json, json};
use uuid::Uuid;

use crate::callable::{NemoRelayLlmExecNextFn, NemoRelayToolExecNextFn};
use crate::convert::nemo_relay_string_free;
use crate::error::{NemoRelayStatus, nemo_relay_last_error};
use crate::types::{
    FfiAtifExporter, FfiEvent, FfiLLMHandle, FfiLLMRequest, FfiOpenTelemetrySubscriber,
    FfiPluginActivation, FfiScopeStack, FfiToolHandle, nemo_relay_atif_exporter_free,
    nemo_relay_event_data, nemo_relay_event_input, nemo_relay_event_metadata,
    nemo_relay_event_model_name, nemo_relay_event_name, nemo_relay_event_output,
    nemo_relay_event_parent_uuid, nemo_relay_event_scope_type, nemo_relay_event_timestamp,
    nemo_relay_event_tool_call_id, nemo_relay_event_uuid, nemo_relay_llm_handle_attributes,
    nemo_relay_llm_handle_free, nemo_relay_llm_handle_name, nemo_relay_llm_handle_parent_uuid,
    nemo_relay_llm_handle_uuid, nemo_relay_llm_request_content, nemo_relay_llm_request_free,
    nemo_relay_llm_request_headers, nemo_relay_llm_request_new, nemo_relay_otel_subscriber_free,
    nemo_relay_scope_handle_attributes, nemo_relay_scope_handle_data, nemo_relay_scope_handle_free,
    nemo_relay_scope_handle_metadata, nemo_relay_scope_handle_name,
    nemo_relay_scope_handle_parent_uuid, nemo_relay_scope_handle_scope_type,
    nemo_relay_scope_handle_uuid, nemo_relay_scope_stack_free, nemo_relay_tool_handle_attributes,
    nemo_relay_tool_handle_free, nemo_relay_tool_handle_name, nemo_relay_tool_handle_parent_uuid,
    nemo_relay_tool_handle_uuid,
};
use crate::{api, callable, types};

static TEST_MUTEX: Mutex<()> = Mutex::new(());
static EVENT_LOG: OnceLock<Mutex<Vec<Json>>> = OnceLock::new();
static COLLECTED_CHUNKS: OnceLock<Mutex<Vec<Json>>> = OnceLock::new();
static FINALIZER_CALLS: OnceLock<Mutex<usize>> = OnceLock::new();
static PLUGIN_FREES: OnceLock<Mutex<usize>> = OnceLock::new();

fn event_log() -> &'static Mutex<Vec<Json>> {
    EVENT_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn collected_chunks() -> &'static Mutex<Vec<Json>> {
    COLLECTED_CHUNKS.get_or_init(|| Mutex::new(Vec::new()))
}

fn finalizer_calls() -> &'static Mutex<usize> {
    FINALIZER_CALLS.get_or_init(|| Mutex::new(0))
}

fn plugin_frees() -> &'static Mutex<usize> {
    PLUGIN_FREES.get_or_init(|| Mutex::new(0))
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

#[allow(clippy::too_many_arguments)]
unsafe fn nemo_relay_push_scope(
    name: *const c_char,
    scope_type: NemoRelayScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    input_json: *const c_char,
    out: *mut *mut FfiScopeHandle,
) -> NemoRelayStatus {
    unsafe {
        api::nemo_relay_push_scope(
            name,
            scope_type,
            parent,
            attributes,
            data_json,
            metadata_json,
            input_json,
            ptr::null(),
            out,
        )
    }
}

unsafe fn nemo_relay_pop_scope(
    handle: *const FfiScopeHandle,
    output_json: *const c_char,
) -> NemoRelayStatus {
    unsafe { api::nemo_relay_pop_scope(handle, output_json, ptr::null(), ptr::null()) }
}

unsafe fn nemo_relay_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoRelayStatus {
    unsafe { api::nemo_relay_event(name, parent, data_json, metadata_json, ptr::null()) }
}

#[allow(clippy::too_many_arguments)]
unsafe fn nemo_relay_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    tool_call_id: *const c_char,
    out: *mut *mut FfiToolHandle,
) -> NemoRelayStatus {
    unsafe {
        api::nemo_relay_tool_call(
            name,
            args_json,
            parent,
            attributes,
            data_json,
            metadata_json,
            tool_call_id,
            ptr::null(),
            out,
        )
    }
}

unsafe fn nemo_relay_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoRelayStatus {
    unsafe {
        api::nemo_relay_tool_call_end(handle, result_json, data_json, metadata_json, ptr::null())
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn nemo_relay_llm_call(
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiLLMHandle,
) -> NemoRelayStatus {
    unsafe {
        api::nemo_relay_llm_call(
            name,
            native_json,
            parent,
            attributes,
            data_json,
            metadata_json,
            model_name,
            ptr::null(),
            out,
        )
    }
}

unsafe fn nemo_relay_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoRelayStatus {
    unsafe {
        api::nemo_relay_llm_call_end(handle, response_json, data_json, metadata_json, ptr::null())
    }
}

unsafe fn take_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { nemo_relay_string_free(ptr) };
    Some(s)
}

unsafe fn read_last_error() -> Option<String> {
    let ptr = nemo_relay_last_error();
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
        unsafe { nemo_relay_scope_stack_create(&mut stack) },
        NemoRelayStatus::Ok
    );
    assert!(!stack.is_null());
    assert_eq!(
        unsafe { nemo_relay_scope_stack_set_thread(stack) },
        NemoRelayStatus::Ok
    );
    stack
}

fn reset_globals() {
    lock_unpoisoned(event_log()).clear();
    lock_unpoisoned(collected_chunks()).clear();
    *lock_unpoisoned(finalizer_calls()) = 0;
    *lock_unpoisoned(plugin_frees()) = 0;
}

unsafe extern "C" fn subscriber_cb(_user_data: *mut libc::c_void, event: *const FfiEvent) {
    let payload = json!({
        "uuid": unsafe { take_string(nemo_relay_event_uuid(event)) }.unwrap_or_default(),
        "name": unsafe { take_string(nemo_relay_event_name(event)) }.unwrap_or_default(),
        "kind": unsafe { take_string(crate::types::nemo_relay_event_kind(event)) }.unwrap_or_default(),
        "json": unsafe { take_string(crate::types::nemo_relay_event_json(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "data": unsafe { take_string(nemo_relay_event_data(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "metadata": unsafe { take_string(nemo_relay_event_metadata(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "timestamp": unsafe { take_string(nemo_relay_event_timestamp(event)) }.unwrap_or_default(),
        "input": unsafe { take_string(nemo_relay_event_input(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "output": unsafe { take_string(nemo_relay_event_output(event)) }
            .map(|s| serde_json::from_str::<Json>(&s).unwrap()),
        "model_name": unsafe { take_string(nemo_relay_event_model_name(event)) },
        "tool_call_id": unsafe { take_string(nemo_relay_event_tool_call_id(event)) },
        "parent_uuid": unsafe { take_string(nemo_relay_event_parent_uuid(event)) },
        "scope_type": unsafe { take_string(nemo_relay_event_scope_type(event)) },
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

unsafe extern "C" fn tool_request_fail_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _args_json: *const c_char,
) -> *mut c_char {
    crate::error::set_last_error("tool sanitize callback failed");
    ptr::null_mut()
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
    CString::new("blocked").unwrap().into_raw()
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

unsafe extern "C" fn tool_exec_fail_cb(
    _user_data: *mut libc::c_void,
    _args_json: *const c_char,
) -> *mut c_char {
    crate::error::set_last_error("tool execution callback failed");
    ptr::null_mut()
}

unsafe extern "C" fn tool_exec_intercept_cb(
    _user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: NemoRelayToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    let result_ptr = unsafe { next_fn(args_json, next_ctx) };
    if result_ptr.is_null() {
        return ptr::null_mut();
    }
    let result: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap()).unwrap();
    unsafe { nemo_relay_string_free(result_ptr) };
    CString::new(json!({ "result": result, "pending_marks": [] }).to_string())
        .unwrap()
        .into_raw()
}

unsafe extern "C" fn llm_request_cb(
    _user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest {
    let request = unsafe { &*request };
    let mut content = request.0.content.clone();
    content["intercepted"] = json!(true);
    Box::into_raw(Box::new(FfiLLMRequest(LlmRequest {
        headers: request.0.headers.clone(),
        content,
    })))
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
    CString::new("blocked").unwrap().into_raw()
}

unsafe extern "C" fn llm_request_intercept_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    request: *const FfiLLMRequest,
    _annotated_json: *const c_char,
    out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    let transformed = unsafe { Box::from_raw(llm_request_cb(ptr::null_mut(), request)) };
    let outcome = json!({
        "request": transformed.0,
        "annotated_request": null,
        "pending_marks": [],
    });
    unsafe { *out_outcome_json = CString::new(outcome.to_string()).unwrap().into_raw() };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_request_intercept_fail_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _request: *const FfiLLMRequest,
    _annotated_json: *const c_char,
    _out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    crate::error::set_last_error("llm request intercept callback failed");
    NemoRelayStatus::Internal
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
    let response = json!({
        "role": "assistant",
        "content": "hello from ffi",
        "tool_calls": [],
        "model_seen": request
            .get("content")
            .and_then(|value| value.get("model"))
            .cloned()
            .unwrap_or(Json::Null),
    });
    CString::new(response.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn llm_exec_fail_cb(
    _user_data: *mut libc::c_void,
    _native_json: *const c_char,
) -> *mut c_char {
    crate::error::set_last_error("llm execution callback failed");
    ptr::null_mut()
}

unsafe extern "C" fn llm_exec_openai_chat_cb(
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
        .unwrap_or(json!("gpt-ffi"));
    let response = json!({
        "id": "chatcmpl-ffi",
        "model": model,
        "choices": [{
            "message": {
                "content": "hello from openai-chat",
                "tool_calls": []
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        }
    });
    CString::new(response.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn codec_decode_cb(
    _user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char {
    let request = unsafe { &*request };
    let model = request
        .0
        .content
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("codec-model");
    let prompt = request
        .0
        .content
        .get("prompt")
        .and_then(|value| value.as_str())
        .unwrap_or("hello");
    CString::new(
        json!({
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "codec_marker": "decoded"
        })
        .to_string(),
    )
    .unwrap()
    .into_raw()
}

unsafe extern "C" fn codec_encode_cb(
    _user_data: *mut libc::c_void,
    annotated_json: *const c_char,
    _original_request: *const FfiLLMRequest,
) -> *mut c_char {
    let annotated: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(annotated_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    let prompt = annotated
        .get("messages")
        .and_then(|messages| messages.as_array())
        .and_then(|messages| messages.first())
        .and_then(|message| message.get("content"))
        .cloned()
        .unwrap_or(Json::Null);
    CString::new(
        json!({
            "model": annotated["model"].clone(),
            "prompt": prompt,
            "encoded": true
        })
        .to_string(),
    )
    .unwrap()
    .into_raw()
}

unsafe extern "C" fn llm_exec_intercept_cb(
    _user_data: *mut libc::c_void,
    native_json: *const c_char,
    next_fn: NemoRelayLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    unsafe { next_fn(native_json, next_ctx) }
}

unsafe extern "C" fn collector_cb(chunk_json: *const c_char) {
    let chunk: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(chunk_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    lock_unpoisoned(collected_chunks()).push(chunk);
}

unsafe extern "C" fn finalizer_cb() -> *mut c_char {
    *lock_unpoisoned(finalizer_calls()) += 1;
    CString::new(json!({"finalized": true}).to_string())
        .unwrap()
        .into_raw()
}

unsafe extern "C" fn plugin_free(user_data: *mut libc::c_void) {
    *lock_unpoisoned(plugin_frees()) += 1;
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data as *mut usize) });
    }
}

unsafe extern "C" fn plugin_validate_warn(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
) -> *mut c_char {
    CString::new(
        json!([{
            "level": "warning",
            "code": "plugin.warning",
            "component": "ffi.plugin",
            "message": "plugin validation ran"
        }])
        .to_string(),
    )
    .unwrap()
    .into_raw()
}

unsafe extern "C" fn plugin_validate_invalid(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
) -> *mut c_char {
    CString::new("not-json").unwrap().into_raw()
}

unsafe extern "C" fn plugin_validate_null(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
) -> *mut c_char {
    ptr::null_mut()
}

unsafe extern "C" fn plugin_register_subscriber(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
    ctx: *mut FfiPluginContext,
) -> NemoRelayStatus {
    let name = CString::new("subscriber").unwrap();
    unsafe {
        nemo_relay_plugin_context_register_subscriber(
            ctx,
            name.as_ptr(),
            subscriber_cb,
            ptr::null_mut(),
            None,
        )
    }
}

unsafe extern "C" fn plugin_register_fail(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
    _ctx: *mut FfiPluginContext,
) -> NemoRelayStatus {
    NemoRelayStatus::Internal
}

unsafe extern "C" fn plugin_register_fail_with_last_error(
    _user_data: *mut libc::c_void,
    _plugin_config_json: *const c_char,
    _ctx: *mut FfiPluginContext,
) -> NemoRelayStatus {
    set_last_error("plugin register callback set last error explicitly");
    NemoRelayStatus::Internal
}

#[path = "api/core_tests.rs"]
mod core_tests;
#[path = "api/coverage_sweeps_tests.rs"]
mod coverage_sweeps_tests;
#[path = "api/execution_tests.rs"]
mod execution_tests;
#[path = "api/plugin_tests.rs"]
mod plugin_tests;
#[path = "api/registry_tests.rs"]
mod registry_tests;
