// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for callable in the NeMo Flow FFI crate.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_flow::api::event::Event;
use nemo_flow::api::llm::{LlmAttributes, LlmHandle};
use serde_json::json;
use tokio_stream::StreamExt;

extern "C" fn free_arc_counter(user_data: *mut libc::c_void) {
    let counter = unsafe { Box::from_raw(user_data as *mut Arc<AtomicUsize>) };
    counter.fetch_add(1, Ordering::SeqCst);
}

fn user_data_counter() -> (*mut libc::c_void, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let ptr = Box::into_raw(Box::new(counter.clone())) as *mut libc::c_void;
    (ptr, counter)
}

unsafe extern "C" fn tool_sanitize_cb(
    user_data: *mut libc::c_void,
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    let counter = unsafe { &*(user_data as *const Arc<AtomicUsize>) };
    counter.fetch_add(1, Ordering::SeqCst);
    let mut args: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(args_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    args["name"] = json!(unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default());
    CString::new(args.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn tool_conditional_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    let args: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(args_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap();
    if args["block"] == json!(true) {
        CString::new("blocked").unwrap().into_raw()
    } else {
        std::ptr::null_mut()
    }
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

unsafe extern "C" fn tool_exec_error_cb(
    _user_data: *mut libc::c_void,
    _args_json: *const c_char,
) -> *mut c_char {
    set_last_error("tool callback failed");
    std::ptr::null_mut()
}

unsafe extern "C" fn tool_exec_intercept_cb(
    _user_data: *mut libc::c_void,
    args_json: *const c_char,
    next_fn: NemoFlowToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    let result_ptr = unsafe { next_fn(args_json, next_ctx) };
    if result_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let mut result: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap()).unwrap();
    unsafe { nemo_flow_string_free_internal(result_ptr) };
    result["intercepted"] = json!(true);
    CString::new(result.to_string()).unwrap().into_raw()
}

/// Intercept-specific callback with the unified annotated-aware signature
/// for callable.rs unit tests.
unsafe extern "C" fn llm_request_intercept_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    request: *const FfiLLMRequest,
    annotated_json: *const c_char,
    out_request: *mut *mut FfiLLMRequest,
    out_annotated_json: *mut *mut c_char,
) -> NemoFlowStatus {
    let mut req = unsafe { (&*request).0.clone() };
    req.content["intercepted"] = json!(true);
    unsafe { *out_request = Box::into_raw(Box::new(FfiLLMRequest(req))) };
    if annotated_json.is_null() {
        unsafe { *out_annotated_json = std::ptr::null_mut() };
    } else {
        let s = unsafe { CStr::from_ptr(annotated_json) }
            .to_string_lossy()
            .into_owned();
        unsafe { *out_annotated_json = CString::new(s).unwrap().into_raw() };
    }
    NemoFlowStatus::Ok
}

unsafe extern "C" fn llm_request_null_cb(
    _user_data: *mut libc::c_void,
    _request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest {
    std::ptr::null_mut()
}

unsafe extern "C" fn llm_conditional_cb(
    _user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut c_char {
    if unsafe { (&*request).0.content.get("block").cloned() } == Some(json!(true)) {
        CString::new("blocked llm").unwrap().into_raw()
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn json_cb(_user_data: *mut libc::c_void, json: *const c_char) -> *mut c_char {
    let mut value: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
    value["wrapped"] = json!(true);
    CString::new(value.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn llm_exec_cb(
    _user_data: *mut libc::c_void,
    native_json: *const c_char,
) -> *mut c_char {
    let request: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(native_json) }.to_str().unwrap()).unwrap();
    let response = json!({
        "model": request["content"]["model"].clone(),
        "ok": true,
    });
    CString::new(response.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn llm_exec_error_cb(
    _user_data: *mut libc::c_void,
    _native_json: *const c_char,
) -> *mut c_char {
    set_last_error("llm callback failed");
    std::ptr::null_mut()
}

unsafe extern "C" fn llm_exec_intercept_cb(
    _user_data: *mut libc::c_void,
    native_json: *const c_char,
    next_fn: NemoFlowLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    let result_ptr = unsafe { next_fn(native_json, next_ctx) };
    if result_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let mut value: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap()).unwrap();
    unsafe { nemo_flow_string_free_internal(result_ptr) };
    value["intercepted"] = json!(true);
    CString::new(value.to_string()).unwrap().into_raw()
}

unsafe extern "C" fn llm_exec_short_circuit_cb(
    _user_data: *mut libc::c_void,
    native_json: *const c_char,
    _next_fn: NemoFlowLlmExecNextFn,
    _next_ctx: *mut libc::c_void,
) -> *mut c_char {
    let request: Json =
        serde_json::from_str(unsafe { CStr::from_ptr(native_json) }.to_str().unwrap()).unwrap();
    let response = json!({
        "model": request["content"]["model"].clone(),
        "intercepted": true,
    });
    CString::new(response.to_string()).unwrap().into_raw()
}

static COLLECTED_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn collector_cb(_chunk: *const c_char) {
    COLLECTED_COUNT.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn finalizer_cb() -> *mut c_char {
    CString::new(r#"{"done":true}"#).unwrap().into_raw()
}

unsafe extern "C" fn subscriber_cb(user_data: *mut libc::c_void, event: *const FfiEvent) {
    let counter = unsafe { &*(user_data as *const Arc<AtomicUsize>) };
    if unsafe { (&*event).0.name() } == "ffi-event" {
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

fn make_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "test-model"}),
    }
}

#[test]
fn test_wrap_tool_request_and_conditional_callbacks() {
    let (user_data, called) = user_data_counter();
    let wrapped = wrap_tool_sanitize_fn(tool_sanitize_cb, user_data, Some(free_arc_counter));
    let result = wrapped("tool-name", json!({"value": 1}));
    assert_eq!(result["value"], json!(1));
    assert_eq!(result["name"], json!("tool-name"));
    assert_eq!(called.load(Ordering::SeqCst), 1);
    drop(wrapped);
    assert_eq!(called.load(Ordering::SeqCst), 2);

    let wrapped_conditional =
        wrap_tool_conditional_fn(tool_conditional_cb, std::ptr::null_mut(), None);
    assert_eq!(
        wrapped_conditional("tool", &json!({"block": true})).unwrap(),
        Some("blocked".into())
    );
    assert_eq!(
        wrapped_conditional("tool", &json!({"block": false})).unwrap(),
        None
    );
}

#[test]
fn test_wrap_tool_exec_and_intercept_callbacks() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let exec = wrap_tool_exec_fn(tool_exec_cb, std::ptr::null_mut(), None);
    let result = runtime.block_on(exec(json!({"value": 2}))).unwrap();
    assert_eq!(result["executed"], json!(true));

    let exec_err = wrap_tool_exec_fn(tool_exec_error_cb, std::ptr::null_mut(), None);
    let err = runtime.block_on(exec_err(json!({}))).unwrap_err();
    assert!(err.to_string().contains("tool callback failed"));

    let intercept = wrap_tool_exec_intercept_fn(tool_exec_intercept_cb, std::ptr::null_mut(), None);
    let next: ToolExecutionNextFn =
        Arc::new(|args| Box::pin(async move { Ok(json!({"from_next": args})) }));
    let intercepted = runtime
        .block_on(intercept("tool", json!({"v": 1}), next))
        .unwrap();
    assert_eq!(intercepted["intercepted"], json!(true));
    assert_eq!(intercepted["from_next"]["v"], json!(1));

    let failing_intercept =
        wrap_tool_exec_intercept_fn(tool_exec_intercept_cb, std::ptr::null_mut(), None);
    let failing_next: ToolExecutionNextFn =
        Arc::new(|_| Box::pin(async { Err(FlowError::Internal("next failed".into())) }));
    let err = runtime
        .block_on(failing_intercept("tool", json!({"v": 2}), failing_next))
        .unwrap_err();
    assert!(err.to_string().contains("next failed"));
}

#[test]
fn test_wrap_llm_request_response_and_conditional_callbacks() {
    let request_intercept =
        wrap_llm_request_intercept_fn(llm_request_intercept_cb, std::ptr::null_mut(), None);
    let (intercepted, _annotated) = request_intercept("llm", make_request(), None).unwrap();
    assert_eq!(intercepted.content["intercepted"], json!(true));

    let sanitize_request =
        wrap_llm_sanitize_request_fn(llm_request_null_cb, std::ptr::null_mut(), None);
    let sanitized = sanitize_request(make_request());
    assert_eq!(sanitized.headers.len(), 0);
    assert_eq!(sanitized.content, Json::Null);

    let conditional = wrap_llm_conditional_fn(llm_conditional_cb, std::ptr::null_mut(), None);
    assert_eq!(
        conditional(&LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"block": true}),
        })
        .unwrap(),
        Some("blocked llm".into())
    );
    assert_eq!(conditional(&make_request()).unwrap(), None);

    let wrapped_json = wrap_json_fn(json_cb, std::ptr::null_mut(), None);
    assert_eq!(wrapped_json(json!({"value": 1}))["wrapped"], json!(true));

    let wrapped_response = wrap_llm_response_fn(json_cb, std::ptr::null_mut(), None);
    assert_eq!(
        wrapped_response(json!({"value": 2}))["wrapped"],
        json!(true)
    );
}

#[test]
fn test_wrap_llm_request_intercept_with_annotated_input() {
    let request_intercept =
        wrap_llm_request_intercept_fn(llm_request_intercept_cb, std::ptr::null_mut(), None);
    let annotated = nemo_flow::codec::request::AnnotatedLlmRequest {
        messages: vec![],
        model: Some("test-model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::from_iter([("annotated".into(), json!(true))]),
    };
    let (intercepted, annotated_out) =
        request_intercept("llm", make_request(), Some(annotated)).unwrap();
    assert_eq!(intercepted.content["intercepted"], json!(true));
    let annotated_out = annotated_out.expect("expected annotated request output");
    assert_eq!(annotated_out.model.as_deref(), Some("test-model"));
    assert_eq!(annotated_out.extra.get("annotated"), Some(&json!(true)));
}

#[test]
fn test_wrap_llm_exec_stream_and_event_callbacks() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let exec = wrap_llm_exec_fn(llm_exec_cb, std::ptr::null_mut(), None);
    let result = runtime.block_on(exec(make_request())).unwrap();
    assert_eq!(result["ok"], json!(true));
    assert_eq!(result["model"], json!("test-model"));

    let exec_err = wrap_llm_exec_fn(llm_exec_error_cb, std::ptr::null_mut(), None);
    let err = runtime.block_on(exec_err(make_request())).unwrap_err();
    assert!(err.to_string().contains("llm callback failed"));

    let intercept = wrap_llm_exec_intercept_fn(llm_exec_intercept_cb, std::ptr::null_mut(), None);
    let next: LlmExecutionNextFn =
        Arc::new(|request| Box::pin(async move { Ok(json!({"model": request.content["model"]})) }));
    let intercepted = runtime
        .block_on(intercept("llm", make_request(), next))
        .unwrap();
    assert_eq!(intercepted["intercepted"], json!(true));

    let stream_exec = wrap_llm_stream_exec_fn(llm_exec_cb, std::ptr::null_mut(), None);
    let mut stream = runtime.block_on(stream_exec(make_request())).unwrap();
    let first = runtime.block_on(async { stream.next().await.unwrap().unwrap() });
    assert_eq!(first["ok"], json!(true));

    let stream_intercept =
        wrap_llm_stream_exec_intercept_fn(llm_exec_short_circuit_cb, std::ptr::null_mut(), None);
    let next_stream: LlmStreamExecutionNextFn = Arc::new(|_request| {
        Box::pin(async {
            Ok(
                Box::pin(tokio_stream::iter(vec![Ok(json!({"ignored": true}))]))
                    as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
            )
        })
    });
    let mut intercepted_stream = runtime
        .block_on(stream_intercept("llm", make_request(), next_stream))
        .unwrap();
    let first = runtime.block_on(async { intercepted_stream.next().await.unwrap().unwrap() });
    assert_eq!(first["intercepted"], json!(true));

    let stream_intercept_with_next =
        wrap_llm_stream_exec_intercept_fn(llm_exec_intercept_cb, std::ptr::null_mut(), None);
    let next_stream: LlmStreamExecutionNextFn = Arc::new(|request| {
        Box::pin(async move {
            Ok(Box::pin(tokio_stream::iter(vec![Ok(json!({
                "model": request.content["model"].clone()
            }))]))
                as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        })
    });
    let mut intercepted_stream = runtime
        .block_on(stream_intercept_with_next(
            "llm",
            make_request(),
            next_stream,
        ))
        .unwrap();
    let first = runtime.block_on(async { intercepted_stream.next().await.unwrap().unwrap() });
    assert_eq!(first["intercepted"], json!(true));
    assert_eq!(first["model"], json!("test-model"));

    COLLECTED_COUNT.store(0, Ordering::SeqCst);
    let mut collector = wrap_collector_fn(collector_cb);
    collector(json!({"chunk": 1})).unwrap();
    assert_eq!(COLLECTED_COUNT.load(Ordering::SeqCst), 1);

    let finalizer = wrap_finalizer_fn(finalizer_cb);
    assert_eq!(finalizer(), json!({"done": true}));

    let (user_data, seen) = user_data_counter();
    let subscriber = wrap_event_subscriber(subscriber_cb, user_data, Some(free_arc_counter));
    let event = Event::Scope(nemo_flow::api::event::ScopeEvent::new(
        nemo_flow::api::event::BaseEvent::builder()
            .name("ffi-event")
            .build(),
        nemo_flow::api::event::ScopeCategory::Start,
        Vec::new(),
        nemo_flow::api::event::EventCategory::llm(),
        Some(
            nemo_flow::api::event::CategoryProfile::builder()
                .model_name("test-model")
                .build(),
        ),
    ));
    subscriber(&event);
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    drop(subscriber);
    assert_eq!(seen.load(Ordering::SeqCst), 2);

    let handle = LlmHandle::builder()
        .name("llm")
        .attributes(LlmAttributes::STATEFUL)
        .build();
    assert_eq!(handle.name, "llm");
}
