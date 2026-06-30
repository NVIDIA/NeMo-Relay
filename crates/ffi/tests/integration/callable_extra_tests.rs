// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for callable extra in the NeMo Relay FFI crate.

use super::*;
use std::ptr;

use tokio_stream::StreamExt;

unsafe extern "C" fn tool_conditional_error_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _args_json: *const c_char,
) -> *mut c_char {
    set_last_error("tool conditional failed");
    ptr::null_mut()
}

unsafe extern "C" fn tool_exec_intercept_null_next_cb(
    _user_data: *mut libc::c_void,
    _args_json: *const c_char,
    next_fn: NemoRelayToolExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    unsafe { next_fn(ptr::null(), next_ctx) }
}

unsafe extern "C" fn llm_exec_intercept_null_next_cb(
    _user_data: *mut libc::c_void,
    _native_json: *const c_char,
    next_fn: NemoRelayLlmExecNextFn,
    next_ctx: *mut libc::c_void,
) -> *mut c_char {
    unsafe { next_fn(ptr::null(), next_ctx) }
}

unsafe extern "C" fn llm_request_intercept_status_error_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _request: *const FfiLLMRequest,
    _annotated_json: *const c_char,
    _out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    NemoRelayStatus::Internal
}

unsafe extern "C" fn llm_request_intercept_null_out_request_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _request: *const FfiLLMRequest,
    _annotated_json: *const c_char,
    _out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_request_intercept_invalid_annotated_cb(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    _request: *const FfiLLMRequest,
    _annotated_json: *const c_char,
    out_outcome_json: *mut *mut c_char,
) -> NemoRelayStatus {
    unsafe { *out_outcome_json = CString::new("not-json").unwrap().into_raw() };
    NemoRelayStatus::Ok
}

unsafe extern "C" fn llm_request_passthrough_cb(
    _user_data: *mut libc::c_void,
    request: *const FfiLLMRequest,
) -> *mut FfiLLMRequest {
    Box::into_raw(Box::new(FfiLLMRequest(unsafe { (&*request).0.clone() })))
}

unsafe extern "C" fn llm_conditional_error_cb(
    _user_data: *mut libc::c_void,
    _request: *const FfiLLMRequest,
) -> *mut c_char {
    set_last_error("llm conditional failed");
    ptr::null_mut()
}

unsafe extern "C" fn codec_decode_null_cb(
    _user_data: *mut libc::c_void,
    _request: *const FfiLLMRequest,
) -> *mut c_char {
    ptr::null_mut()
}

unsafe extern "C" fn codec_decode_invalid_json_cb(
    _user_data: *mut libc::c_void,
    _request: *const FfiLLMRequest,
) -> *mut c_char {
    CString::new("not-json").unwrap().into_raw()
}

unsafe extern "C" fn codec_encode_null_cb(
    _user_data: *mut libc::c_void,
    _annotated_json: *const c_char,
    _original_request: *const FfiLLMRequest,
) -> *mut c_char {
    ptr::null_mut()
}

unsafe extern "C" fn codec_encode_invalid_json_cb(
    _user_data: *mut libc::c_void,
    _annotated_json: *const c_char,
    _original_request: *const FfiLLMRequest,
) -> *mut c_char {
    CString::new("not-json").unwrap().into_raw()
}

unsafe extern "C" fn finalizer_null_cb() -> *mut c_char {
    ptr::null_mut()
}

fn make_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "test-model"}),
    }
}

#[test]
fn test_callable_extra_trampoline_and_helper_paths() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let conditional = wrap_tool_conditional_fn(tool_conditional_error_cb, ptr::null_mut(), None);
    let conditional_err = conditional("tool", &json!({})).unwrap_err();
    assert!(
        conditional_err
            .to_string()
            .contains("tool conditional failed")
    );

    let tool_intercept =
        wrap_tool_exec_intercept_fn(tool_exec_intercept_null_next_cb, ptr::null_mut(), None);
    let tool_next: ToolExecutionNextFn = Arc::new(|args| {
        Box::pin(async move {
            assert_eq!(args, Json::Null);
            Err(FlowError::Internal("tool next failed".into()))
        })
    });
    let tool_err = runtime
        .block_on(tool_intercept("tool", json!({"value": 1}), tool_next))
        .unwrap_err();
    assert!(tool_err.to_string().contains("tool next failed"));

    let llm_intercept =
        wrap_llm_exec_intercept_fn(llm_exec_intercept_null_next_cb, ptr::null_mut(), None);
    let llm_next: LlmExecutionNextFn = Arc::new(|request| {
        Box::pin(async move {
            assert_eq!(request.content, Json::Null);
            Err(FlowError::Internal("llm next failed".into()))
        })
    });
    let llm_err = runtime
        .block_on(llm_intercept("llm", make_request(), llm_next))
        .unwrap_err();
    assert!(llm_err.to_string().contains("llm next failed"));

    let llm_stream_intercept =
        wrap_llm_stream_exec_intercept_fn(llm_exec_intercept_null_next_cb, ptr::null_mut(), None);
    let empty_next: LlmStreamExecutionNextFn = Arc::new(|request| {
        Box::pin(async move {
            assert_eq!(request.content, Json::Null);
            Ok(Box::pin(tokio_stream::empty()) as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        })
    });
    let mut empty_stream = runtime
        .block_on(llm_stream_intercept("llm", make_request(), empty_next))
        .unwrap();
    let empty_item = runtime
        .block_on(async { empty_stream.next().await })
        .unwrap()
        .unwrap();
    assert_eq!(empty_item, Json::Null);

    let err_next: LlmStreamExecutionNextFn = Arc::new(|_request| {
        Box::pin(async move { Err(FlowError::Internal("stream next failed".into())) })
    });
    let stream_err = match runtime.block_on(llm_stream_intercept("llm", make_request(), err_next)) {
        Ok(_) => panic!("expected llm stream intercept error"),
        Err(err) => err,
    };
    assert!(stream_err.to_string().contains("stream next failed"));

    let finalizer = wrap_finalizer_fn(finalizer_null_cb);
    assert_eq!(finalizer(), Json::Null);
}

#[test]
fn test_callable_extra_request_intercept_and_codec_paths() {
    let request = make_request();

    let intercept_error =
        wrap_llm_request_intercept_fn(llm_request_intercept_status_error_cb, ptr::null_mut(), None);
    let err = intercept_error("llm", request.clone(), None).unwrap_err();
    assert!(
        err.to_string()
            .contains("request intercept callback failed")
    );

    let intercept_null = wrap_llm_request_intercept_fn(
        llm_request_intercept_null_out_request_cb,
        ptr::null_mut(),
        None,
    );
    let err = intercept_null("llm", request.clone(), None).unwrap_err();
    assert!(err.to_string().contains("null out_outcome_json"));

    let intercept_invalid_annotated = wrap_llm_request_intercept_fn(
        llm_request_intercept_invalid_annotated_cb,
        ptr::null_mut(),
        None,
    );
    let err = intercept_invalid_annotated("llm", request.clone(), None).unwrap_err();
    assert!(
        err.to_string()
            .contains("invalid LLM request intercept outcome JSON")
    );

    let sanitize = wrap_llm_sanitize_request_fn(llm_request_passthrough_cb, ptr::null_mut(), None);
    let sanitized = sanitize(request.clone());
    assert_eq!(sanitized.content, request.content);

    let conditional = wrap_llm_conditional_fn(llm_conditional_error_cb, ptr::null_mut(), None);
    let conditional_err = conditional(&request).unwrap_err();
    assert!(
        conditional_err
            .to_string()
            .contains("llm conditional failed")
    );

    let null_decode = wrap_codec_fn(
        codec_decode_null_cb,
        codec_encode_invalid_json_cb,
        ptr::null_mut(),
        None,
    );
    let decode_err = null_decode.decode(&request).unwrap_err();
    assert!(decode_err.to_string().contains("returned null"));

    let invalid_decode = wrap_codec_fn(
        codec_decode_invalid_json_cb,
        codec_encode_invalid_json_cb,
        ptr::null_mut(),
        None,
    );
    let decode_err = invalid_decode.decode(&request).unwrap_err();
    assert!(decode_err.to_string().contains("invalid JSON"));

    let annotated = AnnotatedLLMRequest {
        model: Some("test-model".into()),
        messages: vec![],
        params: None,
        tools: Some(vec![]),
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
        extra: serde_json::Map::new(),
    };

    let null_encode = wrap_codec_fn(
        codec_decode_invalid_json_cb,
        codec_encode_null_cb,
        ptr::null_mut(),
        None,
    );
    let encode_err = null_encode.encode(&annotated, &request).unwrap_err();
    assert!(encode_err.to_string().contains("returned null"));

    let invalid_encode = wrap_codec_fn(
        codec_decode_invalid_json_cb,
        codec_encode_invalid_json_cb,
        ptr::null_mut(),
        None,
    );
    let encode_err = invalid_encode.encode(&annotated, &request).unwrap_err();
    assert!(encode_err.to_string().contains("invalid result JSON"));
}
