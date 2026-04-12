// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{CStr, CString};
use std::ptr;

use nemo_flow_ffi::api::{
    nemo_flow_scope_stack_active, nemo_flow_scope_stack_create, nemo_flow_scope_stack_set_thread,
};
use nemo_flow_ffi::convert::nemo_flow_string_free;
use nemo_flow_ffi::error::{NemoFlowStatus, nemo_flow_last_error};
use nemo_flow_ffi::types::{
    FfiScopeStack, nemo_flow_llm_request_content, nemo_flow_llm_request_free,
    nemo_flow_llm_request_headers, nemo_flow_llm_request_new, nemo_flow_scope_stack_free,
};
use serde_json::json;

fn cstring(value: &str) -> CString {
    CString::new(value).unwrap()
}

unsafe fn take_string(ptr: *mut libc::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let value = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { nemo_flow_string_free(ptr) };
    Some(value)
}

#[test]
fn scope_stack_api_round_trip() {
    let mut stack: *mut FfiScopeStack = ptr::null_mut();

    let create_status = unsafe { nemo_flow_scope_stack_create(&mut stack) };
    assert_eq!(create_status, NemoFlowStatus::Ok);
    assert!(!stack.is_null());

    let bind_status = unsafe { nemo_flow_scope_stack_set_thread(stack) };
    assert_eq!(bind_status, NemoFlowStatus::Ok);
    assert!(nemo_flow_scope_stack_active());

    unsafe { nemo_flow_scope_stack_free(stack) };
}

#[test]
fn llm_request_accessors_round_trip() {
    let headers = cstring(r#"{"x-trace":"1"}"#);
    let content = cstring(r#"{"model":"test-model","messages":[]}"#);

    let request = unsafe { nemo_flow_llm_request_new(headers.as_ptr(), content.as_ptr()) };
    assert!(!request.is_null());

    let headers_json = unsafe { take_string(nemo_flow_llm_request_headers(request)) }.unwrap();
    let content_json = unsafe { take_string(nemo_flow_llm_request_content(request)) }.unwrap();

    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&headers_json).unwrap(),
        json!({"x-trace": "1"})
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&content_json).unwrap(),
        json!({"model": "test-model", "messages": []})
    );

    unsafe { nemo_flow_llm_request_free(request) };
}

#[test]
fn scope_stack_create_reports_null_pointer_errors() {
    let status = unsafe { nemo_flow_scope_stack_create(ptr::null_mut()) };
    assert_eq!(status, NemoFlowStatus::NullPointer);

    let message = unsafe { CStr::from_ptr(nemo_flow_last_error()) }
        .to_string_lossy()
        .into_owned();
    assert!(message.contains("out pointer is null"));
}
