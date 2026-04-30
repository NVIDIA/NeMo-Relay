// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for llm in the NeMo Flow WebAssembly crate.

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use nemo_flow_wasm::api::*;
use nemo_flow_wasm::types::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn js_fn1(arg: &str, body: &str) -> js_sys::Function {
    js_sys::Function::new_with_args(arg, body)
}

fn parse_json(s: &str) -> JsValue {
    js_sys::JSON::parse(s).unwrap()
}

fn make_request() -> JsValue {
    parse_json(r#"{"headers":{},"content":{"messages":[],"model":"test-model"}}"#)
}

fn parent_handle(handle: Option<ScopeHandle>) -> JsValue {
    handle.map(JsValue::from).unwrap_or(JsValue::NULL)
}

fn push_scope(
    name: &str,
    scope_type: ScopeType,
    handle: Option<ScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
) -> Result<ScopeHandle, JsValue> {
    nemo_flow_wasm::api::push_scope(
        name,
        scope_type,
        parent_handle(handle),
        attributes,
        data,
        metadata,
        JsValue::NULL,
        None,
    )
}

fn pop_scope(handle: &ScopeHandle) -> Result<(), JsValue> {
    nemo_flow_wasm::api::pop_scope(handle, JsValue::NULL, None)
}

fn llm_call(
    name: &str,
    request: JsValue,
    handle: Option<ScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
) -> Result<LlmHandle, JsValue> {
    nemo_flow_wasm::api::llm_call(
        name,
        request,
        parent_handle(handle),
        attributes,
        data,
        metadata,
        model_name,
        None,
    )
}

fn llm_call_end(
    handle: &LlmHandle,
    response: JsValue,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    nemo_flow_wasm::api::llm_call_end(handle, response, data, metadata, None)
}

#[allow(clippy::too_many_arguments)]
async fn llm_call_execute(
    name: &str,
    request: JsValue,
    func: js_sys::Function,
    handle: Option<ScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
    model_name: Option<String>,
    codec_decode: Option<js_sys::Function>,
    codec_encode: Option<js_sys::Function>,
    response_codec_decode: Option<js_sys::Function>,
) -> Result<JsValue, JsValue> {
    nemo_flow_wasm::api::llm_call_execute(
        name,
        request,
        func,
        parent_handle(handle),
        attributes,
        data,
        metadata,
        model_name,
        codec_decode,
        codec_encode,
        response_codec_decode,
    )
    .await
}

// ===========================================================================
// LLM lifecycle
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_call_and_end() {
    let request = make_request();
    let handle = llm_call(
        "test_llm",
        request,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    assert_eq!(handle.name(), "test_llm");
    assert!(!handle.uuid().is_empty());

    let response = parse_json(r#"{"choices":[{"text":"hello"}]}"#);
    llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_with_attributes() {
    let request = make_request();
    let handle = llm_call(
        "attr_llm",
        request,
        None,
        Some(LLM_STATEFUL | LLM_STREAMING),
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    assert_eq!(handle.attributes(), LLM_STATEFUL | LLM_STREAMING);

    let response = parse_json(r#"{}"#);
    llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_with_parent() {
    let scope = push_scope(
        "llm_parent",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let scope_uuid = scope.uuid();
    let request = make_request();
    let handle = llm_call(
        "parented_llm",
        request,
        Some(scope),
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    assert_eq!(
        handle.parent_uuid().as_string().as_deref(),
        Some(scope_uuid.as_str())
    );

    let response = parse_json(r#"{}"#);
    llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();

    let current = get_handle().unwrap();
    pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_with_data_metadata() {
    let request = make_request();
    let data = parse_json(r#"{"info":"llm_test"}"#);
    let meta = parse_json(r#"{"version":"2.0"}"#);
    let handle = llm_call("data_llm", request, None, None, data, meta, None).unwrap();

    let response = parse_json(r#"{}"#);
    let end_data = parse_json(r#"{"tokens":100}"#);
    llm_call_end(&handle, response, end_data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_generates_events() {
    js_sys::eval("globalThis.__llm_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__llm_events.push(event)");
    register_subscriber("wasm_llm_evt_sub", cb).unwrap();

    let request = make_request();
    let handle = llm_call(
        "evt_llm",
        request,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    let response = parse_json(r#"{}"#);
    llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__llm_events").unwrap();
    let arr = js_sys::Array::from(&events);
    assert!(
        arr.length() >= 2,
        "Expected at least 2 events for llm call/end"
    );

    deregister_subscriber("wasm_llm_evt_sub").unwrap();
    js_sys::eval("delete globalThis.__llm_events").unwrap();
}

// ===========================================================================
// LLM execute
// ===========================================================================

#[wasm_bindgen_test]
async fn test_llm_execute_basic() {
    let func = js_fn1("native", "return {response: 'hello from llm'}");
    let request = make_request();
    let result = llm_call_execute(
        "exec_llm",
        request,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let r = js_sys::Reflect::get(&result, &"response".into()).unwrap();
    assert_eq!(r.as_string().unwrap(), "hello from llm");
}

#[wasm_bindgen_test]
async fn test_llm_execute_promise() {
    let func = js_fn1("native", "return Promise.resolve({async: true})");
    let request = make_request();
    let result = llm_call_execute(
        "async_llm",
        request,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let a = js_sys::Reflect::get(&result, &"async".into()).unwrap();
    assert!(a.as_bool().unwrap());
}

// ===========================================================================
// LLM guardrails
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_sanitize_request_guardrail() {
    let guardrail = js_fn1("request", "request.extra = 'sanitized'; return request");
    register_llm_sanitize_request_guardrail("wasm_llm_san_req", 10, guardrail).unwrap();
    deregister_llm_sanitize_request_guardrail("wasm_llm_san_req").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_sanitize_response_guardrail() {
    let guardrail = js_fn1("response", "response.sanitized = true; return response");
    register_llm_sanitize_response_guardrail("wasm_llm_san_resp", 10, guardrail).unwrap();
    deregister_llm_sanitize_response_guardrail("wasm_llm_san_resp").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_conditional_guardrail() {
    let guardrail = js_fn1("request", "return null");
    register_llm_conditional_execution_guardrail("wasm_llm_cond", 10, guardrail).unwrap();
    deregister_llm_conditional_execution_guardrail("wasm_llm_cond").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_conditional_guardrail_blocks() {
    let guardrail = js_fn1("request", "return 'blocked'");
    register_llm_conditional_execution_guardrail("wasm_llm_block", 10, guardrail).unwrap();
    deregister_llm_conditional_execution_guardrail("wasm_llm_block").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_llm_guardrail_fails() {
    let g1 = js_fn1("request", "return request");
    let g2 = js_fn1("request", "return request");
    register_llm_sanitize_request_guardrail("wasm_llm_dup_guard", 10, g1).unwrap();
    let result = register_llm_sanitize_request_guardrail("wasm_llm_dup_guard", 20, g2);
    assert!(result.is_err());
    deregister_llm_sanitize_request_guardrail("wasm_llm_dup_guard").unwrap();
}

// ===========================================================================
// LLM intercepts
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_request_intercept() {
    let func = js_sys::Function::new_with_args(
        "name, native, annotated",
        "native.content.intercepted = true; return {request: native, annotated: annotated}",
    );
    register_llm_request_intercept("wasm_llm_req_int", 10, false, func).unwrap();
    deregister_llm_request_intercept("wasm_llm_req_int").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_execution_intercept() {
    let exec = js_fn1("native", "return {replaced: true}");
    register_llm_execution_intercept("wasm_llm_exec_int", 10, exec).unwrap();
    deregister_llm_execution_intercept("wasm_llm_exec_int").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_stream_execution_intercept() {
    let exec = js_fn1("native", "return {stream_result: true}");
    register_llm_stream_execution_intercept("wasm_llm_stream_exec", 10, exec).unwrap();
    deregister_llm_stream_execution_intercept("wasm_llm_stream_exec").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_request_intercept_break_chain() {
    let func = js_sys::Function::new_with_args(
        "name, native, annotated",
        "return {request: native, annotated: annotated}",
    );
    register_llm_request_intercept("wasm_llm_break", 10, true, func).unwrap();
    deregister_llm_request_intercept("wasm_llm_break").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_llm_intercept_fails() {
    let f1 = js_sys::Function::new_with_args(
        "name, native, annotated",
        "return {request: native, annotated: annotated}",
    );
    let f2 = js_sys::Function::new_with_args(
        "name, native, annotated",
        "return {request: native, annotated: annotated}",
    );
    register_llm_request_intercept("wasm_llm_dup_int", 10, false, f1).unwrap();
    let result = register_llm_request_intercept("wasm_llm_dup_int", 20, false, f2);
    assert!(result.is_err());
    deregister_llm_request_intercept("wasm_llm_dup_int").unwrap();
}

#[wasm_bindgen_test]
async fn test_llm_request_intercept_modifies_request() {
    let intercept = js_sys::Function::new_with_args(
        "name, native, annotated",
        "native.content.intercepted = true; return {request: native, annotated: annotated}",
    );
    register_llm_request_intercept("wasm_llm_req_mod", 10, false, intercept).unwrap();

    let func = js_fn1(
        "native",
        "return {saw_intercepted: (native.content && native.content.intercepted) || false}",
    );
    let request = make_request();
    let result = llm_call_execute(
        "mod_llm",
        request,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let saw = js_sys::Reflect::get(&result, &"saw_intercepted".into()).unwrap();
    assert!(saw.as_bool().unwrap());

    deregister_llm_request_intercept("wasm_llm_req_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_llm_execution_intercept_replaces_func() {
    let intercept_exec = js_fn1("native", "return {replaced: true}");
    register_llm_execution_intercept("wasm_llm_exec_repl", 10, intercept_exec).unwrap();

    let original = js_fn1("native", "return {original: true}");
    let request = make_request();
    let result = llm_call_execute(
        "repl_llm",
        request,
        original,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let replaced = js_sys::Reflect::get(&result, &"replaced".into()).unwrap();
    assert!(replaced.as_bool().unwrap());

    deregister_llm_execution_intercept("wasm_llm_exec_repl").unwrap();
}
