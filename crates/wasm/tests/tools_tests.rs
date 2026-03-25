// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use nvidia_nat_nexus_wasm::api::*;
use nvidia_nat_nexus_wasm::types::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn js_fn1(arg: &str, body: &str) -> js_sys::Function {
    js_sys::Function::new_with_args(arg, body)
}

fn js_fn2(args: &str, body: &str) -> js_sys::Function {
    js_sys::Function::new_with_args(args, body)
}

fn parse_json(s: &str) -> JsValue {
    js_sys::JSON::parse(s).unwrap()
}

// ===========================================================================
// Tool lifecycle
// ===========================================================================

#[wasm_bindgen_test]
fn test_tool_call_and_end() {
    let args = parse_json(r#"{"x": 1}"#);
    let handle = nat_nexus_tool_call(
        "test_tool",
        args,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    assert_eq!(handle.name(), "test_tool");
    assert!(!handle.uuid().is_empty());

    let result = parse_json(r#"{"result": 42}"#);
    nat_nexus_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_with_attributes() {
    let args = parse_json(r#"{}"#);
    let handle = nat_nexus_tool_call(
        "attr_tool",
        args,
        None,
        Some(TOOL_LOCAL),
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    assert_eq!(handle.attributes(), TOOL_LOCAL);

    let result = parse_json(r#"{}"#);
    nat_nexus_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_with_data_metadata() {
    let args = parse_json(r#"{}"#);
    let data = parse_json(r#"{"info":"test"}"#);
    let meta = parse_json(r#"{"version":"1.0"}"#);
    let handle = nat_nexus_tool_call("data_tool", args, None, None, data, meta, None).unwrap();

    let result = parse_json(r#"{}"#);
    let end_data = parse_json(r#"{"done":true}"#);
    nat_nexus_tool_call_end(&handle, result, end_data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_with_parent() {
    let scope = nat_nexus_push_scope(
        "tool_parent",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let scope_uuid = scope.uuid();
    let args = parse_json(r#"{}"#);
    let handle = nat_nexus_tool_call(
        "parented_tool",
        args,
        Some(scope),
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    assert_eq!(handle.parent_uuid().unwrap(), scope_uuid);

    let result = parse_json(r#"{}"#);
    nat_nexus_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();

    let current = nat_nexus_get_handle().unwrap();
    nat_nexus_pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_generates_events() {
    js_sys::eval("globalThis.__tool_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__tool_events.push(event)");
    register_subscriber("wasm_tool_evt_sub", cb).unwrap();

    let args = parse_json(r#"{}"#);
    let handle = nat_nexus_tool_call(
        "evt_tool",
        args,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
        None,
    )
    .unwrap();
    let result = parse_json(r#"{}"#);
    nat_nexus_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__tool_events").unwrap();
    let arr = js_sys::Array::from(&events);
    assert!(
        arr.length() >= 2,
        "Expected at least 2 events for tool call/end"
    );

    deregister_subscriber("wasm_tool_evt_sub").unwrap();
    js_sys::eval("delete globalThis.__tool_events").unwrap();
}

// ===========================================================================
// Tool execute
// ===========================================================================

#[wasm_bindgen_test]
async fn test_tool_execute_basic() {
    let func = js_fn1("args", "return {result: args.x + 1}");
    let args = parse_json(r#"{"x": 10}"#);
    let result = nat_nexus_tool_call_execute(
        "exec_tool",
        args,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let r = js_sys::Reflect::get(&result, &"result".into()).unwrap();
    assert_eq!(r.as_f64().unwrap(), 11.0);
}

#[wasm_bindgen_test]
async fn test_tool_execute_with_attributes() {
    let func = js_fn1("args", "return {ok: true}");
    let args = parse_json(r#"{}"#);
    let result = nat_nexus_tool_call_execute(
        "exec_attr_tool",
        args,
        func,
        None,
        Some(TOOL_LOCAL),
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let ok = js_sys::Reflect::get(&result, &"ok".into()).unwrap();
    assert!(ok.as_bool().unwrap());
}

#[wasm_bindgen_test]
async fn test_tool_execute_promise() {
    let func = js_fn1("args", "return Promise.resolve({async_result: args.v * 2})");
    let args = parse_json(r#"{"v": 5}"#);
    let result = nat_nexus_tool_call_execute(
        "promise_tool",
        args,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let r = js_sys::Reflect::get(&result, &"async_result".into()).unwrap();
    assert_eq!(r.as_f64().unwrap(), 10.0);
}

// ===========================================================================
// Tool guardrails
// ===========================================================================

#[wasm_bindgen_test]
fn test_tool_sanitize_request_guardrail() {
    let guardrail = js_fn2("name, args", "args.sanitized = true; return args");
    register_tool_sanitize_request_guardrail("wasm_tool_san_req", 10, guardrail).unwrap();
    deregister_tool_sanitize_request_guardrail("wasm_tool_san_req").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_sanitize_response_guardrail() {
    let guardrail = js_fn2("name, result", "result.checked = true; return result");
    register_tool_sanitize_response_guardrail("wasm_tool_san_resp", 10, guardrail).unwrap();
    deregister_tool_sanitize_response_guardrail("wasm_tool_san_resp").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_conditional_guardrail() {
    let guardrail = js_fn2("name, args", "return null");
    register_tool_conditional_execution_guardrail("wasm_tool_cond", 10, guardrail).unwrap();
    deregister_tool_conditional_execution_guardrail("wasm_tool_cond").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_conditional_guardrail_blocks() {
    let guardrail = js_fn2("name, args", "return 'blocked by guardrail'");
    register_tool_conditional_execution_guardrail("wasm_tool_block", 10, guardrail).unwrap();
    deregister_tool_conditional_execution_guardrail("wasm_tool_block").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_tool_guardrail_fails() {
    let g1 = js_fn2("name, args", "return args");
    let g2 = js_fn2("name, args", "return args");
    register_tool_sanitize_request_guardrail("wasm_dup_guard", 10, g1).unwrap();
    let result = register_tool_sanitize_request_guardrail("wasm_dup_guard", 20, g2);
    assert!(result.is_err());
    deregister_tool_sanitize_request_guardrail("wasm_dup_guard").unwrap();
}

// ===========================================================================
// Tool intercepts
// ===========================================================================

#[wasm_bindgen_test]
fn test_tool_request_intercept() {
    let func = js_fn2("name, args", "args.intercepted = true; return args");
    register_tool_request_intercept("wasm_tool_req_int", 10, false, func).unwrap();
    deregister_tool_request_intercept("wasm_tool_req_int").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_response_intercept() {
    let func = js_fn2("name, result", "result.processed = true; return result");
    register_tool_response_intercept("wasm_tool_resp_int", 10, false, func).unwrap();
    deregister_tool_response_intercept("wasm_tool_resp_int").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_execution_intercept() {
    let exec = js_fn1("args", "return {intercepted: true}");
    register_tool_execution_intercept("wasm_tool_exec_int", 10, exec).unwrap();
    deregister_tool_execution_intercept("wasm_tool_exec_int").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_request_intercept_break_chain() {
    let func = js_fn2("name, args", "return args");
    register_tool_request_intercept("wasm_tool_break", 10, true, func).unwrap();
    deregister_tool_request_intercept("wasm_tool_break").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_tool_intercept_fails() {
    let f1 = js_fn2("name, args", "return args");
    let f2 = js_fn2("name, args", "return args");
    register_tool_request_intercept("wasm_dup_int", 10, false, f1).unwrap();
    let result = register_tool_request_intercept("wasm_dup_int", 20, false, f2);
    assert!(result.is_err());
    deregister_tool_request_intercept("wasm_dup_int").unwrap();
}

#[wasm_bindgen_test]
async fn test_tool_request_intercept_modifies_args() {
    let func = js_fn2("name, args", "args.added = 'yes'; return args");
    register_tool_request_intercept("wasm_tool_req_mod", 10, false, func).unwrap();

    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{"original": true}"#);
    let result = nat_nexus_tool_call_execute(
        "mod_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let added = js_sys::Reflect::get(&result, &"added".into()).unwrap();
    assert_eq!(added.as_string().unwrap(), "yes");

    deregister_tool_request_intercept("wasm_tool_req_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_tool_response_intercept_modifies_result() {
    let func = js_fn2(
        "name, result",
        "result.post_processed = true; return result",
    );
    register_tool_response_intercept("wasm_tool_resp_mod", 10, false, func).unwrap();

    let exec = js_fn1("args", "return {value: 42}");
    let args = parse_json(r#"{}"#);
    let result = nat_nexus_tool_call_execute(
        "resp_mod_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let pp = js_sys::Reflect::get(&result, &"post_processed".into()).unwrap();
    assert!(pp.as_bool().unwrap());

    deregister_tool_response_intercept("wasm_tool_resp_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_tool_execution_intercept_replaces_func() {
    let intercept_exec = js_fn1("args", "return {replaced: true}");
    register_tool_execution_intercept("wasm_tool_exec_repl", 10, intercept_exec).unwrap();

    let original = js_fn1("args", "return {original: true}");
    let args = parse_json(r#"{}"#);
    let result = nat_nexus_tool_call_execute(
        "replaced_tool",
        args,
        original,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let replaced = js_sys::Reflect::get(&result, &"replaced".into()).unwrap();
    assert!(replaced.as_bool().unwrap());

    deregister_tool_execution_intercept("wasm_tool_exec_repl").unwrap();
}
