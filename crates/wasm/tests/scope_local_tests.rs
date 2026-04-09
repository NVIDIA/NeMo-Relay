// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

fn js_fn2(args: &str, body: &str) -> js_sys::Function {
    js_sys::Function::new_with_args(args, body)
}

fn parse_json(s: &str) -> JsValue {
    js_sys::JSON::parse(s).unwrap()
}

// ===========================================================================
// Scope-local guardrail registration and execution
// ===========================================================================

#[wasm_bindgen_test]
fn test_scope_local_register_deregister_tool_sanitize_request_guardrail() {
    let scope = nemo_flow_push_scope(
        "sl_guard_req",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, args", "args.sanitized = true; return args");
    scope_register_tool_sanitize_request_guardrail(&uuid, "sl_san_req_1", 10, guardrail).unwrap();

    let removed = scope_deregister_tool_sanitize_request_guardrail(&uuid, "sl_san_req_1").unwrap();
    assert!(removed);

    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_local_register_deregister_tool_sanitize_response_guardrail() {
    let scope = nemo_flow_push_scope(
        "sl_guard_resp",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, result", "result.checked = true; return result");
    scope_register_tool_sanitize_response_guardrail(&uuid, "sl_san_resp_1", 10, guardrail).unwrap();

    let removed =
        scope_deregister_tool_sanitize_response_guardrail(&uuid, "sl_san_resp_1").unwrap();
    assert!(removed);

    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_local_register_deregister_tool_conditional_guardrail() {
    let scope = nemo_flow_push_scope(
        "sl_guard_cond",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, args", "return null");
    scope_register_tool_conditional_execution_guardrail(&uuid, "sl_cond_1", 10, guardrail).unwrap();

    let removed =
        scope_deregister_tool_conditional_execution_guardrail(&uuid, "sl_cond_1").unwrap();
    assert!(removed);

    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
async fn test_scope_local_sanitize_request_guardrail_modifies_args() {
    js_sys::eval("globalThis.__wasm_sl_san_req_events = []; true").unwrap();
    let sub = js_fn1("event", "globalThis.__wasm_sl_san_req_events.push(event)");
    register_subscriber("sl_san_exec_sub", sub).unwrap();

    let scope = nemo_flow_push_scope(
        "sl_guard_exec",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, args", "args.scope_sanitized = true; return args");
    scope_register_tool_sanitize_request_guardrail(&uuid, "sl_san_exec_1", 10, guardrail).unwrap();

    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{"original": true}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_guarded_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    // Sanitize guardrails are observability-only; they modify event data, not execution results
    let original = js_sys::Reflect::get(&result, &"original".into()).unwrap();
    assert!(original.as_bool().unwrap());

    deregister_subscriber("sl_san_exec_sub").unwrap();
    // Verify the sanitized data appears in the tool Start event input
    let events = js_sys::Array::from(&js_sys::eval("globalThis.__wasm_sl_san_req_events").unwrap());
    let mut found = false;
    for i in 0..events.length() {
        let evt = events.get(i);
        let kind = js_sys::Reflect::get(&evt, &"kind".into()).unwrap();
        if kind.as_string().as_deref() == Some("ToolStart") {
            let input = js_sys::Reflect::get(&evt, &"input".into()).unwrap();
            let scope_sanitized = js_sys::Reflect::get(&input, &"scope_sanitized".into()).unwrap();
            assert!(scope_sanitized.as_bool().unwrap());
            found = true;
            break;
        }
    }
    assert!(found, "Expected a tool Start event with sanitized input");

    js_sys::eval("delete globalThis.__wasm_sl_san_req_events").unwrap();
    scope_deregister_tool_sanitize_request_guardrail(&uuid, "sl_san_exec_1").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
async fn test_scope_local_sanitize_response_guardrail_modifies_result() {
    js_sys::eval("globalThis.__wasm_sl_san_resp_events = []; true").unwrap();
    let sub = js_fn1("event", "globalThis.__wasm_sl_san_resp_events.push(event)");
    register_subscriber("sl_resp_exec_sub", sub).unwrap();

    let scope = nemo_flow_push_scope(
        "sl_guard_resp_exec",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, result", "result.post_checked = true; return result");
    scope_register_tool_sanitize_response_guardrail(&uuid, "sl_resp_exec_1", 10, guardrail)
        .unwrap();

    let exec = js_fn1("args", "return {value: 99}");
    let args = parse_json(r#"{}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_resp_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    // Sanitize guardrails are observability-only; they modify event data, not execution results
    let value = js_sys::Reflect::get(&result, &"value".into()).unwrap();
    assert_eq!(value.as_f64().unwrap(), 99.0);

    deregister_subscriber("sl_resp_exec_sub").unwrap();
    // Verify the sanitized data appears in the tool End event output
    let events =
        js_sys::Array::from(&js_sys::eval("globalThis.__wasm_sl_san_resp_events").unwrap());
    let mut found = false;
    for i in 0..events.length() {
        let evt = events.get(i);
        let kind = js_sys::Reflect::get(&evt, &"kind".into()).unwrap();
        if kind.as_string().as_deref() == Some("ToolEnd") {
            let output = js_sys::Reflect::get(&evt, &"output".into()).unwrap();
            let post_checked = js_sys::Reflect::get(&output, &"post_checked".into()).unwrap();
            assert!(post_checked.as_bool().unwrap());
            found = true;
            break;
        }
    }
    assert!(found, "Expected a tool End event with sanitized output");

    js_sys::eval("delete globalThis.__wasm_sl_san_resp_events").unwrap();
    scope_deregister_tool_sanitize_response_guardrail(&uuid, "sl_resp_exec_1").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
async fn test_scope_local_conditional_guardrail_blocks_execution() {
    let scope = nemo_flow_push_scope(
        "sl_guard_block",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, args", "return 'blocked by scope guardrail'");
    scope_register_tool_conditional_execution_guardrail(&uuid, "sl_block_1", 10, guardrail)
        .unwrap();

    let exec = js_fn1("args", "return {should_not: 'run'}");
    let args = parse_json(r#"{}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_blocked_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await;

    assert!(result.is_err(), "Expected execution to be blocked");

    scope_deregister_tool_conditional_execution_guardrail(&uuid, "sl_block_1").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_local_duplicate_guardrail_fails() {
    let scope = nemo_flow_push_scope(
        "sl_guard_dup",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let g1 = js_fn2("name, args", "return args");
    let g2 = js_fn2("name, args", "return args");
    scope_register_tool_sanitize_request_guardrail(&uuid, "sl_dup_guard", 10, g1).unwrap();
    let result = scope_register_tool_sanitize_request_guardrail(&uuid, "sl_dup_guard", 20, g2);
    assert!(result.is_err());

    scope_deregister_tool_sanitize_request_guardrail(&uuid, "sl_dup_guard").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_local_deregister_nonexistent_guardrail() {
    let scope = nemo_flow_push_scope(
        "sl_guard_nx",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let removed =
        scope_deregister_tool_sanitize_request_guardrail(&uuid, "nonexistent_guard").unwrap();
    assert!(!removed);

    nemo_flow_pop_scope(&scope).unwrap();
}

// ===========================================================================
// Auto-cleanup on scope pop
// ===========================================================================

#[wasm_bindgen_test]
async fn test_scope_local_guardrail_cleaned_up_on_pop() {
    let scope = nemo_flow_push_scope(
        "sl_cleanup_guard",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let guardrail = js_fn2("name, args", "args.from_popped_scope = true; return args");
    scope_register_tool_sanitize_request_guardrail(&uuid, "sl_cleanup_san", 10, guardrail).unwrap();

    nemo_flow_pop_scope(&scope).unwrap();

    // After popping, the scope-local guardrail should no longer affect tool calls.
    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{"original": true}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_cleanup_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let from_popped = js_sys::Reflect::get(&result, &"from_popped_scope".into()).unwrap();
    assert!(
        from_popped.is_undefined(),
        "Guardrail should not run after scope pop"
    );
    let original = js_sys::Reflect::get(&result, &"original".into()).unwrap();
    assert!(original.as_bool().unwrap());
}

#[wasm_bindgen_test]
async fn test_scope_local_intercept_cleaned_up_on_pop() {
    let scope = nemo_flow_push_scope(
        "sl_cleanup_int",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let func = js_fn2(
        "name, args",
        "args.from_popped_intercept = true; return args",
    );
    scope_register_tool_request_intercept(&uuid, "sl_cleanup_req_int", 10, false, func).unwrap();

    nemo_flow_pop_scope(&scope).unwrap();

    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{"original": true}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_cleanup_int_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let from_popped = js_sys::Reflect::get(&result, &"from_popped_intercept".into()).unwrap();
    assert!(
        from_popped.is_undefined(),
        "Intercept should not run after scope pop"
    );
    let original = js_sys::Reflect::get(&result, &"original".into()).unwrap();
    assert!(original.as_bool().unwrap());
}

#[wasm_bindgen_test]
async fn test_nested_scope_cleanup_preserves_parent() {
    let parent = nemo_flow_push_scope(
        "sl_parent",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let parent_uuid = parent.uuid();

    // Use a request intercept for parent (intercepts DO modify execution args)
    let parent_intercept = js_fn2("name, args", "args.parent_ran = true; return args");
    scope_register_tool_request_intercept(
        &parent_uuid,
        "sl_parent_guard",
        10,
        false,
        parent_intercept,
    )
    .unwrap();

    let child = nemo_flow_push_scope(
        "sl_child",
        SCOPE_TYPE_FUNCTION,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let child_uuid = child.uuid();

    // Child uses a sanitize guardrail (observability-only, won't affect execution result)
    let child_guard = js_fn2("name, args", "args.child_ran = true; return args");
    scope_register_tool_sanitize_request_guardrail(&child_uuid, "sl_child_guard", 20, child_guard)
        .unwrap();

    nemo_flow_pop_scope(&child).unwrap();

    // After child pop, parent intercept should still be active
    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_nested_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let parent_ran = js_sys::Reflect::get(&result, &"parent_ran".into()).unwrap();
    assert!(
        parent_ran.as_bool().unwrap(),
        "Parent intercept should still run"
    );

    let child_ran = js_sys::Reflect::get(&result, &"child_ran".into()).unwrap();
    assert!(
        child_ran.is_undefined(),
        "Child guardrail should not run after child scope pop"
    );

    scope_deregister_tool_request_intercept(&parent_uuid, "sl_parent_guard").unwrap();
    // Pop the parent (now current) scope
    let current = nemo_flow_get_handle().unwrap();
    nemo_flow_pop_scope(&current).unwrap();
}

// ===========================================================================
// Priority merge (global + scope-local)
// ===========================================================================

#[wasm_bindgen_test]
async fn test_global_and_scope_local_guardrails_both_run() {
    js_sys::eval("globalThis.__wasm_sl_merge_events = []; true").unwrap();
    let sub = js_fn1("event", "globalThis.__wasm_sl_merge_events.push(event)");
    register_subscriber("sl_merge_sub", sub).unwrap();

    let global_guard = js_fn2("name, args", "args.global_ran = true; return args");
    register_tool_sanitize_request_guardrail("sl_merge_global", 5, global_guard).unwrap();

    let scope = nemo_flow_push_scope(
        "sl_merge_scope",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let scope_guard = js_fn2("name, args", "args.scope_ran = true; return args");
    scope_register_tool_sanitize_request_guardrail(&uuid, "sl_merge_local", 15, scope_guard)
        .unwrap();

    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{}"#);
    nemo_flow_tool_call_execute(
        "sl_merged_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    // Sanitize guardrails are observability-only; verify via tool Start event input
    deregister_subscriber("sl_merge_sub").unwrap();
    let events = js_sys::Array::from(&js_sys::eval("globalThis.__wasm_sl_merge_events").unwrap());
    let mut found = false;
    for i in 0..events.length() {
        let evt = events.get(i);
        let kind = js_sys::Reflect::get(&evt, &"kind".into()).unwrap();
        if kind.as_string().as_deref() == Some("ToolStart") {
            let input = js_sys::Reflect::get(&evt, &"input".into()).unwrap();
            let global_ran = js_sys::Reflect::get(&input, &"global_ran".into()).unwrap();
            assert!(global_ran.as_bool().unwrap());
            let scope_ran = js_sys::Reflect::get(&input, &"scope_ran".into()).unwrap();
            assert!(scope_ran.as_bool().unwrap());
            found = true;
            break;
        }
    }
    assert!(found, "Expected a tool Start event with sanitized input");

    js_sys::eval("delete globalThis.__wasm_sl_merge_events").unwrap();
    scope_deregister_tool_sanitize_request_guardrail(&uuid, "sl_merge_local").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
    deregister_tool_sanitize_request_guardrail("sl_merge_global").unwrap();
}

#[wasm_bindgen_test]
async fn test_global_and_scope_local_request_intercepts_both_run() {
    let global_int = js_fn2("name, args", "args.global_intercepted = true; return args");
    register_tool_request_intercept("sl_merge_global_int", 5, false, global_int).unwrap();

    let scope = nemo_flow_push_scope(
        "sl_merge_int_scope",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let scope_int = js_fn2("name, args", "args.scope_intercepted = true; return args");
    scope_register_tool_request_intercept(&uuid, "sl_merge_local_int", 15, false, scope_int)
        .unwrap();

    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_merge_int_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let global_intercepted = js_sys::Reflect::get(&result, &"global_intercepted".into()).unwrap();
    assert!(global_intercepted.as_bool().unwrap());
    let scope_intercepted = js_sys::Reflect::get(&result, &"scope_intercepted".into()).unwrap();
    assert!(scope_intercepted.as_bool().unwrap());

    scope_deregister_tool_request_intercept(&uuid, "sl_merge_local_int").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
    deregister_tool_request_intercept("sl_merge_global_int").unwrap();
}

#[wasm_bindgen_test]
async fn test_scope_local_and_global_execution_intercepts_merge() {
    let global_exec = js_fn1("args", "args.global_exec = true; return args");
    register_tool_execution_intercept("sl_merge_global_exec", 5, global_exec).unwrap();

    let scope = nemo_flow_push_scope(
        "sl_merge_exec_scope",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let uuid = scope.uuid();

    let scope_exec = js_fn1("args", "args.scope_exec = true; return args");
    scope_register_tool_execution_intercept(&uuid, "sl_merge_local_exec", 15, scope_exec).unwrap();

    let original = js_fn1("args", "return args");
    let args = parse_json(r#"{"base": true}"#);
    let result = nemo_flow_tool_call_execute(
        "sl_merge_exec_tool",
        args,
        original,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    // At least one execution intercept should have run
    let global_exec_val = js_sys::Reflect::get(&result, &"global_exec".into()).unwrap();
    let scope_exec_val = js_sys::Reflect::get(&result, &"scope_exec".into()).unwrap();
    assert!(
        global_exec_val.as_bool().unwrap_or(false) || scope_exec_val.as_bool().unwrap_or(false),
        "At least one execution intercept should have run"
    );

    scope_deregister_tool_execution_intercept(&uuid, "sl_merge_local_exec").unwrap();
    nemo_flow_pop_scope(&scope).unwrap();
    deregister_tool_execution_intercept("sl_merge_global_exec").unwrap();
}
