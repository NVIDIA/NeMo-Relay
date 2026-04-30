// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for scope in the NeMo Flow WebAssembly crate.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
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

fn with_scope(
    name: &str,
    scope_type: ScopeType,
    callback: &js_sys::Function,
    handle: Option<ScopeHandle>,
    attributes: Option<u32>,
    data: JsValue,
    metadata: JsValue,
) -> Result<js_sys::Promise, JsValue> {
    nemo_flow_wasm::api::with_scope(
        name,
        scope_type,
        callback,
        parent_handle(handle),
        attributes,
        data,
        metadata,
        JsValue::NULL,
    )
}

fn pop_scope(handle: &ScopeHandle) -> Result<(), JsValue> {
    nemo_flow_wasm::api::pop_scope(handle, JsValue::NULL, None)
}

fn pop_scope_with_output(handle: &ScopeHandle, output: JsValue) -> Result<(), JsValue> {
    nemo_flow_wasm::api::pop_scope(handle, output, None)
}

fn event(
    name: &str,
    handle: Option<ScopeHandle>,
    data: JsValue,
    metadata: JsValue,
) -> Result<(), JsValue> {
    nemo_flow_wasm::api::event(name, parent_handle(handle), data, metadata, None)
}

// ===========================================================================
// Scope operations
// ===========================================================================

#[wasm_bindgen_test]
fn test_get_handle_returns_root() {
    let handle = get_handle().unwrap();
    assert!(!handle.uuid().is_empty());
}

#[wasm_bindgen_test]
fn test_push_pop_scope() {
    let scope = push_scope(
        "test_wasm_scope",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(scope.name(), "test_wasm_scope");
    assert_eq!(scope.scope_type(), ScopeType::Agent);
    pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_attributes() {
    let scope = push_scope(
        "attr_scope",
        ScopeType::Function,
        None,
        Some(SCOPE_PARALLEL | SCOPE_RELOCATABLE),
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(scope.attributes(), SCOPE_PARALLEL | SCOPE_RELOCATABLE);
    pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_parent() {
    let parent = push_scope(
        "parent_scope",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let parent_uuid = parent.uuid();
    let child = push_scope(
        "child_scope",
        ScopeType::Function,
        Some(parent),
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(
        child.parent_uuid().as_string().as_deref(),
        Some(parent_uuid.as_str())
    );
    pop_scope(&child).unwrap();
    let current = get_handle().unwrap();
    assert_eq!(current.uuid(), parent_uuid);
    pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_nesting() {
    let s1 = push_scope(
        "nest_1",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let s2 = push_scope(
        "nest_2",
        ScopeType::Function,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let s3 = push_scope(
        "nest_3",
        ScopeType::Tool,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    pop_scope(&s3).unwrap();
    pop_scope(&s2).unwrap();
    pop_scope(&s1).unwrap();
}

#[wasm_bindgen_test]
fn test_all_scope_types() {
    let types = [
        (ScopeType::Agent, "agent_s"),
        (ScopeType::Function, "function_s"),
        (ScopeType::Tool, "tool_s"),
        (ScopeType::Llm, "llm_s"),
        (ScopeType::Retriever, "retriever_s"),
        (ScopeType::Embedder, "embedder_s"),
        (ScopeType::Reranker, "reranker_s"),
        (ScopeType::Guardrail, "guardrail_s"),
        (ScopeType::Evaluator, "evaluator_s"),
        (ScopeType::Custom, "custom_s"),
        (ScopeType::Unknown, "unknown_s"),
    ];
    for (st, name) in types {
        let scope = push_scope(name, st, None, None, JsValue::NULL, JsValue::NULL).unwrap();
        assert_eq!(scope.scope_type(), st);
        pop_scope(&scope).unwrap();
    }
}

// ===========================================================================
// withScope (context manager)
// ===========================================================================

#[wasm_bindgen_test]
async fn test_with_scope_normal_return() {
    let before = get_handle().unwrap();
    let before_uuid = before.uuid();

    // Callback that returns the handle's uuid
    let cb = js_fn1("handle", "return handle.uuid");
    let result = with_scope(
        "with_scope_test",
        ScopeType::Agent,
        &cb,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let resolved = JsFuture::from(result).await.unwrap();

    // The callback should have received a handle with a uuid
    assert!(resolved.is_string(), "Expected string uuid from callback");

    // Scope should be popped
    let after = get_handle().unwrap();
    assert_eq!(
        after.uuid(),
        before_uuid,
        "Scope should be popped after withScope"
    );
}

#[wasm_bindgen_test]
fn test_with_scope_callback_receives_handle() {
    // Store handle properties in a global for inspection
    js_sys::eval("globalThis.__wasm_ws_handle = null; true").unwrap();
    let cb = js_fn1("handle", "globalThis.__wasm_ws_handle = handle");
    let _ = with_scope(
        "handle_check",
        ScopeType::Function,
        &cb,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();

    let handle = js_sys::eval("globalThis.__wasm_ws_handle").unwrap();
    assert!(
        !handle.is_null() && !handle.is_undefined(),
        "Handle should be set"
    );

    // Check that the handle has expected properties (WasmScopeHandle getters)
    let uuid = js_sys::Reflect::get(&handle, &"uuid".into()).unwrap();
    assert!(uuid.is_string(), "Handle should have uuid string");

    let name = js_sys::Reflect::get(&handle, &"name".into()).unwrap();
    assert_eq!(name.as_string().unwrap(), "handle_check");

    let scope_type = js_sys::Reflect::get(&handle, &"scopeType".into()).unwrap();
    assert_eq!(
        scope_type.as_f64().unwrap() as i32,
        ScopeType::Function as i32
    );

    js_sys::eval("delete globalThis.__wasm_ws_handle").unwrap();
}

#[wasm_bindgen_test]
fn test_with_scope_pops_on_throw() {
    let before = get_handle().unwrap();
    let before_uuid = before.uuid();

    let cb = js_fn1("handle", "throw new Error('test error')");
    let result = with_scope(
        "throw_test",
        ScopeType::Tool,
        &cb,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    );

    // Should have returned an error
    assert!(result.is_err(), "Expected error from throwing callback");

    // Scope should still be popped
    let after = get_handle().unwrap();
    assert_eq!(
        after.uuid(),
        before_uuid,
        "Scope should be popped after throw"
    );
}

#[wasm_bindgen_test]
async fn test_with_scope_nested() {
    let before = get_handle().unwrap();
    let before_uuid = before.uuid();

    // Push outer scope manually so we can nest a withScope inside it.
    let outer = push_scope(
        "outer",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let outer_uuid = outer.uuid();

    // Use withScope for the inner scope — the callback returns parentUuid.
    let inner_cb = js_fn1("handle", "return handle.parentUuid");
    let inner_parent = JsFuture::from(
        with_scope(
            "inner",
            ScopeType::Function,
            &inner_cb,
            None,
            None,
            JsValue::NULL,
            JsValue::NULL,
        )
        .unwrap(),
    )
    .await
    .unwrap()
    .as_string()
    .unwrap_or_default();

    // The inner scope's parent should be the outer scope.
    assert_eq!(
        inner_parent, outer_uuid,
        "Inner scope's parent should be the outer scope"
    );

    // After withScope returns, the inner scope is popped; outer should be on top.
    let current = get_handle().unwrap();
    assert_eq!(
        current.uuid(),
        outer_uuid,
        "Outer scope should be on top after inner withScope completes"
    );

    // Pop the outer scope.
    pop_scope(&outer).unwrap();

    // Stack should be back to original.
    let after = get_handle().unwrap();
    assert_eq!(after.uuid(), before_uuid, "All scopes should be popped");

    // Clean up globals.
    let _ =
        js_sys::Reflect::delete_property(&js_sys::global(), &JsValue::from_str("__wasm_inner_cb"));
    let _ = js_sys::Reflect::delete_property(
        &js_sys::global(),
        &JsValue::from_str("__wasm_inner_parent"),
    );
    let _ = js_sys::Reflect::delete_property(
        &js_sys::global(),
        &JsValue::from_str("__wasm_outer_uuid"),
    );
}

// ===========================================================================
// Events
// ===========================================================================

#[wasm_bindgen_test]
fn test_event_basic() {
    event("test_event", None, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_data() {
    let data = parse_json(r#"{"key":"value"}"#);
    event("data_event", None, data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_parent() {
    let scope = push_scope(
        "event_parent",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let scope_uuid = scope.uuid();
    event("child_event", Some(scope), JsValue::NULL, JsValue::NULL).unwrap();
    let current = get_handle().unwrap();
    assert_eq!(current.uuid(), scope_uuid);
    pop_scope(&current).unwrap();
}

// ===========================================================================
// Subscribers
// ===========================================================================

#[wasm_bindgen_test]
fn test_register_deregister_subscriber() {
    let cb = js_fn1("event", "");
    register_subscriber("wasm_sub_1", cb).unwrap();
    let removed = deregister_subscriber("wasm_sub_1").unwrap();
    assert!(removed);
}

#[wasm_bindgen_test]
fn test_duplicate_subscriber_fails() {
    let cb1 = js_fn1("event", "");
    let cb2 = js_fn1("event", "");
    register_subscriber("wasm_dup_sub", cb1).unwrap();
    let result = register_subscriber("wasm_dup_sub", cb2);
    assert!(result.is_err());
    deregister_subscriber("wasm_dup_sub").unwrap();
}

#[wasm_bindgen_test]
fn test_deregister_nonexistent_subscriber() {
    let removed = deregister_subscriber("nonexistent_sub").unwrap();
    assert!(!removed);
}

#[wasm_bindgen_test]
fn test_subscriber_receives_events() {
    js_sys::eval("globalThis.__wasm_test_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__wasm_test_events.push(event)");
    register_subscriber("wasm_event_collector", cb).unwrap();

    let scope = push_scope(
        "sub_test",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    pop_scope(&scope).unwrap();

    let events = js_sys::eval("globalThis.__wasm_test_events").unwrap();
    let arr = js_sys::Array::from(&events);
    assert!(arr.length() > 0, "Expected at least one event");

    deregister_subscriber("wasm_event_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_test_events").unwrap();
}

#[wasm_bindgen_test]
fn test_subscriber_event_properties() {
    js_sys::eval("globalThis.__wasm_evt_props = null; true").unwrap();
    let cb = js_fn1(
        "event",
        "if (!globalThis.__wasm_evt_props) globalThis.__wasm_evt_props = event",
    );
    register_subscriber("wasm_prop_collector", cb).unwrap();

    let scope = push_scope(
        "prop_test",
        ScopeType::Function,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    pop_scope(&scope).unwrap();

    let event = js_sys::eval("globalThis.__wasm_evt_props").unwrap();
    assert!(
        !event.is_null() && !event.is_undefined(),
        "Expected an event"
    );

    let uuid = js_sys::Reflect::get(&event, &"uuid".into()).unwrap();
    assert!(uuid.is_string(), "Event should have uuid string");

    let timestamp = js_sys::Reflect::get(&event, &"timestamp".into()).unwrap();
    assert!(timestamp.is_string(), "Event should have timestamp string");

    let kind = js_sys::Reflect::get(&event, &"kind".into()).unwrap();
    assert!(kind.is_string(), "Event should have kind string");

    deregister_subscriber("wasm_prop_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_evt_props").unwrap();
}

#[wasm_bindgen_test]
fn test_subscriber_receives_scope_output_payload() {
    js_sys::eval("globalThis.__wasm_scope_end_events = []; true").unwrap();
    let cb = js_fn1(
        "event",
        "if (event.kind === 'scope' && event.scope_category === 'end') globalThis.__wasm_scope_end_events.push(event)",
    );
    register_subscriber("wasm_scope_end_collector", cb).unwrap();

    let scope = push_scope(
        "output_scope",
        ScopeType::Agent,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    pop_scope_with_output(
        &scope,
        parse_json(r#"{"status":"done","metrics":{"tokens":42}}"#),
    )
    .unwrap();

    let events = js_sys::eval("globalThis.__wasm_scope_end_events").unwrap();
    let arr = js_sys::Array::from(&events);
    assert_eq!(arr.length(), 1, "Expected one scope end event");

    let event = arr.get(0);
    let output = js_sys::Reflect::get(&event, &"data".into()).unwrap();
    let output_json =
        serde_wasm_bindgen::from_value::<serde_json::Value>(output).expect("output should be JSON");
    assert_eq!(
        output_json,
        serde_json::json!({"status":"done","metrics":{"tokens":42}})
    );

    deregister_subscriber("wasm_scope_end_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_scope_end_events").unwrap();
}

#[wasm_bindgen_test]
fn test_event_mark() {
    js_sys::eval("globalThis.__wasm_mark_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__wasm_mark_events.push(event)");
    register_subscriber("wasm_mark_collector", cb).unwrap();

    let data = parse_json(r#"{"marker":"test"}"#);
    event("mark_event", None, data, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__wasm_mark_events").unwrap();
    let arr = js_sys::Array::from(&events);
    let found = (0..arr.length()).any(|i| {
        let e = arr.get(i);
        let kind_value = js_sys::Reflect::get(&e, &"kind".into())
            .unwrap()
            .as_string();
        let kind = kind_value.as_deref();
        kind == Some("mark")
    });
    assert!(found, "Expected a Mark event");

    deregister_subscriber("wasm_mark_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_mark_events").unwrap();
}
