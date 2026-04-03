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

fn parse_json(s: &str) -> JsValue {
    js_sys::JSON::parse(s).unwrap()
}

// ===========================================================================
// Scope operations
// ===========================================================================

#[wasm_bindgen_test]
fn test_get_handle_returns_root() {
    let handle = nat_nexus_get_handle().unwrap();
    assert!(!handle.uuid().is_empty());
}

#[wasm_bindgen_test]
fn test_push_pop_scope() {
    let scope = nat_nexus_push_scope(
        "test_wasm_scope",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(scope.name(), "test_wasm_scope");
    assert_eq!(scope.scope_type(), SCOPE_TYPE_AGENT);
    nat_nexus_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_attributes() {
    let scope = nat_nexus_push_scope(
        "attr_scope",
        SCOPE_TYPE_FUNCTION,
        None,
        Some(SCOPE_PARALLEL | SCOPE_RELOCATABLE),
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(scope.attributes(), SCOPE_PARALLEL | SCOPE_RELOCATABLE);
    nat_nexus_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_parent() {
    let parent = nat_nexus_push_scope(
        "parent_scope",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let parent_uuid = parent.uuid();
    let child = nat_nexus_push_scope(
        "child_scope",
        SCOPE_TYPE_FUNCTION,
        Some(parent),
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(child.parent_uuid().unwrap(), parent_uuid);
    nat_nexus_pop_scope(&child).unwrap();
    let current = nat_nexus_get_handle().unwrap();
    assert_eq!(current.uuid(), parent_uuid);
    nat_nexus_pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_nesting() {
    let s1 = nat_nexus_push_scope(
        "nest_1",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let s2 = nat_nexus_push_scope(
        "nest_2",
        SCOPE_TYPE_FUNCTION,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let s3 = nat_nexus_push_scope(
        "nest_3",
        SCOPE_TYPE_TOOL,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    nat_nexus_pop_scope(&s3).unwrap();
    nat_nexus_pop_scope(&s2).unwrap();
    nat_nexus_pop_scope(&s1).unwrap();
}

#[wasm_bindgen_test]
fn test_all_scope_types() {
    let types = [
        (SCOPE_TYPE_AGENT, "agent_s"),
        (SCOPE_TYPE_FUNCTION, "function_s"),
        (SCOPE_TYPE_TOOL, "tool_s"),
        (SCOPE_TYPE_LLM, "llm_s"),
        (SCOPE_TYPE_RETRIEVER, "retriever_s"),
        (SCOPE_TYPE_EMBEDDER, "embedder_s"),
        (SCOPE_TYPE_RERANKER, "reranker_s"),
        (SCOPE_TYPE_GUARDRAIL, "guardrail_s"),
        (SCOPE_TYPE_EVALUATOR, "evaluator_s"),
        (SCOPE_TYPE_CUSTOM, "custom_s"),
        (SCOPE_TYPE_UNKNOWN, "unknown_s"),
    ];
    for (st, name) in types {
        let scope =
            nat_nexus_push_scope(name, st, None, None, JsValue::NULL, JsValue::NULL).unwrap();
        assert_eq!(scope.scope_type(), st);
        nat_nexus_pop_scope(&scope).unwrap();
    }
}

// ===========================================================================
// withScope (context manager)
// ===========================================================================

#[wasm_bindgen_test]
fn test_with_scope_normal_return() {
    let before = nat_nexus_get_handle().unwrap();
    let before_uuid = before.uuid();

    // Callback that returns the handle's uuid
    let cb = js_fn1("handle", "return handle.uuid");
    let result = nat_nexus_with_scope(
        "with_scope_test",
        SCOPE_TYPE_AGENT,
        &cb,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();

    // The callback should have received a handle with a uuid
    assert!(result.is_string(), "Expected string uuid from callback");

    // Scope should be popped
    let after = nat_nexus_get_handle().unwrap();
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
    nat_nexus_with_scope(
        "handle_check",
        SCOPE_TYPE_FUNCTION,
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
    assert_eq!(scope_type.as_f64().unwrap() as i32, SCOPE_TYPE_FUNCTION);

    js_sys::eval("delete globalThis.__wasm_ws_handle").unwrap();
}

#[wasm_bindgen_test]
fn test_with_scope_pops_on_throw() {
    let before = nat_nexus_get_handle().unwrap();
    let before_uuid = before.uuid();

    let cb = js_fn1("handle", "throw new Error('test error')");
    let result = nat_nexus_with_scope(
        "throw_test",
        SCOPE_TYPE_TOOL,
        &cb,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    );

    // Should have returned an error
    assert!(result.is_err(), "Expected error from throwing callback");

    // Scope should still be popped
    let after = nat_nexus_get_handle().unwrap();
    assert_eq!(
        after.uuid(),
        before_uuid,
        "Scope should be popped after throw"
    );
}

#[wasm_bindgen_test]
fn test_with_scope_nested() {
    let before = nat_nexus_get_handle().unwrap();
    let before_uuid = before.uuid();

    // Push outer scope manually so we can nest a withScope inside it.
    let outer = nat_nexus_push_scope(
        "outer",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let outer_uuid = outer.uuid();

    // Use withScope for the inner scope — the callback returns parentUuid.
    let inner_cb = js_fn1("handle", "return handle.parentUuid");
    let inner_parent = nat_nexus_with_scope(
        "inner",
        SCOPE_TYPE_FUNCTION,
        &inner_cb,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap()
    .as_string()
    .unwrap_or_default();

    // The inner scope's parent should be the outer scope.
    assert_eq!(
        inner_parent, outer_uuid,
        "Inner scope's parent should be the outer scope"
    );

    // After withScope returns, the inner scope is popped; outer should be on top.
    let current = nat_nexus_get_handle().unwrap();
    assert_eq!(
        current.uuid(),
        outer_uuid,
        "Outer scope should be on top after inner withScope completes"
    );

    // Pop the outer scope.
    nat_nexus_pop_scope(&outer).unwrap();

    // Stack should be back to original.
    let after = nat_nexus_get_handle().unwrap();
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
    nat_nexus_event("test_event", None, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_data() {
    let data = parse_json(r#"{"key":"value"}"#);
    nat_nexus_event("data_event", None, data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_parent() {
    let scope = nat_nexus_push_scope(
        "event_parent",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    let scope_uuid = scope.uuid();
    nat_nexus_event("child_event", Some(scope), JsValue::NULL, JsValue::NULL).unwrap();
    let current = nat_nexus_get_handle().unwrap();
    assert_eq!(current.uuid(), scope_uuid);
    nat_nexus_pop_scope(&current).unwrap();
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

    let scope = nat_nexus_push_scope(
        "sub_test",
        SCOPE_TYPE_AGENT,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    nat_nexus_pop_scope(&scope).unwrap();

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

    let scope = nat_nexus_push_scope(
        "prop_test",
        SCOPE_TYPE_FUNCTION,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    nat_nexus_pop_scope(&scope).unwrap();

    let event = js_sys::eval("globalThis.__wasm_evt_props").unwrap();
    assert!(
        !event.is_null() && !event.is_undefined(),
        "Expected an event"
    );

    let uuid = js_sys::Reflect::get(&event, &"uuid".into()).unwrap();
    assert!(uuid.is_string(), "Event should have uuid string");

    let timestamp = js_sys::Reflect::get(&event, &"timestamp".into()).unwrap();
    assert!(timestamp.is_string(), "Event should have timestamp string");

    let event_type = js_sys::Reflect::get(&event, &"event_type".into()).unwrap();
    assert!(
        event_type.as_f64().is_some(),
        "Event should have event_type number"
    );

    deregister_subscriber("wasm_prop_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_evt_props").unwrap();
}

#[wasm_bindgen_test]
fn test_event_mark() {
    js_sys::eval("globalThis.__wasm_mark_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__wasm_mark_events.push(event)");
    register_subscriber("wasm_mark_collector", cb).unwrap();

    let data = parse_json(r#"{"marker":"test"}"#);
    nat_nexus_event("mark_event", None, data, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__wasm_mark_events").unwrap();
    let arr = js_sys::Array::from(&events);
    let found = (0..arr.length()).any(|i| {
        let e = arr.get(i);
        let et = js_sys::Reflect::get(&e, &"event_type".into())
            .unwrap()
            .as_f64()
            .unwrap_or(-1.0);
        et == 2.0 // Mark = 2
    });
    assert!(found, "Expected a Mark event (event_type=2)");

    deregister_subscriber("wasm_mark_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_mark_events").unwrap();
}
