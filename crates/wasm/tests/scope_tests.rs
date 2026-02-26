// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use nvagentrt_wasm::api::*;
use nvagentrt_wasm::types::*;

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
    let handle = nvagentrt_get_handle().unwrap();
    assert!(!handle.uuid().is_empty());
}

#[wasm_bindgen_test]
fn test_push_pop_scope() {
    let scope = nvagentrt_push_scope("test_wasm_scope", SCOPE_TYPE_AGENT, None, None).unwrap();
    assert_eq!(scope.name(), "test_wasm_scope");
    assert_eq!(scope.scope_type(), SCOPE_TYPE_AGENT);
    nvagentrt_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_attributes() {
    let scope = nvagentrt_push_scope(
        "attr_scope",
        SCOPE_TYPE_FUNCTION,
        None,
        Some(SCOPE_PARALLEL | SCOPE_RELOCATABLE),
    )
    .unwrap();
    assert_eq!(scope.attributes(), SCOPE_PARALLEL | SCOPE_RELOCATABLE);
    nvagentrt_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_parent() {
    let parent = nvagentrt_push_scope("parent_scope", SCOPE_TYPE_AGENT, None, None).unwrap();
    let parent_uuid = parent.uuid();
    let child =
        nvagentrt_push_scope("child_scope", SCOPE_TYPE_FUNCTION, Some(parent), None).unwrap();
    assert_eq!(child.parent_uuid().unwrap(), parent_uuid);
    nvagentrt_pop_scope(&child).unwrap();
    let current = nvagentrt_get_handle().unwrap();
    assert_eq!(current.uuid(), parent_uuid);
    nvagentrt_pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_nesting() {
    let s1 = nvagentrt_push_scope("nest_1", SCOPE_TYPE_AGENT, None, None).unwrap();
    let s2 = nvagentrt_push_scope("nest_2", SCOPE_TYPE_FUNCTION, None, None).unwrap();
    let s3 = nvagentrt_push_scope("nest_3", SCOPE_TYPE_TOOL, None, None).unwrap();
    nvagentrt_pop_scope(&s3).unwrap();
    nvagentrt_pop_scope(&s2).unwrap();
    nvagentrt_pop_scope(&s1).unwrap();
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
        let scope = nvagentrt_push_scope(name, st, None, None).unwrap();
        assert_eq!(scope.scope_type(), st);
        nvagentrt_pop_scope(&scope).unwrap();
    }
}

// ===========================================================================
// Events
// ===========================================================================

#[wasm_bindgen_test]
fn test_event_basic() {
    nvagentrt_event("test_event", None, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_data() {
    let data = parse_json(r#"{"key":"value"}"#);
    nvagentrt_event("data_event", None, data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_parent() {
    let scope = nvagentrt_push_scope("event_parent", SCOPE_TYPE_AGENT, None, None).unwrap();
    let scope_uuid = scope.uuid();
    nvagentrt_event("child_event", Some(scope), JsValue::NULL, JsValue::NULL).unwrap();
    let current = nvagentrt_get_handle().unwrap();
    assert_eq!(current.uuid(), scope_uuid);
    nvagentrt_pop_scope(&current).unwrap();
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

    let scope = nvagentrt_push_scope("sub_test", SCOPE_TYPE_AGENT, None, None).unwrap();
    nvagentrt_pop_scope(&scope).unwrap();

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

    let scope = nvagentrt_push_scope("prop_test", SCOPE_TYPE_FUNCTION, None, None).unwrap();
    nvagentrt_pop_scope(&scope).unwrap();

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
    nvagentrt_event("mark_event", None, data, JsValue::NULL).unwrap();

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
