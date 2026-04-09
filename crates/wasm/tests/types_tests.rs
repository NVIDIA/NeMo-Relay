// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use nemo_flow_wasm::types::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
fn empty_obj() -> JsValue {
    js_sys::Object::new().into()
}

fn parse_json(s: &str) -> JsValue {
    js_sys::JSON::parse(s).unwrap()
}

// ===========================================================================
// Type constants
// ===========================================================================

#[wasm_bindgen_test]
fn test_scope_type_constants() {
    assert_eq!(SCOPE_TYPE_AGENT, 0);
    assert_eq!(SCOPE_TYPE_FUNCTION, 1);
    assert_eq!(SCOPE_TYPE_TOOL, 2);
    assert_eq!(SCOPE_TYPE_LLM, 3);
    assert_eq!(SCOPE_TYPE_RETRIEVER, 4);
    assert_eq!(SCOPE_TYPE_EMBEDDER, 5);
    assert_eq!(SCOPE_TYPE_RERANKER, 6);
    assert_eq!(SCOPE_TYPE_GUARDRAIL, 7);
    assert_eq!(SCOPE_TYPE_EVALUATOR, 8);
    assert_eq!(SCOPE_TYPE_CUSTOM, 9);
    assert_eq!(SCOPE_TYPE_UNKNOWN, 10);
}

#[wasm_bindgen_test]
fn test_attribute_constants() {
    assert_eq!(SCOPE_PARALLEL, 0b01);
    assert_eq!(SCOPE_RELOCATABLE, 0b10);
    assert_eq!(TOOL_LOCAL, 0b01);
    assert_eq!(LLM_STATELESS, 0b01);
    assert_eq!(LLM_STREAMING, 0b10);
}

#[wasm_bindgen_test]
fn test_scope_type_roundtrip() {
    assert_eq!(scope_type_to_i32(i32_to_scope_type(0)), 0);
    assert_eq!(scope_type_to_i32(i32_to_scope_type(9)), 9);
    assert_eq!(scope_type_to_i32(i32_to_scope_type(99)), SCOPE_TYPE_UNKNOWN);
}

// ===========================================================================
// WasmLLMRequest
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_request_construction() {
    let headers = empty_obj();
    let content = parse_json(r#"{"model":"gpt-4"}"#);
    let req = WasmLLMRequest::new(headers, content).unwrap();

    let h = req.headers();
    assert!(h.is_object());

    let c = req.content();
    let model = js_sys::Reflect::get(&c, &"model".into()).unwrap();
    assert_eq!(model.as_string().unwrap(), "gpt-4");
}

#[wasm_bindgen_test]
fn test_llm_request_headers_and_content() {
    let headers = js_sys::Object::new();
    js_sys::Reflect::set(&headers, &"Authorization".into(), &"Bearer tok".into()).unwrap();
    let content = parse_json(r#"{"prompt":"hello"}"#);
    let req = WasmLLMRequest::new(headers.into(), content).unwrap();

    let h = req.headers();
    let auth = js_sys::Reflect::get(&h, &"Authorization".into()).unwrap();
    assert_eq!(auth.as_string().unwrap(), "Bearer tok");

    let c = req.content();
    let prompt = js_sys::Reflect::get(&c, &"prompt".into()).unwrap();
    assert_eq!(prompt.as_string().unwrap(), "hello");
}
