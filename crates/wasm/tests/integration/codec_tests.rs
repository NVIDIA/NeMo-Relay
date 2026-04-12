// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a JS decode function: extracts messages, model, params from content.
fn make_decode_fn() -> js_sys::Function {
    js_sys::Function::new_with_args(
        "request",
        r#"
        var c = request.content;
        return {
            messages: c.messages || [],
            model: c.model || null,
            params: c.temperature !== undefined ? { temperature: c.temperature } : null,
            tools: null,
            tool_choice: null,
            extra: {}
        };
        "#,
    )
}

/// Create a JS encode function: merges annotated back into original.
fn make_encode_fn() -> js_sys::Function {
    js_sys::Function::new_with_args(
        "annotated, original",
        r#"
        var content = Object.assign({}, original.content);
        content.messages = annotated.messages;
        if (annotated.model) content.model = annotated.model;
        if (annotated.params) Object.assign(content, annotated.params);
        if (annotated.extra) Object.assign(content, annotated.extra);
        return { headers: original.headers, content: content };
        "#,
    )
}

// ===========================================================================
// Codec helper functions
// ===========================================================================

#[wasm_bindgen_test]
fn test_make_decode_fn_creates_valid_function() {
    let decode = make_decode_fn();
    // Verify it is a callable JS function
    assert!(decode.is_function());
}

#[wasm_bindgen_test]
fn test_make_encode_fn_creates_valid_function() {
    let encode = make_encode_fn();
    // Verify it is a callable JS function
    assert!(encode.is_function());
}

#[wasm_bindgen_test]
fn test_decode_fn_produces_annotated_structure() {
    use wasm_bindgen::JsValue;

    let decode = make_decode_fn();

    // Create a mock LLMRequest-like object
    let request = js_sys::Object::new();
    let content = js_sys::Object::new();
    let messages = js_sys::Array::new();
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(&msg, &"role".into(), &"user".into()).unwrap();
    js_sys::Reflect::set(&msg, &"content".into(), &"Hello".into()).unwrap();
    messages.push(&msg);
    js_sys::Reflect::set(&content, &"messages".into(), &messages).unwrap();
    js_sys::Reflect::set(&content, &"model".into(), &"gpt-4".into()).unwrap();
    js_sys::Reflect::set(&content, &"temperature".into(), &JsValue::from_f64(0.7)).unwrap();
    js_sys::Reflect::set(&request, &"content".into(), &content).unwrap();

    let result = decode.call1(&JsValue::NULL, &request).unwrap();

    // Verify the annotated structure has expected fields
    let model = js_sys::Reflect::get(&result, &"model".into()).unwrap();
    assert_eq!(model.as_string().unwrap(), "gpt-4");

    let msgs = js_sys::Reflect::get(&result, &"messages".into()).unwrap();
    let msgs_arr = js_sys::Array::from(&msgs);
    assert_eq!(msgs_arr.length(), 1);
}
