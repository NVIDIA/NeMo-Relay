// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use super::*;

fn tool_call(id: &str, name: &str, arguments: Json) -> ResponseToolCall {
    ResponseToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
    }
}

#[test]
fn openai_chat_overlay_truncates_extra_raw_tool_calls() {
    let mut message = json!({
        "tool_calls": [
            {"id": "call_1", "function": {"name": "one", "arguments": "{\"secret\":\"raw-1\"}"}},
            {"id": "call_2", "function": {"name": "two", "arguments": "{\"secret\":\"raw-2\"}"}}
        ]
    })
    .as_object()
    .unwrap()
    .clone();

    overlay_openai_chat_tool_calls(
        &mut message,
        Some(&[tool_call("call_1", "one", json!({"secret": "[REDACTED]"}))]),
    );

    let calls = message["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]["function"]["arguments"],
        json!("{\"secret\":\"[REDACTED]\"}")
    );
}

#[test]
fn openai_chat_overlay_removes_tool_calls_when_typed_entry_has_wrong_shape() {
    let mut message = json!({
        "tool_calls": [
            {"id": "call_1", "arguments": "{\"secret\":\"raw-1\"}"}
        ]
    })
    .as_object()
    .unwrap()
    .clone();

    overlay_openai_chat_tool_calls(
        &mut message,
        Some(&[tool_call("call_1", "one", json!({"secret": "[REDACTED]"}))]),
    );

    assert!(!message.contains_key("tool_calls"));
}

#[test]
fn openai_responses_overlay_removes_extra_function_calls() {
    let mut items = vec![
        json!({"type": "message", "content": [{"type": "output_text", "text": "ok"}]}),
        json!({"type": "function_call", "call_id": "call_1", "name": "one", "arguments": "{\"secret\":\"raw-1\"}"}),
        json!({"type": "function_call", "call_id": "call_2", "name": "two", "arguments": "{\"secret\":\"raw-2\"}"}),
    ];

    overlay_openai_responses_tool_calls(
        &mut items,
        Some(&[tool_call("call_1", "one", json!({"secret": "[REDACTED]"}))]),
    );

    assert_eq!(items.len(), 2);
    assert_eq!(items[1]["type"], json!("function_call"));
    assert_eq!(items[1]["arguments"], json!("{\"secret\":\"[REDACTED]\"}"));
}

#[test]
fn openai_responses_overlay_preserves_full_multiline_text_in_single_output_block() {
    let mut items = vec![json!({
        "type": "message",
        "content": [{"type": "output_text", "text": "raw"}]
    })];

    overlay_output_text_blocks(&mut items, Some("line one\nline two".to_string()));

    assert_eq!(items[0]["content"][0]["text"], json!("line one\nline two"));
}

#[test]
fn anthropic_overlay_removes_tool_use_blocks_when_no_sanitized_calls_exist() {
    let mut blocks = vec![
        json!({"type": "text", "text": "hello"}),
        json!({"type": "tool_use", "id": "call_1", "name": "one", "input": {"secret": "raw-1"}}),
    ];

    overlay_anthropic_tool_calls(&mut blocks, None);

    assert_eq!(blocks, vec![json!({"type": "text", "text": "hello"})]);
}

#[test]
fn anthropic_overlay_preserves_full_multiline_text_in_single_text_block() {
    let mut blocks = vec![json!({"type": "text", "text": "raw"})];

    overlay_anthropic_text_blocks(&mut blocks, Some("line one\nline two".to_string()));

    assert_eq!(blocks[0]["text"], json!("line one\nline two"));
}
