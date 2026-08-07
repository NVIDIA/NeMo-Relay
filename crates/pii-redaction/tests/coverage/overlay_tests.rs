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
fn annotated_message_text_includes_provider_native_text_and_refusal_parts() {
    let content = MessageContent::Parts(vec![
        ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "output_text".into(),
            value: json!({"text": "redacted text"}),
        },
        ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "refusal".into(),
            value: json!({"refusal": "redacted refusal"}),
        },
        ContentPart::ProviderNative {
            provider: "openai_responses".into(),
            kind: "reasoning".into(),
            value: json!({"summary": []}),
        },
    ]);

    assert_eq!(
        annotated_message_text(Some(&content)).as_deref(),
        Some("redacted text\nredacted refusal")
    );
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

#[test]
fn oci_genai_overlay_rewrites_generic_text_and_tool_calls() {
    let payload = json!({
        "modelId": "meta.llama-3.3-70b-instruct",
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "raw secret"}],
                    "toolCalls": [
                        {"id": "call_1", "type": "FUNCTION", "name": "one", "arguments": "{\"secret\":\"raw-1\"}"},
                        {"id": "call_2", "type": "FUNCTION", "name": "two", "arguments": "{\"secret\":\"raw-2\"}"}
                    ]
                },
                "finishReason": "tool_calls"
            }]
        }
    });
    let annotated = AnnotatedLlmResponse {
        model: Some("meta.llama-3.3-70b-instruct".into()),
        message: Some(MessageContent::Text("[REDACTED]".into())),
        tool_calls: Some(vec![tool_call(
            "call_1",
            "one",
            json!({"secret": "[REDACTED]"}),
        )]),
        finish_reason: Some(FinishReason::ToolUse),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    let message = &overlaid["chatResponse"]["choices"][0]["message"];
    assert_eq!(message["content"][0]["text"], json!("[REDACTED]"));
    let calls = message["toolCalls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["arguments"], json!("{\"secret\":\"[REDACTED]\"}"));
    assert_eq!(
        overlaid["chatResponse"]["choices"][0]["finishReason"],
        json!("tool_calls")
    );
}

#[test]
fn oci_genai_overlay_rewrites_cohere_text() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "raw secret",
            "finishReason": "COMPLETE"
        }
    });
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        finish_reason: Some(FinishReason::Complete),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    assert_eq!(overlaid["chatResponse"]["text"], json!("[REDACTED]"));
    assert_eq!(overlaid["chatResponse"]["finishReason"], json!("COMPLETE"));
}

#[test]
fn oci_genai_overlay_rewrites_each_text_part_and_keeps_non_text_blocks() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [
                        {"type": "TEXT", "text": "raw one"},
                        {"type": "IMAGE", "imageUrl": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "TEXT", "text": "raw two"}
                    ]
                }
            }]
        }
    });
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text(
            "[REDACTED ONE]\n[REDACTED TWO]\nwith remainder".into(),
        )),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    let content = &overlaid["chatResponse"]["choices"][0]["message"]["content"];
    assert_eq!(content[0]["text"], json!("[REDACTED ONE]"));
    assert_eq!(
        content[1],
        json!({"type": "IMAGE", "imageUrl": {"url": "data:image/png;base64,AAAA"}})
    );
    // The final TEXT part keeps any surplus newline-separated text.
    assert_eq!(content[2]["text"], json!("[REDACTED TWO]\nwith remainder"));
}

#[test]
fn oci_genai_overlay_sanitizes_nested_function_tool_calls() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [],
                    "toolCalls": [{
                        "id": "call_1",
                        "type": "FUNCTION",
                        "function": {"name": "one", "arguments": "{\"secret\":\"raw-1\"}"}
                    }]
                }
            }]
        }
    });
    let annotated = AnnotatedLlmResponse {
        tool_calls: Some(vec![tool_call(
            "call_1",
            "one",
            json!({"secret": "[REDACTED]"}),
        )]),
        finish_reason: Some(FinishReason::ToolUse),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    let call = &overlaid["chatResponse"]["choices"][0]["message"]["toolCalls"][0];
    assert_eq!(
        call["function"]["arguments"],
        json!("{\"secret\":\"[REDACTED]\"}")
    );
    assert!(
        call.get("arguments").is_none(),
        "sanitized arguments must land on the nested function object, got {call}"
    );
}
