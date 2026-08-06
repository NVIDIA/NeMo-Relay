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

fn gemini_annotated(
    message: Option<&str>,
    tool_calls: Option<Vec<ResponseToolCall>>,
    id: Option<&str>,
    model: Option<&str>,
) -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: id.map(String::from),
        model: model.map(String::from),
        message: message.map(|t| nemo_relay::codec::request::MessageContent::Text(t.into())),
        tool_calls,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: Default::default(),
    }
}

#[test]
fn gemini_overlay_redacts_candidate_text() {
    let payload = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "raw secret text"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "modelVersion": "gemini-2.0-flash"
    });

    let annotated = gemini_annotated(Some("[REDACTED]"), None, None, None);
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);

    let text = result["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .unwrap();
    assert_eq!(
        text, "[REDACTED]",
        "Gemini overlay must redact candidate text"
    );
}

#[test]
fn gemini_overlay_preserves_embedded_newline_text_and_thought_parts() {
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "raw line one\nraw line two"},
                    {"text": "raw second part"},
                    {"text": "", "thought": true, "thoughtSignature": "sig-THOUGHT"}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = gemini_annotated(Some("[REDACTED]\nkept together"), None, None, None);
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);
    let parts = result["candidates"][0]["content"]["parts"]
        .as_array()
        .expect("Gemini parts array");

    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["text"], json!("[REDACTED]\nkept together"));
    assert_eq!(parts[1]["thought"], json!(true));
    assert_eq!(parts[1]["thoughtSignature"], json!("sig-THOUGHT"));
}

#[test]
fn gemini_overlay_does_not_add_absent_response_id_or_model_version() {
    let payload = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = gemini_annotated(Some("hi"), None, None, None);
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);
    assert!(result.get("responseId").is_none());
    assert!(result.get("modelVersion").is_none());
}

#[test]
fn gemini_overlay_redacts_provider_native_candidate_part() {
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "ran code"},
                    {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "sk-code-secret"}}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "ran code".into(),
                extra: Default::default(),
            },
            ContentPart::ProviderNative {
                provider: "gemini".into(),
                kind: "codeExecutionResult".into(),
                value: json!({
                    "codeExecutionResult": {
                        "outcome": "OUTCOME_OK",
                        "output": "[REDACTED]"
                    }
                }),
            },
        ])),
        ..Default::default()
    };

    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);
    assert_eq!(
        result["candidates"][0]["content"]["parts"][1]["codeExecutionResult"]["output"],
        json!("[REDACTED]"),
        "Gemini overlay must write sanitized provider-native response parts back to raw payload"
    );
}

#[test]
fn gemini_overlay_updates_tool_call_args() {
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "search", "id": "c1", "args": {"secret": "raw"}}}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = gemini_annotated(
        None,
        Some(vec![tool_call(
            "c1",
            "search",
            json!({"secret": "[REDACTED]"}),
        )]),
        None,
        None,
    );
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);

    let args = &result["candidates"][0]["content"]["parts"][0]["functionCall"]["args"];
    assert_eq!(
        args["secret"],
        json!("[REDACTED]"),
        "Gemini overlay must redact functionCall args"
    );
}

#[test]
fn gemini_overlay_does_not_synthesize_missing_function_call_id() {
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "search", "args": {"secret": "raw"}}}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = gemini_annotated(
        None,
        Some(vec![tool_call(
            "search",
            "search",
            json!({"secret": "[REDACTED]"}),
        )]),
        None,
        None,
    );
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);
    let fc = &result["candidates"][0]["content"]["parts"][0]["functionCall"];

    assert!(fc.get("id").is_none());
    assert_eq!(fc["name"], json!("search"));
    assert_eq!(fc["args"]["secret"], json!("[REDACTED]"));
}

#[test]
fn gemini_overlay_removes_extra_function_call_parts() {
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "one", "id": "c1", "args": {"secret": "raw-1"}}},
                    {"functionCall": {"name": "two", "id": "c2", "args": {"secret": "raw-2"}}},
                    {"text": "", "thought": true, "thoughtSignature": "sig-KEEP"}
                ]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = gemini_annotated(
        None,
        Some(vec![tool_call(
            "c1",
            "one",
            json!({"secret": "[REDACTED]"}),
        )]),
        None,
        None,
    );
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);
    let parts = result["candidates"][0]["content"]["parts"]
        .as_array()
        .expect("Gemini parts array");

    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["functionCall"]["id"], json!("c1"));
    assert_eq!(
        parts[0]["functionCall"]["args"]["secret"],
        json!("[REDACTED]")
    );
    assert_eq!(parts[1]["thoughtSignature"], json!("sig-KEEP"));
}

#[test]
fn gemini_overlay_updates_response_id_and_model_version() {
    let payload = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "responseId": "resp-old",
        "modelVersion": "gemini-old"
    });

    // Annotated view carries the sanitizer-approved id/model.
    let annotated = gemini_annotated(Some("hi"), None, Some("resp-abc"), Some("gemini-2.0-flash"));
    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);

    assert_eq!(
        result["responseId"],
        json!("resp-abc"),
        "overlay must write annotated.id to responseId"
    );
    assert_eq!(
        result["modelVersion"],
        json!("gemini-2.0-flash"),
        "overlay must write annotated.model to modelVersion"
    );
}

#[test]
fn gemini_overlay_does_not_overwrite_finish_reason() {
    // A STOP response with a functionCall part: normalized finish_reason is ToolUse,
    // but the raw finishReason in the payload is STOP and must not be overwritten.
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "fn", "id": "c1", "args": {}}}]
            },
            "finishReason": "STOP",
            "index": 0
        }]
    });

    let annotated = AnnotatedLlmResponse {
        finish_reason: Some(nemo_relay::codec::response::FinishReason::ToolUse),
        tool_calls: Some(vec![tool_call("c1", "fn", json!({}))]),
        ..Default::default()
    };

    let result =
        BuiltinCodecName::GeminiGenerateContent.overlay_response_payload(payload, &annotated);

    assert_eq!(
        result["candidates"][0]["finishReason"].as_str(),
        Some("STOP"),
        "Gemini overlay must not overwrite native finishReason with the derived ToolUse value"
    );
}
