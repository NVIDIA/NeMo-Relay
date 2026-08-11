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

#[test]
fn oci_genai_overlay_sanitizes_flat_cohere_tool_calls() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "raw secret",
            "finishReason": "COMPLETE",
            "toolCalls": [
                {"name": "one", "parameters": {"secret": "raw-1"}},
                {"name": "two", "parameters": {"secret": "raw-2"}}
            ]
        }
    });
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        tool_calls: Some(vec![tool_call(
            "call_0",
            "one",
            json!({"secret": "[REDACTED]"}),
        )]),
        finish_reason: Some(FinishReason::Complete),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    let chat_response = &overlaid["chatResponse"];
    assert_eq!(chat_response["text"], json!("[REDACTED]"));
    let calls = chat_response["toolCalls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "dropped sanitized calls must be truncated");
    assert_eq!(calls[0]["parameters"], json!({"secret": "[REDACTED]"}));
    assert!(
        calls[0].get("id").is_none(),
        "COHERE wire tool calls carry no id and must not gain one"
    );
}

#[test]
fn oci_genai_overlay_sanitizes_cohere_v2_root_message() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "COHEREV2",
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": "raw secret"}],
                "toolCalls": [{
                    "id": "call_1",
                    "type": "FUNCTION",
                    "function": {"name": "one", "arguments": "{\"secret\":\"raw-1\"}"}
                }]
            },
            "finishReason": "TOOL_CALL"
        }
    });
    let annotated = AnnotatedLlmResponse {
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

    let message = &overlaid["chatResponse"]["message"];
    assert_eq!(message["content"][0]["text"], json!("[REDACTED]"));
    assert_eq!(
        message["toolCalls"][0]["function"]["arguments"],
        json!("{\"secret\":\"[REDACTED]\"}")
    );
    assert_eq!(overlaid["chatResponse"]["finishReason"], json!("TOOL_CALL"));
}

#[test]
fn oci_genai_overlay_sanitizes_generic_string_content() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {"role": "ASSISTANT", "content": "raw secret"}
            }]
        }
    });
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    assert_eq!(
        overlaid["chatResponse"]["choices"][0]["message"]["content"],
        json!("[REDACTED]")
    );
}

#[test]
fn oci_genai_overlay_drops_cohere_tool_calls_with_non_object_arguments() {
    for arguments in [json!("scalar"), json!([1, 2]), json!(null)] {
        let payload = json!({
            "chatResponse": {
                "apiFormat": "COHERE",
                "text": "ok",
                "toolCalls": [{"name": "one", "parameters": {"secret": "raw-1"}}]
            }
        });
        let annotated = AnnotatedLlmResponse {
            tool_calls: Some(vec![tool_call("call_0", "one", arguments.clone())]),
            ..AnnotatedLlmResponse::default()
        };

        let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

        assert!(
            overlaid["chatResponse"].get("toolCalls").is_none(),
            "non-object sanitized arguments ({arguments}) must drop toolCalls, got {}",
            overlaid["chatResponse"]
        );
    }
}

#[test]
fn oci_genai_overlay_removes_unsanitized_additional_choices() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "raw secret"}]}
                },
                {
                    "index": 1,
                    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "second raw secret"}]}
                }
            ]
        }
    });
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);

    let choices = overlaid["chatResponse"]["choices"].as_array().unwrap();
    assert_eq!(
        choices.len(),
        1,
        "additional raw choices have no sanitized counterpart and must be removed"
    );
    assert_eq!(
        choices[0]["message"]["content"][0]["text"],
        json!("[REDACTED]")
    );
}

#[test]
fn oci_genai_overlay_guards_pass_unrecognized_shapes_through() {
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        ..AnnotatedLlmResponse::default()
    };

    // Non-object payloads and shapes without the expected structure pass
    // through unchanged instead of panicking or half-sanitizing.
    for payload in [
        json!("not an object"),
        json!({"chatResponse": {"apiFormat": "GENERIC"}}),
        json!({"chatResponse": {"apiFormat": "GENERIC", "choices": ["not-an-object"]}}),
        json!({"chatResponse": {"apiFormat": "GENERIC", "choices": [{"index": 0}]}}),
        json!({"chatResponse": {"apiFormat": "COHEREV2"}}),
    ] {
        let overlaid =
            BuiltinCodecName::OCIGenAI.overlay_response_payload(payload.clone(), &annotated);
        assert_eq!(overlaid, payload);
    }
}

#[test]
fn oci_genai_overlay_reaches_bare_chat_response_via_provider_surface() {
    // Envelope-less payload routed through the provider-surface mapping.
    let payload = json!({
        "apiFormat": "COHERE",
        "text": "raw secret",
        "finishReason": "COMPLETE"
    });
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        finish_reason: Some(FinishReason::Complete),
        ..AnnotatedLlmResponse::default()
    };

    let overlaid = BuiltinCodecName::from_provider_surface(ProviderSurface::OCIGenAI)
        .overlay_response_payload(payload, &annotated);

    assert_eq!(overlaid["text"], json!("[REDACTED]"));
}

#[test]
fn oci_genai_overlay_removes_tool_calls_without_sanitized_counterparts() {
    // GENERIC: sanitized None removes the key; a non-object raw call also
    // removes the key rather than leaving unredacted entries behind.
    for (payload_calls, sanitized) in [
        (
            json!([{"id": "call_1", "name": "one", "arguments": "{\"secret\":\"raw\"}"}]),
            None,
        ),
        (
            json!(["not-an-object"]),
            Some(vec![tool_call("call_1", "one", json!({}))]),
        ),
    ] {
        let payload = json!({
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [{
                    "index": 0,
                    "message": {"role": "ASSISTANT", "content": [], "toolCalls": payload_calls}
                }]
            }
        });
        let annotated = AnnotatedLlmResponse {
            tool_calls: sanitized,
            ..AnnotatedLlmResponse::default()
        };
        let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);
        assert!(
            overlaid["chatResponse"]["choices"][0]["message"]
                .get("toolCalls")
                .is_none(),
            "unsanitizable toolCalls must be removed"
        );
    }

    // COHERE: same removal semantics on the flat root-level calls.
    for (payload_calls, sanitized) in [
        (
            json!([{"name": "one", "parameters": {"secret": "raw"}}]),
            None,
        ),
        (
            json!(["not-an-object"]),
            Some(vec![tool_call("call_0", "one", json!({}))]),
        ),
    ] {
        let payload = json!({
            "chatResponse": {"apiFormat": "COHERE", "text": "ok", "toolCalls": payload_calls}
        });
        let annotated = AnnotatedLlmResponse {
            message: Some(MessageContent::Text("ok".into())),
            tool_calls: sanitized,
            ..AnnotatedLlmResponse::default()
        };
        let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);
        assert!(overlaid["chatResponse"].get("toolCalls").is_none());
    }
}

#[test]
fn oci_genai_overlay_multi_part_text_handles_short_and_non_object_blocks() {
    let payload = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [
                        {"type": "TEXT", "text": "raw one"},
                        {"type": "TEXT", "text": "raw two"}
                    ]
                }
            }]
        }
    });
    // A single sanitized line for two TEXT parts: the first block takes the
    // full text (index-0 fallback), the second has no fragment and is
    // left untouched.
    let annotated = AnnotatedLlmResponse {
        message: Some(MessageContent::Text("[REDACTED]".into())),
        ..AnnotatedLlmResponse::default()
    };
    let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(payload, &annotated);
    let content = &overlaid["chatResponse"]["choices"][0]["message"]["content"];
    assert_eq!(content[0]["text"], json!("[REDACTED]"));
}

#[test]
fn oci_genai_overlay_maps_every_finish_reason_variant() {
    for (reason, generic, cohere, v2) in [
        (FinishReason::Complete, "stop", "COMPLETE", "COMPLETE"),
        (FinishReason::Length, "length", "MAX_TOKENS", "MAX_TOKENS"),
        (FinishReason::ToolUse, "tool_calls", "COMPLETE", "TOOL_CALL"),
        (
            FinishReason::ContentFilter,
            "content_filter",
            "COMPLETE",
            "COMPLETE",
        ),
        (
            FinishReason::Unknown("mystery".into()),
            "mystery",
            "mystery",
            "mystery",
        ),
    ] {
        let annotated = AnnotatedLlmResponse {
            finish_reason: Some(reason),
            ..AnnotatedLlmResponse::default()
        };

        let generic_payload = json!({"chatResponse": {"apiFormat": "GENERIC",
            "choices": [{"index": 0, "finishReason": "x", "message": {"role": "ASSISTANT", "content": []}}]}});
        let overlaid =
            BuiltinCodecName::OCIGenAI.overlay_response_payload(generic_payload, &annotated);
        assert_eq!(
            overlaid["chatResponse"]["choices"][0]["finishReason"],
            json!(generic)
        );

        let cohere_payload =
            json!({"chatResponse": {"apiFormat": "COHERE", "text": "ok", "finishReason": "x"}});
        let overlaid =
            BuiltinCodecName::OCIGenAI.overlay_response_payload(cohere_payload, &annotated);
        assert_eq!(overlaid["chatResponse"]["finishReason"], json!(cohere));

        let v2_payload = json!({"chatResponse": {"apiFormat": "COHEREV2", "finishReason": "x",
            "message": {"role": "ASSISTANT", "content": []}}});
        let overlaid = BuiltinCodecName::OCIGenAI.overlay_response_payload(v2_payload, &annotated);
        assert_eq!(overlaid["chatResponse"]["finishReason"], json!(v2));
    }
}
