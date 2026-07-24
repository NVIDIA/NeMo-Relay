// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the OCI Generative AI codec in the NeMo Relay core crate.

use super::*;
use serde_json::json;

use super::super::request::{ContentPart, MessageContent};
use super::super::response::{ApiSpecificResponse, FinishReason};

// -------------------------------------------------------------------
// Helpers and fixtures
// -------------------------------------------------------------------

const DEDICATED_ENDPOINT: &str = "ocid1.generativeaiendpoint.oc1.us-chicago-1.example";

/// Shape observed from a live dedicated-endpoint chat (imported NVIDIA Nemotron 3).
fn generic_chat_result() -> Json {
    json!({
        "modelId": DEDICATED_ENDPOINT,
        "modelVersion": "1.0",
        "chatResponse": {
            "apiFormat": "GENERIC",
            "timeCreated": "2026-07-23T22:59:00.000Z",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": "NEMOTRON3_OK"}]
                    },
                    "finishReason": "stop"
                }
            ],
            "usage": {"promptTokens": 18, "completionTokens": 5, "totalTokens": 23}
        }
    })
}

fn cohere_chat_result() -> Json {
    json!({
        "modelId": "cohere.command-a-03-2025",
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "Sunny and 72.",
            "finishReason": "COMPLETE",
            "usage": {"promptTokens": 12, "completionTokens": 4, "totalTokens": 16}
        }
    })
}

// ===================================================================
// Response decode tests
// ===================================================================

#[test]
fn test_generic_chat_result() {
    let annotated = OCIGenAIChatCodec
        .decode_response(&generic_chat_result())
        .unwrap();

    assert_eq!(annotated.model.as_deref(), Some(DEDICATED_ENDPOINT));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("NEMOTRON3_OK".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));

    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(18));
    assert_eq!(usage.completion_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(23));

    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificResponse::OCIGenAI {
            api_format: Some("GENERIC".into()),
            model_version: Some("1.0".into()),
        })
    );
}

#[test]
fn test_cohere_chat_result() {
    let annotated = OCIGenAIChatCodec
        .decode_response(&cohere_chat_result())
        .unwrap();

    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Sunny and 72.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(annotated.model.as_deref(), Some("cohere.command-a-03-2025"));
    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificResponse::OCIGenAI {
            api_format: Some("COHERE".into()),
            model_version: None,
        })
    );
}

#[test]
fn test_kebab_case_cli_shape() {
    let cli_shaped = json!({
        "model-id": DEDICATED_ENDPOINT,
        "chat-response": {
            "api-format": "GENERIC",
            "choices": [
                {
                    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hello"}]},
                    "finish-reason": "stop"
                }
            ],
            "usage": {"total-tokens": 9}
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&cli_shaped).unwrap();

    assert_eq!(annotated.model.as_deref(), Some(DEDICATED_ENDPOINT));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("hello".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(annotated.usage.as_ref().unwrap().total_tokens, Some(9));
}

#[test]
fn test_non_dict_response() {
    let annotated = OCIGenAIChatCodec
        .decode_response(&json!("plain text"))
        .unwrap();
    assert_eq!(annotated.extra.get("raw"), Some(&json!("plain text")));
    assert_eq!(annotated.message, None);
}

#[test]
fn test_response_tool_calls_parse_string_arguments() {
    let raw = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "message": {
                    "role": "ASSISTANT",
                    "content": [],
                    "toolCalls": [{
                        "id": "call-9",
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": "{\"city\": \"NYC\"}"
                    }]
                },
                "finishReason": "tool_calls"
            }]
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&raw).unwrap();

    assert_eq!(annotated.finish_reason, Some(FinishReason::ToolUse));
    let tool_calls = annotated.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls[0].id, "call-9");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));
    // A tool-call-only message with `"content": []` has no assistant content.
    assert_eq!(annotated.message, None);
}

#[test]
fn test_non_text_parts_preserved_as_provider_native() {
    let response = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "message": {
                    "role": "ASSISTANT",
                    "content": [
                        {"type": "TEXT", "text": "see image"},
                        {"type": "IMAGE", "imageUrl": {"url": "https://example.com/x.png"}}
                    ]
                },
                "finishReason": "stop"
            }]
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&response).unwrap();

    let Some(MessageContent::Parts(parts)) = annotated.message else {
        panic!("expected typed parts, got {:?}", annotated.message);
    };
    assert_eq!(parts.len(), 2);
    assert!(matches!(&parts[0], ContentPart::Text { text, .. } if text == "see image"));
    let ContentPart::ProviderNative {
        provider,
        kind,
        value,
    } = &parts[1]
    else {
        panic!("expected ProviderNative part, got {:?}", parts[1]);
    };
    assert_eq!(provider, "oci_genai");
    assert_eq!(kind, "IMAGE");
    assert_eq!(value["imageUrl"]["url"], json!("https://example.com/x.png"));
}

#[test]
fn test_invalid_generic_content_shape_errors() {
    for bad_content in [json!(42), json!({"type": "TEXT"}), json!([17])] {
        let response = json!({
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [{
                    "message": {"role": "ASSISTANT", "content": bad_content},
                    "finishReason": "stop"
                }]
            }
        });
        let error = OCIGenAIChatCodec.decode_response(&response).unwrap_err();
        assert!(
            matches!(error, crate::error::FlowError::InvalidArgument(_)),
            "expected InvalidArgument, got {error:?}"
        );
    }
}

#[test]
fn test_finish_reason_mapping() {
    for (raw, expected) in [
        ("stop", FinishReason::Complete),
        ("length", FinishReason::Length),
        ("tool_calls", FinishReason::ToolUse),
        ("COMPLETE", FinishReason::Complete),
        ("MAX_TOKENS", FinishReason::Length),
        ("weird", FinishReason::Unknown("weird".into())),
    ] {
        let response = json!({
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [{"message": {"role": "ASSISTANT", "content": []}, "finishReason": raw}]
            }
        });
        let annotated = OCIGenAIChatCodec.decode_response(&response).unwrap();
        assert_eq!(annotated.finish_reason, Some(expected), "for {raw}");
    }
}
