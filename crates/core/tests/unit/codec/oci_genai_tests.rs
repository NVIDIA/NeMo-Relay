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
fn test_cli_data_envelope_unwrapped() {
    // Shape observed from live `oci generative-ai-inference chat-result chat`
    // output: kebab-case keys wrapped in a `data` envelope.
    let cli_output = json!({
        "data": {
            "model-id": "meta.llama-4-maverick-17b-128e-instruct-fp8",
            "model-version": "1.0.0",
            "chat-response": {
                "api-format": "GENERIC",
                "choices": [
                    {
                        "finish-reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "Hello, how are you today?"}],
                            "tool-calls": []
                        }
                    }
                ],
                "usage": {"completion-tokens": 8, "prompt-tokens": 16, "total-tokens": 24}
            }
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&cli_output).unwrap();

    assert_eq!(
        annotated.model.as_deref(),
        Some("meta.llama-4-maverick-17b-128e-instruct-fp8")
    );
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello, how are you today?".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(annotated.usage.as_ref().unwrap().prompt_tokens, Some(16));
    assert_eq!(annotated.tool_calls, None);
}

#[test]
fn test_snake_case_sdk_dict_shape() {
    // Shape observed from live `oci.util.to_dict(response.data)` on a Python
    // SDK ChatResult: snake_case keys, no envelope wrapper.
    let sdk_dict = json!({
        "model_id": "meta.llama-4-maverick-17b-128e-instruct-fp8",
        "model_version": "1.0.0",
        "chat_response": {
            "api_format": "GENERIC",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": "Hello, it's nice to meet."}],
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"completion_tokens": 8, "prompt_tokens": 16, "total_tokens": 24}
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&sdk_dict).unwrap();

    assert_eq!(
        annotated.model.as_deref(),
        Some("meta.llama-4-maverick-17b-128e-instruct-fp8")
    );
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello, it's nice to meet.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(annotated.usage.as_ref().unwrap().total_tokens, Some(24));
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
fn test_cohere_parallel_tool_calls_get_positional_ids() {
    // Shape observed live: COHERE tool calls carry no `id`, so parallel calls
    // must receive distinct synthesized ids.
    let response = json!({
        "modelId": "cohere.command-r-08-2024",
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "I will use the tool for each city.",
            "finishReason": "COMPLETE",
            "toolCalls": [
                {"name": "get_weather", "parameters": {"city": "Paris"}},
                {"name": "get_weather", "parameters": {"city": "Rome"}}
            ]
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&response).unwrap();

    let tool_calls = annotated.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].id, "call_0");
    assert_eq!(tool_calls[1].id, "call_1");
    assert_eq!(tool_calls[0].arguments, json!({"city": "Paris"}));
    assert_eq!(tool_calls[1].arguments, json!({"city": "Rome"}));
}

#[test]
fn test_usage_cached_tokens_mapped_to_cache_read() {
    // Shape observed live from OpenAI and xAI models on OCI: cache hits are
    // reported under `promptTokensDetails.cachedTokens`.
    let response = json!({
        "modelId": "xai.grok-3-mini",
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hi"}]},
                "finishReason": "stop"
            }],
            "usage": {
                "promptTokens": 13,
                "completionTokens": 8,
                "totalTokens": 607,
                "promptTokensDetails": {"cachedTokens": 3},
                "completionTokensDetails": {"reasoningTokens": 586}
            }
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&response).unwrap();

    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(13));
    assert_eq!(usage.cache_read_tokens, Some(3));
    assert_eq!(usage.cache_write_tokens, None);
}

#[test]
fn test_finish_reason_mapping() {
    for (raw, expected) in [
        ("stop", FinishReason::Complete),
        ("length", FinishReason::Length),
        ("tool_calls", FinishReason::ToolUse),
        ("COMPLETE", FinishReason::Complete),
        ("MAX_TOKENS", FinishReason::Length),
        // Live Gemini-on-OCI responses use the lowercase spelling.
        ("max_tokens", FinishReason::Length),
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
