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
fn test_non_wire_renderings_are_not_decoded() {
    // The codec accepts the REST wire format only (camelCase). Alternate
    // renderings from Oracle tooling (CLI kebab-case, SDK-dict snake_case)
    // are the caller's responsibility to convert first.
    let snake_cased = json!({
        "model_id": DEDICATED_ENDPOINT,
        "chat_response": {
            "api_format": "GENERIC",
            "choices": [{
                "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hello"}]},
                "finish_reason": "stop"
            }]
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&snake_cased).unwrap();

    assert_eq!(annotated.model, None);
    assert_eq!(annotated.message, None);
    assert_eq!(annotated.finish_reason, None);

    // CLI output: kebab-case keys wrapped in a `data` envelope.
    let cli_shaped = json!({
        "data": {
            "model-id": DEDICATED_ENDPOINT,
            "chat-response": {
                "api-format": "GENERIC",
                "choices": [{
                    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hello"}]},
                    "finish-reason": "stop"
                }],
                "usage": {"total-tokens": 9}
            }
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&cli_shaped).unwrap();

    assert_eq!(annotated.model, None);
    assert_eq!(annotated.message, None);
    assert_eq!(annotated.finish_reason, None);
    assert_eq!(annotated.usage, None);
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
fn test_cohere_v2_chat_result() {
    // Shape per the OCI `CohereChatResponseV2` schema (apiFormat COHEREV2):
    // a single assistant message with typed content parts and nested-function
    // tool calls. Confirmed against the live service (us-chicago-1,
    // 2026-07-29): the wire matches this schema, including provider-supplied
    // nested-function tool-call ids, JSON-encoded string arguments, and
    // message-level toolPlan/citations.
    let response = json!({
        "modelId": "cohere.command-a-03-2025",
        "modelVersion": "2.0",
        "chatResponse": {
            "apiFormat": "COHEREV2",
            "id": "resp-v2-123",
            "message": {
                "role": "ASSISTANT",
                "content": [
                    {"type": "THINKING", "thinking": "I should call the tool."},
                    {"type": "TEXT", "text": "Checking the weather."}
                ],
                "toolCalls": [{
                    "id": "call-v2-1",
                    "type": "FUNCTION",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
                }],
                // Message-level per the OCI CohereAssistantMessageV2 schema.
                "toolPlan": "I will check the weather.",
                "citations": [{"start": 0, "end": 8, "text": "Checking"}]
            },
            "finishReason": "TOOL_CALL",
            "usage": {"promptTokens": 20, "completionTokens": 15, "totalTokens": 35}
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&response).unwrap();

    // Grounding metadata and the tool plan are not normalized but must
    // survive, namespaced under the message they came from.
    assert_eq!(
        annotated.extra.get("message"),
        Some(&json!({
            "toolPlan": "I will check the weather.",
            "citations": [{"start": 0, "end": 8, "text": "Checking"}]
        }))
    );

    assert_eq!(annotated.id.as_deref(), Some("resp-v2-123"));
    assert_eq!(annotated.model.as_deref(), Some("cohere.command-a-03-2025"));
    assert_eq!(annotated.finish_reason, Some(FinishReason::ToolUse));

    let Some(MessageContent::Parts(parts)) = &annotated.message else {
        panic!("expected typed parts, got {:?}", annotated.message);
    };
    assert_eq!(parts.len(), 2);
    assert!(
        matches!(&parts[0], ContentPart::ProviderNative { kind, .. } if kind == "THINKING"),
        "THINKING content should be preserved as a provider-native part"
    );
    assert!(matches!(&parts[1], ContentPart::Text { text, .. } if text == "Checking the weather."));

    let tool_calls = annotated.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call-v2-1");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "Paris"}));

    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.total_tokens, Some(35));
    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificResponse::OCIGenAI {
            api_format: Some("COHEREV2".into()),
            model_version: Some("2.0".into()),
        })
    );
}

#[test]
fn test_cohere_v2_text_only_flattens() {
    let response = json!({
        "chatResponse": {
            "apiFormat": "COHEREV2",
            "id": "resp-v2-456",
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": "Sunny and 72."}]
            },
            "finishReason": "COMPLETE"
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&response).unwrap();

    assert_eq!(annotated.id.as_deref(), Some("resp-v2-456"));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Sunny and 72.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn test_unmodeled_response_fields_preserved_in_extra() {
    // GENERIC: timeCreated and serviceTier are not normalized but must
    // survive; envelope-level unknown fields likewise.
    let generic = json!({
        "modelId": DEDICATED_ENDPOINT,
        "modelVersion": "1.0",
        "futureEnvelopeField": {"nested": true},
        "chatResponse": {
            "apiFormat": "GENERIC",
            "timeCreated": "2026-07-27T17:27:25.871Z",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "hi"}],
                    "refusal": null,
                    "reasoningContent": "chain of thought"
                },
                "finishReason": "stop",
                // Choice-level per the OCI ChatChoice schema.
                "serviceTier": "default",
                "groundingMetadata": {"sources": ["doc-1"]},
                "logprobs": {"tokenLogprobs": [-0.1]}
            }],
            "usage": {"totalTokens": 9}
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&generic).unwrap();

    assert_eq!(
        annotated.extra.get("timeCreated"),
        Some(&json!("2026-07-27T17:27:25.871Z"))
    );
    assert_eq!(
        annotated.extra.get("futureEnvelopeField"),
        Some(&json!({"nested": true}))
    );
    // Choice- and message-level unmodeled fields are namespaced by origin.
    assert_eq!(
        annotated.extra.get("choice"),
        Some(&json!({
            "serviceTier": "default",
            "groundingMetadata": {"sources": ["doc-1"]},
            "logprobs": {"tokenLogprobs": [-0.1]}
        }))
    );
    assert_eq!(
        annotated.extra.get("message"),
        Some(&json!({"refusal": null, "reasoningContent": "chain of thought"}))
    );
    // Modeled fields stay normalized-only.
    for modeled in [
        "apiFormat",
        "choices",
        "usage",
        "chatResponse",
        "modelId",
        "modelVersion",
    ] {
        assert!(
            !annotated.extra.contains_key(modeled),
            "{modeled} should not be duplicated into extra"
        );
    }

    // COHERE: chatHistory is not normalized and must survive.
    let cohere = json!({
        "modelId": "cohere.command-r-08-2024",
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "hi",
            "chatHistory": [{"role": "USER", "message": "hello"}],
            "finishReason": "COMPLETE"
        }
    });
    let annotated = OCIGenAIChatCodec.decode_response(&cohere).unwrap();
    assert_eq!(
        annotated.extra.get("chatHistory"),
        Some(&json!([{"role": "USER", "message": "hello"}]))
    );
}

#[test]
fn test_finish_reason_mapping() {
    for (raw, expected) in [
        ("stop", FinishReason::Complete),
        ("length", FinishReason::Length),
        ("tool_calls", FinishReason::ToolUse),
        ("content_filter", FinishReason::ContentFilter),
        ("COMPLETE", FinishReason::Complete),
        ("MAX_TOKENS", FinishReason::Length),
        // Live Gemini-on-OCI responses use the lowercase spelling.
        ("max_tokens", FinishReason::Length),
        // COHEREV2 reasons per the OCI CohereChatResponseV2 schema.
        ("TOOL_CALL", FinishReason::ToolUse),
        ("STOP_SEQUENCE", FinishReason::Complete),
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
