// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the OCI Generative AI codec in the NeMo Relay core crate.

use super::*;
use serde_json::json;

use super::super::request::{ContentPart, Message, MessageContent, ToolChoice};
use super::super::resolve::{
    detect_request_surface, detect_request_surface_with_hint, detect_response_surface,
};
use super::super::response::{ApiSpecificResponse, FinishReason};
use super::super::streaming::StreamingCodec;

// -------------------------------------------------------------------
// Helpers and fixtures
// -------------------------------------------------------------------

const DEDICATED_ENDPOINT: &str = "ocid1.generativeaiendpoint.oc1.us-chicago-1.example";

fn make_request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content,
    }
}

fn generic_chat_details() -> Json {
    json!({
        "compartmentId": "ocid1.compartment.oc1..example",
        "servingMode": {"servingType": "DEDICATED", "endpointId": DEDICATED_ENDPOINT},
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": [
                {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "You are terse."}]},
                {"role": "USER", "content": [{"type": "TEXT", "text": "My SSN is 111-22-3333."}]}
            ],
            "maxTokens": 600,
            "temperature": 0.0
        }
    })
}

fn cohere_chat_details() -> Json {
    json!({
        "compartmentId": "ocid1.compartment.oc1..example",
        "servingMode": {"servingType": "ON_DEMAND", "modelId": "cohere.command-a-03-2025"},
        "chatRequest": {
            "apiFormat": "COHERE",
            "preambleOverride": "You are terse.",
            "chatHistory": [
                {"role": "USER", "message": "hello"},
                {"role": "CHATBOT", "message": "hi"}
            ],
            "message": "What is the weather?",
            "maxTokens": 100
        }
    })
}

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

/// Envelope, request, and per-message levels all carry unmodeled fields.
fn unmodeled_generic() -> Json {
    json!({
        "compartmentId": "ocid1.compartment.oc1..example",
        "opcRetryToken": "retry-abc",
        "servingMode": {"servingType": "DEDICATED", "endpointId": DEDICATED_ENDPOINT, "futureFlag": true},
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": [
                {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "Be terse."}], "name": "sys-1"},
                {"role": "USER", "content": [{"type": "TEXT", "text": "hello"}], "unknownPerMessage": 7}
            ],
            "maxTokens": 64,
            "topK": 40,
            "seed": 7,
            "unknownFutureField": {"nested": true}
        }
    })
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Developer { .. } => "developer",
        Message::Assistant { .. } => "assistant",
        Message::Tool { .. } => "tool",
        Message::Function { .. } => "function",
        Message::ToolCallItem { .. } => "tool_call",
        Message::ToolResultItem { .. } => "tool_result",
        Message::ProviderNative { .. } => "provider_native",
    }
}

fn message_text(message: &Message) -> Option<&str> {
    let content = match message {
        Message::System { content, .. }
        | Message::User { content, .. }
        | Message::Tool { content, .. } => content,
        Message::Assistant {
            content: Some(content),
            ..
        } => content,
        _ => return None,
    };
    match content {
        MessageContent::Text(text) => Some(text.as_str()),
        MessageContent::Parts(_) => None,
    }
}

// ===================================================================
// codec_identity
// ===================================================================

#[test]
fn test_codec_identity_is_oci_genai_builtin() {
    let codec = OCIGenAIChatCodec;
    assert_eq!(
        LlmCodec::codec_identity(&codec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI),
        "OCIGenAIChatCodec must not return Opaque; PII sanitization depends on a known identity"
    );
}

#[test]
fn test_response_codec_identity_is_oci_genai_builtin() {
    let codec = OCIGenAIChatCodec;
    assert_eq!(
        <OCIGenAIChatCodec as LlmResponseCodec>::codec_identity(&codec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI),
        "OCIGenAIChatCodec response codec must not return Opaque"
    );
}

// ===================================================================
// GENERIC request decode tests
// ===================================================================

#[test]
fn test_generic_decode_envelope() {
    let annotated = OCIGenAIChatCodec
        .decode(&make_request(generic_chat_details()))
        .unwrap();

    let roles: Vec<_> = annotated.messages.iter().map(message_role).collect();
    assert_eq!(roles, vec!["system", "user"]);
    assert_eq!(
        message_text(&annotated.messages[1]),
        Some("My SSN is 111-22-3333.")
    );
    assert_eq!(annotated.model.as_deref(), Some(DEDICATED_ENDPOINT));

    let params = annotated.params.as_ref().unwrap();
    assert_eq!(params.max_tokens, Some(600));
    assert_eq!(params.temperature, Some(0.0));

    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificRequest::OCIGenAI {
            compartment_id: Some("ocid1.compartment.oc1..example".into()),
            serving_mode: Some(
                json!({"servingType": "DEDICATED", "endpointId": DEDICATED_ENDPOINT})
            ),
            api_format: Some("GENERIC".into()),
        })
    );
}

#[test]
fn test_generic_decode_bare_chat_request() {
    let bare = generic_chat_details().get("chatRequest").cloned().unwrap();
    let annotated = OCIGenAIChatCodec.decode(&make_request(bare)).unwrap();

    let roles: Vec<_> = annotated.messages.iter().map(message_role).collect();
    assert_eq!(roles, vec!["system", "user"]);
    assert_eq!(annotated.model, None);
    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificRequest::OCIGenAI {
            compartment_id: None,
            serving_mode: None,
            api_format: Some("GENERIC".into()),
        })
    );
}

#[test]
fn test_generic_decode_defaults_missing_api_format_to_generic() {
    let annotated = OCIGenAIChatCodec
        .decode(&make_request(json!({
            "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hi"}]}],
            "chatRequest": "not-an-object"
        })))
        .unwrap();
    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificRequest::OCIGenAI {
            compartment_id: None,
            serving_mode: None,
            api_format: Some("GENERIC".into()),
        })
    );
    assert_eq!(message_text(&annotated.messages[0]), Some("hi"));
}

#[test]
fn test_generic_decode_rejects_non_array_messages() {
    let error = OCIGenAIChatCodec
        .decode(&make_request(json!({
            "apiFormat": "GENERIC",
            "messages": "oops"
        })))
        .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)), "{error}");
}

#[test]
fn test_non_wire_request_renderings_are_not_decoded() {
    // The codec accepts the REST wire format only (camelCase). Alternate
    // renderings from Oracle tooling (CLI kebab-case, SDK-dict snake_case)
    // are the caller's responsibility to convert first.
    let annotated = OCIGenAIChatCodec
        .decode(&make_request(json!({
            "compartment-id": "ocid1.compartment.oc1..kebab",
            "serving-mode": {"serving-type": "ON_DEMAND", "model-id": "meta.llama-3.3-70b-instruct"},
            "chat-request": {
                "api-format": "GENERIC",
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hi"}]}],
                "max-tokens": 32
            }
        })))
        .unwrap();
    assert_eq!(annotated.model, None);
    assert!(annotated.messages.is_empty());
    assert_eq!(annotated.params, None);
}

// ===================================================================
// GENERIC request encode tests
// ===================================================================

#[test]
fn test_redaction_round_trip_preserves_envelope() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.messages[1] = Message::User {
        content: MessageContent::Text("My SSN is [REDACTED].".into()),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = encoded.content.get("chatRequest").unwrap();

    assert_eq!(
        chat_request["messages"][1],
        json!({
            "role": "USER",
            "content": [{"type": "TEXT", "text": "My SSN is [REDACTED]."}]
        })
    );
    // Envelope fields survive untouched.
    assert_eq!(
        encoded.content["compartmentId"],
        json!("ocid1.compartment.oc1..example")
    );
    assert_eq!(
        encoded.content["servingMode"],
        json!({"servingType": "DEDICATED", "endpointId": DEDICATED_ENDPOINT})
    );
    assert_eq!(chat_request["maxTokens"], json!(600));
}

#[test]
fn test_tool_calls_round_trip() {
    let payload = json!({
        "apiFormat": "GENERIC",
        "messages": [
            {
                "role": "ASSISTANT",
                "content": [],
                "toolCalls": [
                    {"id": "call-1", "type": "FUNCTION", "name": "get_weather", "arguments": "{}"}
                ]
            },
            {"role": "TOOL", "content": [{"type": "TEXT", "text": "72F"}], "toolCallId": "call-1"}
        ]
    });
    let codec = OCIGenAIChatCodec;
    let original = make_request(payload.clone());
    let annotated = codec.decode(&original).unwrap();

    match &annotated.messages[0] {
        Message::Assistant {
            tool_calls: Some(tool_calls),
            ..
        } => {
            assert_eq!(tool_calls[0].id, "call-1");
            assert_eq!(tool_calls[0].function.name, "get_weather");
            assert_eq!(tool_calls[0].function.arguments, "{}");
        }
        other => panic!("expected assistant with tool calls, got {other:?}"),
    }
    match &annotated.messages[1] {
        Message::Tool { tool_call_id, .. } => assert_eq!(tool_call_id, "call-1"),
        other => panic!("expected tool message, got {other:?}"),
    }

    // Unedited round trip is the identity.
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(encoded.content, payload);

    // Editing the assistant message forces a rebuild through the flat OCI
    // tool-call shape.
    let mut edited = annotated.clone();
    edited.messages[0] = Message::Assistant {
        content: Some(MessageContent::Text("checking".into())),
        tool_calls: match &annotated.messages[0] {
            Message::Assistant { tool_calls, .. } => tool_calls.clone(),
            _ => unreachable!(),
        },
        name: None,
    };
    let encoded = codec.encode(&edited, &original).unwrap();
    assert_eq!(
        encoded.content["messages"][0]["toolCalls"][0],
        json!({"id": "call-1", "type": "FUNCTION", "name": "get_weather", "arguments": "{}"})
    );
    assert_eq!(
        encoded.content["messages"][1]["toolCallId"],
        json!("call-1")
    );
}

// ===================================================================
// COHERE request tests
// ===================================================================

#[test]
fn test_cohere_decode() {
    let annotated = OCIGenAIChatCodec
        .decode(&make_request(cohere_chat_details()))
        .unwrap();

    let roles: Vec<_> = annotated.messages.iter().map(message_role).collect();
    assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
    assert_eq!(message_text(&annotated.messages[0]), Some("You are terse."));
    assert_eq!(
        message_text(annotated.messages.last().unwrap()),
        Some("What is the weather?")
    );
    assert_eq!(annotated.model.as_deref(), Some("cohere.command-a-03-2025"));
    assert_eq!(annotated.params.as_ref().unwrap().max_tokens, Some(100));
    assert!(matches!(
        &annotated.api_specific,
        Some(ApiSpecificRequest::OCIGenAI {
            api_format: Some(api_format),
            ..
        }) if api_format == "COHERE"
    ));
}

#[test]
fn test_cohere_round_trip() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_chat_details());
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();

    // Unedited COHERE requests round-trip to the identical payload.
    assert_eq!(encoded.content, cohere_chat_details());
}

#[test]
fn test_cohere_edit_rebuilds_modeled_fields() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    let last = annotated.messages.len() - 1;
    annotated.messages[last] = Message::User {
        content: MessageContent::Text("What is the weather in [REDACTED]?".into()),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = encoded.content.get("chatRequest").unwrap();
    assert_eq!(
        chat_request["message"],
        json!("What is the weather in [REDACTED]?")
    );
    assert_eq!(chat_request["preambleOverride"], json!("You are terse."));
    assert_eq!(
        chat_request["chatHistory"],
        json!([
            {"role": "USER", "message": "hello"},
            {"role": "CHATBOT", "message": "hi"}
        ])
    );
    assert_eq!(
        encoded.content["servingMode"],
        json!({"servingType": "ON_DEMAND", "modelId": "cohere.command-a-03-2025"})
    );
    assert_eq!(chat_request["maxTokens"], json!(100));
}

#[test]
fn test_cohere_stop_sequences_map_to_stop() {
    let mut payload = cohere_chat_details();
    payload["chatRequest"]["stopSequences"] = json!(["END"]);
    let annotated = OCIGenAIChatCodec.decode(&make_request(payload)).unwrap();
    assert_eq!(
        annotated.params.as_ref().unwrap().stop,
        Some(vec!["END".to_string()])
    );
}

// ===================================================================
// Identity invariant: encode(decode(original), original) == original
// ===================================================================

#[test]
fn test_generic_identity() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(unmodeled_generic());
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();

    assert_eq!(encoded.content, unmodeled_generic());
}

#[test]
fn test_cohere_identity() {
    let mut payload = cohere_chat_details();
    payload["chatRequest"]["isForceSingleStep"] = json!(true);
    let codec = OCIGenAIChatCodec;
    let original = make_request(payload.clone());
    let annotated = codec.decode(&original).unwrap();
    let encoded = codec.encode(&annotated, &original).unwrap();

    assert_eq!(encoded.content, payload);
}

#[test]
fn test_edit_preserves_unmodeled_fields_on_untouched_messages() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(unmodeled_generic());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.messages[1] = Message::User {
        content: MessageContent::Text("redacted".into()),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = encoded.content.get("chatRequest").unwrap();

    // Untouched system message keeps its unmodeled per-message field.
    assert_eq!(
        chat_request["messages"][0],
        unmodeled_generic()["chatRequest"]["messages"][0]
    );
    // Edited message carries the redaction.
    assert_eq!(
        chat_request["messages"][1]["content"],
        json!([{"type": "TEXT", "text": "redacted"}])
    );
    // Unmodeled request-level fields survive.
    assert_eq!(chat_request["topK"], json!(40));
    assert_eq!(chat_request["seed"], json!(7));
    assert_eq!(chat_request["unknownFutureField"], json!({"nested": true}));
    assert_eq!(encoded.content["opcRetryToken"], json!("retry-abc"));
}

#[test]
fn test_param_edit_only_touches_changed_param() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(unmodeled_generic());
    let mut annotated = codec.decode(&original).unwrap();

    let mut params = annotated.params.clone().unwrap_or_default();
    params.max_tokens = Some(128);
    annotated.params = Some(params);

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = encoded.content.get("chatRequest").unwrap();

    assert_eq!(chat_request["maxTokens"], json!(128));
    assert_eq!(
        chat_request["messages"],
        unmodeled_generic()["chatRequest"]["messages"]
    );
}

#[test]
fn test_tool_choice_survives_as_provider_native() {
    let payload = json!({
        "apiFormat": "GENERIC",
        "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hi"}]}],
        "tools": [{"type": "FUNCTION", "name": "get_weather", "parameters": {"type": "object"}}],
        "toolChoice": {"type": "auto"}
    });
    let codec = OCIGenAIChatCodec;
    let original = make_request(payload.clone());
    let annotated = codec.decode(&original).unwrap();

    assert!(matches!(
        &annotated.tool_choice,
        Some(ToolChoice::ProviderNative(native)) if native.provider == "oci_genai"
    ));
    assert_eq!(annotated.tools.as_ref().map(Vec::len), Some(1));

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(encoded.content, payload);
}

#[test]
fn test_model_edit_is_rejected() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();
    annotated.model = Some("other-model".into());

    let error = codec.encode(&annotated, &original).unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)), "{error}");
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

// ===================================================================
// Surface detection tests
// ===================================================================

#[test]
fn test_detect_request_envelope_and_api_format() {
    assert_eq!(
        detect_request_surface(&generic_chat_details()),
        Some(ProviderSurface::OCIGenAI)
    );
    assert_eq!(
        detect_request_surface(&cohere_chat_details()),
        Some(ProviderSurface::OCIGenAI)
    );
    // A bare chatRequest carries the apiFormat discriminator.
    assert_eq!(
        detect_request_surface(&generic_chat_details()["chatRequest"]),
        Some(ProviderSurface::OCIGenAI)
    );
}

#[test]
fn test_detect_request_hint_resolves_bare_chat_request() {
    let bare = json!({"chatRequest": {"messages": []}});
    assert_eq!(detect_request_surface(&bare), None);
    assert_eq!(
        detect_request_surface_with_hint(&bare, Some("oci")),
        Some(ProviderSurface::OCIGenAI)
    );
    assert_eq!(
        detect_request_surface_with_hint(&bare, Some("oci.genai")),
        Some(ProviderSurface::OCIGenAI)
    );
    assert_eq!(detect_request_surface_with_hint(&bare, Some("other")), None);
}

#[test]
fn test_detect_request_does_not_shadow_other_surfaces() {
    assert_eq!(
        detect_request_surface(&json!({"messages": []})),
        Some(ProviderSurface::OpenAIChat)
    );
    assert_eq!(
        detect_request_surface(&json!({"system": "x", "messages": []})),
        Some(ProviderSurface::AnthropicMessages)
    );
    assert_eq!(
        detect_request_surface(&json!({"input": []})),
        Some(ProviderSurface::OpenAIResponses)
    );
}

#[test]
fn test_detect_response_chat_result() {
    assert_eq!(
        detect_response_surface(&generic_chat_result()),
        Some(ProviderSurface::OCIGenAI)
    );
    assert_eq!(
        detect_response_surface(&cohere_chat_result()),
        Some(ProviderSurface::OCIGenAI)
    );
    // A bare COHERE chat response has no `choices`, so it stays unambiguous.
    assert_eq!(
        detect_response_surface(&cohere_chat_result()["chatResponse"]),
        Some(ProviderSurface::OCIGenAI)
    );
}

#[test]
fn test_detect_response_bare_generic_is_ambiguous_with_openai_chat() {
    // A bare GENERIC chat response carries both `apiFormat` and `choices`;
    // strict response detection refuses ambiguous shapes.
    assert_eq!(
        detect_response_surface(&generic_chat_result()["chatResponse"]),
        None
    );
}

#[test]
fn test_detect_response_does_not_shadow_other_surfaces() {
    assert_eq!(
        detect_response_surface(&json!({"choices": []})),
        Some(ProviderSurface::OpenAIChat)
    );
    assert_eq!(
        detect_response_surface(&json!({"type": "message", "content": []})),
        Some(ProviderSurface::AnthropicMessages)
    );
    assert_eq!(
        detect_response_surface(&json!({"output": []})),
        Some(ProviderSurface::OpenAIResponses)
    );
}

// ===================================================================
// Streaming codec tests
// ===================================================================

#[test]
fn oci_streaming_codec_assembles_generic_text_response() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "modelId": DEDICATED_ENDPOINT,
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "Hello, "}]}
            }]
        }
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [{"type": "TEXT", "text": "world."}]}
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": []},
        "finishReason": "stop",
        "usage": {"promptTokens": 12, "completionTokens": 3, "totalTokens": 15}
    }))
    .unwrap();

    let assembled = finalizer();
    // Wire-compatible with a ChatResult — feed it back through the decoder.
    let annotated = OCIGenAIChatCodec.decode_response(&assembled).unwrap();
    assert_eq!(annotated.model.as_deref(), Some(DEDICATED_ENDPOINT));
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Hello, world.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    let usage = annotated.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(12));
    assert_eq!(usage.completion_tokens, Some(3));
    assert_eq!(usage.total_tokens, Some(15));
}

#[test]
fn oci_streaming_codec_accumulates_generic_tool_call_arguments() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "apiFormat": "GENERIC",
        "index": 0,
        "message": {
            "role": "ASSISTANT",
            "content": [],
            "toolCalls": [{"id": "call-1", "type": "FUNCTION", "name": "get_weather", "arguments": "{\"city\":"}]
        }
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [], "toolCalls": [{"arguments": " \"NYC\"}"}]},
        "finishReason": "tool_calls"
    }))
    .unwrap();

    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    assert_eq!(annotated.finish_reason, Some(FinishReason::ToolUse));
    let tool_calls = annotated.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls[0].id, "call-1");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));
}

#[test]
fn oci_streaming_codec_assembles_cohere_text_response() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({"apiFormat": "COHERE", "text": "Sunny"})).unwrap();
    collector(json!({"apiFormat": "COHERE", "text": " and 72."})).unwrap();
    collector(json!({
        "apiFormat": "COHERE",
        "text": "",
        "finishReason": "COMPLETE",
        "usage": {"promptTokens": 8, "completionTokens": 4, "totalTokens": 12}
    }))
    .unwrap();

    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Sunny and 72.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
    assert_eq!(annotated.usage.as_ref().unwrap().total_tokens, Some(12));
    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificResponse::OCIGenAI {
            api_format: Some("COHERE".into()),
            model_version: None,
        })
    );
}

// ===================================================================
// Encode edge cases surfaced in review
// ===================================================================

#[test]
fn test_cohere_encode_rejects_multimodal_content() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    let last = annotated.messages.len() - 1;
    annotated.messages[last] = Message::User {
        content: MessageContent::Parts(vec![ContentPart::Text {
            text: "described image".into(),
            extra: Default::default(),
        }]),
        name: None,
    };

    let err = codec.encode(&annotated, &original).unwrap_err();
    assert!(matches!(err, FlowError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn test_cohere_encode_requires_trailing_user_message() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.messages.push(Message::Assistant {
        content: Some(MessageContent::Text("appended".into())),
        tool_calls: None,
        name: None,
    });

    let err = codec.encode(&annotated, &original).unwrap_err();
    assert!(matches!(err, FlowError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn test_cohere_encode_rejects_normalized_tool_message() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    let last = annotated.messages.len() - 1;
    annotated.messages.insert(
        last,
        Message::Tool {
            content: MessageContent::Text("72F".into()),
            tool_call_id: "call-1".into(),
        },
    );

    let err = codec.encode(&annotated, &original).unwrap_err();
    assert!(matches!(err, FlowError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn test_tool_call_only_assistant_reencodes_without_null_content_or_empty_id() {
    use super::super::request::{FunctionCall, ToolCall};

    let codec = OCIGenAIChatCodec;
    let mut payload = generic_chat_details();
    payload["chatRequest"]["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "role": "ASSISTANT",
            "content": [],
            "toolCalls": [{"type": "FUNCTION", "name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}]
        }));
    let original = make_request(payload);
    let mut annotated = codec.decode(&original).unwrap();

    let last = annotated.messages.len() - 1;
    let Message::Assistant { tool_calls, .. } = &annotated.messages[last] else {
        panic!("expected an assistant message");
    };
    assert_eq!(tool_calls.as_ref().unwrap()[0].id, "");
    annotated.messages[last] = Message::Assistant {
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: String::new(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: "{\"city\": \"[REDACTED]\"}".into(),
            },
        }]),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    let message = &encoded.content["chatRequest"]["messages"][2];
    assert_eq!(message["content"], json!([]));
    let tool_call = &message["toolCalls"][0];
    assert!(
        tool_call.get("id").is_none(),
        "empty tool-call id must be omitted, got {tool_call}"
    );
    assert_eq!(tool_call["arguments"], json!("{\"city\": \"[REDACTED]\"}"));
}

#[test]
fn test_api_format_edit_is_rejected() {
    // Merge-not-replace encoding patches the raw payload in place, so a format
    // switch could never remove the previous format's modeled fields; the
    // api_format annotation is therefore read-only.
    let codec = OCIGenAIChatCodec;
    for (payload, new_format) in [
        (generic_chat_details(), "COHERE"),
        (generic_chat_details(), "COHEREV2"),
        (cohere_chat_details(), "GENERIC"),
    ] {
        let original = make_request(payload);
        let mut annotated = codec.decode(&original).unwrap();
        let Some(ApiSpecificRequest::OCIGenAI { api_format, .. }) = &mut annotated.api_specific
        else {
            panic!("expected an OCI api_specific annotation");
        };
        *api_format = Some(new_format.into());

        let err = codec.encode(&annotated, &original).unwrap_err();
        assert!(
            matches!(&err, FlowError::InvalidArgument(message)
                if message.contains("api_format cannot be edited")),
            "{new_format}: {err:?}"
        );
    }
}

#[test]
fn oci_streaming_codec_tracks_parallel_tool_calls_by_id() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    // Two parallel calls whose fragments arrive in separate events, each at
    // event-local position 0.
    collector(json!({
        "apiFormat": "GENERIC",
        "index": 0,
        "message": {
            "role": "ASSISTANT",
            "content": [],
            "toolCalls": [{"id": "call-a", "type": "FUNCTION", "name": "get_weather", "arguments": "{\"city\":"}]
        }
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [], "toolCalls": [{"id": "call-b", "type": "FUNCTION", "name": "get_weather", "arguments": "{\"city\":"}]}
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [], "toolCalls": [{"id": "call-a", "arguments": " \"Paris\"}"}]}
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [], "toolCalls": [{"id": "call-b", "arguments": " \"Rome\"}"}]},
        "finishReason": "tool_calls"
    }))
    .unwrap();

    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    let tool_calls = annotated.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].id, "call-a");
    assert_eq!(tool_calls[0].arguments, json!({"city": "Paris"}));
    assert_eq!(tool_calls[1].id, "call-b");
    assert_eq!(tool_calls[1].arguments, json!({"city": "Rome"}));
}

#[test]
fn oci_streaming_codec_tool_call_only_stream_decodes_without_message() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "apiFormat": "GENERIC",
        "index": 0,
        "message": {
            "role": "ASSISTANT",
            "content": [],
            "toolCalls": [{"id": "call-1", "type": "FUNCTION", "name": "get_weather", "arguments": "{}"}]
        },
        "finishReason": "tool_calls"
    }))
    .unwrap();

    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    assert_eq!(annotated.message, None);
    assert_eq!(annotated.tool_calls.as_ref().unwrap().len(), 1);
}

#[test]
fn test_cohere_edit_removes_stale_preamble_override() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    // Drop the leading system message (the decoded preambleOverride).
    assert!(matches!(
        annotated.messages.first(),
        Some(Message::System { .. })
    ));
    annotated.messages.remove(0);

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = encoded.content["chatRequest"].as_object().unwrap();
    assert!(
        !chat_request.contains_key("preambleOverride"),
        "a removed system message must not leave the original preamble on the wire"
    );
}

fn cohere_v2_chat_details() -> Json {
    json!({
        "compartmentId": "ocid1.compartment.oc1..example",
        "servingMode": {"servingType": "ON_DEMAND", "modelId": "cohere.command-a-03-2025"},
        "chatRequest": {
            "apiFormat": "COHEREV2",
            "messages": [
                {"role": "USER", "content": [{"type": "TEXT", "text": "What is the weather?"}]}
            ],
            "maxTokens": 100,
            "stopSequences": ["END"],
            "citationOptions": {"mode": "OFF"}
        }
    })
}

#[test]
fn test_cohere_v2_request_decode_and_identity() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_v2_chat_details());
    let annotated = codec.decode(&original).unwrap();

    assert_eq!(annotated.messages.len(), 1);
    assert!(matches!(&annotated.messages[0], Message::User { .. }));
    let params = annotated.params.as_ref().unwrap();
    assert_eq!(params.stop, Some(vec!["END".to_string()]));
    assert_eq!(params.max_tokens, Some(100));
    // V2-only request fields ride along in extra rather than being dropped.
    assert_eq!(annotated.extra["citationOptions"], json!({"mode": "OFF"}));

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(encoded.content, cohere_v2_chat_details());
}

#[test]
fn test_cohere_v2_request_edit_patches_stop_sequences() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_v2_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.params.as_mut().unwrap().stop = Some(vec!["HALT".to_string()]);
    annotated.messages[0] = Message::User {
        content: MessageContent::Text("What is the weather in [REDACTED]?".into()),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = &encoded.content["chatRequest"];
    assert_eq!(chat_request["stopSequences"], json!(["HALT"]));
    assert!(
        chat_request.get("stop").is_none(),
        "COHEREV2 edits must patch stopSequences, not the GENERIC stop key"
    );
    assert_eq!(
        chat_request["messages"][0]["content"],
        json!([{"type": "TEXT", "text": "What is the weather in [REDACTED]?"}])
    );
    assert_eq!(chat_request["citationOptions"], json!({"mode": "OFF"}));
}

#[test]
fn oci_streaming_codec_does_not_double_cohere_text_on_full_terminal_event() {
    // Live-captured shape: the service's terminal COHERE event repeats the
    // complete response text alongside chatHistory and finishReason.
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    for fragment in ["Rome", " is", " the", " capital", " of", " Italy", "."] {
        collector(json!({"apiFormat": "COHERE", "text": fragment})).unwrap();
    }
    collector(json!({
        "apiFormat": "COHERE",
        "text": "Rome is the capital of Italy.",
        "chatHistory": [
            {"role": "USER", "message": "What is the capital of Italy?"},
            {"role": "CHATBOT", "message": "Rome is the capital of Italy."}
        ],
        "finishReason": "COMPLETE"
    }))
    .unwrap();

    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Rome is the capital of Italy.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn oci_streaming_codec_finalizes_cohere_v2_root_message() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "apiFormat": "COHEREV2",
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "Checking "}]}
    }))
    .unwrap();
    collector(json!({
        "message": {"content": [{"type": "TEXT", "text": "the weather."}]}
    }))
    .unwrap();
    collector(json!({
        "message": {
            "content": [],
            "toolCalls": [{
                "id": "call-1",
                "type": "FUNCTION",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
            }]
        },
        "finishReason": "TOOL_CALL",
        "usage": {"promptTokens": 20, "completionTokens": 9, "totalTokens": 29}
    }))
    .unwrap();

    let assembled = finalizer();
    let chat_response = &assembled["chatResponse"];
    assert!(
        chat_response.get("message").is_some() && chat_response.get("choices").is_none(),
        "COHEREV2 streams must finalize to the root-message shape, got {chat_response}"
    );

    let annotated = OCIGenAIChatCodec.decode_response(&assembled).unwrap();
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("Checking the weather.".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::ToolUse));
    let tool_calls = annotated.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls[0].id, "call-1");
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "Paris"}));
    assert_eq!(annotated.usage.as_ref().unwrap().total_tokens, Some(29));
    assert_eq!(
        annotated.api_specific,
        Some(ApiSpecificResponse::OCIGenAI {
            api_format: Some("COHEREV2".into()),
            model_version: None,
        })
    );
}

#[test]
fn oci_streaming_codec_preserves_non_text_cohere_v2_parts_in_order() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "apiFormat": "COHEREV2",
        "message": {"role": "ASSISTANT", "content": [
            {"type": "THINKING", "thinking": "internal reasoning"}
        ]}
    }))
    .unwrap();
    collector(json!({
        "message": {"content": [{"type": "TEXT", "text": "The answer "}]}
    }))
    .unwrap();
    collector(json!({
        "message": {"content": [{"type": "TEXT", "text": "is 42."}]},
        "finishReason": "COMPLETE"
    }))
    .unwrap();

    let assembled = finalizer();
    // The finalized wire shape keeps the typed parts in arrival order, with
    // consecutive TEXT fragments merged into one part.
    assert_eq!(
        assembled["chatResponse"]["message"]["content"],
        json!([
            {"type": "THINKING", "thinking": "internal reasoning"},
            {"type": "TEXT", "text": "The answer is 42."}
        ])
    );

    let annotated = OCIGenAIChatCodec.decode_response(&assembled).unwrap();
    let Some(MessageContent::Parts(parts)) = &annotated.message else {
        panic!("expected typed parts, got {:?}", annotated.message);
    };
    assert!(matches!(
        &parts[0],
        ContentPart::ProviderNative { kind, .. } if kind == "THINKING"
    ));
    assert!(matches!(
        &parts[1],
        ContentPart::Text { text, .. } if text == "The answer is 42."
    ));
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn oci_streaming_codec_preserves_non_text_generic_parts_in_order() {
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();

    collector(json!({
        "apiFormat": "GENERIC",
        "index": 0,
        "message": {"role": "ASSISTANT", "content": [
            {"type": "THINKING", "thinking": "internal reasoning"}
        ]}
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [{"type": "TEXT", "text": "The answer "}]}
    }))
    .unwrap();
    collector(json!({
        "index": 0,
        "message": {"content": [{"type": "TEXT", "text": "is 42."}]},
        "finishReason": "stop"
    }))
    .unwrap();

    let assembled = finalizer();
    assert_eq!(
        assembled["chatResponse"]["choices"][0]["message"]["content"],
        json!([
            {"type": "THINKING", "thinking": "internal reasoning"},
            {"type": "TEXT", "text": "The answer is 42."}
        ])
    );

    let annotated = OCIGenAIChatCodec.decode_response(&assembled).unwrap();
    let Some(MessageContent::Parts(parts)) = &annotated.message else {
        panic!("expected typed parts, got {:?}", annotated.message);
    };
    assert!(matches!(
        &parts[0],
        ContentPart::ProviderNative { kind, .. } if kind == "THINKING"
    ));
    assert!(matches!(
        &parts[1],
        ContentPart::Text { text, .. } if text == "The answer is 42."
    ));
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
}

#[test]
fn test_non_oci_api_specific_is_rejected() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.api_specific = Some(ApiSpecificRequest::Custom {
        api_name: "not-oci".into(),
        data: Json::Null,
    });

    let err = codec.encode(&annotated, &original).unwrap_err();
    assert!(
        matches!(&err, FlowError::InvalidArgument(message)
            if message.contains("does not match OCI GenAI")),
        "{err:?}"
    );
}

#[test]
fn test_envelope_edit_on_bare_chat_request_is_rejected() {
    let codec = OCIGenAIChatCodec;
    let bare = generic_chat_details().get("chatRequest").cloned().unwrap();
    let original = make_request(bare);
    let mut annotated = codec.decode(&original).unwrap();

    let Some(ApiSpecificRequest::OCIGenAI { compartment_id, .. }) = &mut annotated.api_specific
    else {
        panic!("expected an OCI api_specific annotation");
    };
    *compartment_id = Some("ocid1.compartment.oc1..edited".into());

    let err = codec.encode(&annotated, &original).unwrap_err();
    assert!(
        matches!(&err, FlowError::InvalidArgument(message)
            if message.contains("require a ChatDetails envelope")),
        "{err:?}"
    );
}

#[test]
fn test_cohere_v2_assistant_edit_reencodes_nested_tool_calls() {
    use super::super::request::{FunctionCall, ToolCall};

    let codec = OCIGenAIChatCodec;
    let mut payload = cohere_v2_chat_details();
    payload["chatRequest"]["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "role": "ASSISTANT",
            "content": [],
            "toolCalls": [{
                "id": "call-1",
                "type": "FUNCTION",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}
            }]
        }));
    let original = make_request(payload);
    let mut annotated = codec.decode(&original).unwrap();

    let last = annotated.messages.len() - 1;
    annotated.messages[last] = Message::Assistant {
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: "call-1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: "{\"city\": \"[REDACTED]\"}".into(),
            },
        }]),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    let tool_call = &encoded.content["chatRequest"]["messages"][1]["toolCalls"][0];
    assert_eq!(
        tool_call["function"],
        json!({"name": "get_weather", "arguments": "{\"city\": \"[REDACTED]\"}"}),
        "COHEREV2 edits must re-encode nested-function tool calls, got {tool_call}"
    );
    assert!(
        tool_call.get("name").is_none() && tool_call.get("arguments").is_none(),
        "flat GENERIC keys must not appear on a COHEREV2 tool call, got {tool_call}"
    );
}

#[test]
fn test_non_string_text_part_survives_as_provider_native() {
    let codec = OCIGenAIChatCodec;
    let mut payload = generic_chat_details();
    payload["chatRequest"]["messages"][1]["content"] = json!([
        {"type": "TEXT", "text": {"unexpected": "object"}}
    ]);
    let original = make_request(payload.clone());
    let annotated = codec.decode(&original).unwrap();

    let Some(Message::User {
        content: MessageContent::Parts(parts),
        ..
    }) = annotated.messages.get(1)
    else {
        panic!("expected a user message with typed parts");
    };
    assert!(
        matches!(&parts[0], ContentPart::ProviderNative { .. }),
        "non-string TEXT value must survive as provider-native, got {:?}",
        parts[0]
    );

    // Identity: the raw value re-encodes untouched.
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(encoded.content, payload);
}

// ===================================================================
// Coverage: edit paths and error paths not exercised elsewhere
// ===================================================================

#[test]
fn test_tools_and_tool_choice_edits_reencode() {
    use super::super::request::{FunctionDefinition, ProviderNativeComponent, ToolDefinition};

    let codec = OCIGenAIChatCodec;
    let mut payload = generic_chat_details();
    payload["chatRequest"]["tools"] = json!([
        {"type": "FUNCTION", "name": "old_tool", "parameters": {"type": "object"}}
    ]);
    payload["chatRequest"]["toolChoice"] = json!({"type": "AUTO"});
    let original = make_request(payload);
    let mut annotated = codec.decode(&original).unwrap();

    annotated.tools = Some(vec![ToolDefinition::Function {
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            parameters: Some(json!({"type": "object", "properties": {}})),
            strict: None,
            extra: Default::default(),
        },
        extra: Default::default(),
    }]);
    annotated.tool_choice = Some(ToolChoice::ProviderNative(ProviderNativeComponent {
        provider: "oci_genai".into(),
        kind: "tool_choice".into(),
        value: json!({"type": "REQUIRED"}),
    }));

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = &encoded.content["chatRequest"];
    assert_eq!(
        chat_request["tools"],
        json!([{
            "type": "FUNCTION",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}}
        }])
    );
    assert_eq!(chat_request["toolChoice"], json!({"type": "REQUIRED"}));
}

#[test]
fn test_dropping_tools_removes_the_wire_key() {
    let codec = OCIGenAIChatCodec;
    let mut payload = generic_chat_details();
    payload["chatRequest"]["tools"] = json!([
        {"type": "FUNCTION", "name": "old_tool", "parameters": {"type": "object"}}
    ]);
    let original = make_request(payload);
    let mut annotated = codec.decode(&original).unwrap();

    annotated.tools = None;

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert!(
        encoded.content["chatRequest"].get("tools").is_none(),
        "dropping the tools annotation must remove the wire key"
    );
}

#[test]
fn test_non_native_tool_choice_edit_is_rejected() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.tool_choice = Some(ToolChoice::Auto);

    let err = codec.encode(&annotated, &original).unwrap_err();
    assert!(matches!(err, FlowError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn test_parts_content_edit_reencodes_typed_parts() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.messages[1] = Message::User {
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "look at this".into(),
                extra: Default::default(),
            },
            ContentPart::ProviderNative {
                provider: "oci_genai".into(),
                kind: "IMAGE".into(),
                value: json!({"type": "IMAGE", "imageUrl": {"url": "data:image/png;base64,AA"}}),
            },
        ]),
        name: None,
    };

    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content["chatRequest"]["messages"][1]["content"],
        json!([
            {"type": "TEXT", "text": "look at this"},
            {"type": "IMAGE", "imageUrl": {"url": "data:image/png;base64,AA"}}
        ])
    );
}

#[test]
fn test_top_p_edit_patches_only_top_p() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    let params = annotated.params.as_mut().unwrap();
    params.top_p = Some(0.5);

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = &encoded.content["chatRequest"];
    assert_eq!(chat_request["topP"], json!(0.5));
    assert_eq!(chat_request["temperature"], json!(0.0));
    assert_eq!(chat_request["maxTokens"], json!(600));
}

#[test]
fn test_clearing_params_removes_provider_fields() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    let params = annotated.params.as_mut().unwrap();
    params.temperature = None;

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = &encoded.content["chatRequest"];
    assert!(
        chat_request.get("temperature").is_none(),
        "clearing temperature must remove the provider field"
    );
    // Untouched params keep their raw values.
    assert_eq!(chat_request["maxTokens"], json!(600));
}

#[test]
fn test_clearing_all_params_removes_all_provider_fields() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(generic_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.params = None;

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = &encoded.content["chatRequest"];
    assert!(chat_request.get("temperature").is_none());
    assert!(chat_request.get("maxTokens").is_none());
    // Non-param request fields survive a full params clear.
    assert_eq!(chat_request["apiFormat"], json!("GENERIC"));
    assert!(chat_request.get("messages").is_some());
}

#[test]
fn test_cohere_v2_clearing_stop_removes_stop_sequences() {
    let codec = OCIGenAIChatCodec;
    let original = make_request(cohere_v2_chat_details());
    let mut annotated = codec.decode(&original).unwrap();

    annotated.params.as_mut().unwrap().stop = None;

    let encoded = codec.encode(&annotated, &original).unwrap();
    let chat_request = &encoded.content["chatRequest"];
    assert!(
        chat_request.get("stopSequences").is_none(),
        "clearing stop must remove the COHERE stopSequences field"
    );
    assert_eq!(chat_request["maxTokens"], json!(100));
    assert_eq!(chat_request["citationOptions"], json!({"mode": "OFF"}));
}

#[test]
fn test_decode_error_paths() {
    let codec = OCIGenAIChatCodec;

    // Request content that is not an object.
    assert!(codec.decode(&make_request(json!("nope"))).is_err());

    // A stop list that is not a string array.
    let mut payload = generic_chat_details();
    payload["chatRequest"]["stop"] = json!("HALT");
    assert!(codec.decode(&make_request(payload)).is_err());

    // A toolCalls entry that is not an object.
    let mut payload = generic_chat_details();
    payload["chatRequest"]["messages"] = json!([
        {"role": "ASSISTANT", "content": [], "toolCalls": ["nope"]}
    ]);
    assert!(codec.decode(&make_request(payload)).is_err());

    // A GENERIC message that is not an object.
    let mut payload = generic_chat_details();
    payload["chatRequest"]["messages"] = json!(["nope"]);
    assert!(codec.decode(&make_request(payload)).is_err());

    // A COHERE chatHistory turn that is not an object.
    let mut payload = cohere_chat_details();
    payload["chatRequest"]["chatHistory"] = json!(["nope"]);
    assert!(codec.decode(&make_request(payload)).is_err());
}

#[test]
fn test_unknown_generic_role_survives_as_provider_native() {
    let codec = OCIGenAIChatCodec;
    let mut payload = generic_chat_details();
    payload["chatRequest"]["messages"] = json!([
        {"role": "MODERATOR", "content": [{"type": "TEXT", "text": "hi"}]}
    ]);
    let original = make_request(payload.clone());
    let annotated = codec.decode(&original).unwrap();

    assert!(matches!(
        &annotated.messages[0],
        Message::ProviderNative { provider, .. } if provider == "oci_genai"
    ));
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(encoded.content, payload);
}

#[test]
fn oci_streaming_codec_infers_api_format_from_event_shape() {
    // GENERIC inferred from a bare choice delta with no apiFormat anywhere.
    let codec = OCIGenAIStreamingCodec::default();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();
    collector(json!({
        "index": 0,
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hi"}]},
        "finishReason": "stop"
    }))
    .unwrap();
    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    assert_eq!(annotated.message, Some(MessageContent::Text("hi".into())));

    // COHERE inferred from a bare text fragment with no apiFormat anywhere.
    let codec = OCIGenAIStreamingCodec::new();
    let mut collector = codec.collector();
    let finalizer = codec.finalizer();
    collector(json!({"text": "hello"})).unwrap();
    collector(json!({"text": "!"})).unwrap();
    // The live terminal event repeats the complete text.
    collector(json!({"text": "hello!", "finishReason": "COMPLETE"})).unwrap();
    let annotated = OCIGenAIChatCodec.decode_response(&finalizer()).unwrap();
    assert_eq!(
        annotated.message,
        Some(MessageContent::Text("hello!".into()))
    );
    assert_eq!(annotated.finish_reason, Some(FinishReason::Complete));
}
