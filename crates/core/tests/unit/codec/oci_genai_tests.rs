// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the OCI Generative AI codec in the NeMo Relay core crate.

use super::*;
use serde_json::json;

use super::super::request::{Message, MessageContent, ToolChoice};
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
fn test_generic_decode_kebab_case_keys() {
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
    assert_eq!(
        annotated.model.as_deref(),
        Some("meta.llama-3.3-70b-instruct")
    );
    assert_eq!(annotated.params.as_ref().unwrap().max_tokens, Some(32));
    assert_eq!(message_text(&annotated.messages[0]), Some("hi"));
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
