// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the Amazon Bedrock Converse codec.

use super::*;
use serde_json::json;

use super::super::streaming::StreamingCodec;

fn request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::from_iter([(
            "x-test-header".into(),
            Json::String("preserved".into()),
        )]),
        content,
    }
}

fn full_request() -> LlmRequest {
    request(json!({
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "system": [
            {"text": "Follow policy."},
            {"cachePoint": {"type": "default"}}
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Weather in Montréal?"}
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "call-1",
                        "name": "weather",
                        "input": {"city": "Montréal"}
                    }
                }]
            },
            {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": "call-1",
                        "content": [{"json": {"temperature": 20}}],
                        "status": "success"
                    }
                }]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.2,
            "topP": 0.9,
            "stopSequences": ["END"]
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "weather",
                        "description": "Read the weather",
                        "strict": true,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}}
                            }
                        }
                    }
                },
                {"cachePoint": {"type": "default"}}
            ],
            "toolChoice": {"auto": {}}
        },
        "additionalModelRequestFields": {"thinking": {"type": "enabled"}},
        "requestMetadata": {"tenant": "example"}
    }))
}

#[test]
fn codec_identity_is_bedrock_converse_builtin() {
    let codec = BedrockConverseCodec;
    assert_eq!(
        LlmCodec::codec_identity(&codec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::BedrockConverse)
    );
}

#[test]
fn response_codec_identity_is_bedrock_converse_builtin() {
    let codec = BedrockConverseCodec;
    assert_eq!(
        <BedrockConverseCodec as LlmResponseCodec>::codec_identity(&codec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::BedrockConverse)
    );
}

#[test]
fn detector_accepts_the_strong_model_id_shape_and_explicit_hints() {
    let body = full_request().content;
    let obj = body.as_object().unwrap();
    assert!((PROVIDER_SURFACE.detect_request)(obj, None));
    assert!((PROVIDER_SURFACE.detect_request)(
        obj,
        Some("openai.chat_completions")
    ));
    assert!((PROVIDER_SURFACE.detect_request)(
        obj,
        Some("aws.bedrock.converse")
    ));
    assert!((PROVIDER_SURFACE.detect_request)(
        obj,
        Some("bedrock.converse")
    ));
}

#[test]
#[allow(clippy::cognitive_complexity)] // Verifies one decoded envelope across every mapped field.
fn decode_maps_messages_tools_params_and_provider_extras() {
    let decoded = BedrockConverseCodec.decode(&full_request()).unwrap();
    assert_eq!(
        decoded.model.as_deref(),
        Some("anthropic.claude-3-5-sonnet-20241022-v2:0")
    );
    assert_eq!(decoded.messages.len(), 3);
    assert!(matches!(decoded.messages[0], Message::User { .. }));
    assert!(matches!(decoded.messages[1], Message::Assistant { .. }));
    assert!(matches!(decoded.messages[2], Message::User { .. }));
    let Some(MessageContent::Parts(instructions)) = decoded.instructions.as_ref() else {
        panic!("expected multipart system context")
    };
    assert_eq!(instructions.len(), 2);
    assert!(matches!(
        &instructions[0],
        ContentPart::Text { text, .. } if text == "Follow policy."
    ));
    assert!(matches!(
        &instructions[1],
        ContentPart::ProviderNative { kind, .. } if kind == "cachePoint"
    ));

    let params = decoded.params.unwrap();
    assert_eq!(params.max_tokens, Some(512));
    assert_eq!(params.temperature, Some(0.2));
    assert_eq!(params.top_p, Some(0.9));
    assert_eq!(params.stop, Some(vec!["END".into()]));

    let tools = decoded.tools.unwrap();
    assert_eq!(tools.len(), 2);
    match &tools[0] {
        ToolDefinition::Function { function, extra } => {
            assert_eq!(function.name, "weather");
            assert_eq!(function.description.as_deref(), Some("Read the weather"));
            assert_eq!(function.strict, Some(true));
            assert_eq!(
                function.parameters,
                Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }))
            );
            assert!(function.extra.is_empty());
            assert!(extra.is_empty());
        }
        other => panic!("expected function tool, got {other:?}"),
    }
    assert!(matches!(tools[1], ToolDefinition::ProviderNative { .. }));
    assert_eq!(decoded.tool_choice, Some(ToolChoice::Auto));
    assert!(decoded.extra.contains_key("additionalModelRequestFields"));
    assert!(decoded.extra.contains_key("requestMetadata"));
}

#[test]
fn unchanged_encode_is_byte_for_byte_json_equivalent() {
    let original = full_request();
    let decoded = BedrockConverseCodec.decode(&original).unwrap();
    let encoded = BedrockConverseCodec.encode(&decoded, &original).unwrap();
    assert_eq!(encoded, original);
}

#[test]
fn encode_overlays_edits_without_dropping_provider_fields() {
    let original = full_request();
    let mut decoded = BedrockConverseCodec.decode(&original).unwrap();
    decoded.model = Some("amazon.nova-pro-v1:0".into());
    decoded.params.as_mut().unwrap().max_tokens = Some(1024);
    let Some(MessageContent::Parts(instructions)) = decoded.instructions.as_mut() else {
        panic!("expected multipart system context")
    };
    let ContentPart::Text { text, .. } = &mut instructions[0] else {
        panic!("expected system text")
    };
    *text = "Follow the updated policy.".into();
    let Message::User { content, .. } = &mut decoded.messages[0] else {
        panic!("expected user message")
    };
    let MessageContent::Parts(parts) = content else {
        panic!("expected multipart message")
    };
    let ContentPart::Text { text, .. } = &mut parts[0] else {
        panic!("expected text part")
    };
    *text = "Weather in Toronto?".into();

    let encoded = BedrockConverseCodec.encode(&decoded, &original).unwrap();
    assert_eq!(encoded.headers, original.headers);
    assert_eq!(encoded.content["modelId"], "amazon.nova-pro-v1:0");
    assert_eq!(
        encoded.content["messages"][0]["content"][0]["text"],
        "Weather in Toronto?"
    );
    assert_eq!(
        encoded.content["system"][0]["text"],
        "Follow the updated policy."
    );
    assert_eq!(
        encoded.content["system"][1]["cachePoint"]["type"],
        "default"
    );
    assert_eq!(encoded.content["inferenceConfig"]["maxTokens"], 1024);
    assert_eq!(
        encoded.content["toolConfig"]["tools"][1]["cachePoint"]["type"],
        "default"
    );
    assert_eq!(
        encoded.content["additionalModelRequestFields"],
        original.content["additionalModelRequestFields"]
    );
}

#[test]
fn synthetic_future_fields_survive_unrelated_edits() {
    // These fields deliberately model a future SDK schema. They are separate
    // from `full_request`, which stays valid against today's tagged unions.
    let original = request(json!({
        "modelId": "amazon.nova-pro-v1:0",
        "messages": [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"toolUse": {
                "toolUseId": "call-future",
                "name": "lookup",
                "input": {"query": "hello"},
                "futureToolUseField": "keep"
            }}]}
        ],
        "inferenceConfig": {"maxTokens": 32, "futureInferenceField": "keep"},
        "futureTopLevelField": {"keep": true}
    }));
    let mut decoded = BedrockConverseCodec.decode(&original).unwrap();
    decoded.params.as_mut().unwrap().max_tokens = Some(64);
    let encoded = BedrockConverseCodec.encode(&decoded, &original).unwrap();
    assert_eq!(encoded.content["inferenceConfig"]["maxTokens"], 64);
    assert_eq!(
        encoded.content["inferenceConfig"]["futureInferenceField"],
        "keep"
    );
    assert_eq!(
        encoded.content["messages"][1]["content"][0]["toolUse"]["futureToolUseField"],
        "keep"
    );
    assert_eq!(encoded.content["futureTopLevelField"]["keep"], true);
}

#[test]
fn unmodeled_content_blocks_remain_provider_native_and_round_trip() {
    let blocks = vec![
        json!({
            "audio": {
                "format": "mp3",
                "source": {"bytes": "base64-audio"}
            }
        }),
        json!({
            "searchResult": {
                "title": "Relay",
                "source": "https://example.test/relay",
                "content": [{"text": "Search result text"}]
            }
        }),
        json!({"toolAddition": {"tool": {"name": "get_weather"}}}),
        json!({"toolRemoval": {"tool": {"name": "get_time"}}}),
    ];
    let original = request(json!({
        "modelId": "amazon.nova-pro-v1:0",
        "messages": [{"role": "user", "content": blocks}]
    }));

    let decoded = BedrockConverseCodec.decode(&original).unwrap();
    let Message::User { content, .. } = &decoded.messages[0] else {
        panic!("expected user message")
    };
    let MessageContent::Parts(parts) = content else {
        panic!("expected multipart message")
    };
    for (part, expected_kind) in
        parts
            .iter()
            .zip(["audio", "searchResult", "toolAddition", "toolRemoval"])
    {
        assert!(matches!(
            part,
            ContentPart::ProviderNative { provider, kind, .. }
                if provider == BEDROCK_PROVIDER && kind == expected_kind
        ));
    }

    assert_eq!(
        BedrockConverseCodec.encode(&decoded, &original).unwrap(),
        original
    );
}

#[test]
fn mixed_content_block_unions_remain_provider_native_and_round_trip() {
    let blocks = vec![
        json!({"text": "do not normalize", "audio": {"format": "mp3"}}),
        json!({
            "text": "do not normalize",
            "searchResult": {
                "title": "Relay",
                "source": "https://example.test/relay",
                "content": [{"text": "Search result text"}]
            }
        }),
        json!({
            "text": "do not normalize",
            "toolAddition": {"tool": {"name": "get_weather"}}
        }),
        json!({
            "text": "do not normalize",
            "toolRemoval": {"tool": {"name": "get_time"}}
        }),
    ];
    let original = request(json!({
        "modelId": "amazon.nova-pro-v1:0",
        "messages": [{"role": "user", "content": blocks}]
    }));

    let decoded = BedrockConverseCodec.decode(&original).unwrap();
    let Message::User { content, .. } = &decoded.messages[0] else {
        panic!("expected user message")
    };
    let MessageContent::Parts(parts) = content else {
        panic!("expected multipart message")
    };
    assert!(
        parts
            .iter()
            .all(|part| matches!(part, ContentPart::ProviderNative { .. }))
    );

    assert_eq!(
        BedrockConverseCodec.encode(&decoded, &original).unwrap(),
        original
    );
}

#[test]
fn decode_response_maps_text_tools_stop_reason_and_usage() {
    let raw = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Checking now."},
                    {"toolUse": {
                        "toolUseId": "call-2",
                        "name": "weather",
                        "input": {"city": "Toronto"}
                    }},
                    {"reasoningContent": {"reasoningText": {"text": "hidden"}}}
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 25,
            "totalTokens": 125,
            "cacheReadInputTokens": 20,
            "cacheWriteInputTokens": 5,
            "cacheDetails": [{"ttl": "1h", "inputTokens": 5}]
        },
        "metrics": {"latencyMs": 321},
        "trace": {"guardrail": {"actionReason": "NONE"}}
    });
    let decoded = BedrockConverseCodec.decode_response(&raw).unwrap();
    assert_eq!(decoded.finish_reason, Some(FinishReason::ToolUse));
    assert!(matches!(
        decoded.message,
        Some(MessageContent::Parts(ref parts)) if parts.len() == 3
    ));
    let call = &decoded.tool_calls.unwrap()[0];
    assert_eq!(call.id, "call-2");
    assert_eq!(call.name, "weather");
    assert_eq!(call.arguments, json!({"city": "Toronto"}));
    let usage = decoded.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(100));
    assert_eq!(usage.completion_tokens, Some(25));
    assert_eq!(usage.total_tokens, Some(125));
    assert_eq!(usage.cache_read_tokens, Some(20));
    assert_eq!(usage.cache_write_tokens, Some(5));
    assert_eq!(decoded.extra["metrics"]["latencyMs"], 321);
    match decoded.api_specific.unwrap() {
        ApiSpecificResponse::Custom { api_name, data } => {
            assert_eq!(api_name, BEDROCK_PROVIDER);
            assert_eq!(
                data,
                json!({
                    "stopReason": "tool_use",
                    "usage": {
                        "inputTokens": 100,
                        "outputTokens": 25,
                        "totalTokens": 125,
                        "cacheReadInputTokens": 20,
                        "cacheWriteInputTokens": 5,
                        "cacheDetails": [{"ttl": "1h", "inputTokens": 5}]
                    }
                })
            );
        }
        other => panic!("expected custom Bedrock response data, got {other:?}"),
    }
}

#[test]
fn streaming_accumulator_builds_a_decodable_converse_response() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    for event in [
        json!({"messageStart": {"role": "assistant"}}),
        json!({"contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"text": "Hello "}
        }}),
        json!({"contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"text": "world"}
        }}),
        json!({"contentBlockStop": {"contentBlockIndex": 0}}),
        json!({"contentBlockStart": {
            "contentBlockIndex": 1,
            "start": {"toolUse": {"toolUseId": "call-3", "name": "weather"}}
        }}),
        json!({"contentBlockDelta": {
            "contentBlockIndex": 1,
            "delta": {"toolUse": {"input": "{\"city\":"}}
        }}),
        json!({"contentBlockDelta": {
            "contentBlockIndex": 1,
            "delta": {"toolUse": {"input": "\"London\"}"}}
        }}),
        json!({"contentBlockStop": {"contentBlockIndex": 1}}),
        json!({"messageStop": {
            "stopReason": "tool_use",
            "additionalModelResponseFields": {"provider": "field"}
        }}),
        json!({"metadata": {
            "usage": {"inputTokens": 4, "outputTokens": 6, "totalTokens": 10},
            "metrics": {"latencyMs": 12},
            "trace": {"key": "value"}
        }}),
    ] {
        collect(event).unwrap();
    }
    let aggregate = codec.finalizer()();
    assert_eq!(
        aggregate["output"]["message"]["content"][0]["text"],
        "Hello world"
    );
    assert_eq!(
        aggregate["output"]["message"]["content"][1]["toolUse"]["input"],
        json!({"city": "London"})
    );
    assert_eq!(
        aggregate["additionalModelResponseFields"]["provider"],
        "field"
    );
    assert_eq!(aggregate["trace"]["key"], "value");

    let decoded = BedrockConverseCodec.decode_response(&aggregate).unwrap();
    assert_eq!(decoded.finish_reason, Some(FinishReason::ToolUse));
    assert_eq!(decoded.usage.unwrap().total_tokens, Some(10));
    assert_eq!(decoded.tool_calls.unwrap()[0].name, "weather");
}

#[test]
fn streaming_sparse_huge_indices_do_not_allocate_by_index() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    for event in [
        json!({"contentBlockDelta": {
            "contentBlockIndex": u64::MAX,
            "delta": {"text": "tail"}
        }}),
        json!({"contentBlockStop": {"contentBlockIndex": u64::MAX}}),
        json!({"contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"text": "head"}
        }}),
        json!({"contentBlockStop": {"contentBlockIndex": 0}}),
    ] {
        collect(event).unwrap();
    }

    let aggregate = codec.finalizer()();
    assert_eq!(
        aggregate["output"]["message"]["content"],
        json!([{"text": "head"}, {"text": "tail"}])
    );
}

#[test]
fn streaming_exception_is_not_silently_ignored() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    let error = collect(json!({
        "throttlingException": {"message": "slow down"}
    }))
    .unwrap_err();
    match error {
        FlowError::Upstream(failure) => {
            assert_eq!(failure.status, Some(429));
            assert_eq!(failure.class, UpstreamFailureClass::RetryableStatus);
            assert!(failure.body.contains("throttlingException"));
            assert!(failure.body.contains("slow down"));
        }
        other => panic!("expected a typed upstream failure, got {other:?}"),
    }
}

#[test]
fn streaming_exception_body_is_bounded_on_utf8_boundaries() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    let error = collect(json!({
        "throttlingException": {"message": "é".repeat(40 * 1024)}
    }))
    .unwrap_err();
    let FlowError::Upstream(failure) = error else {
        panic!("expected typed upstream failure")
    };
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.class, UpstreamFailureClass::RetryableStatus);
    assert!(failure.body.len() <= MAX_UPSTREAM_ERROR_BODY_BYTES);
    assert!(
        failure
            .body
            .starts_with("Bedrock ConverseStream throttlingException:")
    );
}

#[test]
fn malformed_provider_fields_are_rejected() {
    assert!(
        BedrockConverseCodec
            .decode(&request(json!({"modelId": 1, "messages": []})))
            .is_err()
    );
    assert!(
        BedrockConverseCodec
            .decode_response(&json!({
                "output": {"message": {"content": []}},
                "stopReason": 1
            }))
            .is_err()
    );
}

#[test]
fn response_never_invents_or_reads_a_model_id() {
    let decoded = BedrockConverseCodec
        .decode_response(&json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "modelId": "nonstandard-response-field"
        }))
        .unwrap();
    assert_eq!(decoded.model, None);
    assert_eq!(decoded.extra["modelId"], "nonstandard-response-field");
}

#[test]
fn response_maps_all_documented_stop_reason_classes() {
    for (reason, expected) in [
        ("end_turn", FinishReason::Complete),
        ("stop_sequence", FinishReason::Complete),
        ("max_tokens", FinishReason::Length),
        ("tool_use", FinishReason::ToolUse),
        ("guardrail_intervened", FinishReason::ContentFilter),
        ("content_filtered", FinishReason::ContentFilter),
        (
            "malformed_tool_use",
            FinishReason::Unknown("malformed_tool_use".into()),
        ),
        (
            "model_context_window_exceeded",
            FinishReason::Unknown("model_context_window_exceeded".into()),
        ),
        (
            "malformed_model_output",
            FinishReason::Unknown("malformed_model_output".into()),
        ),
    ] {
        let decoded = BedrockConverseCodec
            .decode_response(&json!({
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": reason
            }))
            .unwrap();
        assert_eq!(decoded.finish_reason, Some(expected));
    }
}

#[test]
fn prompt_resource_arn_rejects_forbidden_inline_configuration() {
    let original = request(json!({
        "modelId": "arn:aws:bedrock:us-east-1:123456789012:prompt/EXAMPLE",
        "promptVariables": {"name": {"text": "Relay"}}
    }));
    let decoded = BedrockConverseCodec.decode(&original).unwrap();
    assert_eq!(
        BedrockConverseCodec.encode(&decoded, &original).unwrap(),
        original
    );

    let mut edited = decoded;
    edited.params = Some(GenerationParams {
        max_tokens: Some(32),
        ..Default::default()
    });
    let error = BedrockConverseCodec.encode(&edited, &original).unwrap_err();
    assert!(error.to_string().contains("prompt resource ARN"));

    let original = request(json!({
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [],
        "inferenceConfig": {"futureOnlyField": true}
    }));
    let mut edited = BedrockConverseCodec.decode(&original).unwrap();
    assert_eq!(edited.params, None);
    edited.model = Some("arn:aws:bedrock:us-east-1:123456789012:prompt/EXAMPLE".into());
    let error = BedrockConverseCodec.encode(&edited, &original).unwrap_err();
    assert!(error.to_string().contains("prompt resource ARN"));
}

#[test]
fn stream_omits_aggregation_for_unknown_or_unsupported_union_members() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    collect(json!({"SDK_UNKNOWN_MEMBER": {"name": "futureEvent"}})).unwrap();
    assert!(codec.finalizer()().is_null());

    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    collect(json!({"contentBlockDelta": {
        "contentBlockIndex": 0,
        "delta": {"reasoningContent": {"text": "hidden"}}
    }}))
    .unwrap();
    assert!(codec.finalizer()().is_null());
}

#[test]
fn unsupported_stream_content_releases_and_stops_rebuilding_aggregate_state() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    collect(json!({"contentBlockDelta": {
        "contentBlockIndex": 0,
        "delta": {"text": "discard me"}
    }}))
    .unwrap();
    assert_eq!(codec.state.lock().unwrap().blocks.len(), 1);

    collect(json!({"contentBlockDelta": {
        "contentBlockIndex": 1,
        "delta": {"reasoningContent": {"text": "unsupported"}}
    }}))
    .unwrap();
    {
        let state = codec.state.lock().unwrap();
        assert!(!state.aggregation_supported);
        assert!(state.blocks.is_empty());
    }

    collect(json!({"contentBlockDelta": {
        "contentBlockIndex": 2,
        "delta": {"text": "do not retain me"}
    }}))
    .unwrap();
    assert!(codec.state.lock().unwrap().blocks.is_empty());

    let error = collect(json!({
        "throttlingException": {"message": "still surface provider failures"}
    }))
    .unwrap_err();
    assert!(matches!(error, FlowError::Upstream(_)));
    assert!(codec.finalizer()().is_null());
}

#[test]
fn stream_forwards_incomplete_tool_json_without_a_partial_aggregate() {
    let codec = BedrockConverseStreamingCodec::new();
    let mut collect = codec.collector();
    collect(json!({"contentBlockStart": {
        "contentBlockIndex": 0,
        "start": {"toolUse": {"toolUseId": "call-1", "name": "weather"}}
    }}))
    .unwrap();
    collect(json!({"contentBlockDelta": {
        "contentBlockIndex": 0,
        "delta": {"toolUse": {"input": "{\"city\":"}}
    }}))
    .unwrap();
    collect(json!({"contentBlockStop": {"contentBlockIndex": 0}})).unwrap();
    assert!(codec.finalizer()().is_null());
}
