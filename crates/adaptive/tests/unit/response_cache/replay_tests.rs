// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache streaming replay in the NeMo Relay adaptive crate.

use super::*;

#[test]
fn stream_metadata_fields_do_not_make_a_real_body_lossy() {
    // Real OpenAI buffered bodies carry `system_fingerprint`, `service_tier`
    // and null `logprobs` that the streaming collector does not aggregate;
    // none of them changes what a streaming caller receives.
    let real = json!({"id": "c1", "object": "chat.completion", "created": 1,
        "model": "m", "system_fingerprint": "fp_abc", "service_tier": "default",
        "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "logprobs": null,
            "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}});
    assert!(
        !replay_is_lossy(&real),
        "stream-metadata fields must not disable the streaming replay"
    );
}

/// The replay synthesizer must emit chunks the provider's own streaming codec
/// reassembles into EXACTLY the stored aggregate — the property that makes a
/// replayed hit indistinguishable from a live stream to a strict client.
#[test]
fn replay_chunks_roundtrip_through_the_codecs() {
    let anthropic = json!({"id": "msg_1", "type": "message", "role": "assistant",
        "model": "m",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "get", "input": {"q": "x"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 9, "output_tokens": 3}});
    let chat = json!({"id": "c1", "object": "chat.completion", "created": 1,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}});
    let responses = json!({"id": "r1", "object": "response", "status": "completed",
        "model": "m",
        "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}]}],
        "usage": {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12}});
    for (aggregate, surface, codec_name) in [
        (
            anthropic,
            ProviderSurface::AnthropicMessages,
            "anthropic_messages",
        ),
        (chat, ProviderSurface::OpenAIChat, "openai_chat"),
        (
            responses,
            ProviderSurface::OpenAIResponses,
            "openai_responses",
        ),
    ] {
        let codec = streaming_codec(surface);
        let chunks =
            synthesize_replay_chunks(&aggregate).expect("aggregate shape must be recognized");
        assert!(
            chunks.len() > 1,
            "{codec_name}: a native replay must be a chunk sequence, not one frame"
        );
        let mut collect = codec.collector();
        for chunk in &chunks {
            collect(chunk.clone()).expect("codec must accept its own native chunk shape");
        }
        let reassembled = codec.finalizer()();
        assert_eq!(
            reassembled, aggregate,
            "{codec_name}: replayed chunks must reassemble to the stored aggregate"
        );
    }
}

#[test]
fn gemini_replay_uses_a_valid_native_stream_event() {
    let aggregate = json!({
        "candidates": [{
            "index": 0,
            "content": {
                "role": "model",
                "parts": [
                    {"text": "hello", "thoughtSignature": "sig_TEXT=="},
                    {
                        "functionCall": {
                            "id": "call_1",
                            "name": "lookup",
                            "args": {"q": "x"}
                        },
                        "thoughtSignature": "sig_CALL=="
                    }
                ]
            },
            "finishReason": "STOP",
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}
            ],
            "groundingMetadata": {"webSearchQueries": ["example query"]}
        }],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 3,
            "totalTokenCount": 12
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "resp_1"
    });
    let chunks = synthesize_replay_chunks(&aggregate).expect("gemini shape");
    assert_eq!(
        chunks,
        vec![aggregate.clone()],
        "a GenerateContentResponse aggregate is already a native Gemini stream event"
    );
    assert!(
        !replay_is_lossy(&aggregate),
        "the Gemini streaming codec must reassemble the native replay exactly"
    );
}

#[test]
fn gemini_replay_rejects_multi_candidate_aggregates_as_lossy() {
    let aggregate = json!({
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "first"}]},
                "finishReason": "STOP"
            },
            {
                "index": 1,
                "content": {"role": "model", "parts": [{"text": "second"}]},
                "finishReason": "STOP"
            }
        ]
    });
    assert!(
        replay_is_lossy(&aggregate),
        "Gemini streaming replay must not serve aggregates with candidates the collector cannot preserve"
    );
}

#[test]
fn responses_replay_sequence_numbers_are_contiguous() {
    let aggregate = json!({
        "id": "r1",
        "object": "response",
        "status": "completed",
        "model": "m",
        "output": [
            {"type": "message", "role": "assistant", "content": []},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
        ]
    });

    let chunks = synthesize_responses_chunks(&aggregate);
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk["type"].as_str().expect("event type"))
            .collect::<Vec<_>>(),
        [
            "response.created",
            "response.output_item.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| {
                chunk["sequence_number"]
                    .as_u64()
                    .expect("every Responses event needs a sequence_number")
            })
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
}

/// A chat replay must carry tool calls in the streaming delta shape a client
/// can accumulate (full arguments in one fragment is spec-valid).
#[test]
fn chat_replay_streams_tool_calls_as_deltas() {
    let aggregate = json!({"id": "c1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null,
            "tool_calls": [{"id": "call1", "type": "function",
                "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
            "finish_reason": "tool_calls"}]});
    let chunks = synthesize_replay_chunks(&aggregate).expect("chat shape");
    let tool_delta = chunks
        .iter()
        .find(|chunk| chunk.pointer("/choices/0/delta/tool_calls").is_some())
        .expect("a tool_calls delta chunk must be synthesized");
    assert_eq!(
        tool_delta.pointer("/choices/0/delta/tool_calls/0/function/arguments"),
        Some(&json!("{\"a\":1}"))
    );
}

/// An unknown aggregate shape has no native chunk synthesis, so the streaming
/// tier must treat it as lossy and run live rather than serve one
/// aggregate-shaped frame to a strict streaming client.
#[test]
fn replay_of_an_unknown_shape_is_lossy_for_the_streaming_tier() {
    assert!(synthesize_replay_chunks(&json!({"weird": true})).is_none());
    assert!(synthesize_replay_chunks(&json!("bare string")).is_none());
    assert!(replay_is_lossy(&json!({"weird": true})));
    assert!(replay_is_lossy(&json!("bare string")));
}

#[test]
fn stripping_stream_metadata_leaves_nonobject_frames_unchanged() {
    // The helper also runs against collector output. A malformed non-object
    // frame must be a harmless no-op rather than preventing the lossiness
    // check from completing.
    let mut frame = json!("not an aggregate");
    strip_stream_metadata(&mut frame);
    assert_eq!(frame, json!("not an aggregate"));
}

#[test]
fn anthropic_replay_keeps_complete_unknown_blocks_and_stop_sequences() {
    // Blocks without a delta representation (such as thinking/server blocks)
    // must be sent intact at content-block start, while stop_sequence remains
    // visible to strict Anthropic stream consumers.
    let aggregate = json!({
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "thinking", "thinking": "reasoning"}],
        "stop_reason": "end_turn",
        "stop_sequence": "<END>",
        "usage": {"input_tokens": 3, "output_tokens": 2}
    });

    let chunks = synthesize_anthropic_chunks(&aggregate);
    assert_eq!(chunks[1]["type"], json!("content_block_start"));
    assert_eq!(chunks[1]["content_block"], aggregate["content"][0]);
    assert_eq!(chunks[2]["type"], json!("content_block_stop"));
    let message_delta = chunks
        .iter()
        .find(|chunk| chunk["type"] == "message_delta")
        .expect("replay must finish with a message_delta");
    assert_eq!(
        message_delta.pointer("/delta/stop_sequence"),
        Some(&json!("<END>"))
    );
}

#[test]
fn responses_replay_omits_item_events_for_a_nonarray_output() {
    // A partially formed stored Responses aggregate is still replayed with
    // lifecycle framing, but only real output arrays produce item-done events.
    let aggregate = json!({
        "id": "resp_2",
        "object": "response",
        "model": "gpt-test",
        "output": {"unexpected": true}
    });

    let chunks = synthesize_responses_chunks(&aggregate);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["type"], json!("response.created"));
    assert_eq!(chunks[1]["type"], json!("response.completed"));
    assert_eq!(chunks[1]["sequence_number"], json!(1));
}
