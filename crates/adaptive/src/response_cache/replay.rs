// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay of a stored aggregate as provider-native streaming chunks.

use nemo_relay::api::runtime::LlmJsonStream;
use nemo_relay::codec::resolve::{ProviderSurface, detect_response_surface, streaming_codec};
use nemo_relay::error::FlowError;
use serde_json::{Map, Value as Json, json};

/// Replays a stored aggregate response as a stream of **provider-native chunks**.
///
/// A strict streaming client parses only its provider's wire chunks (Anthropic
/// `message_start → content_block_delta → … → message_stop`, OpenAI Chat
/// `chat.completion.chunk` deltas, OpenAI Responses lifecycle events) — replaying
/// the aggregate as one frame breaks such clients even though the body is correct.
/// The surface is detected from the stored aggregate's own shape (the codec
/// finalizer's output, or a buffered body of the same shape), so no codec handle
/// is needed at the call sites. An unrecognized shape falls back to the original
/// single-frame replay rather than dropping the answer.
pub(crate) fn replay_aggregate(response: Json) -> LlmJsonStream {
    let chunks = synthesize_replay_chunks(&response).unwrap_or_else(|| vec![response]);
    Box::pin(tokio_stream::iter(
        chunks.into_iter().map(Ok::<Json, FlowError>),
    ))
}

/// True when replaying `aggregate` as provider-native chunks would lose
/// content: the synthesized chunk sequence, re-aggregated through the same
/// streaming codec the live tee uses, must reassemble to the exact stored
/// aggregate. Buffered bodies can carry fields the synthesizers have no delta
/// for (a chat `refusal` or `audio` message), which would otherwise replay as
/// an empty stream. An unrecognized shape replays as one full-body frame and
/// is never lossy.
pub(crate) fn replay_is_lossy(aggregate: &Json) -> bool {
    let Some(surface) = detect_response_surface(aggregate) else {
        return false;
    };
    let codec = streaming_codec(surface);
    let mut collect = codec.collector();
    for chunk in synthesize_replay_chunks(aggregate).unwrap_or_default() {
        if collect(chunk).is_err() {
            return true;
        }
    }
    let mut reassembled = codec.finalizer()();
    let mut expected = aggregate.clone();
    strip_stream_metadata(&mut reassembled);
    strip_stream_metadata(&mut expected);
    reassembled != expected
}

/// Drops stream-metadata fields the chunk collectors do not aggregate and that
/// carry no answer content — `system_fingerprint`, `service_tier`, and a null
/// `logprobs` — so their absence from a replay does not count as loss.
fn strip_stream_metadata(aggregate: &mut Json) {
    let Some(object) = aggregate.as_object_mut() else {
        return;
    };
    object.remove("system_fingerprint");
    object.remove("service_tier");
    if let Some(choices) = object.get_mut("choices").and_then(Json::as_array_mut) {
        for choice in choices {
            if let Some(choice) = choice.as_object_mut()
                && choice.get("logprobs").is_some_and(Json::is_null)
            {
                choice.remove("logprobs");
            }
        }
    }
}

/// Synthesizes the native chunk sequence for the aggregate's detected surface.
/// `None` when the shape is not recognized.
fn synthesize_replay_chunks(aggregate: &Json) -> Option<Vec<Json>> {
    Some(match detect_response_surface(aggregate)? {
        ProviderSurface::AnthropicMessages => synthesize_anthropic_chunks(aggregate),
        ProviderSurface::OpenAIChat => synthesize_chat_chunks(aggregate),
        ProviderSurface::OpenAIResponses => synthesize_responses_chunks(aggregate),
    })
}

/// Anthropic Messages: `message_start`, then per content block a
/// `content_block_start`/delta/`content_block_stop` triple (text and `tool_use`
/// stream their payload as a native delta; other block types ship complete at
/// start, as the live API does), then `message_delta` carrying the stored usage
/// verbatim (the collector replaces usage wholesale, so the reassembled aggregate
/// round-trips), then `message_stop`.
fn synthesize_anthropic_chunks(aggregate: &Json) -> Vec<Json> {
    let mut start_message = Map::new();
    start_message.insert("type".to_string(), json!("message"));
    for key in ["id", "role", "model"] {
        if let Some(value) = aggregate.get(key) {
            start_message.insert(key.to_string(), value.clone());
        }
    }
    start_message.insert("content".to_string(), json!([]));
    start_message.insert("stop_reason".to_string(), Json::Null);
    start_message.insert("stop_sequence".to_string(), Json::Null);
    let input_tokens = aggregate
        .pointer("/usage/input_tokens")
        .cloned()
        .unwrap_or(json!(0));
    start_message.insert(
        "usage".to_string(),
        json!({"input_tokens": input_tokens, "output_tokens": 0}),
    );
    let mut chunks = vec![json!({"type": "message_start", "message": start_message})];

    let blocks = aggregate
        .get("content")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    for (index, block) in blocks.iter().enumerate() {
        let block_type = block.get("type").and_then(Json::as_str).unwrap_or("");
        match block_type {
            "text" => {
                let mut skeleton = block.clone();
                let text = skeleton
                    .as_object_mut()
                    .and_then(|map| map.insert("text".to_string(), json!("")))
                    .and_then(|old| old.as_str().map(str::to_string))
                    .unwrap_or_default();
                chunks.push(json!({"type": "content_block_start", "index": index,
                    "content_block": skeleton}));
                chunks.push(json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": text}}));
            }
            "tool_use" => {
                let mut skeleton = block.clone();
                let input = skeleton
                    .as_object_mut()
                    .and_then(|map| map.insert("input".to_string(), json!({})))
                    .unwrap_or(json!({}));
                let partial = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                chunks.push(json!({"type": "content_block_start", "index": index,
                    "content_block": skeleton}));
                chunks.push(json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": partial}}));
            }
            // Thinking, server_tool_use, and other block types ship complete at
            // start (the collector keeps the skeleton verbatim).
            _ => {
                chunks.push(json!({"type": "content_block_start", "index": index,
                    "content_block": block.clone()}));
            }
        }
        chunks.push(json!({"type": "content_block_stop", "index": index}));
    }

    let mut delta = Map::new();
    if let Some(reason) = aggregate.get("stop_reason") {
        delta.insert("stop_reason".to_string(), reason.clone());
    }
    if let Some(sequence) = aggregate.get("stop_sequence") {
        delta.insert("stop_sequence".to_string(), sequence.clone());
    }
    let mut message_delta = Map::new();
    message_delta.insert("type".to_string(), json!("message_delta"));
    message_delta.insert("delta".to_string(), Json::Object(delta));
    if let Some(usage) = aggregate.get("usage") {
        message_delta.insert("usage".to_string(), usage.clone());
    }
    chunks.push(Json::Object(message_delta));
    chunks.push(json!({"type": "message_stop"}));
    chunks
}

/// OpenAI Chat: per choice a role delta, a content delta (when the message has
/// text), a `tool_calls` delta (full arguments in one fragment — spec-valid), and
/// a finish chunk; then a final usage-bearing chunk with empty `choices` (the
/// `include_usage` wire shape). Top-level id/created/model ride on every chunk.
fn synthesize_chat_chunks(aggregate: &Json) -> Vec<Json> {
    let base = |choices: Json| -> Json {
        let mut chunk = Map::new();
        for key in ["id", "created", "model"] {
            if let Some(value) = aggregate.get(key) {
                chunk.insert(key.to_string(), value.clone());
            }
        }
        chunk.insert("object".to_string(), json!("chat.completion.chunk"));
        chunk.insert("choices".to_string(), choices);
        Json::Object(chunk)
    };
    let mut chunks = Vec::new();
    let choices = aggregate
        .get("choices")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    for (position, choice) in choices.iter().enumerate() {
        let index = choice
            .get("index")
            .and_then(Json::as_u64)
            .unwrap_or(position as u64);
        let message = choice.get("message").cloned().unwrap_or(json!({}));
        if let Some(role) = message.get("role") {
            chunks.push(base(
                json!([{"index": index, "delta": {"role": role}, "finish_reason": null}]),
            ));
        }
        if let Some(content) = message.get("content").and_then(Json::as_str)
            && !content.is_empty()
        {
            chunks.push(base(
                json!([{"index": index, "delta": {"content": content}, "finish_reason": null}]),
            ));
        }
        if let Some(tool_calls) = message.get("tool_calls").and_then(Json::as_array) {
            let deltas: Vec<Json> = tool_calls
                .iter()
                .enumerate()
                .map(|(call_index, call)| {
                    let mut delta = call.clone();
                    if let Some(map) = delta.as_object_mut() {
                        map.entry("index".to_string())
                            .or_insert(json!(call_index as u64));
                    }
                    delta
                })
                .collect();
            chunks.push(base(
                json!([{"index": index, "delta": {"tool_calls": deltas}, "finish_reason": null}]),
            ));
        }
        let finish = choice.get("finish_reason").cloned().unwrap_or(Json::Null);
        chunks.push(base(
            json!([{"index": index, "delta": {}, "finish_reason": finish}]),
        ));
    }
    if let Some(usage) = aggregate.get("usage") {
        let mut usage_chunk = base(json!([]));
        if let Some(map) = usage_chunk.as_object_mut() {
            map.insert("usage".to_string(), usage.clone());
        }
        chunks.push(usage_chunk);
    }
    chunks
}

/// OpenAI Responses: a `response.created` snapshot, one `response.output_item.done`
/// per output item, and a `response.completed` carrying the full stored aggregate
/// (the collector keeps the last snapshot wholesale, so reassembly is exact).
fn synthesize_responses_chunks(aggregate: &Json) -> Vec<Json> {
    let mut created = Map::new();
    for key in ["id", "object", "model"] {
        if let Some(value) = aggregate.get(key) {
            created.insert(key.to_string(), value.clone());
        }
    }
    created.insert("status".to_string(), json!("in_progress"));
    let mut chunks = vec![json!({"type": "response.created", "response": created})];
    if let Some(items) = aggregate.get("output").and_then(Json::as_array) {
        for (index, item) in items.iter().enumerate() {
            chunks.push(json!({"type": "response.output_item.done",
                "output_index": index, "item": item.clone()}));
        }
    }
    chunks.push(json!({"type": "response.completed", "response": aggregate.clone()}));
    chunks
}

#[cfg(test)]
mod tests {
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

    /// An unknown aggregate shape must fall back to the single-frame replay,
    /// never drop the answer.
    #[test]
    fn replay_of_an_unknown_shape_falls_back_to_one_frame() {
        assert!(synthesize_replay_chunks(&json!({"weird": true})).is_none());
        assert!(synthesize_replay_chunks(&json!("bare string")).is_none());
    }
}
