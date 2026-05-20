// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response wrapper.
//!
//! This module provides [`LlmStreamWrapper`], a [`Stream`] adapter
//! that sits between the raw stream from an LLM API and the consumer. It
//! feeds chunks to a user-supplied collector, and automatically emits
//! lifecycle events when the stream ends.
//!
//! ## Pipeline
//!
//! ```text
//! raw chunk (Json) -> collector(chunk) -> Ok(()) -> yield chunk
//!                                      -> Err(e) -> terminate stream with error
//! upstream error -> terminate stream with error -> finalizer() -> Json -> SanitizeResponseGuardrails -> END event
//! stream ends -> finalizer() -> Json -> SanitizeResponseGuardrails -> END event
//! ```
//!
//! The **collector** receives each chunk (Json) and can accumulate state
//! (e.g., concatenating tokens). If the collector returns `Err`, the stream
//! terminates immediately with that error. Upstream stream errors also
//! terminate the stream immediately. The **finalizer** is called once when the
//! stream terminates and returns the aggregated response as [`Json`]. That
//! aggregated response then flows through sanitize response guardrails before
//! being included in the END event.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio_stream::Stream;

use crate::api::event::{BaseEvent, MarkEvent};
use crate::api::llm::LlmHandle;
use crate::api::runtime::NemoFlowContextState;
use crate::api::runtime::global_context;
use crate::api::runtime::{ScopeStackHandle, current_scope_stack};
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::LlmResponseCodec;
use crate::error::Result;
use crate::json::Json;
use serde_json::Map;

/// Wraps an inner `Stream<Item = Result<Json>>` of raw chunks and:
///
/// 1. Passes each chunk to the user-supplied **collector** closure.
///    If the collector returns `Err`, the stream terminates with that error.
/// 2. On stream exhaustion, calls the **finalizer** to produce an aggregated
///    [`Json`] response, runs sanitize response guardrails on it, then emits
///    the LLM END event.
///
/// This type is returned by [`crate::api::llm::llm_stream_call_execute`] and
/// is usually consumed as an ordinary async stream. The wrapper preserves the
/// originating scope stack so end-of-stream bookkeeping still uses the correct
/// scope-local middleware and subscribers even when polling happens elsewhere.
pub struct LlmStreamWrapper {
    inner: Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
    handle: LlmHandle,
    scope_stack: ScopeStackHandle,
    collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
    finalizer: Option<Box<dyn FnOnce() -> Json + Send>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    metadata: Option<Json>,
    chunk_index: u64,
    ended: bool,
}

impl LlmStreamWrapper {
    /// Create a new `LlmStreamWrapper` around the given raw stream.
    ///
    /// Captures the current [`ScopeStackHandle`] at creation time so the
    /// correct scope stack is used when the stream is later polled, even if
    /// polling happens on a different task or thread.
    ///
    /// # Parameters
    /// - `inner`: Raw stream of JSON chunks from the provider callback.
    /// - `handle`: [`LlmHandle`] identifying the managed LLM span.
    /// - `collector`: Per-chunk callback used to accumulate stream state or
    ///   forward chunks elsewhere. Returning `Err` terminates the stream.
    /// - `finalizer`: One-shot callback invoked when the stream finishes to
    ///   synthesize the aggregated response payload.
    /// - `data`: Retained compatibility payload; Agent Trajectory
    ///   Observability Format (ATOF) end data is the finalized response.
    /// - `metadata`: Optional event metadata merged into the emitted LLM-end event.
    /// - `response_codec`: Optional codec used to derive annotated response
    ///   metadata from the aggregated final payload.
    ///
    /// # Returns
    /// A new [`LlmStreamWrapper`] ready to be polled.
    pub fn new(
        inner: Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
        handle: LlmHandle,
        collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
        finalizer: Box<dyn FnOnce() -> Json + Send>,
        _data: Option<Json>,
        metadata: Option<Json>,
        response_codec: Option<Arc<dyn LlmResponseCodec>>,
    ) -> Self {
        Self {
            inner,
            handle,
            scope_stack: current_scope_stack(),
            collector,
            finalizer: Some(finalizer),
            response_codec,
            metadata,
            chunk_index: 0,
            ended: false,
        }
    }

    /// Return the captured scope stack handle for this stream.
    ///
    /// Callers can use this to bind the correct scope stack when spawning
    /// the stream on a different task via `TASK_SCOPE_STACK.scope(...)`.
    ///
    /// # Returns
    /// A shared reference to the [`ScopeStackHandle`] captured when the stream
    /// wrapper was created.
    pub fn scope_stack(&self) -> &ScopeStackHandle {
        &self.scope_stack
    }

    fn finish(&mut self) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.emit_end_event();
    }

    /// Emit the LLM END event with aggregated response data.
    ///
    /// Calls the finalizer to produce the aggregated response, runs sanitize
    /// response guardrails, and emits the END event.
    fn emit_end_event(&mut self) {
        let aggregated = match self.finalizer.take() {
            Some(finalizer) => finalizer(),
            None => Json::Null,
        };

        // Decode aggregated response if response codec is present (non-fatal)
        let annotated_response: Option<Arc<AnnotatedLlmResponse>> = self
            .response_codec
            .as_ref()
            .and_then(|c| c.decode_response(&aggregated).ok())
            .map(Arc::new);

        let event_snapshot = {
            let ss_guard = self.scope_stack.read().expect("scope stack lock poisoned");
            let sl =
                ss_guard.collect_scope_local_registries(|r| &r.llm_sanitize_response_guardrails);
            let sl_subs = ss_guard.collect_scope_local_subscribers();
            let ctx = global_context();
            let state = ctx.read();
            match state {
                Ok(state) => {
                    let subscribers = state.collect_event_subscribers(&sl_subs);
                    let sanitized = state.llm_sanitize_response_chain(aggregated, &sl);
                    let data = if sanitized.is_null() {
                        self.handle.data.clone()
                    } else {
                        Some(sanitized)
                    };
                    let event = state.end_llm_handle(
                        &self.handle,
                        data,
                        self.metadata.clone(),
                        annotated_response,
                    );
                    Some((event, subscribers))
                }
                Err(_) => None,
            }
        };
        if let Some((event, subscribers)) = event_snapshot {
            NemoFlowContextState::emit_event(&event, &subscribers);
        }
    }

    /// Emit a compact per-chunk receipt mark before collector processing.
    fn emit_chunk_mark(&self, chunk_index: u64, raw_chunk: &Json) {
        let data = llm_chunk_mark_data(chunk_index, raw_chunk);
        let event_snapshot = {
            let ss_guard = self.scope_stack.read().expect("scope stack lock poisoned");
            let sl_subs = ss_guard.collect_scope_local_subscribers();
            let ctx = global_context();
            let state = ctx.read();
            match state {
                Ok(state) => {
                    let subscribers = state.collect_event_subscribers(&sl_subs);
                    let event = state.create_event(MarkEvent::new(
                        BaseEvent::builder()
                            .name("llm.chunk")
                            .parent_uuid(self.handle.uuid)
                            .data(data)
                            .build(),
                        None,
                        None,
                    ));
                    Some((event, subscribers))
                }
                Err(_) => None,
            }
        };
        if let Some((event, subscribers)) = event_snapshot {
            NemoFlowContextState::emit_event(&event, &subscribers);
        }
    }
}

impl Stream for LlmStreamWrapper {
    type Item = Result<Json>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.ended {
            return Poll::Ready(None);
        }

        // Poll the inner stream
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(raw_chunk))) => {
                let chunk_index = this.chunk_index;
                this.chunk_index += 1;
                this.emit_chunk_mark(chunk_index, &raw_chunk);
                // Feed chunk to the collector; if it returns Err, terminate the stream
                match (this.collector)(raw_chunk.clone()) {
                    Ok(()) => Poll::Ready(Some(Ok(raw_chunk))),
                    Err(e) => {
                        this.finish();
                        Poll::Ready(Some(Err(e)))
                    }
                }
            }
            Poll::Ready(Some(Err(e))) => {
                this.finish();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                this.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn llm_chunk_mark_data(chunk_index: u64, raw_chunk: &Json) -> Json {
    if let Some(data) = summarize_openai_chat_chunk(chunk_index, raw_chunk) {
        return data;
    }
    if let Some(data) = summarize_openai_responses_chunk(chunk_index, raw_chunk) {
        return data;
    }
    if let Some(data) = summarize_anthropic_messages_chunk(chunk_index, raw_chunk) {
        return data;
    }
    Json::Object(base_chunk_mark_data(chunk_index, "unknown"))
}

fn base_chunk_mark_data(chunk_index: u64, provider: &str) -> Map<String, Json> {
    let mut data = Map::new();
    data.insert("chunk_index".into(), Json::from(chunk_index));
    data.insert("provider".into(), Json::String(provider.to_string()));
    data
}

fn summarize_openai_chat_chunk(chunk_index: u64, raw_chunk: &Json) -> Option<Json> {
    let object = raw_chunk.get("object").and_then(Json::as_str);
    let choices = raw_chunk.get("choices").and_then(Json::as_array);
    if object != Some("chat.completion.chunk") {
        return None;
    }

    let mut data = base_chunk_mark_data(chunk_index, "openai_chat_completions");
    if let Some(object) = object {
        data.insert("event_type".into(), Json::String(object.to_string()));
    }
    if let Some(choices) = choices {
        let choice_indices: Vec<Json> = choices
            .iter()
            .filter_map(|choice| choice.get("index").and_then(Json::as_u64).map(Json::from))
            .collect();
        if !choice_indices.is_empty() {
            data.insert("choice_indices".into(), Json::Array(choice_indices));
        }

        let finish_reasons: Vec<Json> = choices
            .iter()
            .filter_map(|choice| {
                let reason = choice.get("finish_reason").and_then(Json::as_str)?;
                let mut item = Map::new();
                if let Some(index) = choice.get("index").and_then(Json::as_u64) {
                    item.insert("choice_index".into(), Json::from(index));
                }
                item.insert("finish_reason".into(), Json::String(reason.to_string()));
                Some(Json::Object(item))
            })
            .collect();
        if !finish_reasons.is_empty() {
            data.insert("finish_reasons".into(), Json::Array(finish_reasons));
        }
    }
    if let Some(usage) = raw_chunk.get("usage").and_then(normalize_openai_chat_usage) {
        data.insert("usage".into(), usage);
    }

    Some(Json::Object(data))
}

fn summarize_openai_responses_chunk(chunk_index: u64, raw_chunk: &Json) -> Option<Json> {
    let event_type = raw_chunk.get("type").and_then(Json::as_str)?;
    if !event_type.starts_with("response.") {
        return None;
    }

    let mut data = base_chunk_mark_data(chunk_index, "openai_responses");
    data.insert("event_type".into(), Json::String(event_type.to_string()));
    insert_index_fields(&mut data, raw_chunk, &["output_index", "content_index"]);

    if let Some(status) = raw_chunk
        .get("response")
        .and_then(|response| response.get("status"))
        .or_else(|| raw_chunk.get("status"))
        .and_then(Json::as_str)
    {
        data.insert("status".into(), Json::String(status.to_string()));
    }
    if let Some(reason) = raw_chunk
        .get("response")
        .and_then(|response| response.get("incomplete_details"))
        .and_then(|details| details.get("reason"))
        .and_then(Json::as_str)
    {
        data.insert("finish_reason".into(), Json::String(reason.to_string()));
    }
    if let Some(usage) = raw_chunk
        .get("usage")
        .or_else(|| {
            raw_chunk
                .get("response")
                .and_then(|response| response.get("usage"))
        })
        .and_then(normalize_openai_responses_usage)
    {
        data.insert("usage".into(), usage);
    }

    Some(Json::Object(data))
}

fn summarize_anthropic_messages_chunk(chunk_index: u64, raw_chunk: &Json) -> Option<Json> {
    let event_type = raw_chunk.get("type").and_then(Json::as_str)?;
    if !matches!(
        event_type,
        "message_start"
            | "content_block_start"
            | "content_block_delta"
            | "content_block_stop"
            | "message_delta"
            | "message_stop"
            | "ping"
    ) {
        return None;
    }

    let mut data = base_chunk_mark_data(chunk_index, "anthropic_messages");
    data.insert("event_type".into(), Json::String(event_type.to_string()));
    insert_index_fields(&mut data, raw_chunk, &["index"]);

    if let Some(stop_reason) = raw_chunk
        .get("delta")
        .and_then(|delta| delta.get("stop_reason"))
        .or_else(|| {
            raw_chunk
                .get("message")
                .and_then(|message| message.get("stop_reason"))
        })
        .and_then(Json::as_str)
    {
        data.insert("stop_reason".into(), Json::String(stop_reason.to_string()));
    }
    if let Some(usage) = raw_chunk
        .get("usage")
        .or_else(|| {
            raw_chunk
                .get("message")
                .and_then(|message| message.get("usage"))
        })
        .and_then(normalize_anthropic_usage)
    {
        data.insert("usage".into(), usage);
    }

    Some(Json::Object(data))
}

fn insert_index_fields(data: &mut Map<String, Json>, raw_chunk: &Json, field_names: &[&str]) {
    let mut indices = Map::new();
    for field_name in field_names {
        if let Some(index) = raw_chunk.get(*field_name).and_then(Json::as_u64) {
            indices.insert((*field_name).to_string(), Json::from(index));
        }
    }
    if !indices.is_empty() {
        data.insert("indices".into(), Json::Object(indices));
    }
}

fn normalize_openai_chat_usage(usage: &Json) -> Option<Json> {
    let mut normalized = Map::new();
    insert_u64_field(&mut normalized, usage, "prompt_tokens", "prompt_tokens");
    insert_u64_field(
        &mut normalized,
        usage,
        "completion_tokens",
        "completion_tokens",
    );
    insert_u64_field(&mut normalized, usage, "total_tokens", "total_tokens");
    if let Some(cached_tokens) = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Json::as_u64)
    {
        normalized.insert("cache_read_tokens".into(), Json::from(cached_tokens));
    }
    non_empty_object(normalized)
}

fn normalize_openai_responses_usage(usage: &Json) -> Option<Json> {
    let mut normalized = Map::new();
    insert_u64_field(&mut normalized, usage, "input_tokens", "prompt_tokens");
    insert_u64_field(&mut normalized, usage, "output_tokens", "completion_tokens");
    insert_u64_field(&mut normalized, usage, "total_tokens", "total_tokens");
    if let Some(cached_tokens) = usage
        .get("input_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Json::as_u64)
    {
        normalized.insert("cache_read_tokens".into(), Json::from(cached_tokens));
    }
    non_empty_object(normalized)
}

fn normalize_anthropic_usage(usage: &Json) -> Option<Json> {
    let mut normalized = Map::new();
    let prompt_tokens = usage.get("input_tokens").and_then(Json::as_u64);
    let completion_tokens = usage.get("output_tokens").and_then(Json::as_u64);
    if let Some(prompt_tokens) = prompt_tokens {
        normalized.insert("prompt_tokens".into(), Json::from(prompt_tokens));
    }
    if let Some(completion_tokens) = completion_tokens {
        normalized.insert("completion_tokens".into(), Json::from(completion_tokens));
    }
    if let Some(total_tokens) = prompt_tokens
        .and_then(|prompt| completion_tokens.and_then(|completion| prompt.checked_add(completion)))
    {
        normalized.insert("total_tokens".into(), Json::from(total_tokens));
    }
    insert_u64_field(
        &mut normalized,
        usage,
        "cache_read_input_tokens",
        "cache_read_tokens",
    );
    insert_u64_field(
        &mut normalized,
        usage,
        "cache_creation_input_tokens",
        "cache_write_tokens",
    );
    non_empty_object(normalized)
}

fn insert_u64_field(
    output: &mut Map<String, Json>,
    input: &Json,
    input_field: &str,
    output_field: &str,
) {
    if let Some(value) = input.get(input_field).and_then(Json::as_u64) {
        output.insert(output_field.to_string(), Json::from(value));
    }
}

fn non_empty_object(object: Map<String, Json>) -> Option<Json> {
    if object.is_empty() {
        None
    } else {
        Some(Json::Object(object))
    }
}

impl Drop for LlmStreamWrapper {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_no_top_level_fields(data: &Json, fields: &[&str]) {
        let object = data.as_object().expect("mark data must be an object");
        for field in fields {
            assert!(
                !object.contains_key(*field),
                "unexpected top-level field {field}"
            );
        }
    }

    #[test]
    fn openai_chat_chunk_summary_keeps_only_compact_metadata() {
        let data = llm_chunk_mark_data(
            7,
            &json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "classified text",
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "{\"secret\":true}"}
                        }]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 3}
                }
            }),
        );

        assert_eq!(
            data,
            json!({
                "chunk_index": 7,
                "provider": "openai_chat_completions",
                "event_type": "chat.completion.chunk",
                "choice_indices": [0],
                "finish_reasons": [{"choice_index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "cache_read_tokens": 3
                }
            })
        );
        assert_no_top_level_fields(&data, &["choices", "delta", "content", "tool_calls"]);
        assert!(!data.to_string().contains("classified text"));
        assert!(!data.to_string().contains("secret"));
    }

    #[test]
    fn openai_responses_chunk_summary_omits_response_output_item_and_delta() {
        let delta_data = llm_chunk_mark_data(
            0,
            &json!({
                "type": "response.output_text.delta",
                "output_index": 1,
                "content_index": 2,
                "delta": "classified text"
            }),
        );
        assert_eq!(
            delta_data,
            json!({
                "chunk_index": 0,
                "provider": "openai_responses",
                "event_type": "response.output_text.delta",
                "indices": {"output_index": 1, "content_index": 2}
            })
        );
        assert_no_top_level_fields(&delta_data, &["response", "output", "item", "delta"]);
        assert!(!delta_data.to_string().contains("classified text"));

        let completed_data = llm_chunk_mark_data(
            1,
            &json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "classified text"}]
                    }],
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "total_tokens": 17,
                        "input_tokens_details": {"cached_tokens": 8}
                    }
                }
            }),
        );
        assert_eq!(
            completed_data,
            json!({
                "chunk_index": 1,
                "provider": "openai_responses",
                "event_type": "response.completed",
                "status": "completed",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                    "cache_read_tokens": 8
                }
            })
        );
        assert_no_top_level_fields(&completed_data, &["response", "output", "item", "delta"]);
        assert!(!completed_data.to_string().contains("classified text"));
    }

    #[test]
    fn anthropic_chunk_summary_omits_content_blocks_and_delta_payloads() {
        let delta_data = llm_chunk_mark_data(
            3,
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "classified text"},
                "delta": {"type": "input_json_delta", "partial_json": "{\"secret\":true}"}
            }),
        );
        assert_eq!(
            delta_data,
            json!({
                "chunk_index": 3,
                "provider": "anthropic_messages",
                "event_type": "content_block_delta",
                "indices": {"index": 0}
            })
        );
        assert_no_top_level_fields(
            &delta_data,
            &["content_block", "delta", "text", "partial_json"],
        );
        assert!(!delta_data.to_string().contains("classified text"));
        assert!(!delta_data.to_string().contains("secret"));

        let usage_data = llm_chunk_mark_data(
            4,
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 11,
                    "cache_creation_input_tokens": 2
                }
            }),
        );
        assert_eq!(
            usage_data,
            json!({
                "chunk_index": 4,
                "provider": "anthropic_messages",
                "event_type": "message_delta",
                "stop_reason": "end_turn",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "cache_read_tokens": 11,
                    "cache_write_tokens": 2
                }
            })
        );
    }

    #[test]
    fn unknown_chunk_summary_only_records_receipt() {
        let data = llm_chunk_mark_data(
            2,
            &json!({
                "delta": "classified text",
                "choices": [{"index": 0, "delta": {"content": "still unknown"}}],
                "response": {"output": [{"item": {"text": "secret"}}]}
            }),
        );

        assert_eq!(data, json!({"chunk_index": 2, "provider": "unknown"}));
    }
}
