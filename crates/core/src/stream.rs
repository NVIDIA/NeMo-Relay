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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio_stream::Stream;

use crate::api::event::{BaseEvent, MarkEvent};
use crate::api::llm::LlmHandle;
use crate::api::llm::emit_reserved_optimization_marks;
use crate::api::optimization::finalize_optimization_summary;
use crate::api::registry::Guardrail;
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::runtime::subscriber_dispatcher;
use crate::api::runtime::{
    EventSubscriberFn, LlmJsonStream, LlmStreamInner, ScopeStackHandle, TASK_SCOPE_STACK,
    current_scope_stack,
};
use crate::api::runtime::{LlmSanitizeResponseContext, LlmSanitizeResponseFn};
use crate::api::shared::{
    metadata_with_otel_error, metadata_with_otel_status, snapshot_event_sanitizers,
};
use crate::codec::response::{
    AnnotatedLlmResponse, FinishReason, attach_estimated_cost_for_provider,
};
use crate::codec::traits::LlmResponseCodec;
use crate::error::{FlowError, Result};
use crate::json::Json;
use serde_json::Map;

/// Wraps an inner `Stream<Item = Result<Json>>` of raw chunks and:
///
/// 1. Passes each chunk to the user-supplied **collector** closure.
///    If the collector returns `Err`, the stream terminates with that error.
/// 2. On stream exhaustion or explicit close, calls the **finalizer** to
///    produce an aggregated [`Json`] response, runs sanitize response
///    guardrails on it, then emits the LLM END event. Explicit close marks the
///    end event as interrupted and waits for producer cleanup.
///
/// This type is returned by [`crate::api::llm::llm_stream_call_execute`] and
/// is usually consumed as an ordinary async stream. Consumers that stop early
/// should call [`LlmJsonStream::close`] to perform deterministic cleanup. The
/// wrapper preserves the originating scope stack so end-of-stream bookkeeping
/// still uses the correct scope-local middleware and subscribers even when
/// polling happens elsewhere.
pub struct LlmStreamWrapper {
    inner: LlmJsonStream,
    handle: LlmHandle,
    scope_stack: ScopeStackHandle,
    collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
    finalizer: Option<Box<dyn FnOnce() -> Json + Send>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    sanitize_context: LlmSanitizeResponseContext,
    metadata: Option<Json>,
    subscribers: Vec<EventSubscriberFn>,
    chunk_index: u64,
    ended: bool,
    close_result: Option<Result<()>>,
    finalization: Option<tokio::task::JoinHandle<()>>,
    terminal_result: Option<Result<Json>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamTermination {
    Complete,
    Failed,
    Dropped,
}

impl StreamTermination {
    const fn is_interrupted(self) -> bool {
        !matches!(self, Self::Complete)
    }
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
        inner: LlmJsonStream,
        handle: LlmHandle,
        collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
        finalizer: Box<dyn FnOnce() -> Json + Send>,
        _data: Option<Json>,
        metadata: Option<Json>,
        response_codec: Option<Arc<dyn LlmResponseCodec>>,
    ) -> Self {
        let subscribers = {
            let scope_stack = current_scope_stack();
            let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
            let scope_subscribers = scope_guard.collect_scope_local_subscribers();
            let context = global_context();
            context
                .read()
                .map(|state| state.collect_event_subscribers(&scope_subscribers))
                .unwrap_or_default()
        };
        Self::new_managed(
            inner,
            handle,
            collector,
            finalizer,
            metadata,
            response_codec,
            subscribers,
        )
    }

    pub(crate) fn new_managed(
        inner: LlmJsonStream,
        handle: LlmHandle,
        collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
        finalizer: Box<dyn FnOnce() -> Json + Send>,
        metadata: Option<Json>,
        response_codec: Option<Arc<dyn LlmResponseCodec>>,
        subscribers: Vec<EventSubscriberFn>,
    ) -> Self {
        let scope_stack = handle.captured_scope_stack().clone();
        let sanitize_context =
            LlmSanitizeResponseContext::for_response_codec(response_codec.clone());
        Self {
            inner,
            handle,
            scope_stack,
            collector,
            finalizer: Some(finalizer),
            response_codec,
            sanitize_context,
            metadata,
            subscribers,
            chunk_index: 0,
            ended: false,
            close_result: None,
            finalization: None,
            terminal_result: None,
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

    fn finish(&mut self, background_thread: bool) {
        if self.ended {
            return;
        }
        self.ended = true;
        let metadata = metadata_with_otel_status(
            self.metadata.clone(),
            "ERROR",
            Some("stream dropped before clean completion".to_string()),
        );
        // Drop cannot await the async finalizer. Seal contribution acceptance
        // immediately, but let the finalizer decide whether authoritative
        // terminal usage means the stream should be marked interrupted.
        self.handle
            .optimization_recorder
            .close_for_finalization(None);
        self.finalization =
            self.emit_end_event(metadata, StreamTermination::Dropped, background_thread);
    }

    fn finish_cleanly(&mut self) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.inner.terminalize();
        let metadata = metadata_with_otel_status(self.metadata.clone(), "OK", None);
        self.finalization = self.emit_end_event(metadata, StreamTermination::Complete, false);
    }

    fn finish_with_error(&mut self, error: &FlowError) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.inner.terminalize();
        let metadata = metadata_with_otel_error(self.metadata.clone(), error);
        self.finalization = self.emit_end_event(metadata, StreamTermination::Failed, false);
    }

    /// Emit the LLM END event with aggregated response data.
    ///
    /// Calls the finalizer to produce the aggregated response, runs sanitize
    /// response guardrails, and emits the END event.
    fn emit_end_event(
        &mut self,
        metadata: Option<Json>,
        termination: StreamTermination,
        background_thread: bool,
    ) -> Option<tokio::task::JoinHandle<()>> {
        // The finalizer below runs on the caller's Tokio runtime. Register a
        // dispatcher barrier before spawning it so a synchronous subscriber
        // flush after this stream is dropped cannot overtake the END event.
        let publication_barrier = subscriber_dispatcher::register_async_publication();
        let aggregated = match self.finalizer.take() {
            Some(finalizer) => finalizer(),
            None => Json::Null,
        };
        let response_was_null_without_fallback = aggregated.is_null() && self.handle.data.is_none();
        let response = if aggregated.is_null() {
            self.handle.data.clone().unwrap_or(aggregated)
        } else {
            aggregated
        };

        let (entries, sanitizer_snapshot_failed) =
            snapshot_stream_end_sanitizers(&self.scope_stack);
        let handle = self.handle.clone();
        let scope_stack = self.scope_stack.clone();
        let finalization_scope_stack = scope_stack.clone();
        let subscribers = self.subscribers.clone();
        let response_codec = self.response_codec.clone();
        let sanitize_context = self.sanitize_context.clone();
        let finalize = async move {
            let sanitized = (!sanitizer_snapshot_failed).then(|| {
                NemoRelayContextState::llm_sanitize_response_snapshot_chain(
                    response,
                    sanitize_context,
                    &entries,
                )
            });
            let sanitized = match sanitized {
                Some(sanitized) => sanitized.await,
                None => None,
            };
            let data = match sanitized {
                Some(response) if response_was_null_without_fallback && response.is_null() => None,
                response => response,
            };
            let annotation_omitted = data.as_ref().is_none_or(Json::is_null);
            let mut annotated_response: Option<AnnotatedLlmResponse> = (!annotation_omitted)
                .then(|| {
                    data.as_ref().and_then(|response| {
                        response_codec.as_ref().and_then(|codec| {
                            let mut decoded = codec.decode_response(response).ok()?;
                            attach_estimated_cost_for_provider(&mut decoded, Some(&handle.name));
                            Some(decoded)
                        })
                    })
                })
                .flatten();
            let metadata = if termination == StreamTermination::Dropped
                && has_authoritative_terminal_outcome(annotated_response.as_ref())
            {
                metadata_with_otel_status(metadata, "OK", None)
            } else {
                metadata
            };
            let interruption = (termination.is_interrupted()
                && !has_authoritative_final_usage(annotated_response.as_ref()))
            .then_some("stream_interrupted");
            handle
                .optimization_recorder
                .close_for_finalization(interruption);
            emit_reserved_optimization_marks(&handle, &subscribers).await;
            let pricing = crate::codec::response::active_pricing_resolver();
            let summary = finalize_optimization_summary(
                &handle.optimization_recorder,
                annotated_response.as_mut(),
                handle.model_name.as_deref(),
                &pricing,
            );
            if !annotation_omitted
                && annotated_response.is_none()
                && let Some(summary) = summary
            {
                annotated_response = Some(AnnotatedLlmResponse {
                    optimization_summary: Some(summary),
                    ..AnnotatedLlmResponse::default()
                });
            }
            let annotated_response = annotated_response.map(Arc::new);
            let event_snapshot = {
                let ctx = global_context();
                let state = ctx.read();
                match state {
                    Ok(state) => {
                        Some(state.end_llm_handle(&handle, data, metadata, annotated_response))
                    }
                    Err(_) => None,
                }
            };
            if let Some(event) = event_snapshot {
                let sanitizers =
                    snapshot_event_sanitizers(&event, &scope_stack).unwrap_or_default();
                let _ = subscriber_dispatcher::dispatch_reserved_sanitized_event(
                    event,
                    sanitizers,
                    &subscribers,
                    scope_stack.clone(),
                );
            }
        };
        let finalize = TASK_SCOPE_STACK.scope(finalization_scope_stack, finalize);
        let publication_context = subscriber_dispatcher::capture_publication_context();
        let finalize = subscriber_dispatcher::with_task_publication_context(
            publication_context,
            subscriber_dispatcher::with_async_publication_context(publication_barrier, finalize),
        );
        if background_thread {
            // `Drop` cannot await middleware and may run while the caller's
            // executor is synchronously flushing subscribers. A process-local
            // executor polls all detached finalizers on one shared OS thread.
            // Pending middleware therefore does not create one thread per
            // abandoned stream.
            let _ = subscriber_dispatcher::spawn_background_publication(finalize);
            return None;
        }
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => Some(handle.spawn(finalize)),
            Err(_) => {
                let _ = subscriber_dispatcher::spawn_background_publication(finalize);
                None
            }
        }
    }

    /// Emit a compact per-chunk receipt mark before collector processing.
    fn emit_chunk_mark(&self, chunk_index: u64, raw_chunk: &Json) {
        let data = llm_chunk_mark_data(chunk_index, raw_chunk);
        let event_snapshot = {
            let ctx = global_context();
            let state = ctx.read();
            match state {
                Ok(state) => {
                    let event = state.create_event(MarkEvent::new(
                        BaseEvent::builder()
                            .name("llm.chunk")
                            .parent_uuid(self.handle.uuid)
                            .data(data)
                            .build(),
                        None,
                        None,
                    ));
                    Some(event)
                }
                Err(_) => None,
            }
        };
        if let Some(event) = event_snapshot {
            let sanitizers =
                snapshot_event_sanitizers(&event, &self.scope_stack).unwrap_or_default();
            let _ = subscriber_dispatcher::dispatch_sanitized_event(
                event,
                sanitizers,
                &self.subscribers,
                self.scope_stack.clone(),
            );
        }
    }
}

fn snapshot_stream_end_sanitizers(
    scope_stack: &ScopeStackHandle,
) -> (Vec<Guardrail<LlmSanitizeResponseFn>>, bool) {
    let entries = scope_stack.read().ok().and_then(|scope_guard| {
        let scope_locals = scope_guard
            .collect_scope_local_registries(|registry| &registry.llm_sanitize_response_guardrails);
        global_context()
            .read()
            .ok()
            .map(|state| state.llm_sanitize_response_entries(&scope_locals))
    });
    match entries {
        Some(entries) => (entries, false),
        None => {
            log::error!(
                target: "nemo_relay.runtime",
                event = "stream_end_sanitizer_snapshot_failed";
                "LLM stream END sanitizer snapshot failed; omitting the observability payload"
            );
            (Vec::new(), true)
        }
    }
}

impl Stream for LlmStreamWrapper {
    type Item = Result<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();

        // The END event runs async because response and event sanitizers may
        // await. Do not expose stream termination until that work has queued
        // the event: callers commonly flush subscribers immediately after
        // exhausting a stream, and that flush must include its END event.
        if let Some(finalization) = this.finalization.as_mut() {
            return match Pin::new(finalization).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Ok(())) => {
                    this.finalization = None;
                    match this.terminal_result.take() {
                        Some(result) => Poll::Ready(Some(result)),
                        None => Poll::Ready(None),
                    }
                }
                Poll::Ready(Err(error)) => {
                    this.finalization = None;
                    Poll::Ready(Some(Err(FlowError::Internal(format!(
                        "stream finalization task failed: {error}"
                    )))))
                }
            };
        }

        if this.ended {
            return match this.terminal_result.take() {
                Some(result) => Poll::Ready(Some(result)),
                None => Poll::Ready(None),
            };
        }

        // Poll the inner stream
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(raw_chunk))) => {
                let chunk_index = this.chunk_index;
                this.chunk_index += 1;
                this.emit_chunk_mark(chunk_index, &raw_chunk);
                // Feed chunk to the collector; if it returns Err, terminate the stream
                match (this.collector)(raw_chunk.clone()) {
                    Ok(()) => Poll::Ready(Some(Ok(raw_chunk))),
                    Err(e) => {
                        this.finish_with_error(&e);
                        this.terminal_result = Some(Err(e));
                        self.poll_next(cx)
                    }
                }
            }
            Poll::Ready(Some(Err(e))) => {
                this.finish_with_error(&e);
                this.terminal_result = Some(Err(e));
                self.poll_next(cx)
            }
            Poll::Ready(None) => {
                this.finish_cleanly();
                self.poll_next(cx)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl LlmStreamInner for LlmStreamWrapper {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        Box::pin(async move {
            if let Some(result) = &this.close_result {
                return result.clone();
            }
            let result = this.inner.close().await;
            this.finish(false);
            if let Some(finalization) = this.finalization.take() {
                finalization.await.map_err(|error| {
                    FlowError::Internal(format!("stream finalization task failed: {error}"))
                })?;
            }
            this.close_result = Some(result.clone());
            this.close_result
                .as_ref()
                .expect("close result was just stored")
                .clone()
        })
    }
}

fn has_authoritative_final_usage(response: Option<&AnnotatedLlmResponse>) -> bool {
    response.is_some_and(|response| {
        response.finish_reason.is_some()
            && response.usage.as_ref().is_some_and(|usage| {
                usage.total_tokens.is_some()
                    || (usage.prompt_tokens.is_some() && usage.completion_tokens.is_some())
            })
    })
}

fn has_authoritative_terminal_outcome(response: Option<&AnnotatedLlmResponse>) -> bool {
    has_authoritative_final_usage(response)
        && response.is_some_and(|response| {
            response
                .finish_reason
                .as_ref()
                .is_some_and(|reason| !matches!(reason, FinishReason::Unknown(_)))
        })
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
        self.finish(true);
    }
}

#[cfg(test)]
#[path = "../tests/unit/stream_tests.rs"]
mod tests;
