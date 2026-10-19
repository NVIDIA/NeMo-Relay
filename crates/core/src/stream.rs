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
//! upstream error -> terminate stream with error -> finalizer() -> queue END sanitization
//! stream ends -> finalizer() -> queue END sanitization
//! ```
//!
//! The **collector** receives each chunk (Json) and can accumulate state
//! (e.g., concatenating tokens). If the collector returns `Err`, the stream
//! terminates immediately with that error. Upstream stream errors also
//! terminate the stream immediately. The **finalizer** is called once when the
//! stream terminates and returns the aggregated response as [`Json`]. That
//! aggregated response is queued for sanitize response guardrails before being
//! included in the END event. Stream termination awaits an asynchronous
//! finalizer, not that queued observability work.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use chrono::Utc;
use futures_util::FutureExt;
use tokio::sync::oneshot;
use tokio_stream::Stream;

use crate::api::event::{BaseEvent, MarkEvent};
use crate::api::llm::emit_reserved_optimization_marks;
use crate::api::llm::{EndLlmHandleParams, LlmHandle};
use crate::api::optimization::finalize_optimization_summary;
use crate::api::registry::{Guardrail, RuntimeRegistrationKind};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::callbacks::LlmStreamResponseCompletionFn;
use crate::api::runtime::global_context;
use crate::api::runtime::subscriber_dispatcher;
use crate::api::runtime::{
    EventSubscriberFn, LlmBorrowedCollectorFn, LlmCollector, LlmCollectorFuture, LlmFinalizer,
    LlmFinalizerFuture, LlmJsonStream, LlmStreamInner, ScopeStackHandle, TASK_SCOPE_STACK,
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
///    produce an aggregated [`Json`] response, then queues sanitize response
///    guardrails and LLM END event publication. Explicit close marks the end
///    event as interrupted and waits for producer cleanup.
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
    collector: LlmChunkCollector,
    finalizer: Option<LlmFinalizer>,
    response_completion: Option<LlmStreamResponseCompletionFn>,
    pending_collector: Option<(Json, LlmCollectorFuture)>,
    pending_end: Option<oneshot::Receiver<()>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    sanitize_context: LlmSanitizeResponseContext,
    metadata: Option<Json>,
    subscribers: Vec<EventSubscriberFn>,
    stream_started_at: Instant,
    first_chunk_elapsed: Option<Duration>,
    chunk_index: u64,
    ended: bool,
    close_result: Option<Result<()>>,
    undelivered_error: Option<FlowError>,
}

pub(crate) enum LlmChunkCollector {
    Owned(LlmCollector),
    Borrowed(LlmBorrowedCollectorFn),
}

pub(crate) struct ManagedLlmStreamTelemetry {
    response_completion: Option<LlmStreamResponseCompletionFn>,
    subscribers: Vec<EventSubscriberFn>,
    stream_started_at: Instant,
}

impl ManagedLlmStreamTelemetry {
    pub(crate) fn new(subscribers: Vec<EventSubscriberFn>, stream_started_at: Instant) -> Self {
        Self {
            subscribers,
            stream_started_at,
            response_completion: None,
        }
    }

    pub(crate) fn with_response_completion(
        mut self,
        completion: Option<LlmStreamResponseCompletionFn>,
    ) -> Self {
        self.response_completion = completion;
        self
    }
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

enum EndPublication {
    Reserved {
        aggregate: Result<Json>,
        barrier: Option<subscriber_dispatcher::AsyncPublication>,
    },
    Awaiting {
        finalizer: LlmFinalizerFuture,
        finalized: oneshot::Sender<()>,
    },
}

impl EndPublication {
    fn reserve(aggregate: Result<Json>) -> Self {
        // Register a dispatcher barrier before spawning finalization so a synchronous
        // subscriber flush after this stream is dropped cannot overtake the END event.
        let barrier = subscriber_dispatcher::register_async_publication();
        Self::Reserved { aggregate, barrier }
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
        collector: impl Into<LlmCollector>,
        finalizer: impl Into<LlmFinalizer>,
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
                .map(|state| state.registry_snapshot(&[RuntimeRegistrationKind::Subscriber]))
                .map(|state| state.collect_event_subscribers(&scope_subscribers))
                .unwrap_or_default()
        };
        Self::new_managed(
            inner,
            handle,
            LlmChunkCollector::Owned(collector.into()),
            finalizer.into(),
            metadata,
            response_codec,
            ManagedLlmStreamTelemetry::new(subscribers, Instant::now()),
        )
    }

    pub(crate) fn new_managed(
        inner: LlmJsonStream,
        handle: LlmHandle,
        collector: LlmChunkCollector,
        finalizer: LlmFinalizer,
        metadata: Option<Json>,
        response_codec: Option<Arc<dyn LlmResponseCodec>>,
        telemetry: ManagedLlmStreamTelemetry,
    ) -> Self {
        let ManagedLlmStreamTelemetry {
            subscribers,
            stream_started_at,
            response_completion,
        } = telemetry;
        let scope_stack = handle.captured_scope_stack().clone();
        let sanitize_context =
            LlmSanitizeResponseContext::for_response_codec(response_codec.clone());
        Self {
            inner,
            handle,
            scope_stack,
            collector,
            finalizer: Some(finalizer),
            pending_collector: None,
            pending_end: None,
            response_codec,
            sanitize_context,
            metadata,
            subscribers,
            stream_started_at,
            response_completion,
            first_chunk_elapsed: None,
            chunk_index: 0,
            ended: false,
            close_result: None,
            undelivered_error: None,
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
        self.emit_end_event(metadata, StreamTermination::Dropped);
    }

    fn finish_cleanly(&mut self) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.inner.terminalize();
        self.handle
            .optimization_recorder
            .close_for_finalization(None);
        let metadata = metadata_with_otel_status(self.metadata.clone(), "OK", None);
        self.emit_end_event(metadata, StreamTermination::Complete);
    }

    fn finish_with_error(&mut self, error: &FlowError) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.inner.terminalize();
        self.handle
            .optimization_recorder
            .close_for_finalization(None);
        let metadata = metadata_with_otel_error(self.metadata.clone(), error);
        self.emit_end_event(metadata, StreamTermination::Failed);
    }

    /// Emit the LLM END event with aggregated response data.
    ///
    /// Calls the finalizer and queues response sanitization and END publication.
    fn emit_end_event(&mut self, metadata: Option<Json>, termination: StreamTermination) {
        let timestamp = Utc::now();
        let (publication, pending_publication) = match self.finalizer.take() {
            Some(LlmFinalizer::Async(finalizer)) => {
                // A pending permit makes flushes wait for END without parking the dispatcher.
                let finalizer = finalizer();
                let (finalized, pending_end) = oneshot::channel();
                self.pending_end = Some(pending_end);
                let pending_publication = (!self.subscribers.is_empty())
                    .then(subscriber_dispatcher::register_pending_publication)
                    .flatten();
                (
                    EndPublication::Awaiting {
                        finalizer,
                        finalized,
                    },
                    pending_publication,
                )
            }
            // Runs before the END barrier exists, so a flush inside a synchronous finalizer completes.
            Some(LlmFinalizer::Sync(finalizer)) => (EndPublication::reserve(Ok(finalizer())), None),
            Some(LlmFinalizer::Fallible(finalizer)) => (EndPublication::reserve(finalizer()), None),
            None => (EndPublication::reserve(Ok(Json::Null)), None),
        };

        let (entries, sanitizer_snapshot_failed) =
            snapshot_stream_end_sanitizers(&self.scope_stack);
        let handle = self.handle.clone();
        let scope_stack = self.scope_stack.clone();
        let finalization_scope_stack = scope_stack.clone();
        let subscribers = self.subscribers.clone();
        let response_codec = self.response_codec.clone();
        let sanitize_context = self.sanitize_context.clone();
        let time_to_first_chunk = self
            .first_chunk_elapsed
            .map(|elapsed| elapsed.as_secs_f64());
        let response_completion = self.response_completion.take();
        let publish = move |finalizer_result: Result<Json>| async move {
            let (aggregated, finalizer_failure) = match finalizer_result {
                Ok(aggregated) => (aggregated, None),
                Err(error) => {
                    log::warn!(
                        target: "nemo_relay.runtime",
                        event = "stream_finalizer_failed";
                        "LLM stream finalizer failed; omitting the aggregated response: {error}"
                    );
                    (Json::Null, Some(error))
                }
            };
            if let Some(completion) = response_completion {
                let outcome = AssertUnwindSafe(async { completion(&aggregated).await })
                    .catch_unwind()
                    .await;
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => log::error!(
                        target: "nemo_relay.runtime",
                        event = "stream_response_completion_failed";
                        "Streaming response completion failed: {error}"
                    ),
                    Err(_) => log::error!(
                        target: "nemo_relay.runtime",
                        event = "stream_response_completion_panicked";
                        "Streaming response completion panicked"
                    ),
                }
            }
            let finalizer_failed = finalizer_failure.is_some();
            let metadata = match (&finalizer_failure, termination) {
                (Some(error), StreamTermination::Complete) => {
                    metadata_with_otel_error(metadata, error)
                }
                _ => metadata,
            };
            let response_was_null_without_fallback = aggregated.is_null() && handle.data.is_none();
            let response = if aggregated.is_null() && !finalizer_failed {
                handle.data.clone().unwrap_or(aggregated)
            } else {
                aggregated
            };
            let sanitized = (!sanitizer_snapshot_failed && !finalizer_failed).then(|| {
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
                            if codec.allows_estimated_cost(response) {
                                attach_estimated_cost_for_provider(
                                    &mut decoded,
                                    Some(&handle.name),
                                );
                            }
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
                    Ok(state) => Some(
                        state.build_llm_end_event(
                            EndLlmHandleParams::builder()
                                .handle(&handle)
                                .data_opt(data)
                                .metadata_opt(metadata)
                                .annotated_response_opt(annotated_response)
                                .time_to_first_chunk_opt(time_to_first_chunk)
                                .timestamp(timestamp)
                                .build(),
                        ),
                    ),
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
        let finalize = async move {
            let (aggregate, barrier) = match publication {
                EndPublication::Reserved { aggregate, barrier } => (aggregate, barrier),
                EndPublication::Awaiting {
                    finalizer,
                    finalized,
                } => {
                    let aggregate = finalizer.await;
                    // Reserve the END position before the consumer observes the end.
                    let barrier = subscriber_dispatcher::register_async_publication();
                    let _ = finalized.send(());
                    (aggregate, barrier)
                }
            };
            subscriber_dispatcher::with_async_publication_context(barrier, publish(aggregate))
                .await;
        };
        let finalize = TASK_SCOPE_STACK.scope(finalization_scope_stack, finalize);
        let publication_context = subscriber_dispatcher::capture_publication_context();
        let finalize = subscriber_dispatcher::with_task_publication_context(
            publication_context,
            subscriber_dispatcher::with_pending_publication(pending_publication, finalize),
        );
        // Stream finalization is observability-only. Queue it on the shared
        // publication executor so stream termination does not await response
        // or event sanitizers. The reserved barrier or pending permit keeps
        // subscriber flushes ordered behind this END event.
        let _ = subscriber_dispatcher::spawn_background_publication(finalize);
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
    let scope_locals = scope_stack.read().ok().map(|scope_guard| {
        scope_guard
            .snapshot_scope_local_registries(|registry| &registry.llm_sanitize_response_guardrails)
    });
    let entries = scope_locals.and_then(|scope_locals| {
        let scope_local_refs = scope_locals.iter().collect::<Vec<_>>();
        global_context()
            .read()
            .ok()
            .map(|state| {
                state.registry_snapshot(&[RuntimeRegistrationKind::LlmSanitizeResponseGuardrail])
            })
            .map(|state| state.llm_sanitize_response_entries(&scope_local_refs))
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

        if let Some((chunk, mut collecting)) = this.pending_collector.take() {
            match collecting.as_mut().poll(cx) {
                Poll::Pending => {
                    this.pending_collector = Some((chunk, collecting));
                    return Poll::Pending;
                }
                Poll::Ready(Ok(())) => return Poll::Ready(Some(Ok(chunk))),
                Poll::Ready(Err(e)) => {
                    this.finish_with_error(&e);
                    this.undelivered_error = Some(e);
                    return self.poll_next(cx);
                }
            }
        }

        if this.ended {
            if let Some(pending_end) = this.pending_end.as_mut() {
                if Pin::new(pending_end).poll(cx).is_pending() {
                    return Poll::Pending;
                }
                this.pending_end = None;
            }
            return match this.undelivered_error.take() {
                Some(error) => Poll::Ready(Some(Err(error))),
                None => Poll::Ready(None),
            };
        }

        // Poll the inner stream
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(raw_chunk))) => {
                this.first_chunk_elapsed
                    .get_or_insert_with(|| this.stream_started_at.elapsed());
                let chunk_index = this.chunk_index;
                this.chunk_index += 1;
                this.emit_chunk_mark(chunk_index, &raw_chunk);
                // Feed chunk to the collector; if it returns Err, terminate the stream
                match &mut this.collector {
                    LlmChunkCollector::Borrowed(collector) => match collector(&raw_chunk) {
                        Ok(()) => Poll::Ready(Some(Ok(raw_chunk))),
                        Err(error) => {
                            this.finish_with_error(&error);
                            this.undelivered_error = Some(error);
                            self.poll_next(cx)
                        }
                    },
                    LlmChunkCollector::Owned(LlmCollector::Sync(collector)) => {
                        match collector(raw_chunk.clone()) {
                            Ok(()) => Poll::Ready(Some(Ok(raw_chunk))),
                            Err(e) => {
                                this.finish_with_error(&e);
                                this.undelivered_error = Some(e);
                                self.poll_next(cx)
                            }
                        }
                    }
                    LlmChunkCollector::Owned(LlmCollector::Async(collector)) => {
                        let collecting = collector(raw_chunk.clone());
                        this.pending_collector = Some((raw_chunk, collecting));
                        self.poll_next(cx)
                    }
                }
            }
            Poll::Ready(Some(Err(e))) => {
                this.finish_with_error(&e);
                this.undelivered_error = Some(e);
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
            let cleanup = this.inner.close().await;
            // Close the producer first so a collector waiting on its abort can settle.
            if let Some((_, collecting)) = this.pending_collector.take()
                && let Err(error) = collecting.await
            {
                this.finish_with_error(&error);
                this.undelivered_error = Some(error);
            }
            this.finish();
            if let Some(pending_end) = this.pending_end.take() {
                let _ = pending_end.await;
            }
            let result = match (this.undelivered_error.take(), cleanup) {
                (Some(error), cleanup) => {
                    if let Err(cleanup) = cleanup {
                        log::warn!(
                            target: "nemo_relay.runtime",
                            event = "stream_cleanup_failed";
                            "LLM stream cleanup failed after the stream failed: {cleanup}"
                        );
                    }
                    Err(error)
                }
                (None, cleanup) => cleanup,
            };
            this.close_result = Some(result.clone());
            result
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
        self.finish();
    }
}

#[cfg(test)]
#[path = "../tests/unit/stream_tests.rs"]
mod tests;
