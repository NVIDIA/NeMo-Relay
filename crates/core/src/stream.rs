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

use crate::api::llm::LlmHandle;
use crate::api::runtime::NemoFlowContextState;
use crate::api::runtime::global_context;
use crate::api::runtime::{ScopeStackHandle, current_scope_stack};
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::LlmResponseCodec;
use crate::error::Result;
use crate::json::Json;

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
    /// - `data`: Retained compatibility payload; ATOF end data is the finalized response.
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

impl Drop for LlmStreamWrapper {
    fn drop(&mut self) {
        self.finish();
    }
}
