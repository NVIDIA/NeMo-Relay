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

use crate::codec::{AnnotatedLLMResponse, LlmResponseCodec};
use crate::context::{NemoFlowContextState, ScopeStackHandle, current_scope_stack, global_context};
use crate::error::Result;
use crate::json::Json;
use crate::types::*;

/// Wraps an inner `Stream<Item = Result<Json>>` of raw chunks and:
///
/// 1. Passes each chunk to the user-supplied **collector** closure.
///    If the collector returns `Err`, the stream terminates with that error.
/// 2. On stream exhaustion, calls the **finalizer** to produce an aggregated
///    [`Json`] response, runs sanitize response guardrails on it, then emits
///    the LLM END event.
pub struct LlmStreamWrapper {
    inner: Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
    handle: LLMHandle,
    scope_stack: ScopeStackHandle,
    collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
    finalizer: Option<Box<dyn FnOnce() -> Json + Send>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    data: Option<Json>,
    metadata: Option<Json>,
    ended: bool,
}

impl LlmStreamWrapper {
    /// Creates a new `LlmStreamWrapper` around the given raw stream.
    ///
    /// Captures the current [`ScopeStackHandle`] at creation time so the
    /// correct scope stack is used when the stream is later polled, even if
    /// polling happens on a different task or thread.
    ///
    /// - `inner` -- the raw stream of Json chunks from the LLM provider.
    /// - `handle` -- the [`LLMHandle`] for this call (used for the `End` event).
    /// - `collector` -- called with each chunk; use this to accumulate
    ///   streaming tokens or forward them to another sink. Return `Ok(())`
    ///   to continue the stream, or `Err(FlowError)` to terminate it.
    /// - `finalizer` -- called once when the stream is exhausted; must return the
    ///   aggregated response as [`Json`]. The returned value flows through
    ///   sanitize response guardrails.
    /// - `data` / `metadata` -- optional values passed through to the `End` event.
    pub fn new(
        inner: Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
        handle: LLMHandle,
        collector: Box<dyn FnMut(Json) -> Result<()> + Send>,
        finalizer: Box<dyn FnOnce() -> Json + Send>,
        data: Option<Json>,
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
            data,
            metadata,
            ended: false,
        }
    }

    /// Returns the captured scope stack handle for this stream.
    ///
    /// Callers can use this to bind the correct scope stack when spawning
    /// the stream on a different task via `TASK_SCOPE_STACK.scope(...)`.
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
        let annotated_response: Option<Arc<AnnotatedLLMResponse>> = self
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
                    let event = state.end_llm_handle(
                        &self.handle,
                        self.data.clone(),
                        self.metadata.clone(),
                        Some(sanitized),
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
