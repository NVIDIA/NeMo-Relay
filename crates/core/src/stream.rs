// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response wrapper.
//!
//! This module provides [`LlmStreamWrapper`], a [`Stream`] adapter
//! that sits between the raw stream from an LLM API and the consumer. It
//! applies per-chunk interception, feeds intercepted chunks to a user-supplied
//! collector, and automatically emits lifecycle events when the stream ends.
//!
//! ## Pipeline
//!
//! ```text
//! raw chunk → StreamResponseIntercepts → collector(chunk) + yield chunk
//! stream ends → finalizer() → ResponseIntercepts → SanitizeResponseGuardrails → END event
//! ```
//!
//! The **collector** receives each intercepted chunk and can accumulate state
//! (e.g., concatenating tokens). The **finalizer** is called once when the
//! stream is exhausted and returns the aggregated response as [`Json`]. That
//! aggregated response then flows through response intercepts and sanitize
//! response guardrails before being included in the END event.

use std::pin::Pin;
use std::task::{Context, Poll};

use tokio_stream::Stream;

use crate::context::{current_scope_stack, global_context, ScopeStackHandle};
use crate::error::Result;
use crate::json::Json;
use crate::types::*;

/// Wraps an inner `Stream<Item = Result<String>>` of raw chunks and:
///
/// 1. Runs the stream response intercept chain on each chunk as a plain `String`.
/// 2. Passes each intercepted chunk to the user-supplied **collector** closure.
/// 3. On stream exhaustion, calls the **finalizer** to produce an aggregated
///    [`Json`] response, runs response intercepts and sanitize response
///    guardrails on it, then emits the LLM END event.
pub struct LlmStreamWrapper {
    inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
    handle: LLMHandle,
    scope_stack: ScopeStackHandle,
    collector: Box<dyn FnMut(String) + Send>,
    finalizer: Option<Box<dyn FnOnce() -> Json + Send>>,
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
    /// - `inner` — the raw stream of text chunks from the LLM provider.
    /// - `handle` — the [`LLMHandle`] for this call (used for the `End` event).
    /// - `collector` — called with each intercepted chunk; use this to accumulate
    ///   streaming tokens or forward them to another sink.
    /// - `finalizer` — called once when the stream is exhausted; must return the
    ///   aggregated response as [`Json`]. The returned value flows through
    ///   response intercepts and sanitize response guardrails.
    /// - `data` / `metadata` — optional values passed through to the `End` event.
    pub fn new(
        inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
        handle: LLMHandle,
        collector: Box<dyn FnMut(String) + Send>,
        finalizer: Box<dyn FnOnce() -> Json + Send>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self {
            inner,
            handle,
            scope_stack: current_scope_stack(),
            collector,
            finalizer: Some(finalizer),
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

    /// Emit the LLM END event with aggregated response data.
    ///
    /// Calls the finalizer to produce the aggregated response, then runs
    /// the response intercept chain on it. The result is passed to
    /// [`nvagentrt_llm_call_end`](crate::api::nvagentrt_llm_call_end)
    /// which applies sanitize response guardrails and emits the END event.
    fn emit_end_event(&mut self) {
        let aggregated = if let Some(finalizer) = self.finalizer.take() {
            finalizer()
        } else {
            Json::Null
        };

        // Run response intercepts on the aggregated response
        let response = if let Ok(mut ctx) = global_context().write() {
            ctx.llm_response_intercepts_chain(aggregated)
        } else {
            aggregated
        };

        // nvagentrt_llm_call_end applies sanitize response guardrails and emits the END event
        let _ = crate::api::nvagentrt_llm_call_end(
            &self.handle,
            response,
            self.data.clone(),
            self.metadata.clone(),
        );
    }
}

impl Stream for LlmStreamWrapper {
    type Item = Result<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.ended {
            return Poll::Ready(None);
        }

        // Poll the inner stream
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(raw_chunk))) => {
                // Run stream response intercept chain on the raw chunk
                let intercepted = if let Ok(mut ctx) = global_context().write() {
                    ctx.llm_stream_response_intercepts_chain(raw_chunk)
                } else {
                    raw_chunk
                };

                // Feed intercepted chunk to the collector
                (this.collector)(intercepted.clone());
                Poll::Ready(Some(Ok(intercepted)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                this.ended = true;
                this.emit_end_event();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
