//! Streaming LLM response wrapper.
//!
//! This module provides [`LlmStreamWrapper`], a [`Stream`] adapter
//! that sits between the raw SSE byte stream from an LLM API and the consumer. It
//! handles SSE parsing, per-event interception, event aggregation, and automatic
//! lifecycle event emission.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio_stream::Stream;

use crate::context::{current_scope_stack, global_context, ScopeStackHandle};
use crate::error::Result;
use crate::json::Json;
use crate::types::*;

/// Wraps an inner `Stream<Item = Result<String>>` of raw SSE text and:
/// 1. Buffers incoming raw text, splits on `\n\n` boundaries
/// 2. Parses each complete event into `SseEvent`
/// 3. Runs the stream response intercept chain on each `SseEvent`
/// 4. Serializes the intercepted `SseEvent` back to raw SSE text
/// 5. Collects all events for aggregation (for the END event)
/// 6. On stream exhaustion, emits the LLM END event with aggregated response
pub struct LlmStreamWrapper {
    inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
    handle: LLMHandle,
    scope_stack: ScopeStackHandle,
    buffer: String,
    collected: Vec<SseEvent>,
    pending: VecDeque<String>,
    data: Option<Json>,
    metadata: Option<Json>,
    ended: bool,
}

impl LlmStreamWrapper {
    /// Creates a new `LlmStreamWrapper` around the given raw SSE stream.
    ///
    /// Captures the current [`ScopeStackHandle`] at creation time so the
    /// correct scope stack is used when the stream is later polled, even if
    /// polling happens on a different task or thread.
    ///
    /// - `inner` — the raw stream of SSE text chunks from the LLM provider.
    /// - `handle` — the [`LLMHandle`] for this call (used for the `End` event).
    /// - `data` / `metadata` — optional values passed through to the `End` event.
    pub fn new(
        inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
        handle: LLMHandle,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self {
            inner,
            handle,
            scope_stack: current_scope_stack(),
            buffer: String::new(),
            collected: Vec::new(),
            pending: VecDeque::new(),
            data,
            metadata,
            ended: false,
        }
    }

    /// Extract complete SSE events from buffer, leaving any partial trailing data.
    fn drain_buffer(&mut self) {
        while let Some(pos) = self.buffer.find("\n\n") {
            let block = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            if block.trim().is_empty() {
                continue;
            }

            let mut event = SseEvent::parse(&block);

            // Run stream response intercept chain
            if let Ok(mut ctx) = global_context().write() {
                event = ctx.llm_stream_response_intercepts_chain(event);
            }

            self.collected.push(event.clone());
            self.pending.push_back(event.to_sse_string());
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
    fn emit_end_event(&self) {
        let aggregated: Vec<Json> = self
            .collected
            .iter()
            .filter_map(|e| serde_json::from_str(&e.data).ok())
            .collect();
        let response = Json::Array(aggregated);
        // Ignoring errors here — best-effort end event emission
        let _ = crate::api::nv_agentrt_llm_call_end(
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

        // If we have pending processed events, yield them first
        if let Some(text) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(text)));
        }

        if this.ended {
            return Poll::Ready(None);
        }

        // Poll the inner stream
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(raw_text))) => {
                this.buffer.push_str(&raw_text);
                this.drain_buffer();

                if let Some(text) = this.pending.pop_front() {
                    Poll::Ready(Some(Ok(text)))
                } else {
                    // Got data but no complete event yet, wake and retry
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                // Inner stream exhausted — drain any remaining buffer
                if !this.buffer.is_empty() {
                    // Force a final boundary if buffer has content
                    this.buffer.push_str("\n\n");
                    this.drain_buffer();
                }

                this.ended = true;
                this.emit_end_event();

                // Yield any final pending events
                if let Some(text) = this.pending.pop_front() {
                    Poll::Ready(Some(Ok(text)))
                } else {
                    Poll::Ready(None)
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
