// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for stream in the NeMo Relay core crate.

#![allow(clippy::await_holding_lock)]

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{LlmAttributes, LlmHandle, LlmRequest};
use nemo_relay::api::llm::{LlmCallParams, llm_call};
use nemo_relay::api::optimization::LlmOptimizationRecorder;
use nemo_relay::api::registry::{
    deregister_llm_sanitize_response_guardrail, register_llm_sanitize_response_guardrail,
};
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::subscriber_dispatcher::flush_subscribers_with_started_signal;
use nemo_relay::api::runtime::{
    LlmCollector, LlmFinalizer, LlmJsonStream, LlmStreamInner, NemoRelayContextState,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_responses::{OpenAIResponsesCodec, OpenAIResponsesStreamingCodec};
use nemo_relay::codec::optimization::LlmOptimizationContribution;
use nemo_relay::codec::response::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay::codec::streaming::StreamingCodec;
use nemo_relay::error::FlowError;
use nemo_relay::error::Result;
use nemo_relay::json::Json;
use nemo_relay::stream::LlmStreamWrapper;
use serde_json::json;
use tokio_stream::{Stream, StreamExt};

// Serialize all tests since they share global state
static TEST_MUTEX: Mutex<()> = Mutex::new(());

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn is_llm_end(event: &Event) -> bool {
    event.scope_type() == Some(nemo_relay::api::scope::ScopeType::Llm)
        && event.scope_category() == Some(ScopeCategory::End)
}

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn make_llm_handle(name: &str) -> LlmHandle {
    LlmHandle::builder()
        .name(name.to_string())
        .attributes(LlmAttributes::STREAMING)
        .build()
}

fn make_optimized_llm_handle(name: &str, producer: &str) -> (LlmHandle, LlmOptimizationRecorder) {
    let handle = make_llm_handle(name);
    let recorder = handle.optimization_recorder.clone();
    assert!(recorder.record(LlmOptimizationContribution::new(producer, "stream_test")));
    (handle, recorder)
}

fn make_stream(items: Vec<Result<Json>>) -> LlmJsonStream {
    LlmJsonStream::new(tokio_stream::iter(items))
}

fn install_anthropic_fallback_pricing() {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [{
                "provider": "anthropic",
                "model_id": "claude-fallback",
                "pricing_as_of": "2026-09-25",
                "pricing_source": "test",
                "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
                "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
            }]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

struct CloseTrackingStream {
    stream: Pin<Box<dyn Stream<Item = Result<Json>> + Send>>,
    close_calls: Arc<AtomicUsize>,
    close_error: Option<FlowError>,
    closed: bool,
}

impl Stream for CloseTrackingStream {
    type Item = Result<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.closed {
            Poll::Ready(None)
        } else {
            self.stream.as_mut().poll_next(cx)
        }
    }
}

impl LlmStreamInner for CloseTrackingStream {
    fn close(mut self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        self.closed = true;
        self.close_calls.fetch_add(1, Ordering::SeqCst);
        let close_error = self.close_error.clone();
        Box::pin(async move { close_error.map_or(Ok(()), Err) })
    }
}

struct DropTrackingStream {
    drops: Arc<AtomicUsize>,
}

impl Stream for DropTrackingStream {
    type Item = Result<Json>;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl Drop for DropTrackingStream {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn captured_snapshot<T: Clone>(items: &Arc<Mutex<Vec<T>>>) -> Vec<T> {
    flush_subscribers().unwrap();
    items.lock().unwrap().clone()
}

/// Helper that creates a collector/finalizer pair backed by a shared `Vec<Json>`.
///
/// Returns `(collector, finalizer, collected_chunks)` where `collected_chunks`
/// can be inspected after the stream is consumed.
#[allow(clippy::type_complexity)]
fn make_collector_finalizer() -> (
    Box<dyn FnMut(Json) -> Result<()> + Send>,
    Box<dyn FnOnce() -> Json + Send>,
    Arc<Mutex<Vec<Json>>>,
) {
    let collected = Arc::new(Mutex::new(Vec::<Json>::new()));
    let cc = collected.clone();
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(move |chunk| {
        cc.lock().unwrap().push(chunk);
        Ok(())
    });
    let fc = collected.clone();
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
        let chunks = fc.lock().unwrap();
        Json::Array(chunks.clone())
    });
    (collector, finalizer, collected)
}

#[tokio::test]
async fn test_stream_wrapper_basic_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![Ok(json!({"token": "hello"})), Ok(json!({"token": "world"}))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["token"], "hello");
    assert_eq!(chunks[1]["token"], "world");
}

#[tokio::test]
async fn explicit_close_finalizes_once_and_exhausts_the_managed_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "explicit_close_finalizes_once",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let close_calls = Arc::new(AtomicUsize::new(0));
    let finalizer_calls = Arc::new(AtomicUsize::new(0));
    let finalizer_count = Arc::clone(&finalizer_calls);
    let inner = LlmJsonStream::from_closeable(CloseTrackingStream {
        stream: Box::pin(tokio_stream::iter(vec![
            Ok(json!({"chunk": "first"})),
            Ok(json!({"chunk": "unread"})),
        ])),
        close_calls: Arc::clone(&close_calls),
        close_error: None,
        closed: false,
    });
    let wrapper = LlmStreamWrapper::new(
        inner,
        make_llm_handle("explicit-close"),
        Box::new(|_| Ok(())),
        Box::new(move || {
            finalizer_count.fetch_add(1, Ordering::SeqCst);
            json!({"partial": true})
        }),
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    assert_eq!(stream.next().await.unwrap().unwrap()["chunk"], "first");
    stream.close().await.unwrap();
    stream.close().await.unwrap();

    assert_eq!(close_calls.load(Ordering::SeqCst), 1);
    assert_eq!(finalizer_calls.load(Ordering::SeqCst), 1);
    assert!(stream.next().await.is_none());

    flush_subscribers().unwrap();
    let end_events = events.lock().unwrap();
    assert_eq!(
        end_events.iter().filter(|event| is_llm_end(event)).count(),
        1
    );
    assert!(deregister_subscriber("explicit_close_finalizes_once").unwrap());
}

#[tokio::test]
async fn explicit_close_caches_cleanup_errors() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let close_calls = Arc::new(AtomicUsize::new(0));
    let inner = LlmJsonStream::from_closeable(CloseTrackingStream {
        stream: Box::pin(tokio_stream::empty()),
        close_calls: Arc::clone(&close_calls),
        close_error: Some(FlowError::NotFound("producer cleanup failed".into())),
        closed: false,
    });
    let wrapper = LlmStreamWrapper::new(
        inner,
        make_llm_handle("explicit-close-error"),
        Box::new(|_| Ok(())),
        Box::new(|| Json::Null),
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    for _ in 0..2 {
        let error = stream.close().await.expect_err("close should fail");
        assert!(error.to_string().contains("producer cleanup failed"));
        assert!(matches!(error, FlowError::NotFound(_)));
    }
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn terminal_error_close_reaches_and_caches_real_producer_cleanup() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let close_calls = Arc::new(AtomicUsize::new(0));
    let inner = LlmJsonStream::from_closeable(CloseTrackingStream {
        stream: Box::pin(tokio_stream::iter(vec![Err(FlowError::Internal(
            "provider stream failed".into(),
        ))])),
        close_calls: Arc::clone(&close_calls),
        close_error: Some(FlowError::NotFound("producer cleanup failed".into())),
        closed: false,
    });
    let wrapper = LlmStreamWrapper::new(
        inner,
        make_llm_handle("terminal-error-close"),
        Box::new(|_| Ok(())),
        Box::new(|| Json::Null),
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    let terminal_error = stream.next().await.unwrap().unwrap_err();
    assert!(
        terminal_error
            .to_string()
            .contains("provider stream failed")
    );
    for _ in 0..2 {
        let close_error = stream
            .close()
            .await
            .expect_err("producer cleanup should fail");
        assert!(matches!(
            close_error,
            FlowError::NotFound(ref message) if message == "producer cleanup failed"
        ));
    }
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn default_stream_close_drops_the_wrapped_stream() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut stream = LlmJsonStream::new(DropTrackingStream {
        drops: Arc::clone(&drops),
    });

    stream.close().await.unwrap();

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn test_stream_wrapper_passthrough() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Any Json content should pass through unchanged
    let items = vec![Ok(json!("data: partial")), Ok(json!("more data"))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], json!("data: partial"));
    assert_eq!(chunks[1], json!("more data"));
}

#[tokio::test]
async fn test_stream_wrapper_empty_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let inner = LlmJsonStream::new(tokio_stream::empty());
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let mut count = 0;
    while let Some(_item) = wrapper.next().await {
        count += 1;
    }
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_stream_wrapper_single_chunk() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![Ok(json!("only chunk"))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], json!("only chunk"));
}

#[tokio::test]
async fn test_stream_wrapper_emits_end_event() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "stream_end_test",
        Arc::new(move |e: &Event| {
            let phase = match e.scope_category() {
                Some(ScopeCategory::Start) => "start",
                Some(ScopeCategory::End) => "end",
                None => e.kind(),
            };
            ec.lock().unwrap().push((phase.to_string(), e.scope_type()));
        }),
    )
    .unwrap();

    let items = vec![Ok(json!({"token": "hi"}))];
    let inner = make_stream(items);

    // Use the real API to create the handle so events are properly tracked
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("test_llm")
            .request(&request)
            .attributes(LlmAttributes::STREAMING)
            .build(),
    )
    .unwrap();

    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    let captured = captured_snapshot(&events);
    // Should have: START (from llm_call) + END (from stream wrapper exhaustion)
    assert!(captured.len() >= 2);
    assert_eq!(captured[0].0, "start");
    // The last event should be END
    assert_eq!(captured.last().unwrap().0, "end");

    deregister_subscriber("stream_end_test").unwrap();
}

#[tokio::test]
async fn anthropic_stream_finalization_does_not_estimate_cross_model_iteration_cost() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let _pricing_guard = ResetPricingResolverGuard;
    reset_global();
    install_anthropic_fallback_pricing();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "anthropic_cross_model_stream_cost",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let response = json!({
        "id": "msg_fallback",
        "type": "message",
        "role": "assistant",
        "model": "claude-fallback",
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 412,
            "output_tokens": 264,
            "iterations": [
                {"type": "message", "model": "claude-primary", "input_tokens": 535, "output_tokens": 0},
                {"type": "fallback_message", "model": "claude-fallback", "input_tokens": 412, "output_tokens": 264}
            ]
        }
    });
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("anthropic")
            .request(&request)
            .attributes(LlmAttributes::STREAMING)
            .build(),
    )
    .unwrap();
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"type": "message_stop"}))]),
        handle,
        Box::new(|_| Ok(())),
        Box::new(move || response),
        None,
        None,
        Some(Arc::new(AnthropicMessagesCodec)),
    );

    while let Some(item) = wrapper.next().await {
        item.unwrap();
    }

    let events = captured_snapshot(&events);
    let end_event = events
        .iter()
        .find(|event| is_llm_end(event))
        .expect("stream should emit an LLM end event");
    let response = end_event
        .annotated_response()
        .expect("stream end event should contain an annotated response");
    let usage = response
        .usage
        .as_ref()
        .expect("stream annotated response should contain usage");
    assert_eq!(usage.cost.as_ref(), None);

    assert!(deregister_subscriber("anthropic_cross_model_stream_cost").unwrap());
}

#[tokio::test]
async fn test_stream_wrapper_drop_emits_end_event_for_partial_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    register_subscriber(
        "stream_drop_end_test",
        Arc::new(move |e: &Event| {
            captured.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let inner = make_stream(vec![
        Ok(json!({"token": "partial"})),
        Ok(json!({"token": "unread"})),
    ]);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("stream_drop_llm")
            .request(&request)
            .attributes(LlmAttributes::STREAMING)
            .build(),
    )
    .unwrap();

    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    assert_eq!(
        wrapper.next().await.unwrap().unwrap(),
        json!({"token": "partial"})
    );
    drop(wrapper);

    let events = captured_snapshot(&events);
    let end_event = events
        .iter()
        .find(|event| is_llm_end(event))
        .expect("expected END event when a partial stream is dropped");
    assert_eq!(end_event.output(), Some(&json!([{"token": "partial"}])));
    assert_eq!(
        end_event.metadata().unwrap()["otel.status_code"],
        json!("ERROR")
    );
    assert!(
        end_event.annotated_response().is_none(),
        "an interrupted stream without optimization evidence must not manufacture a summary"
    );

    deregister_subscriber("stream_drop_end_test").unwrap();
}

#[tokio::test]
async fn dropped_stream_after_terminal_response_emits_success() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    register_subscriber(
        "stream_terminal_drop_status_test",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let terminal_event = json!({
        "type": "response.completed",
        "response": {
            "id": "resp_complete",
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        }
    });
    let streaming_codec = OpenAIResponsesStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"input": "finish"}),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("openai.responses")
            .request(&request)
            .attributes(LlmAttributes::STREAMING)
            .build(),
    )
    .unwrap();
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(terminal_event.clone())]),
        handle,
        collector,
        finalizer,
        None,
        None,
        Some(Arc::new(OpenAIResponsesCodec)),
    );

    assert_eq!(wrapper.next().await.unwrap().unwrap(), terminal_event);
    drop(wrapper);

    let events = captured_snapshot(&events);
    let end_event = events
        .iter()
        .find(|event| is_llm_end(event))
        .expect("expected END event after the terminal response was dropped");
    let metadata = end_event.metadata().unwrap();
    assert_eq!(metadata["otel.status_code"], json!("OK"));
    assert!(metadata.get("otel.status_description").is_none());

    deregister_subscriber("stream_terminal_drop_status_test").unwrap();
}

#[tokio::test]
async fn stream_termination_modes_close_accounting_without_losing_evidence() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "stream_termination_accounting",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (clean_handle, clean_recorder) = make_optimized_llm_handle("stream-clean", "test.clean");
    let (collector, finalizer, _) = make_collector_finalizer();
    let mut clean = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"chunk": "clean"}))]),
        clean_handle,
        collector,
        finalizer,
        None,
        None,
        None,
    );
    while let Some(item) = clean.next().await {
        item.unwrap();
    }
    flush_subscribers().unwrap();
    assert!(!clean_recorder.record(LlmOptimizationContribution::new("late", "test")));

    let (before_error_handle, before_error_recorder) =
        make_optimized_llm_handle("stream-error-before", "test.error_before");
    let (collector, finalizer, _) = make_collector_finalizer();
    let mut error_before = LlmStreamWrapper::new(
        make_stream(vec![Err(FlowError::Internal("before first".into()))]),
        before_error_handle,
        collector,
        finalizer,
        None,
        None,
        None,
    );
    assert!(error_before.next().await.unwrap().is_err());
    flush_subscribers().unwrap();
    assert!(!before_error_recorder.record(LlmOptimizationContribution::new("late", "test")));

    let (after_error_handle, after_error_recorder) =
        make_optimized_llm_handle("stream-error-after", "test.error_after");
    let (collector, finalizer, _) = make_collector_finalizer();
    let mut error_after = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"chunk": "committed"})),
            Err(FlowError::Internal("after first".into())),
        ]),
        after_error_handle,
        collector,
        finalizer,
        None,
        None,
        None,
    );
    assert!(error_after.next().await.unwrap().is_ok());
    assert!(error_after.next().await.unwrap().is_err());
    flush_subscribers().unwrap();
    assert!(!after_error_recorder.record(LlmOptimizationContribution::new("late", "test")));

    let (drop_before_handle, drop_before_recorder) =
        make_optimized_llm_handle("stream-drop-before", "test.drop_before");
    let (collector, finalizer, _) = make_collector_finalizer();
    let drop_before = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"chunk": "unread"}))]),
        drop_before_handle,
        collector,
        finalizer,
        None,
        None,
        None,
    );
    drop(drop_before);
    flush_subscribers().unwrap();
    assert!(!drop_before_recorder.record(LlmOptimizationContribution::new("late", "test")));

    let (drop_after_handle, drop_after_recorder) =
        make_optimized_llm_handle("stream-drop-after", "test.route_commit");
    let drop_after_uuid = drop_after_handle.uuid;
    let (collector, finalizer, _) = make_collector_finalizer();
    let mut drop_after = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"chunk": "route committed"})),
            Ok(json!({"chunk": "unread"})),
        ]),
        drop_after_handle,
        collector,
        finalizer,
        None,
        None,
        None,
    );
    assert!(drop_after.next().await.unwrap().is_ok());
    drop(drop_after);
    flush_subscribers().unwrap();
    assert!(!drop_after_recorder.record(LlmOptimizationContribution::new("late", "test")));

    flush_subscribers().unwrap();
    let events = captured_snapshot(&events);
    let summary_for = |name: &str| {
        events
            .iter()
            .find(|event| event.name() == name && is_llm_end(event))
            .and_then(Event::annotated_response)
            .and_then(|response| response.optimization_summary.as_ref())
            .unwrap_or_else(|| panic!("missing optimization summary for {name}"))
    };
    assert!(
        !summary_for("stream-clean")
            .limitations
            .iter()
            .any(|item| item == "stream_interrupted")
    );
    for name in [
        "stream-error-before",
        "stream-error-after",
        "stream-drop-before",
        "stream-drop-after",
    ] {
        assert!(
            summary_for(name)
                .limitations
                .iter()
                .any(|item| item == "stream_interrupted"),
            "{name} should be marked interrupted"
        );
    }
    assert_eq!(
        summary_for("stream-drop-after").contributions[0].producer,
        "test.route_commit"
    );
    let committed_mark = events
        .iter()
        .find(|event| {
            event.name() == "nemo_relay.llm.optimization"
                && event.parent_uuid() == Some(drop_after_uuid)
        })
        .expect("route-commit contribution mark should survive an interrupted stream");
    assert_eq!(
        committed_mark.data().unwrap()["producer"],
        "test.route_commit"
    );

    deregister_subscriber("stream_termination_accounting").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_error_propagation() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items: Vec<Result<Json>> = vec![
        Ok(json!("good chunk")),
        Err(FlowError::Internal("stream error".into())),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let first = wrapper.next().await.unwrap();
    assert!(first.is_ok());
    assert_eq!(first.unwrap(), json!("good chunk"));

    let second = wrapper.next().await.unwrap();
    assert!(second.is_err());
}

#[tokio::test]
async fn test_stream_wrapper_json_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![Ok(json!({"token": "hello"})), Ok(json!({"token": "world"}))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["token"], "hello");
    assert_eq!(chunks[1]["token"], "world");
}

#[tokio::test]
async fn test_stream_wrapper_collector_receives_all_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![
        Ok(json!("chunk1")),
        Ok(json!("chunk2")),
        Ok(json!("chunk3")),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    let chunks = collected.lock().unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0], json!("chunk1"));
    assert_eq!(chunks[1], json!("chunk2"));
    assert_eq!(chunks[2], json!("chunk3"));
}

#[tokio::test]
async fn test_stream_wrapper_finalizer_called_on_exhaustion() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let finalizer_called = Arc::new(Mutex::new(false));
    let fc = finalizer_called.clone();

    let items = vec![Ok(json!("chunk"))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_| Ok(()));
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
        *fc.lock().unwrap() = true;
        json!({"finalized": true})
    });
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    // Finalizer should not be called yet
    assert!(!*finalizer_called.lock().unwrap());

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    // Finalizer should have been called exactly once
    assert!(*finalizer_called.lock().unwrap());
}

#[tokio::test]
async fn test_stream_wrapper_error_skips_collector_and_finalizes_immediately() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let collector_calls = Arc::new(Mutex::new(0u32));
    let cc = collector_calls.clone();
    let finalizer_called = Arc::new(Mutex::new(false));
    let fc = finalizer_called.clone();

    let items: Vec<Result<Json>> = vec![Err(FlowError::Internal("error".into()))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(move |_| {
        *cc.lock().unwrap() += 1;
        Ok(())
    });
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
        *fc.lock().unwrap() = true;
        Json::Null
    });
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    // Consume the error
    let result = wrapper.next().await.unwrap();
    assert!(result.is_err());

    // Collector should not have been called for the error
    assert_eq!(*collector_calls.lock().unwrap(), 0);

    // Finalizer is called on the first error poll; callers do not need to poll again.
    assert!(*finalizer_called.lock().unwrap());

    // Stream is terminated after the error.
    assert!(wrapper.next().await.is_none());
}

#[tokio::test]
async fn test_stream_wrapper_error_emits_end_event_on_first_error_poll() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    register_subscriber(
        "stream_error_end_test",
        Arc::new(move |e: &Event| {
            captured.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let items: Vec<Result<Json>> = vec![Err(FlowError::Internal("error".into()))];
    let inner = make_stream(items);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("stream_error_llm")
            .request(&request)
            .attributes(LlmAttributes::STREAMING)
            .build(),
    )
    .unwrap();

    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_| Ok(()));
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!({"partial": true}));
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    let result = wrapper.next().await.unwrap();
    assert!(result.is_err());

    let events = captured_snapshot(&events);
    let end_event = events
        .iter()
        .find(|event| is_llm_end(event))
        .expect("expected END event on first error poll");
    assert_eq!(end_event.output(), Some(&json!({"partial": true})));

    deregister_subscriber("stream_error_end_test").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_end_event_contains_intercepted_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "end_event_test",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let items = vec![Ok(json!({"token": "a"})), Ok(json!({"token": "b"}))];
    let inner = make_stream(items);

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("test_llm")
            .request(&request)
            .attributes(LlmAttributes::STREAMING)
            .build(),
    )
    .unwrap();

    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    // The END event output should contain the finalizer's aggregated response
    let captured = captured_snapshot(&events);
    let end_event = captured.iter().find(|e| is_llm_end(e)).unwrap();
    let output = end_event.output().unwrap();
    // The default finalizer collects chunks into an array
    assert!(output.is_array());
    let arr = output.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["token"], "a");
    assert_eq!(arr[1]["token"], "b");

    deregister_subscriber("end_event_test").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_collector_error_terminates_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let collector_calls = Arc::new(Mutex::new(0u32));
    let cc = collector_calls.clone();

    let items = vec![
        Ok(json!("chunk1")),
        Ok(json!("chunk2")),
        Ok(json!("chunk3")),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");

    // Collector that fails on the second chunk
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(move |_chunk| {
        let mut count = cc.lock().unwrap();
        *count += 1;
        if *count >= 2 {
            Err(FlowError::Internal("collector error".into()))
        } else {
            Ok(())
        }
    });
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| Json::Null);
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None, None);

    // First chunk should succeed
    let first = wrapper.next().await;
    assert!(first.is_some());
    assert!(first.unwrap().is_ok());

    // Second chunk: collector returns Err, stream should yield the error
    let second = wrapper.next().await;
    assert!(second.is_some());
    let second_result = second.unwrap();
    assert!(second_result.is_err());
    match second_result {
        Err(FlowError::Internal(msg)) => {
            assert_eq!(msg, "collector error");
        }
        other => panic!("expected Internal error, got {other:?}"),
    }

    // Stream should be terminated (ended = true), yielding None
    let third = wrapper.next().await;
    assert!(third.is_none());

    // Collector was called exactly twice (once for chunk1, once for chunk2)
    assert_eq!(*collector_calls.lock().unwrap(), 2);
}

#[tokio::test]
async fn failed_finalizer_skips_the_data_fallback() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "failed_finalizer_no_fallback",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let handle_with_data = |name: &str| {
        LlmHandle::builder()
            .name(name.to_string())
            .attributes(LlmAttributes::STREAMING)
            .data(json!({"fallback": "not a response"}))
            .build()
    };

    let mut failed = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        handle_with_data("finalizer-failed"),
        Box::new(|_| Ok(())),
        LlmFinalizer::Fallible(Box::new(|| {
            Err(FlowError::Internal("finalizer failed".into()))
        })),
        None,
        None,
        None,
    );
    while failed.next().await.is_some() {}

    let mut null_aggregate = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        handle_with_data("finalizer-null"),
        Box::new(|_| Ok(())),
        Box::new(|| Json::Null),
        None,
        None,
        None,
    );
    while null_aggregate.next().await.is_some() {}

    let events = captured_snapshot(&events);
    let end_data = |name: &str| {
        events
            .iter()
            .find(|event| event.name() == name && is_llm_end(event))
            .expect("END event")
            .data()
            .cloned()
    };
    assert!(end_data("finalizer-failed").is_none_or(|data| data.is_null()));
    assert_eq!(
        end_data("finalizer-null"),
        Some(json!({"fallback": "not a response"}))
    );
    assert!(deregister_subscriber("failed_finalizer_no_fallback").unwrap());
}

#[tokio::test]
async fn failed_finalizer_skips_response_sanitizers() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "failed_finalizer_no_sanitizers",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "failed_finalizer_no_sanitizers",
        1,
        Arc::new(|_response, _context| Box::pin(async { Ok(Some(json!({"sanitized": true}))) })),
    )
    .unwrap();

    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("finalizer-failed-sanitized"),
        Box::new(|_| Ok(())),
        LlmFinalizer::Fallible(Box::new(|| {
            Err(FlowError::Internal("finalizer failed".into()))
        })),
        None,
        None,
        None,
    );
    while wrapper.next().await.is_some() {}

    let events = captured_snapshot(&events);
    let end = events
        .iter()
        .find(|event| event.name() == "finalizer-failed-sanitized" && is_llm_end(event))
        .expect("END event");
    assert!(end.data().is_none_or(Json::is_null));
    assert!(deregister_llm_sanitize_response_guardrail("failed_finalizer_no_sanitizers").unwrap());
    assert!(deregister_subscriber("failed_finalizer_no_sanitizers").unwrap());
}

fn capture_events(name: &'static str) -> Arc<Mutex<Vec<Event>>> {
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        name,
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    events
}

fn end_event(events: &[Event]) -> &Event {
    events
        .iter()
        .find(|event| is_llm_end(event))
        .expect("stream emitted an LLM END event")
}

#[tokio::test]
async fn async_collector_is_awaited_before_its_chunk_is_yielded() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let gate = Arc::new(Mutex::new(Some(gate)));
    let collected = Arc::new(Mutex::new(Vec::<Json>::new()));
    let seen = Arc::clone(&collected);
    let collector = LlmCollector::new_async(move |chunk: Json| {
        let gate = gate.lock().unwrap().take();
        let seen = Arc::clone(&seen);
        async move {
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            seen.lock().unwrap().push(chunk);
            Ok(())
        }
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"token": "first"})),
            Ok(json!({"token": "second"})),
        ]),
        make_llm_handle("async-collector-order"),
        collector,
        Box::new(|| Json::Null),
        None,
        None,
        None,
    );

    let blocked = tokio::time::timeout(tokio::time::Duration::from_millis(100), wrapper.next());
    assert!(
        blocked.await.is_err(),
        "a chunk must wait for its collector"
    );
    assert!(collected.lock().unwrap().is_empty());

    release.send(()).unwrap();
    assert_eq!(wrapper.next().await.unwrap().unwrap()["token"], "first");
    assert_eq!(collected.lock().unwrap().len(), 1);
    assert_eq!(wrapper.next().await.unwrap().unwrap()["token"], "second");
    assert!(wrapper.next().await.is_none());
    assert_eq!(collected.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn async_collector_error_ends_the_stream_with_that_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_collector_error");

    let collector = LlmCollector::new_async(|_chunk: Json| async {
        Err(FlowError::Internal("async collector failed".into()))
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"token": "first"})),
            Ok(json!({"token": "unread"})),
        ]),
        make_llm_handle("async-collector-error"),
        collector,
        Box::new(|| json!({"partial": true})),
        None,
        None,
        None,
    );

    let error = wrapper.next().await.unwrap().unwrap_err();
    assert!(error.to_string().contains("async collector failed"));
    assert!(wrapper.next().await.is_none());

    let events = captured_snapshot(&events);
    let end = end_event(&events);
    assert_eq!(end.metadata().unwrap()["otel.status_code"], json!("ERROR"));
    assert_eq!(end.data().unwrap()["partial"], json!(true));
    assert!(deregister_subscriber("async_collector_error").unwrap());
}

#[tokio::test]
async fn async_finalizer_settles_before_the_stream_ends() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizer_end");

    let finalized = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&finalized);
    let finalizer = LlmFinalizer::new_async(move || async move {
        tokio::task::yield_now().await;
        count.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"aggregate": "done"}))
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("async-finalizer-end"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );

    assert_eq!(wrapper.next().await.unwrap().unwrap()["token"], "only");
    assert!(wrapper.next().await.is_none());
    assert_eq!(finalized.load(Ordering::SeqCst), 1);

    let events = captured_snapshot(&events);
    let end = end_event(&events);
    assert_eq!(end.data().unwrap()["aggregate"], json!("done"));
    assert_eq!(end.metadata().unwrap()["otel.status_code"], json!("OK"));
    assert!(deregister_subscriber("async_finalizer_end").unwrap());
}

#[tokio::test]
async fn close_waits_for_an_async_finalizer() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let finalized = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&finalized);
    let finalizer = LlmFinalizer::new_async(move || async move {
        tokio::task::yield_now().await;
        count.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"partial": true}))
    });
    let wrapper = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"token": "first"})),
            Ok(json!({"token": "unread"})),
        ]),
        make_llm_handle("async-finalizer-close"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    assert_eq!(stream.next().await.unwrap().unwrap()["token"], "first");
    stream.close().await.unwrap();
    assert_eq!(finalized.load(Ordering::SeqCst), 1);
    stream.close().await.unwrap();
    assert_eq!(finalized.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropped_stream_with_an_async_finalizer_still_emits_end() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizer_drop");

    let finalizer = LlmFinalizer::new_async(|| async {
        tokio::task::yield_now().await;
        Ok(json!({"dropped": true}))
    });
    let wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "unread"}))]),
        make_llm_handle("async-finalizer-drop"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );
    drop(wrapper);

    let events = captured_snapshot(&events);
    assert_eq!(end_event(&events).data().unwrap()["dropped"], json!(true));
    assert!(deregister_subscriber("async_finalizer_drop").unwrap());
}

#[tokio::test]
async fn finalizer_that_flushes_subscribers_completes() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("finalizer_flush");

    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("finalizer-flush"),
        Box::new(|_| Ok(())),
        Box::new(|| {
            flush_subscribers().unwrap();
            json!({"flushed": true})
        }),
        None,
        None,
        None,
    );

    let drained = tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
        while wrapper.next().await.is_some() {}
    });
    assert!(
        drained.await.is_ok(),
        "stream end hung on a flush inside the finalizer"
    );

    let events = captured_snapshot(&events);
    assert_eq!(end_event(&events).data().unwrap()["flushed"], json!(true));
    assert!(deregister_subscriber("finalizer_flush").unwrap());
}

#[tokio::test]
async fn async_finalizer_that_flushes_subscribers_completes() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizer_flush");

    let finalizer = LlmFinalizer::new_async(|| async {
        tokio::task::yield_now().await;
        flush_subscribers()?;
        Ok(json!({"flushed": true}))
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("async-finalizer-flush"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );

    let drained = tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
        while wrapper.next().await.is_some() {}
    });
    assert!(
        drained.await.is_ok(),
        "stream end hung on a flush inside the async finalizer"
    );

    let events = captured_snapshot(&events);
    assert_eq!(end_event(&events).data().unwrap()["flushed"], json!(true));
    assert!(deregister_subscriber("async_finalizer_flush").unwrap());
}

#[tokio::test]
async fn close_reports_an_undelivered_async_collector_error_once() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_collector_close");

    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let gate = Arc::new(Mutex::new(Some(gate)));
    let collector = LlmCollector::new_async(move |_chunk: Json| {
        let gate = gate.lock().unwrap().take();
        async move {
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            Err(FlowError::Internal("late collector failure".into()))
        }
    });
    let wrapper = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"token": "first"})),
            Ok(json!({"token": "unread"})),
        ]),
        make_llm_handle("async-collector-close"),
        collector,
        Box::new(|| Json::Null),
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    let blocked = tokio::time::timeout(tokio::time::Duration::from_millis(50), stream.next());
    assert!(blocked.await.is_err(), "the chunk waits for its collector");
    release.send(()).unwrap();

    let error = stream.close().await.unwrap_err();
    assert!(error.to_string().contains("late collector failure"));
    let again = stream.close().await.unwrap_err();
    assert!(again.to_string().contains("late collector failure"));
    assert!(stream.next().await.is_none());

    let events = captured_snapshot(&events);
    assert_eq!(
        end_event(&events).metadata().unwrap()["otel.status_code"],
        json!("ERROR")
    );
    assert!(deregister_subscriber("async_collector_close").unwrap());
}

#[tokio::test]
async fn close_reports_a_stream_error_that_polling_has_not_delivered() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let finalizer = LlmFinalizer::new_async(|| async {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(Json::Null)
    });
    let wrapper = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"token": "first"})),
            Err(FlowError::Internal("producer failed".into())),
        ]),
        make_llm_handle("parked-producer-error"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    assert_eq!(stream.next().await.unwrap().unwrap()["token"], "first");
    // The error is observed here but held until the finalizer settles.
    let parked = tokio::time::timeout(tokio::time::Duration::from_millis(10), stream.next());
    assert!(parked.await.is_err());

    let error = stream.close().await.unwrap_err();
    assert!(error.to_string().contains("producer failed"));
    assert!(stream.close().await.is_err());
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn failed_finalizer_marks_the_end_event_as_an_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("failed_finalizer_status");

    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("finalizer-failed-status"),
        Box::new(|_| Ok(())),
        LlmFinalizer::Fallible(Box::new(|| {
            Err(FlowError::Internal("finalizer failed".into()))
        })),
        None,
        None,
        None,
    );
    while wrapper.next().await.is_some() {}

    let events = captured_snapshot(&events);
    let end = end_event(&events);
    let metadata = end.metadata().unwrap();
    assert_eq!(metadata["otel.status_code"], json!("ERROR"));
    assert!(
        metadata["otel.status_description"]
            .as_str()
            .is_some_and(|description| description.contains("finalizer failed"))
    );
    assert!(metadata["error.type"].is_string());
    assert!(end.data().is_none_or(Json::is_null));
    assert!(deregister_subscriber("failed_finalizer_status").unwrap());
}

#[tokio::test]
async fn failed_async_finalizer_marks_the_end_event_as_an_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("failed_async_finalizer_status");

    let finalizer = LlmFinalizer::new_async(|| async {
        tokio::task::yield_now().await;
        Err(FlowError::Internal("async finalizer failed".into()))
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("async-finalizer-failed-status"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );
    assert_eq!(wrapper.next().await.unwrap().unwrap()["token"], "only");
    assert!(wrapper.next().await.is_none());

    let events = captured_snapshot(&events);
    let end = end_event(&events);
    let metadata = end.metadata().unwrap();
    assert_eq!(metadata["otel.status_code"], json!("ERROR"));
    assert!(
        metadata["otel.status_description"]
            .as_str()
            .is_some_and(|description| description.contains("async finalizer failed"))
    );
    assert!(end.data().is_none_or(Json::is_null));
    assert!(deregister_subscriber("failed_async_finalizer_status").unwrap());
}

#[tokio::test]
async fn failed_finalizer_keeps_the_stream_error_on_a_failed_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("failed_finalizer_failed_stream");

    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![
            Ok(json!({"token": "first"})),
            Err(FlowError::Internal("producer failed".into())),
        ]),
        make_llm_handle("finalizer-failed-after-stream-error"),
        Box::new(|_| Ok(())),
        LlmFinalizer::Fallible(Box::new(|| {
            Err(FlowError::Internal("finalizer failed".into()))
        })),
        None,
        None,
        None,
    );
    assert_eq!(wrapper.next().await.unwrap().unwrap()["token"], "first");
    assert!(wrapper.next().await.unwrap().is_err());
    assert!(wrapper.next().await.is_none());

    let events = captured_snapshot(&events);
    let metadata = end_event(&events).metadata().unwrap();
    assert_eq!(metadata["otel.status_code"], json!("ERROR"));
    let description = metadata["otel.status_description"].as_str().unwrap();
    assert!(description.contains("producer failed"));
    assert!(!description.contains("finalizer failed"));
    assert!(deregister_subscriber("failed_finalizer_failed_stream").unwrap());
}

#[tokio::test]
async fn async_finalizer_does_not_hold_unrelated_event_delivery() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let (fast_ended, fast_end_seen) = tokio::sync::oneshot::channel::<()>();
    let fast_ended = Mutex::new(Some(fast_ended));
    register_subscriber(
        "async_finalizer_unrelated_delivery",
        Arc::new(move |event: &Event| {
            if is_llm_end(event)
                && event.name() == "fast-stream"
                && let Some(fast_ended) = fast_ended.lock().unwrap().take()
            {
                let _ = fast_ended.send(());
            }
        }),
    )
    .unwrap();

    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let slow = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "slow"}))]),
        make_llm_handle("slow-stream"),
        Box::new(|_| Ok(())),
        LlmFinalizer::new_async(move || async move {
            let _ = gate.await;
            Ok(json!({"slow": true}))
        }),
        None,
        None,
        None,
    );
    drop(slow);
    let fast = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "fast"}))]),
        make_llm_handle("fast-stream"),
        Box::new(|_| Ok(())),
        Box::new(|| json!({"fast": true})),
        None,
        None,
        None,
    );
    drop(fast);

    let delivered = tokio::time::timeout(tokio::time::Duration::from_secs(5), fast_end_seen).await;
    assert!(
        delivered.is_ok(),
        "the fast stream's END waited on the slow stream's parked finalizer"
    );

    release.send(()).unwrap();
    flush_subscribers().unwrap();
    assert!(deregister_subscriber("async_finalizer_unrelated_delivery").unwrap());
}

#[tokio::test]
async fn flush_inside_async_finalizer_waits_for_earlier_publications_only() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizer_real_flush");

    let seen_by_finalizer = Arc::new(Mutex::new(Vec::<Event>::new()));
    let observed = Arc::clone(&events);
    let recorded = Arc::clone(&seen_by_finalizer);
    let finalizer = LlmFinalizer::new_async(move || async move {
        flush_subscribers()?;
        *recorded.lock().unwrap() = observed.lock().unwrap().clone();
        Ok(json!({"flushed": true}))
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("async-finalizer-real-flush"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );
    while wrapper.next().await.is_some() {}

    let seen = seen_by_finalizer.lock().unwrap().clone();
    assert!(
        seen.iter().any(|event| event.name() == "llm.chunk"),
        "the flush returned before the chunk mark was delivered"
    );
    assert!(
        !seen.iter().any(is_llm_end),
        "the flush waited on this stream's own END event"
    );

    let events = captured_snapshot(&events);
    assert_eq!(end_event(&events).data().unwrap()["flushed"], json!(true));
    assert!(deregister_subscriber("async_finalizer_real_flush").unwrap());
}

#[tokio::test]
async fn flush_inside_an_async_finalizer_lets_a_queued_end_publication_finish() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizer_flush_executor");

    // This END publication parks in a response sanitizer while its barrier holds the dispatcher.
    let (release_sanitizer, sanitizer_gate) = tokio::sync::oneshot::channel::<()>();
    let sanitizer_gate = Arc::new(Mutex::new(Some(sanitizer_gate)));
    register_llm_sanitize_response_guardrail(
        "async_finalizer_flush_executor",
        1,
        Arc::new(move |response, _context| {
            let gate = sanitizer_gate.lock().unwrap().take();
            Box::pin(async move {
                if let Some(gate) = gate {
                    let _ = gate.await;
                }
                Ok(Some(response))
            })
        }),
    )
    .unwrap();
    drop(LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "parked"}))]),
        make_llm_handle("parked-sanitizer-stream"),
        Box::new(|_| Ok(())),
        Box::new(|| json!({"parked": true})),
        None,
        None,
        None,
    ));

    // Release the sanitizer only once the flush behind that barrier is queued.
    let (flushed_tx, flushed_rx) = mpsc::channel::<()>();
    let finalizer = LlmFinalizer::new_async(move || async move {
        let (started_tx, started_rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            let _ = started_rx.recv();
            let _ = release_sanitizer.send(());
        });
        flush_subscribers_with_started_signal(started_tx)?;
        let _ = flushed_tx.send(());
        Ok(json!({"flushed": true}))
    });
    drop(LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "flushing"}))]),
        make_llm_handle("flushing-finalizer-stream"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    ));

    flushed_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the flush inside the async finalizer stalled the parked END publication");

    let events = captured_snapshot(&events);
    for name in ["parked-sanitizer-stream", "flushing-finalizer-stream"] {
        assert!(
            events
                .iter()
                .any(|event| is_llm_end(event) && event.name() == name),
            "{name} END event was not delivered"
        );
    }
    assert!(deregister_llm_sanitize_response_guardrail("async_finalizer_flush_executor").unwrap());
    assert!(deregister_subscriber("async_finalizer_flush_executor").unwrap());
}

#[tokio::test]
async fn concurrent_async_finalizers_that_flush_do_not_wait_on_each_other() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizers_mutual_flush");

    let rendezvous = Arc::new(tokio::sync::Barrier::new(2));
    let (flushed_tx, flushed_rx) = mpsc::channel::<&'static str>();
    for name in ["first-flushing-finalizer", "second-flushing-finalizer"] {
        let rendezvous = Arc::clone(&rendezvous);
        let flushed_tx = flushed_tx.clone();
        drop(LlmStreamWrapper::new(
            make_stream(vec![Ok(json!({"token": name}))]),
            make_llm_handle(name),
            Box::new(|_| Ok(())),
            LlmFinalizer::new_async(move || async move {
                // Both finalizers hold a pending END permit while they flush.
                rendezvous.wait().await;
                flush_subscribers()?;
                let _ = flushed_tx.send(name);
                Ok(json!({"flushed": name}))
            }),
            None,
            None,
            None,
        ));
    }

    for _ in 0..2 {
        flushed_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("a flush inside one async finalizer waited on the other finalizer's END");
    }

    let events = captured_snapshot(&events);
    assert_eq!(events.iter().filter(|event| is_llm_end(event)).count(), 2);
    assert!(deregister_subscriber("async_finalizers_mutual_flush").unwrap());
}

#[tokio::test]
async fn events_emitted_after_an_async_stream_ends_follow_its_end_event() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let events = capture_events("async_finalizer_end_ordering");

    let finalizer = LlmFinalizer::new_async(|| async {
        tokio::task::yield_now().await;
        Ok(json!({"ordered": true}))
    });
    let mut wrapper = LlmStreamWrapper::new(
        make_stream(vec![Ok(json!({"token": "only"}))]),
        make_llm_handle("async-finalizer-ordering"),
        Box::new(|_| Ok(())),
        finalizer,
        None,
        None,
        None,
    );
    while wrapper.next().await.is_some() {}

    let scope = push_scope(
        PushScopeParams::builder()
            .name("after-stream")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();

    let events = captured_snapshot(&events);
    let end_index = events.iter().position(is_llm_end).unwrap();
    let scope_index = events
        .iter()
        .position(|event| event.name() == "after-stream")
        .unwrap();
    assert!(
        end_index < scope_index,
        "scope events overtook the END event"
    );
    assert!(deregister_subscriber("async_finalizer_end_ordering").unwrap());
}

#[tokio::test]
async fn close_keeps_the_stream_error_variant_when_cleanup_also_fails() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let gate = Arc::new(Mutex::new(Some(gate)));
    let collector = LlmCollector::new_async(move |_chunk: Json| {
        let gate = gate.lock().unwrap().take();
        async move {
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            Err(FlowError::GuardrailRejected(
                "collector rejected the chunk".into(),
            ))
        }
    });
    let close_calls = Arc::new(AtomicUsize::new(0));
    let inner = LlmJsonStream::from_closeable(CloseTrackingStream {
        stream: Box::pin(tokio_stream::iter(vec![Ok::<Json, FlowError>(
            json!({"token": "first"}),
        )])),
        close_calls: Arc::clone(&close_calls),
        close_error: Some(FlowError::NotFound("producer cleanup failed".into())),
        closed: false,
    });
    let wrapper = LlmStreamWrapper::new(
        inner,
        make_llm_handle("collector-then-cleanup-error"),
        collector,
        Box::new(|| Json::Null),
        None,
        None,
        None,
    );
    let mut stream = LlmJsonStream::from_closeable(wrapper);

    let blocked = tokio::time::timeout(tokio::time::Duration::from_millis(50), stream.next());
    assert!(blocked.await.is_err(), "the chunk waits for its collector");
    release.send(()).unwrap();

    for _ in 0..2 {
        let error = stream.close().await.unwrap_err();
        assert!(matches!(
            error,
            FlowError::GuardrailRejected(ref message) if message == "collector rejected the chunk"
        ));
    }
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);
    assert!(stream.next().await.is_none());
}
