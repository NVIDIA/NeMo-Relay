// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for stream in the NeMo Relay core crate.

#![allow(clippy::await_holding_lock)]

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{LlmAttributes, LlmHandle, LlmRequest};
use nemo_relay::api::llm::{LlmCallParams, llm_call};
use nemo_relay::api::optimization::LlmOptimizationRecorder;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{LlmJsonStream, LlmStreamInner, NemoRelayContextState};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::optimization::LlmOptimizationContribution;
use nemo_relay::error::FlowError;
use nemo_relay::error::Result;
use nemo_relay::json::Json;
use nemo_relay::stream::LlmStreamWrapper;
use serde_json::json;
use tokio_stream::{Stream, StreamExt};

// Serialize all tests since they share global state
static TEST_MUTEX: Mutex<()> = Mutex::new(());

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
    assert!(!drop_after_recorder.record(LlmOptimizationContribution::new("late", "test")));

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
