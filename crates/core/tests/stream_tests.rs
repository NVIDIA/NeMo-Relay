// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::await_holding_lock)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use nvagentrt_core::context::*;
use nvagentrt_core::error::Result;
use nvagentrt_core::json::Json;
use nvagentrt_core::stream::LlmStreamWrapper;
use nvagentrt_core::types::*;
use nvagentrt_core::*;
use serde_json::json;
use tokio_stream::{Stream, StreamExt};

// Serialize all tests since they share global state
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NVAgentRTContextState::new();
}

fn make_llm_handle(name: &str) -> LLMHandle {
    LLMHandle::new(name.to_string(), LLMAttributes::STREAMING, None, None, None)
}

fn make_stream(items: Vec<Result<String>>) -> Pin<Box<dyn Stream<Item = Result<String>> + Send>> {
    Box::pin(tokio_stream::iter(items))
}

/// Helper that creates a collector/finalizer pair backed by a shared `Vec<String>`.
///
/// Returns `(collector, finalizer, collected_chunks)` where `collected_chunks`
/// can be inspected after the stream is consumed.
#[allow(clippy::type_complexity)]
fn make_collector_finalizer() -> (
    Box<dyn FnMut(String) + Send>,
    Box<dyn FnOnce() -> Json + Send>,
    Arc<Mutex<Vec<String>>>,
) {
    let collected = Arc::new(Mutex::new(Vec::<String>::new()));
    let cc = collected.clone();
    let collector: Box<dyn FnMut(String) + Send> = Box::new(move |chunk| {
        cc.lock().unwrap().push(chunk);
    });
    let fc = collected.clone();
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
        let chunks = fc.lock().unwrap();
        Json::Array(
            chunks
                .iter()
                .filter_map(|s| serde_json::from_str(s).ok())
                .collect(),
        )
    });
    (collector, finalizer, collected)
}

#[tokio::test]
async fn test_stream_wrapper_basic_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![
        Ok("{\"token\": \"hello\"}".to_string()),
        Ok("{\"token\": \"world\"}".to_string()),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "{\"token\": \"hello\"}");
    assert_eq!(chunks[1], "{\"token\": \"world\"}");
}

#[tokio::test]
async fn test_stream_wrapper_passthrough() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Any string content should pass through unchanged
    let items = vec![Ok("data: partial".to_string()), Ok("more data".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "data: partial");
    assert_eq!(chunks[1], "more data");
}

#[tokio::test]
async fn test_stream_wrapper_empty_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>> = Box::pin(tokio_stream::empty());
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

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

    let items = vec![Ok("only chunk".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], "only chunk");
}

#[tokio::test]
async fn test_stream_wrapper_with_intercept() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Register a stream response intercept that transforms strings
    nvagentrt_register_llm_stream_response_intercept(
        "test_stream_intercept",
        1,
        false,
        Box::new(|chunk: String| format!("[intercepted] {}", chunk)),
    )
    .unwrap();

    let items = vec![Ok("original".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], "[intercepted] original");

    nvagentrt_deregister_llm_stream_response_intercept("test_stream_intercept").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_intercept_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    nvagentrt_register_llm_stream_response_intercept(
        "intercept_a",
        1,
        false,
        Box::new(|chunk: String| format!("A({})", chunk)),
    )
    .unwrap();

    nvagentrt_register_llm_stream_response_intercept(
        "intercept_b",
        2,
        false,
        Box::new(|chunk: String| format!("B({})", chunk)),
    )
    .unwrap();

    let items = vec![Ok("x".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let chunk = wrapper.next().await.unwrap().unwrap();
    // A runs first (priority 1), then B (priority 2)
    assert_eq!(chunk, "B(A(x))");

    nvagentrt_deregister_llm_stream_response_intercept("intercept_a").unwrap();
    nvagentrt_deregister_llm_stream_response_intercept("intercept_b").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_emits_end_event() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    nvagentrt_register_subscriber(
        "stream_end_test",
        Box::new(move |e: &Event| {
            ec.lock().unwrap().push((e.event_type, e.scope_type));
        }),
    )
    .unwrap();

    let items = vec![Ok("{\"token\": \"hi\"}".to_string())];
    let inner = make_stream(items);

    // Use the real API to create the handle so events are properly tracked
    let request = LLMRequest {
        method: "POST".into(),
        url: "https://api.example.com".into(),
        headers: serde_json::Map::new(),
        body: json!({}),
    };
    let handle = nvagentrt_llm_call(
        "test_llm",
        &request,
        None,
        LLMAttributes::STREAMING,
        None,
        None,
        None,
    )
    .unwrap();

    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    let captured = events.lock().unwrap();
    // Should have: START (from llm_call) + END (from stream wrapper exhaustion)
    assert!(captured.len() >= 2);
    assert_eq!(captured[0].0, EventType::Start);
    // The last event should be END
    assert_eq!(captured.last().unwrap().0, EventType::End);

    drop(captured);
    nvagentrt_deregister_subscriber("stream_end_test").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_error_propagation() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items: Vec<Result<String>> = vec![
        Ok("good chunk".to_string()),
        Err(nvagentrt_core::AgentRtError::Internal(
            "stream error".into(),
        )),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let first = wrapper.next().await.unwrap();
    assert!(first.is_ok());
    assert_eq!(first.unwrap(), "good chunk");

    let second = wrapper.next().await.unwrap();
    assert!(second.is_err());
}

#[tokio::test]
async fn test_stream_wrapper_json_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![
        Ok("{\"token\": \"hello\"}".to_string()),
        Ok("{\"token\": \"world\"}".to_string()),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    // Verify chunks are valid JSON
    let _: serde_json::Value = serde_json::from_str(&chunks[0]).unwrap();
    let _: serde_json::Value = serde_json::from_str(&chunks[1]).unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_non_json_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Non-JSON chunks should pass through; they just won't appear in aggregated response
    let items = vec![Ok("plain text".to_string()), Ok("more text".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "plain text");
    assert_eq!(chunks[1], "more text");
}

#[tokio::test]
async fn test_stream_wrapper_collector_receives_all_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![
        Ok("chunk1".to_string()),
        Ok("chunk2".to_string()),
        Ok("chunk3".to_string()),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    let chunks = collected.lock().unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0], "chunk1");
    assert_eq!(chunks[1], "chunk2");
    assert_eq!(chunks[2], "chunk3");
}

#[tokio::test]
async fn test_stream_wrapper_finalizer_called_on_exhaustion() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let finalizer_called = Arc::new(Mutex::new(false));
    let fc = finalizer_called.clone();

    let items = vec![Ok("chunk".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let collector: Box<dyn FnMut(String) + Send> = Box::new(|_| {});
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
        *fc.lock().unwrap() = true;
        json!({"finalized": true})
    });
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Finalizer should not be called yet
    assert!(!*finalizer_called.lock().unwrap());

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    // Finalizer should have been called exactly once
    assert!(*finalizer_called.lock().unwrap());
}

#[tokio::test]
async fn test_stream_wrapper_response_intercepts_on_aggregated() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Register a response intercept that adds a field to the aggregated response
    nvagentrt_register_llm_response_intercept(
        "resp_intercept",
        1,
        false,
        Box::new(|mut resp: Json| {
            resp.as_object_mut()
                .unwrap()
                .insert("intercepted".into(), json!(true));
            resp
        }),
    )
    .unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    nvagentrt_register_subscriber(
        "resp_intercept_test",
        Box::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let items = vec![Ok("{\"token\": \"hi\"}".to_string())];
    let inner = make_stream(items);

    let request = LLMRequest {
        method: "POST".into(),
        url: "https://api.example.com".into(),
        headers: serde_json::Map::new(),
        body: json!({}),
    };
    let handle = nvagentrt_llm_call(
        "test_llm",
        &request,
        None,
        LLMAttributes::STREAMING,
        None,
        None,
        None,
    )
    .unwrap();

    let collector: Box<dyn FnMut(String) + Send> = Box::new(|_| {});
    let finalizer: Box<dyn FnOnce() -> Json + Send> =
        Box::new(|| json!({"aggregated": "response"}));
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    // The END event should contain the response-intercepted output
    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| e.event_type == EventType::End)
        .unwrap();
    let output = end_event.output.as_ref().unwrap();
    assert_eq!(output["aggregated"], "response");
    assert_eq!(output["intercepted"], true);

    drop(captured);
    nvagentrt_deregister_subscriber("resp_intercept_test").unwrap();
    nvagentrt_deregister_llm_response_intercept("resp_intercept").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_collector_receives_intercepted_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Register a stream intercept that transforms chunks
    nvagentrt_register_llm_stream_response_intercept(
        "prefix_intercept",
        1,
        false,
        Box::new(|chunk: String| format!("[prefix] {}", chunk)),
    )
    .unwrap();

    let items = vec![Ok("original".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let (collector, finalizer, collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    // Collector should have received the intercepted (post-stream-intercept) value
    let chunks = collected.lock().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], "[prefix] original");

    nvagentrt_deregister_llm_stream_response_intercept("prefix_intercept").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_error_skips_collector_finalizer() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let collector_calls = Arc::new(Mutex::new(0u32));
    let cc = collector_calls.clone();
    let finalizer_called = Arc::new(Mutex::new(false));
    let fc = finalizer_called.clone();

    let items: Vec<Result<String>> =
        vec![Err(nvagentrt_core::AgentRtError::Internal("error".into()))];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let collector: Box<dyn FnMut(String) + Send> = Box::new(move |_| {
        *cc.lock().unwrap() += 1;
    });
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(move || {
        *fc.lock().unwrap() = true;
        Json::Null
    });
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Consume the error
    let result = wrapper.next().await.unwrap();
    assert!(result.is_err());

    // Collector should not have been called for the error
    assert_eq!(*collector_calls.lock().unwrap(), 0);

    // Stream ends after error, finalizer gets called on None
    let _ = wrapper.next().await;
    // Finalizer is called when stream ends (even after error)
    assert!(*finalizer_called.lock().unwrap());
}

#[tokio::test]
async fn test_stream_wrapper_end_event_contains_intercepted_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    nvagentrt_register_subscriber(
        "end_event_test",
        Box::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let items = vec![
        Ok("{\"token\": \"a\"}".to_string()),
        Ok("{\"token\": \"b\"}".to_string()),
    ];
    let inner = make_stream(items);

    let request = LLMRequest {
        method: "POST".into(),
        url: "https://api.example.com".into(),
        headers: serde_json::Map::new(),
        body: json!({}),
    };
    let handle = nvagentrt_llm_call(
        "test_llm",
        &request,
        None,
        LLMAttributes::STREAMING,
        None,
        None,
        None,
    )
    .unwrap();

    let (collector, finalizer, _collected) = make_collector_finalizer();
    let mut wrapper = LlmStreamWrapper::new(inner, handle, collector, finalizer, None, None);

    // Consume the stream
    while let Some(_item) = wrapper.next().await {}

    // The END event output should contain the finalizer's aggregated response
    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| e.event_type == EventType::End)
        .unwrap();
    let output = end_event.output.as_ref().unwrap();
    // The default finalizer collects JSON-parseable chunks into an array
    assert!(output.is_array());
    let arr = output.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["token"], "a");
    assert_eq!(arr[1]["token"], "b");

    drop(captured);
    nvagentrt_deregister_subscriber("end_event_test").unwrap();
}
