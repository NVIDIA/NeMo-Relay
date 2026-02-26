// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::await_holding_lock)]

use std::pin::Pin;
use std::sync::Mutex;

use nvagentrt_core::context::*;
use nvagentrt_core::error::Result;
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

#[tokio::test]
async fn test_stream_wrapper_basic_events() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![
        Ok("data: {\"token\": \"hello\"}\n\n".to_string()),
        Ok("data: {\"token\": \"world\"}\n\n".to_string()),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    // Each chunk should be a serialized SSE event
    assert!(chunks[0].contains("data:"));
    assert!(chunks[1].contains("data:"));
}

#[tokio::test]
async fn test_stream_wrapper_partial_buffering() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Send data in partial chunks that don't align with \n\n boundaries
    let items = vec![Ok("data: part".to_string()), Ok("ial\n\n".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    // Should produce 1 complete event from the buffered partial data
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].contains("partial"));
}

#[tokio::test]
async fn test_stream_wrapper_multiple_events_in_one_chunk() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![Ok("data: first\n\ndata: second\n\n".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 2);
}

#[tokio::test]
async fn test_stream_wrapper_empty_stream() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>> = Box::pin(tokio_stream::empty());
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let mut count = 0;
    while let Some(_item) = wrapper.next().await {
        count += 1;
    }
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_stream_wrapper_trailing_data_flushed() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Data without trailing \n\n should be flushed when stream ends
    let items = vec![Ok("data: no_boundary".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    // Should flush the remaining buffer as an event
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].contains("no_boundary"));
}

#[tokio::test]
async fn test_stream_wrapper_with_intercept() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Register a stream response intercept
    nvagentrt_register_llm_stream_response_intercept(
        "test_stream_intercept",
        1,
        false,
        Box::new(|mut event: SseEvent| {
            event.data = format!("[intercepted] {}", event.data);
            event
        }),
    )
    .unwrap();

    let items = vec![Ok("data: original\n\n".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let mut chunks = Vec::new();
    while let Some(item) = wrapper.next().await {
        chunks.push(item.unwrap());
    }

    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].contains("[intercepted] original"));

    nvagentrt_deregister_llm_stream_response_intercept("test_stream_intercept").unwrap();
}

#[tokio::test]
async fn test_stream_wrapper_emits_end_event() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let events = std::sync::Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    nvagentrt_register_subscriber(
        "stream_end_test",
        Box::new(move |e: &Event| {
            ec.lock().unwrap().push((e.event_type, e.scope_type));
        }),
    )
    .unwrap();

    let items = vec![Ok("data: {\"token\": \"hi\"}\n\n".to_string())];
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
    )
    .unwrap();

    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

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
        Ok("data: good\n\n".to_string()),
        Err(nvagentrt_core::AgentRtError::Internal(
            "stream error".into(),
        )),
    ];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let first = wrapper.next().await.unwrap();
    assert!(first.is_ok());

    let second = wrapper.next().await.unwrap();
    assert!(second.is_err());
}

#[tokio::test]
async fn test_stream_wrapper_sse_event_with_all_fields() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![Ok(
        "event: chunk\nid: 1\nretry: 5000\ndata: payload\n\n".to_string()
    )];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let chunk = wrapper.next().await.unwrap().unwrap();
    // The output should contain the serialized SSE event
    assert!(chunk.contains("event: chunk"));
    assert!(chunk.contains("data: payload"));
    assert!(chunk.contains("id: 1"));
    assert!(chunk.contains("retry: 5000"));
}

#[tokio::test]
async fn test_stream_wrapper_multiline_data() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let items = vec![Ok("data: line1\ndata: line2\n\n".to_string())];
    let inner = make_stream(items);
    let handle = make_llm_handle("test_llm");
    let mut wrapper = LlmStreamWrapper::new(inner, handle, None, None);

    let chunk = wrapper.next().await.unwrap().unwrap();
    assert!(chunk.contains("data: line1"));
    assert!(chunk.contains("data: line2"));
}
