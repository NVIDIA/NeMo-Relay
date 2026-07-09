// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for LLM API lifecycle behavior.

#![allow(clippy::await_holding_lock)]

use std::sync::{Arc, Mutex};

use serde_json::json;
use tokio_stream::StreamExt;

use super::{
    LlmCallExecuteParams, LlmHandle, LlmRequest, LlmStreamCallExecuteParams,
    emit_optimization_marks_with, llm_call_execute, llm_stream_call_execute,
};
use crate::api::event::ScopeCategory;
use crate::api::optimization::finalize_optimization_summary;
use crate::api::runtime::LlmJsonStream;
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;
use crate::json::Json;
use crate::{codec::optimization::LlmOptimizationContribution, codec::response::PricingResolver};

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": [], "model": "demo"}),
    }
}

#[test]
fn rejected_optimization_mark_queue_keeps_cursor_and_summary_evidence() {
    let _guard = lock_global_runtime();
    reset_global();

    let handle = LlmHandle::builder().name("queue-rejection-test").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new("test", "queue_rejection"))
    );
    emit_optimization_marks_with(&handle, &[], Some, |_event, _subscribers| false);
    assert_eq!(handle.optimization_recorder.unemitted().len(), 1);

    let summary = finalize_optimization_summary(
        &handle.optimization_recorder,
        None,
        None,
        &PricingResolver::default(),
    )
    .expect("queue rejection must not discard close-time evidence");
    assert_eq!(summary.contributions.len(), 1);
    assert_eq!(summary.contributions[0].producer, "test");
}

#[test]
fn unavailable_mark_sanitizer_does_not_acknowledge_the_delivery_cursor() {
    let _guard = lock_global_runtime();
    reset_global();

    let handle = LlmHandle::builder().name("sanitizer-unavailable").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new("test", "sanitize_retry"))
    );
    handle.optimization_recorder.close_for_finalization(None);
    emit_optimization_marks_with(
        &handle,
        &[],
        |_event| None,
        |_event, _subscribers| panic!("unavailable sanitization must not enqueue"),
    );
    assert_eq!(handle.optimization_recorder.unemitted().len(), 1);

    emit_optimization_marks_with(&handle, &[], Some, |_event, _subscribers| true);
    assert!(handle.optimization_recorder.unemitted().is_empty());
}

#[test]
fn close_boundary_freezes_identical_mark_and_summary_contributions() {
    let _guard = lock_global_runtime();
    reset_global();

    let handle = LlmHandle::builder().name("close-boundary").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new(
                "accepted",
                "close_boundary"
            ))
    );
    assert!(handle.optimization_recorder.close_for_finalization(None));
    assert!(
        !handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new("late", "close_boundary"))
    );

    let mut marks = Vec::new();
    emit_optimization_marks_with(&handle, &[], Some, |event, _subscribers| {
        marks.push(event.clone());
        true
    });
    let summary = finalize_optimization_summary(
        &handle.optimization_recorder,
        None,
        None,
        &PricingResolver::default(),
    )
    .unwrap();
    assert_eq!(marks.len(), 1);
    assert_eq!(summary.contributions.len(), 1);
    assert_eq!(marks[0].data().unwrap()["producer"], "accepted");
    assert_eq!(
        marks[0].data().unwrap()["id"],
        json!(summary.contributions[0].id.unwrap())
    );
}

#[test]
fn llm_call_execute_adds_otel_status_metadata_to_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "llm-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let response = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("llm-ok")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Ok(json!({"ok": true})) })
                }))
                .metadata(json!({"caller": "llm-ok", "otel.status_code": "USER"}))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(response, json!({"ok": true}));

        let error = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("llm-error")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Err(FlowError::Internal("llm boom".to_string())) })
                }))
                .metadata(json!({"caller": "llm-error"}))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("llm boom"));
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("llm-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let metadata_for = |name: &str| {
        events
            .iter()
            .find(|event| event.0 == name)
            .and_then(|event| event.1.as_ref())
            .unwrap_or_else(|| panic!("missing end event metadata for {name}"))
    };

    let success_metadata = metadata_for("llm-ok");
    assert_eq!(success_metadata["caller"], json!("llm-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());

    let error_metadata = metadata_for("llm-error");
    assert_eq!(error_metadata["caller"], json!("llm-error"));
    assert_eq!(error_metadata["otel.status_code"], json!("ERROR"));
    assert!(
        error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("llm boom")
    );
}

#[test]
fn llm_stream_call_execute_adds_otel_status_metadata_to_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "llm-stream-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream-ok")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async {
                        Ok(
                            Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": true}))]))
                                as LlmJsonStream,
                        )
                    })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| json!({"ok": true})))
                .metadata(json!({"caller": "llm-stream-ok", "otel.status_code": "USER"}))
                .build(),
        )
        .await
        .unwrap();

        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("llm-stream-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let success_metadata = events
        .iter()
        .find(|event| event.0 == "llm-stream-ok")
        .and_then(|event| event.1.as_ref())
        .unwrap_or_else(|| panic!("missing stream end event metadata"));
    assert_eq!(success_metadata["caller"], json!("llm-stream-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());
}

#[test]
fn llm_stream_call_execute_adds_otel_error_metadata_to_failed_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "llm-stream-error-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut upstream_error_stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream-upstream-error")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async {
                        Ok(Box::pin(tokio_stream::iter(vec![Err(FlowError::Internal(
                            "stream boom".to_string(),
                        ))])) as LlmJsonStream)
                    })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| json!({"partial": true})))
                .metadata(
                    json!({"caller": "llm-stream-upstream-error", "otel.status_code": "USER"}),
                )
                .build(),
        )
        .await
        .unwrap();
        let upstream_error = upstream_error_stream.next().await.unwrap().unwrap_err();
        assert!(upstream_error.to_string().contains("stream boom"));

        let mut collector_error_stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream-collector-error")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async {
                        Ok(
                            Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": true}))]))
                                as LlmJsonStream,
                        )
                    })
                }))
                .collector(Box::new(|_chunk| {
                    Err(FlowError::Internal("collector boom".to_string()))
                }))
                .finalizer(Box::new(|| json!({"partial": true})))
                .metadata(
                    json!({"caller": "llm-stream-collector-error", "otel.status_code": "USER"}),
                )
                .build(),
        )
        .await
        .unwrap();
        let collector_error = collector_error_stream.next().await.unwrap().unwrap_err();
        assert!(collector_error.to_string().contains("collector boom"));
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("llm-stream-error-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let metadata_for = |name: &str| {
        events
            .iter()
            .find(|event| event.0 == name)
            .and_then(|event| event.1.as_ref())
            .unwrap_or_else(|| panic!("missing stream end event metadata for {name}"))
    };

    let upstream_error_metadata = metadata_for("llm-stream-upstream-error");
    assert_eq!(
        upstream_error_metadata["caller"],
        json!("llm-stream-upstream-error")
    );
    assert_eq!(upstream_error_metadata["otel.status_code"], json!("ERROR"));
    assert!(
        upstream_error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("stream boom")
    );

    let collector_error_metadata = metadata_for("llm-stream-collector-error");
    assert_eq!(
        collector_error_metadata["caller"],
        json!("llm-stream-collector-error")
    );
    assert_eq!(collector_error_metadata["otel.status_code"], json!("ERROR"));
    assert!(
        collector_error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("collector boom")
    );
}
