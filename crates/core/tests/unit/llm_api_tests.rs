// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for LLM API lifecycle behavior.

#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use serde_json::json;
use tokio_stream::StreamExt;

use super::{
    LlmCallExecuteParams, LlmCallParams, LlmHandle, LlmRequest, LlmStreamCallExecuteParams,
    emit_optimization_marks_with, llm_call, llm_call_execute, llm_stream_call_execute,
    project_llm_request_to_current_user_turn,
};
use crate::api::event::{Event, ScopeCategory};
use crate::api::optimization::finalize_optimization_summary;
use crate::api::runtime::LlmJsonStream;
use crate::api::runtime::{
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use crate::api::scope::{COMPACTION_EVENT_NAME, EmitMarkEventParams, event};
use crate::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::request::{AnnotatedLlmRequest, Message};
use crate::codec::traits::LlmCodec;
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

fn multi_turn_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "demo",
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "latest question"}
            ]
        }),
    }
}

fn multi_turn_annotation() -> Arc<AnnotatedLlmRequest> {
    Arc::new(OpenAIChatCodec.decode(&multi_turn_request()).unwrap())
}

struct ProjectionFailingCodec {
    projection_attempts: Arc<AtomicUsize>,
}

impl LlmCodec for ProjectionFailingCodec {
    fn decode(&self, request: &LlmRequest) -> crate::error::Result<AnnotatedLlmRequest> {
        OpenAIChatCodec.decode(request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> crate::error::Result<LlmRequest> {
        let original_messages = original.content["messages"].as_array().map_or(0, Vec::len);
        if annotated.messages.len() < original_messages {
            self.projection_attempts.fetch_add(1, Ordering::Relaxed);
            return Err(FlowError::Internal("projection encode failed".into()));
        }
        OpenAIChatCodec.encode(annotated, original)
    }
}

fn emit_compaction() {
    event(
        EmitMarkEventParams::builder()
            .name(COMPACTION_EVENT_NAME)
            .build(),
    )
    .unwrap();
}

#[test]
fn freshness_culls_annotations_and_repeated_compactions_are_idempotent() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let raw_request = multi_turn_request();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "freshness-culling",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let emit = |name| {
        llm_call(
            LlmCallParams::builder()
                .name(name)
                .request(&raw_request)
                .annotated_request(multi_turn_annotation())
                .build(),
        )
        .unwrap();
    };
    emit("fresh-start");
    emit("stale-start");
    // PreCompact and PostCompact both normalize to this canonical mark.
    emit_compaction();
    emit_compaction();
    emit("post-compaction-start");
    emit("post-compaction-stale");

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("freshness-culling").unwrap());
    let starts = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
        .map(|event| {
            assert_eq!(event.input().unwrap()["content"], raw_request.content);
            (
                event.name().to_string(),
                event.annotated_request().unwrap().messages.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(starts[0].0, "fresh-start");
    assert_eq!(starts[0].1.len(), 4);
    assert_eq!(starts[1].0, "stale-start");
    assert_eq!(starts[1].1.len(), 2);
    assert!(matches!(starts[1].1[0], Message::System { .. }));
    assert!(matches!(starts[1].1[1], Message::User { .. }));
    assert_eq!(starts[2].0, "post-compaction-start");
    assert_eq!(starts[2].1.len(), 4);
    assert_eq!(starts[3].0, "post-compaction-stale");
    assert_eq!(starts[3].1.len(), 2);
}

#[test]
fn request_projection_handles_history_edge_cases() {
    let project = |messages: Json| {
        let annotated: AnnotatedLlmRequest =
            serde_json::from_value(json!({"messages": messages})).unwrap();
        let mut annotated = Some(Arc::new(annotated));
        let mut request = LlmRequest {
            headers: serde_json::Map::new(),
            content: Json::Null,
        };
        project_llm_request_to_current_user_turn(&mut request, &mut annotated, None);
        annotated.unwrap().messages.clone()
    };

    assert!(project(json!([])).is_empty());

    let no_user = project(json!([
        {"role": "assistant", "content": "tool request"},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"}
    ]));
    assert_eq!(no_user.len(), 1);
    assert!(matches!(no_user[0], Message::Tool { .. }));

    let instructions_only = project(json!([
        {"role": "system", "content": "first instruction"},
        {"role": "system", "content": "second instruction"}
    ]));
    assert_eq!(instructions_only.len(), 2);
    assert!(
        instructions_only
            .iter()
            .all(|message| matches!(message, Message::System { .. }))
    );

    let current_turn = project(json!([
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"}
    ]));
    assert_eq!(current_turn.len(), 4);
    assert!(matches!(current_turn[0], Message::System { .. }));
    assert!(matches!(current_turn[1], Message::User { .. }));
    assert!(matches!(current_turn[2], Message::Assistant { .. }));
    assert!(matches!(current_turn[3], Message::Tool { .. }));

    let original = Arc::new(
        serde_json::from_value::<AnnotatedLlmRequest>(json!({
            "messages": [{"role": "system", "content": "instructions"}]
        }))
        .unwrap(),
    );
    let mut annotated = Some(original.clone());
    let mut request = LlmRequest {
        headers: serde_json::Map::new(),
        content: Json::Null,
    };
    project_llm_request_to_current_user_turn(&mut request, &mut annotated, Some(&OpenAIChatCodec));
    assert!(Arc::ptr_eq(annotated.as_ref().unwrap(), &original));
}

#[test]
fn managed_and_streaming_calls_cull_event_inputs_and_annotations_with_real_codec() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "managed-streaming-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    let codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for name in ["managed-fresh", "managed-stale"] {
            let response = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            Ok(json!({
                                "provider_message_count": request.content["messages"]
                                    .as_array()
                                    .unwrap()
                                    .len()
                            }))
                        })
                    }))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(response["provider_message_count"], 4);
        }

        emit_compaction();
        for name in ["stream-fresh", "stream-stale"] {
            let mut stream = llm_stream_call_execute(
                LlmStreamCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            assert_eq!(request.content["messages"].as_array().unwrap().len(), 4);
                            Ok(
                                Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": true}))]))
                                    as LlmJsonStream,
                            )
                        })
                    }))
                    .collector(Box::new(|_| Ok(())))
                    .finalizer(Box::new(|| json!({"done": true})))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
        }
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("managed-streaming-freshness").unwrap());
    let events = events.lock().unwrap();
    for (name, expected_messages) in [
        ("managed-fresh", 4),
        ("managed-stale", 2),
        ("stream-fresh", 4),
        ("stream-stale", 2),
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.input().unwrap()["content"]["messages"]
                .as_array()
                .unwrap()
                .len(),
            expected_messages,
            "unexpected raw event history for {name}"
        );
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            expected_messages,
            "unexpected annotation history for {name}"
        );
    }
}

#[test]
fn projection_encode_failures_do_not_block_managed_or_streaming_calls() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "projection-encode-failure",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let projection_attempts = Arc::new(AtomicUsize::new(0));
    let codec: Arc<dyn LlmCodec> = Arc::new(ProjectionFailingCodec {
        projection_attempts: projection_attempts.clone(),
    });
    let request = multi_turn_request();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for name in ["managed-encode-fresh", "managed-encode-stale"] {
            let response = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            Ok(json!({
                                "provider_message_count": request.content["messages"]
                                    .as_array()
                                    .unwrap()
                                    .len()
                            }))
                        })
                    }))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(response["provider_message_count"], 4);
        }

        emit_compaction();
        for name in ["stream-encode-fresh", "stream-encode-stale"] {
            let mut stream = llm_stream_call_execute(
                LlmStreamCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            assert_eq!(request.content["messages"].as_array().unwrap().len(), 4);
                            Ok(
                                Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": true}))]))
                                    as LlmJsonStream,
                            )
                        })
                    }))
                    .collector(Box::new(|_| Ok(())))
                    .finalizer(Box::new(|| json!({"done": true})))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
        }
    });

    assert_eq!(projection_attempts.load(Ordering::Relaxed), 2);
    flush_subscribers().unwrap();
    assert!(deregister_subscriber("projection-encode-failure").unwrap());
    let events = events.lock().unwrap();
    for name in [
        "managed-encode-fresh",
        "managed-encode-stale",
        "stream-encode-fresh",
        "stream-encode-stale",
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.input().unwrap()["content"]["messages"]
                .as_array()
                .unwrap()
                .len(),
            4,
            "encode failure should preserve the full event input for {name}"
        );
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            4,
            "encode failure should preserve the full annotation for {name}"
        );
    }
}

#[test]
fn nested_agents_track_freshness_independently_end_to_end() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "nested-agent-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    let emit = |name| {
        llm_call(
            LlmCallParams::builder()
                .name(name)
                .request(&request)
                .annotated_request(multi_turn_annotation())
                .build(),
        )
        .unwrap();
    };

    let parent = push_scope(
        PushScopeParams::builder()
            .name("parent-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    emit("parent-fresh");
    emit("parent-stale");

    let child = push_scope(
        PushScopeParams::builder()
            .name("child-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    emit("child-fresh");
    emit("child-stale");
    emit_compaction();
    emit_compaction();
    emit("child-after-compaction");
    emit("child-after-compaction-stale");
    pop_scope(PopScopeParams::builder().handle_uuid(&child.uuid).build()).unwrap();

    emit("parent-after-child-compaction");
    emit_compaction();
    emit("parent-after-compaction");
    pop_scope(PopScopeParams::builder().handle_uuid(&parent.uuid).build()).unwrap();

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("nested-agent-freshness").unwrap());
    let events = events.lock().unwrap();
    for (name, expected_messages) in [
        ("parent-fresh", 4),
        ("parent-stale", 2),
        ("child-fresh", 4),
        ("child-stale", 2),
        ("child-after-compaction", 4),
        ("child-after-compaction-stale", 2),
        ("parent-after-child-compaction", 2),
        ("parent-after-compaction", 4),
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            expected_messages,
            "unexpected annotation history for {name}"
        );
    }
}

#[test]
fn non_agent_scopes_share_the_implicit_root_freshness_budget() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "implicit-root-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    for (scope_name, scope_type, event_name) in [
        ("request-a", ScopeType::Custom, "root-fresh"),
        ("request-b", ScopeType::Function, "root-stale"),
    ] {
        let scope = push_scope(
            PushScopeParams::builder()
                .name(scope_name)
                .scope_type(scope_type)
                .build(),
        )
        .unwrap();
        llm_call(
            LlmCallParams::builder()
                .name(event_name)
                .request(&request)
                .annotated_request(multi_turn_annotation())
                .build(),
        )
        .unwrap();
        pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();
    }

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("implicit-root-freshness").unwrap());
    let events = events.lock().unwrap();
    for (name, expected_messages) in [("root-fresh", 4), ("root-stale", 2)] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            expected_messages,
            "non-agent scopes should inherit the implicit root freshness for {name}"
        );
    }
}

#[test]
fn concurrent_starts_consume_freshness_exactly_once_without_ordering() {
    let _guard = lock_global_runtime();
    reset_global();

    let shared_stack = create_scope_stack();
    set_thread_scope_stack(shared_stack.clone());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "concurrent-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for name in ["concurrent-a", "concurrent-b"] {
        let stack = shared_stack.clone();
        let barrier = barrier.clone();
        workers.push(std::thread::spawn(move || {
            set_thread_scope_stack(stack);
            barrier.wait();
            let request = multi_turn_request();
            llm_call(
                LlmCallParams::builder()
                    .name(name)
                    .request(&request)
                    .annotated_request(multi_turn_annotation())
                    .build(),
            )
            .unwrap();
        }));
    }
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("concurrent-freshness").unwrap());
    let events = events.lock().unwrap();
    let mut history_lengths = events
        .iter()
        .filter(|event| {
            event.name().starts_with("concurrent-")
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .map(|event| event.annotated_request().unwrap().messages.len())
        .collect::<Vec<_>>();
    history_lengths.sort_unstable();
    assert_eq!(history_lengths, vec![2, 4]);
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
