// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for api surface in the NeMo Relay core crate.

#![allow(clippy::await_holding_lock)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, TimeDelta, Utc};
use futures::StreamExt;
use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmCallParams, LlmStreamCallExecuteParams, llm_call, llm_call_end,
    llm_call_execute, llm_conditional_execution, llm_request_intercepts, llm_stream_call_execute,
};
use nemo_relay::api::registry::{
    deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
    deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_response_guardrail, deregister_llm_stream_execution_intercept,
    deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
    deregister_tool_request_intercept, deregister_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_response_guardrail, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
    register_llm_stream_execution_intercept, register_tool_conditional_execution_guardrail,
    register_tool_execution_intercept, register_tool_request_intercept,
    register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
    scope_deregister_llm_conditional_execution_guardrail, scope_deregister_llm_execution_intercept,
    scope_deregister_llm_request_intercept, scope_deregister_llm_sanitize_request_guardrail,
    scope_deregister_llm_sanitize_response_guardrail,
    scope_deregister_llm_stream_execution_intercept,
    scope_deregister_tool_conditional_execution_guardrail,
    scope_deregister_tool_execution_intercept, scope_deregister_tool_request_intercept,
    scope_deregister_tool_sanitize_request_guardrail,
    scope_deregister_tool_sanitize_response_guardrail,
    scope_register_llm_conditional_execution_guardrail, scope_register_llm_execution_intercept,
    scope_register_llm_request_intercept, scope_register_llm_sanitize_request_guardrail,
    scope_register_llm_sanitize_response_guardrail, scope_register_llm_stream_execution_intercept,
    scope_register_tool_conditional_execution_guardrail, scope_register_tool_execution_intercept,
    scope_register_tool_request_intercept, scope_register_tool_sanitize_request_guardrail,
    scope_register_tool_sanitize_response_guardrail,
};
use nemo_relay::api::runtime::NemoRelayContextState;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::runtime::{create_scope_stack, set_thread_scope_stack};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::api::scope::{event, pop_scope, push_scope};
use nemo_relay::api::subscriber::{
    deregister_subscriber, register_subscriber, scope_deregister_subscriber,
    scope_register_subscriber,
};
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::api::tool::{
    tool_call, tool_call_end, tool_call_execute, tool_conditional_execution,
    tool_request_intercepts,
};
use nemo_relay::error::{FlowError, Result};
use nemo_relay::json::Json;
use serde_json::{Map, json};
use tokio_stream::Stream;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn make_llm_request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content,
    }
}

fn utc_timestamp(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .unwrap()
        .with_timezone(&Utc)
}

fn capture_events(name: &str) -> Arc<Mutex<Vec<Event>>> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = events.clone();
    register_subscriber(
        name,
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    events
}

fn expect_already_exists(error: FlowError, needle: &str) {
    match error {
        FlowError::AlreadyExists(message) => assert!(message.contains(needle)),
        other => panic!("expected AlreadyExists, got {other}"),
    }
}

fn expect_not_found(error: FlowError, needle: &str) {
    match error {
        FlowError::NotFound(message) => assert!(message.contains(needle)),
        other => panic!("expected NotFound, got {other}"),
    }
}

#[test]
fn test_manual_lifecycle_timestamp_overrides() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("timestamp-api-events");
    let scope_start = utc_timestamp("2026-01-01T00:00:00.123456Z");
    let mark_timestamp = utc_timestamp("2026-01-01T00:00:01.223456Z");
    let tool_start = utc_timestamp("2026-01-01T00:00:02.323456Z");
    let tool_end = utc_timestamp("2026-01-01T00:00:03.423456Z");
    let llm_start = utc_timestamp("2026-01-01T00:00:04.523456Z");
    let llm_end = utc_timestamp("2026-01-01T00:00:05.623456Z");
    let scope_end = utc_timestamp("2026-01-01T00:00:06.723456Z");

    let scope_handle = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("timestamp-scope")
            .scope_type(ScopeType::Agent)
            .timestamp(scope_start)
            .build(),
    )
    .unwrap();
    event(
        nemo_relay::api::scope::EmitMarkEventParams::builder()
            .name("timestamp-mark")
            .parent(&scope_handle)
            .timestamp(mark_timestamp)
            .build(),
    )
    .unwrap();
    let tool_handle = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("timestamp-tool")
            .args(json!({"x": 1}))
            .timestamp(tool_start)
            .build(),
    )
    .unwrap();
    tool_call_end(
        nemo_relay::api::tool::ToolCallEndParams::builder()
            .handle(&tool_handle)
            .result(json!({"ok": true}))
            .timestamp(tool_end)
            .build(),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": []}));
    let llm_handle = llm_call(
        LlmCallParams::builder()
            .name("timestamp-llm")
            .request(&request)
            .timestamp(llm_start)
            .build(),
    )
    .unwrap();
    llm_call_end(
        nemo_relay::api::llm::LlmCallEndParams::builder()
            .handle(&llm_handle)
            .response(json!({"ok": true}))
            .timestamp(llm_end)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope_handle.uuid)
            .timestamp(scope_end)
            .build(),
    )
    .unwrap();

    let captured = events.lock().unwrap();
    let observed: Vec<_> = captured
        .iter()
        .map(|event| (event.name().to_owned(), *event.timestamp()))
        .collect();
    assert_eq!(
        observed,
        vec![
            ("timestamp-scope".to_string(), scope_start),
            ("timestamp-mark".to_string(), mark_timestamp),
            ("timestamp-tool".to_string(), tool_start),
            ("timestamp-tool".to_string(), tool_end),
            ("timestamp-llm".to_string(), llm_start),
            ("timestamp-llm".to_string(), llm_end),
            ("timestamp-scope".to_string(), scope_end),
        ]
    );
    drop(captured);

    deregister_subscriber("timestamp-api-events").unwrap();
}

#[test]
fn test_manual_lifecycle_default_end_timestamps_follow_explicit_starts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("default-end-timestamp-events");
    let scope_start = utc_timestamp("2099-02-01T00:00:00.111111Z");
    let tool_start = utc_timestamp("2099-02-01T00:00:01.222222Z");
    let llm_start = utc_timestamp("2099-02-01T00:00:02.333333Z");

    let scope_handle = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("default_ts_scope")
            .scope_type(ScopeType::Agent)
            .timestamp(scope_start)
            .build(),
    )
    .unwrap();
    let tool_handle = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("default_ts_tool")
            .args(json!({"x": 1}))
            .timestamp(tool_start)
            .build(),
    )
    .unwrap();
    tool_call_end(
        nemo_relay::api::tool::ToolCallEndParams::builder()
            .handle(&tool_handle)
            .result(json!({"ok": true}))
            .build(),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": []}));
    let llm_handle = llm_call(
        LlmCallParams::builder()
            .name("default_ts_llm")
            .request(&request)
            .timestamp(llm_start)
            .build(),
    )
    .unwrap();
    llm_call_end(
        nemo_relay::api::llm::LlmCallEndParams::builder()
            .handle(&llm_handle)
            .response(json!({"ok": true}))
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope_handle.uuid)
            .build(),
    )
    .unwrap();

    let captured = events.lock().unwrap();
    let observed: Vec<_> = captured
        .iter()
        .filter(|event| event.name().starts_with("default_ts_"))
        .map(|event| (event.name().to_owned(), *event.timestamp()))
        .collect();
    let one_microsecond = TimeDelta::microseconds(1);
    assert_eq!(
        observed,
        vec![
            ("default_ts_scope".to_string(), scope_start),
            ("default_ts_tool".to_string(), tool_start),
            ("default_ts_tool".to_string(), tool_start + one_microsecond),
            ("default_ts_llm".to_string(), llm_start),
            ("default_ts_llm".to_string(), llm_start + one_microsecond),
            (
                "default_ts_scope".to_string(),
                scope_start + one_microsecond
            ),
        ]
    );
    drop(captured);

    deregister_subscriber("default-end-timestamp-events").unwrap();
}

fn noop_tool_exec() -> ToolExecutionNextFn {
    Arc::new(|args| Box::pin(async move { Ok(args) }))
}

fn failing_tool_exec() -> ToolExecutionNextFn {
    Arc::new(|_args| Box::pin(async { Err(FlowError::Internal("tool execution failed".into())) }))
}

fn noop_llm_exec() -> LlmExecutionNextFn {
    Arc::new(|request| Box::pin(async move { Ok(request.content) }))
}

fn failing_llm_exec() -> LlmExecutionNextFn {
    Arc::new(|_request| Box::pin(async { Err(FlowError::Internal("llm execution failed".into())) }))
}

fn noop_llm_stream_exec() -> LlmStreamExecutionNextFn {
    Arc::new(|request| {
        Box::pin(async move {
            Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        })
    })
}

fn fixed_llm_stream_exec(chunks: Vec<Json>) -> LlmStreamExecutionNextFn {
    let chunks = Arc::new(chunks);
    Arc::new(move |_request| {
        let chunks = chunks.clone();
        Box::pin(async move {
            let items = chunks.iter().cloned().map(Ok).collect::<Vec<_>>();
            Ok(Box::pin(tokio_stream::iter(items))
                as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
        })
    })
}

fn failing_llm_stream_exec() -> LlmStreamExecutionNextFn {
    Arc::new(|_request| {
        Box::pin(async { Err(FlowError::Internal("llm stream execution failed".into())) })
    })
}

#[test]
fn test_global_registry_and_subscriber_wrappers_cover_success_and_duplicates() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_sanitize_request_guardrail(
        "tool-sanitize-request",
        1,
        Arc::new(|_name, args| args),
    )
    .unwrap();
    expect_already_exists(
        register_tool_sanitize_request_guardrail(
            "tool-sanitize-request",
            1,
            Arc::new(|_name, args| args),
        )
        .unwrap_err(),
        "tool-sanitize-request",
    );
    assert!(deregister_tool_sanitize_request_guardrail("tool-sanitize-request").unwrap());
    assert!(!deregister_tool_sanitize_request_guardrail("tool-sanitize-request").unwrap());

    register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Arc::new(|_name, args| args),
    )
    .unwrap();
    assert!(deregister_tool_sanitize_response_guardrail("tool-sanitize-response").unwrap());

    register_tool_conditional_execution_guardrail(
        "tool-conditional",
        1,
        Arc::new(|_name, _args| Ok(None)),
    )
    .unwrap();
    assert!(deregister_tool_conditional_execution_guardrail("tool-conditional").unwrap());

    register_tool_request_intercept("tool-request", 1, false, Arc::new(|_name, args| Ok(args)))
        .unwrap();
    assert!(deregister_tool_request_intercept("tool-request").unwrap());

    register_tool_execution_intercept(
        "tool-execution",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    assert!(deregister_tool_execution_intercept("tool-execution").unwrap());

    register_llm_sanitize_request_guardrail("llm-sanitize-request", 1, Arc::new(|request| request))
        .unwrap();
    assert!(deregister_llm_sanitize_request_guardrail("llm-sanitize-request").unwrap());

    register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Arc::new(|response| response),
    )
    .unwrap();
    assert!(deregister_llm_sanitize_response_guardrail("llm-sanitize-response").unwrap());

    register_llm_conditional_execution_guardrail(
        "llm-conditional",
        1,
        Arc::new(|_request| Ok(None)),
    )
    .unwrap();
    assert!(deregister_llm_conditional_execution_guardrail("llm-conditional").unwrap());

    register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| Ok((request, annotated))),
    )
    .unwrap();
    assert!(deregister_llm_request_intercept("llm-request").unwrap());

    register_llm_execution_intercept(
        "llm-execution",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    assert!(deregister_llm_execution_intercept("llm-execution").unwrap());

    register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                    as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
            })
        }),
    )
    .unwrap();
    assert!(deregister_llm_stream_execution_intercept("llm-stream").unwrap());

    register_subscriber("global-subscriber", Arc::new(|_event| {})).unwrap();
    expect_already_exists(
        register_subscriber("global-subscriber", Arc::new(|_event| {})).unwrap_err(),
        "global-subscriber",
    );
    assert!(deregister_subscriber("global-subscriber").unwrap());
    assert!(!deregister_subscriber("global-subscriber").unwrap());
}

#[test]
fn test_scope_registry_and_subscriber_wrappers_cover_success_duplicates_and_missing_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let scope = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("scope-registry")
            .scope_type(ScopeType::Function)
            .build(),
    )
    .unwrap();

    scope_register_tool_sanitize_request_guardrail(
        &scope.uuid,
        "tool-sanitize-request",
        1,
        Arc::new(|_name, args| args),
    )
    .unwrap();
    expect_already_exists(
        scope_register_tool_sanitize_request_guardrail(
            &scope.uuid,
            "tool-sanitize-request",
            1,
            Arc::new(|_name, args| args),
        )
        .unwrap_err(),
        "tool-sanitize-request",
    );
    assert!(
        scope_deregister_tool_sanitize_request_guardrail(&scope.uuid, "tool-sanitize-request")
            .unwrap()
    );

    scope_register_tool_sanitize_response_guardrail(
        &scope.uuid,
        "tool-sanitize-response",
        1,
        Arc::new(|_name, args| args),
    )
    .unwrap();
    assert!(
        scope_deregister_tool_sanitize_response_guardrail(&scope.uuid, "tool-sanitize-response")
            .unwrap()
    );

    scope_register_tool_conditional_execution_guardrail(
        &scope.uuid,
        "tool-conditional",
        1,
        Arc::new(|_name, _args| Ok(None)),
    )
    .unwrap();
    assert!(
        scope_deregister_tool_conditional_execution_guardrail(&scope.uuid, "tool-conditional")
            .unwrap()
    );

    scope_register_tool_request_intercept(
        &scope.uuid,
        "tool-request",
        1,
        false,
        Arc::new(|_name, args| Ok(args)),
    )
    .unwrap();
    assert!(scope_deregister_tool_request_intercept(&scope.uuid, "tool-request").unwrap());

    scope_register_tool_execution_intercept(
        &scope.uuid,
        "tool-execution",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    assert!(scope_deregister_tool_execution_intercept(&scope.uuid, "tool-execution").unwrap());

    scope_register_llm_sanitize_request_guardrail(
        &scope.uuid,
        "llm-sanitize-request",
        1,
        Arc::new(|request| request),
    )
    .unwrap();
    assert!(
        scope_deregister_llm_sanitize_request_guardrail(&scope.uuid, "llm-sanitize-request")
            .unwrap()
    );

    scope_register_llm_sanitize_response_guardrail(
        &scope.uuid,
        "llm-sanitize-response",
        1,
        Arc::new(|response| response),
    )
    .unwrap();
    assert!(
        scope_deregister_llm_sanitize_response_guardrail(&scope.uuid, "llm-sanitize-response")
            .unwrap()
    );

    scope_register_llm_conditional_execution_guardrail(
        &scope.uuid,
        "llm-conditional",
        1,
        Arc::new(|_request| Ok(None)),
    )
    .unwrap();
    assert!(
        scope_deregister_llm_conditional_execution_guardrail(&scope.uuid, "llm-conditional")
            .unwrap()
    );

    scope_register_llm_request_intercept(
        &scope.uuid,
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| Ok((request, annotated))),
    )
    .unwrap();
    assert!(scope_deregister_llm_request_intercept(&scope.uuid, "llm-request").unwrap());

    scope_register_llm_execution_intercept(
        &scope.uuid,
        "llm-execution",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    assert!(scope_deregister_llm_execution_intercept(&scope.uuid, "llm-execution").unwrap());

    scope_register_llm_stream_execution_intercept(
        &scope.uuid,
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                    as Pin<Box<dyn Stream<Item = Result<Json>> + Send>>)
            })
        }),
    )
    .unwrap();
    assert!(scope_deregister_llm_stream_execution_intercept(&scope.uuid, "llm-stream").unwrap());

    scope_register_subscriber(&scope.uuid, "scope-subscriber", Arc::new(|_event| {})).unwrap();
    expect_already_exists(
        scope_register_subscriber(&scope.uuid, "scope-subscriber", Arc::new(|_event| {}))
            .unwrap_err(),
        "scope-subscriber",
    );
    assert!(scope_deregister_subscriber(&scope.uuid, "scope-subscriber").unwrap());
    assert!(!scope_deregister_subscriber(&scope.uuid, "scope-subscriber").unwrap());

    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope.uuid)
            .build(),
    )
    .unwrap();

    expect_not_found(
        scope_register_tool_sanitize_request_guardrail(
            &scope.uuid,
            "missing-tool-sanitize",
            1,
            Arc::new(|_name, args| args),
        )
        .unwrap_err(),
        "scope",
    );
    expect_not_found(
        scope_register_tool_request_intercept(
            &scope.uuid,
            "missing-tool-request",
            1,
            false,
            Arc::new(|_name, args| Ok(args)),
        )
        .unwrap_err(),
        "scope",
    );
    expect_not_found(
        scope_register_tool_execution_intercept(
            &scope.uuid,
            "missing-tool-exec",
            1,
            Arc::new(|_name, args, _next| Box::pin(async move { Ok(args) })),
        )
        .unwrap_err(),
        "scope",
    );
    expect_not_found(
        scope_register_subscriber(&scope.uuid, "missing-subscriber", Arc::new(|_event| {}))
            .unwrap_err(),
        "scope",
    );
}

#[tokio::test]
async fn test_tool_api_emits_sanitized_events_and_covers_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("tool-api-events");

    register_tool_sanitize_request_guardrail(
        "tool-sanitize-request",
        1,
        Arc::new(|_name, mut args| {
            args.as_object_mut()
                .unwrap()
                .insert("sanitized_request".into(), json!(true));
            args
        }),
    )
    .unwrap();
    register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Arc::new(|_name, mut result| {
            result
                .as_object_mut()
                .unwrap()
                .insert("sanitized_response".into(), json!(true));
            result
        }),
    )
    .unwrap();

    let handle = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool-api")
            .args(json!({"value": 1}))
            .attributes(ToolAttributes::REMOTE)
            .data(json!({"phase": "start"}))
            .metadata(json!({"meta": "tool"}))
            .tool_call_id("tool-call-id")
            .build(),
    )
    .unwrap();
    tool_call_end(
        nemo_relay::api::tool::ToolCallEndParams::builder()
            .handle(&handle)
            .result(json!({"ok": true}))
            .data(json!({"phase": "end"}))
            .metadata(json!({"meta": "tool"}))
            .build(),
    )
    .unwrap();

    let captured = events.lock().unwrap().clone();
    assert_eq!(captured[0].kind(), "scope");
    assert_eq!(captured[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(captured[0].category().unwrap().as_str(), "tool");
    assert_eq!(
        captured[0].input().unwrap()["sanitized_request"],
        json!(true)
    );
    assert_eq!(captured[0].tool_call_id(), Some("tool-call-id"));
    assert_eq!(captured[1].kind(), "scope");
    assert_eq!(captured[1].scope_category(), Some(ScopeCategory::End));
    assert_eq!(captured[1].category().unwrap().as_str(), "tool");
    assert_eq!(
        captured[1].output().unwrap()["sanitized_response"],
        json!(true)
    );
    assert_eq!(captured[1].tool_call_id(), Some("tool-call-id"));
    drop(captured);

    deregister_tool_sanitize_request_guardrail("tool-sanitize-request").unwrap();
    deregister_tool_sanitize_response_guardrail("tool-sanitize-response").unwrap();

    register_tool_request_intercept(
        "tool-request",
        1,
        false,
        Arc::new(|_name, mut args| {
            args.as_object_mut()
                .unwrap()
                .insert("intercepted".into(), json!(true));
            Ok(args)
        }),
    )
    .unwrap();
    assert_eq!(
        tool_request_intercepts("tool-api", json!({"value": 2})).unwrap()["intercepted"],
        json!(true)
    );
    deregister_tool_request_intercept("tool-request").unwrap();

    register_tool_conditional_execution_guardrail(
        "tool-reject",
        1,
        Arc::new(|_name, _args| Ok(Some("tool denied".into()))),
    )
    .unwrap();
    assert!(matches!(
        tool_conditional_execution("tool-api", &json!({"value": 3})),
        Err(FlowError::GuardrailRejected(reason)) if reason == "tool denied"
    ));
    assert!(matches!(
        tool_call_execute(
            nemo_relay::api::tool::ToolCallExecuteParams::builder()
                .name("tool-api")
                .args(json!({"value": 3}))
                .func(noop_tool_exec())
                .data(json!({"request": "rejected"}))
                .build()
        )
        .await,
        Err(FlowError::GuardrailRejected(reason)) if reason == "tool denied"
    ));
    let rejection_events = events.lock().unwrap().clone();
    let mark = rejection_events.last().unwrap();
    assert_eq!(mark.kind(), "mark");
    assert_eq!(mark.data().unwrap()["rejected"], json!(true));
    assert_eq!(
        mark.data().unwrap()["rejection_reason"],
        json!("tool denied")
    );
    drop(rejection_events);
    deregister_tool_conditional_execution_guardrail("tool-reject").unwrap();

    let baseline = events.lock().unwrap().len();
    assert!(matches!(
        tool_call_execute(
            nemo_relay::api::tool::ToolCallExecuteParams::builder()
                .name("tool-api")
                .args(json!({"value": 4}))
                .func(failing_tool_exec())
                .data(json!({"request": "failed"}))
                .build()
        )
        .await,
        Err(FlowError::Internal(message)) if message == "tool execution failed"
    ));
    let failed_events = events.lock().unwrap();
    assert_eq!(failed_events[baseline].kind(), "scope");
    assert_eq!(
        failed_events[baseline].scope_category(),
        Some(ScopeCategory::Start)
    );
    assert_eq!(failed_events[baseline].category().unwrap().as_str(), "tool");
    assert_eq!(failed_events[baseline + 1].kind(), "scope");
    assert_eq!(
        failed_events[baseline + 1].scope_category(),
        Some(ScopeCategory::End)
    );
    assert_eq!(
        failed_events[baseline + 1].category().unwrap().as_str(),
        "tool"
    );
    assert_eq!(
        failed_events[baseline + 1].output().unwrap(),
        &json!({"request": "failed"})
    );
    drop(failed_events);

    deregister_subscriber("tool-api-events").unwrap();
}

#[tokio::test]
async fn test_llm_api_emits_sanitized_events_and_covers_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("llm-api-events");

    register_llm_sanitize_request_guardrail(
        "llm-sanitize-request",
        1,
        Arc::new(|mut request| {
            request.headers.insert("x-sanitized".into(), json!(true));
            request
        }),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Arc::new(|mut response| {
            response
                .as_object_mut()
                .unwrap()
                .insert("sanitized_response".into(), json!(true));
            response
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]}));
    let handle = llm_call(
        LlmCallParams::builder()
            .name("llm-api")
            .request(&request)
            .attributes(LlmAttributes::STATEFUL)
            .data(json!({"phase": "start"}))
            .metadata(json!({"meta": "llm"}))
            .model_name("test-model")
            .build(),
    )
    .unwrap();
    llm_call_end(
        nemo_relay::api::llm::LlmCallEndParams::builder()
            .handle(&handle)
            .response(json!({"response": "ok"}))
            .data(json!({"phase": "end"}))
            .metadata(json!({"meta": "llm"}))
            .build(),
    )
    .unwrap();

    let captured = events.lock().unwrap().clone();
    assert_eq!(captured[0].kind(), "scope");
    assert_eq!(captured[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(captured[0].category().unwrap().as_str(), "llm");
    assert_eq!(
        captured[0].input().unwrap()["headers"]["x-sanitized"],
        json!(true)
    );
    assert_eq!(captured[0].model_name(), Some("test-model"));
    assert_eq!(captured[1].kind(), "scope");
    assert_eq!(captured[1].scope_category(), Some(ScopeCategory::End));
    assert_eq!(captured[1].category().unwrap().as_str(), "llm");
    assert_eq!(
        captured[1].output().unwrap()["sanitized_response"],
        json!(true)
    );
    assert_eq!(captured[1].model_name(), Some("test-model"));
    drop(captured);

    deregister_llm_sanitize_request_guardrail("llm-sanitize-request").unwrap();
    deregister_llm_sanitize_response_guardrail("llm-sanitize-response").unwrap();

    register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            request.headers.insert("x-intercepted".into(), json!(true));
            Ok((request, annotated))
        }),
    )
    .unwrap();
    let intercepted = llm_request_intercepts(
        "llm-api",
        make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]})),
    )
    .unwrap();
    assert_eq!(intercepted.headers.get("x-intercepted"), Some(&json!(true)));
    deregister_llm_request_intercept("llm-request").unwrap();

    register_llm_conditional_execution_guardrail(
        "llm-reject",
        1,
        Arc::new(|_request| Ok(Some("llm denied".into()))),
    )
    .unwrap();
    assert!(matches!(
        llm_conditional_execution(&make_llm_request(json!({"messages": []}))),
        Err(FlowError::GuardrailRejected(reason)) if reason == "llm denied"
    ));
    assert!(matches!(
        llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("llm-api")
                .request(make_llm_request(json!({"messages": []})))
                .func(noop_llm_exec())
                .data(json!({"request": "rejected"}))
                .model_name("reject-model")
                .build(),
        )
        .await,
        Err(FlowError::GuardrailRejected(reason)) if reason == "llm denied"
    ));
    let rejection_events = events.lock().unwrap().clone();
    let mark = rejection_events.last().unwrap();
    assert_eq!(mark.kind(), "mark");
    assert_eq!(mark.data().unwrap()["rejected"], json!(true));
    assert_eq!(
        mark.data().unwrap()["rejection_reason"],
        json!("llm denied")
    );
    drop(rejection_events);
    deregister_llm_conditional_execution_guardrail("llm-reject").unwrap();

    let baseline = events.lock().unwrap().len();
    assert!(matches!(
        llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("llm-api")
                .request(make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]})))
                .func(failing_llm_exec())
                .data(json!({"request": "failed"}))
                .model_name("error-model")
                .build(),
        )
        .await,
        Err(FlowError::Internal(message)) if message == "llm execution failed"
    ));
    let failed_events = events.lock().unwrap();
    assert_eq!(failed_events[baseline].kind(), "scope");
    assert_eq!(
        failed_events[baseline].scope_category(),
        Some(ScopeCategory::Start)
    );
    assert_eq!(failed_events[baseline].category().unwrap().as_str(), "llm");
    assert_eq!(failed_events[baseline + 1].kind(), "scope");
    assert_eq!(
        failed_events[baseline + 1].scope_category(),
        Some(ScopeCategory::End)
    );
    assert_eq!(
        failed_events[baseline + 1].category().unwrap().as_str(),
        "llm"
    );
    assert_eq!(
        failed_events[baseline + 1].output().unwrap(),
        &json!({"request": "failed"})
    );
    assert_eq!(
        failed_events[baseline + 1].model_name(),
        Some("error-model")
    );
    drop(failed_events);

    deregister_subscriber("llm-api-events").unwrap();
}

#[tokio::test]
async fn test_llm_stream_chunk_marks_track_successful_chunks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("llm-stream-chunk-mark-events");
    let raw_chunks = vec![
        json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]
        }),
        json!({"unexpected": true}),
    ];
    let collected = Arc::new(Mutex::new(Vec::<Json>::new()));

    let collector_state = collected.clone();
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(move |chunk| {
        collector_state.lock().unwrap().push(chunk);
        Ok(())
    });
    let finalizer_state = collected.clone();
    let finalizer: Box<dyn FnOnce() -> Json + Send> =
        Box::new(move || Json::Array(finalizer_state.lock().unwrap().clone()));

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("llm-stream")
            .request(make_llm_request(json!({"messages": []})))
            .func(fixed_llm_stream_exec(raw_chunks.clone()))
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .data(json!({"request": "stream"}))
            .model_name("stream-model")
            .build(),
    )
    .await
    .unwrap();

    let mut yielded = Vec::new();
    while let Some(item) = stream.next().await {
        yielded.push(item.unwrap());
    }
    assert_eq!(yielded, raw_chunks);

    let captured = events.lock().unwrap().clone();
    assert_eq!(captured.len(), 4);
    assert_eq!(captured[0].kind(), "scope");
    assert_eq!(captured[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(captured[0].category().unwrap().as_str(), "llm");
    assert_eq!(captured[1].kind(), "mark");
    assert_eq!(captured[1].name(), "llm.chunk");
    assert_eq!(captured[2].kind(), "mark");
    assert_eq!(captured[2].name(), "llm.chunk");
    assert_eq!(captured[3].kind(), "scope");
    assert_eq!(captured[3].scope_category(), Some(ScopeCategory::End));
    assert_eq!(captured[3].output().unwrap(), &Json::Array(raw_chunks));

    let llm_uuid = captured[0].uuid();
    for mark in [&captured[1], &captured[2]] {
        assert_eq!(mark.parent_uuid(), Some(llm_uuid));
        assert_eq!(mark.category(), None);
        assert_eq!(mark.scope_type(), None);
    }
    assert_eq!(
        captured[1].data().unwrap(),
        &json!({
            "chunk_index": 0,
            "provider": "openai_chat_completions",
            "event_type": "chat.completion.chunk",
            "choice_indices": [0]
        })
    );
    assert_eq!(
        captured[2].data().unwrap(),
        &json!({"chunk_index": 1, "provider": "unknown"})
    );

    deregister_subscriber("llm-stream-chunk-mark-events").unwrap();
}

#[tokio::test]
async fn test_llm_stream_chunk_mark_survives_collector_failure() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("llm-stream-collector-failure-events");
    let raw_chunk = json!({"object": "chat.completion.chunk", "choices": []});
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> =
        Box::new(|_chunk| Err(FlowError::Internal("collector failed".into())));
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!({"finalized": true}));

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("llm-stream")
            .request(make_llm_request(json!({"messages": []})))
            .func(fixed_llm_stream_exec(vec![raw_chunk]))
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .data(json!({"request": "stream"}))
            .model_name("stream-model")
            .build(),
    )
    .await
    .unwrap();

    let item = stream.next().await.unwrap();
    assert!(matches!(
        item,
        Err(FlowError::Internal(message)) if message == "collector failed"
    ));
    assert!(stream.next().await.is_none());

    let captured = events.lock().unwrap().clone();
    assert_eq!(captured.len(), 3);
    assert_eq!(captured[0].kind(), "scope");
    assert_eq!(captured[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(captured[1].kind(), "mark");
    assert_eq!(captured[1].name(), "llm.chunk");
    assert_eq!(captured[1].parent_uuid(), Some(captured[0].uuid()));
    assert_eq!(
        captured[1].data().unwrap(),
        &json!({
            "chunk_index": 0,
            "provider": "openai_chat_completions",
            "event_type": "chat.completion.chunk"
        })
    );
    assert_eq!(captured[2].kind(), "scope");
    assert_eq!(captured[2].scope_category(), Some(ScopeCategory::End));
    assert_eq!(captured[2].output().unwrap(), &json!({"finalized": true}));

    deregister_subscriber("llm-stream-collector-failure-events").unwrap();
}

#[tokio::test]
async fn test_llm_stream_api_covers_success_rejection_and_execution_error_paths() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = capture_events("llm-stream-events");
    let collected = Arc::new(Mutex::new(Vec::<Json>::new()));

    let collector_state = collected.clone();
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(move |chunk| {
        collector_state.lock().unwrap().push(chunk);
        Ok(())
    });
    let finalizer_state = collected.clone();
    let finalizer: Box<dyn FnOnce() -> Json + Send> =
        Box::new(move || Json::Array(finalizer_state.lock().unwrap().clone()));

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("llm-stream")
            .request(make_llm_request(
                json!({"messages": [{"role": "user", "content": "hello"}]}),
            ))
            .func(noop_llm_stream_exec())
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .data(json!({"request": "stream"}))
            .model_name("stream-model")
            .build(),
    )
    .await
    .unwrap();

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.unwrap());
    }
    assert_eq!(
        chunks,
        vec![json!({"messages": [{"role": "user", "content": "hello"}]})]
    );

    let success_events = events.lock().unwrap().clone();
    assert_eq!(success_events[0].kind(), "scope");
    assert_eq!(
        success_events[0].scope_category(),
        Some(ScopeCategory::Start)
    );
    assert_eq!(success_events[0].category().unwrap().as_str(), "llm");
    assert_eq!(success_events.last().unwrap().kind(), "scope");
    assert_eq!(
        success_events.last().unwrap().scope_category(),
        Some(ScopeCategory::End)
    );
    assert_eq!(
        success_events.last().unwrap().category().unwrap().as_str(),
        "llm"
    );
    assert_eq!(
        success_events.last().unwrap().output().unwrap(),
        &json!([{"messages": [{"role": "user", "content": "hello"}]}])
    );
    drop(success_events);

    register_llm_conditional_execution_guardrail(
        "llm-stream-reject",
        1,
        Arc::new(|_request| Ok(Some("stream denied".into()))),
    )
    .unwrap();
    let reject_collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let reject_finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!(null));
    assert!(matches!(
        llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream")
                .request(make_llm_request(json!({"messages": []})))
                .func(noop_llm_stream_exec())
                .collector(reject_collector)
                .finalizer(reject_finalizer)
                .attributes(LlmAttributes::STREAMING)
                .data(json!({"request": "rejected"}))
                .model_name("stream-model")
                .build(),
        )
        .await,
        Err(FlowError::GuardrailRejected(reason)) if reason == "stream denied"
    ));
    let rejection_events = events.lock().unwrap().clone();
    assert_eq!(rejection_events.last().unwrap().kind(), "mark");
    deregister_llm_conditional_execution_guardrail("llm-stream-reject").unwrap();

    let error_collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let error_finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!(null));
    let baseline = events.lock().unwrap().len();
    assert!(matches!(
        llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream")
                .request(make_llm_request(json!({"messages": []})))
                .func(failing_llm_stream_exec())
                .collector(error_collector)
                .finalizer(error_finalizer)
                .attributes(LlmAttributes::STREAMING)
                .data(json!({"request": "failed"}))
                .model_name("stream-error-model")
                .build(),
        )
        .await,
        Err(FlowError::Internal(message)) if message == "llm stream execution failed"
    ));
    let failed_events = events.lock().unwrap();
    assert_eq!(failed_events[baseline].kind(), "scope");
    assert_eq!(
        failed_events[baseline].scope_category(),
        Some(ScopeCategory::Start)
    );
    assert_eq!(failed_events[baseline].category().unwrap().as_str(), "llm");
    assert_eq!(failed_events[baseline + 1].kind(), "scope");
    assert_eq!(
        failed_events[baseline + 1].scope_category(),
        Some(ScopeCategory::End)
    );
    assert_eq!(
        failed_events[baseline + 1].category().unwrap().as_str(),
        "llm"
    );
    assert_eq!(
        failed_events[baseline + 1].output().unwrap(),
        &json!({"request": "failed"})
    );
    assert_eq!(
        failed_events[baseline + 1].model_name(),
        Some("stream-error-model")
    );
    drop(failed_events);

    event(
        nemo_relay::api::scope::EmitMarkEventParams::builder()
            .name("standalone-mark")
            .data(json!({"seen": true}))
            .build(),
    )
    .unwrap();
    let marked_events = events.lock().unwrap();
    assert_eq!(marked_events.last().unwrap().name(), "standalone-mark");
    assert_eq!(marked_events.last().unwrap().kind(), "mark");
    drop(marked_events);

    deregister_subscriber("llm-stream-events").unwrap();
}
