// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::await_holding_lock)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use nemo_flow::api::llm::{
    llm_call, llm_call_end, llm_call_execute, llm_conditional_execution, llm_request_intercepts,
    llm_stream_call_execute,
};
use nemo_flow::api::registry::{
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
use nemo_flow::api::scope::{event, pop_scope, push_scope};
use nemo_flow::api::subscriber::{
    deregister_subscriber, register_subscriber, scope_deregister_subscriber,
    scope_register_subscriber,
};
use nemo_flow::api::tool::{
    tool_call, tool_call_end, tool_call_execute, tool_conditional_execution,
    tool_request_intercepts,
};
use nemo_flow::context::callbacks::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn,
};
use nemo_flow::context::global::global_context;
use nemo_flow::context::scope_stack::{create_scope_stack, set_thread_scope_stack};
use nemo_flow::context::state::NemoFlowContextState;
use nemo_flow::error::{FlowError, Result};
use nemo_flow::json::Json;
use nemo_flow::types::event::Event;
use nemo_flow::types::llm::{LLMAttributes, LLMRequest};
use nemo_flow::types::scope::{ScopeAttributes, ScopeType};
use nemo_flow::types::tool::ToolAttributes;
use serde_json::{Map, json};
use tokio_stream::Stream;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoFlowContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn make_llm_request(content: Json) -> LLMRequest {
    LLMRequest {
        headers: Map::new(),
        content,
    }
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
        Box::new(|_name, args| args),
    )
    .unwrap();
    expect_already_exists(
        register_tool_sanitize_request_guardrail(
            "tool-sanitize-request",
            1,
            Box::new(|_name, args| args),
        )
        .unwrap_err(),
        "tool-sanitize-request",
    );
    assert!(deregister_tool_sanitize_request_guardrail("tool-sanitize-request").unwrap());
    assert!(!deregister_tool_sanitize_request_guardrail("tool-sanitize-request").unwrap());

    register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Box::new(|_name, args| args),
    )
    .unwrap();
    assert!(deregister_tool_sanitize_response_guardrail("tool-sanitize-response").unwrap());

    register_tool_conditional_execution_guardrail(
        "tool-conditional",
        1,
        Box::new(|_name, _args| Ok(None)),
    )
    .unwrap();
    assert!(deregister_tool_conditional_execution_guardrail("tool-conditional").unwrap());

    register_tool_request_intercept("tool-request", 1, false, Box::new(|_name, args| Ok(args)))
        .unwrap();
    assert!(deregister_tool_request_intercept("tool-request").unwrap());

    register_tool_execution_intercept(
        "tool-execution",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    assert!(deregister_tool_execution_intercept("tool-execution").unwrap());

    register_llm_sanitize_request_guardrail("llm-sanitize-request", 1, Box::new(|request| request))
        .unwrap();
    assert!(deregister_llm_sanitize_request_guardrail("llm-sanitize-request").unwrap());

    register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Box::new(|response| response),
    )
    .unwrap();
    assert!(deregister_llm_sanitize_response_guardrail("llm-sanitize-response").unwrap());

    register_llm_conditional_execution_guardrail(
        "llm-conditional",
        1,
        Box::new(|_request| Ok(None)),
    )
    .unwrap();
    assert!(deregister_llm_conditional_execution_guardrail("llm-conditional").unwrap());

    register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Box::new(|_name, request, annotated| Ok((request, annotated))),
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
        "scope-registry",
        ScopeType::Function,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();

    scope_register_tool_sanitize_request_guardrail(
        &scope.uuid,
        "tool-sanitize-request",
        1,
        Box::new(|_name, args| args),
    )
    .unwrap();
    expect_already_exists(
        scope_register_tool_sanitize_request_guardrail(
            &scope.uuid,
            "tool-sanitize-request",
            1,
            Box::new(|_name, args| args),
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
        Box::new(|_name, args| args),
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
        Box::new(|_name, _args| Ok(None)),
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
        Box::new(|_name, args| Ok(args)),
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
        Box::new(|request| request),
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
        Box::new(|response| response),
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
        Box::new(|_request| Ok(None)),
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
        Box::new(|_name, request, annotated| Ok((request, annotated))),
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

    pop_scope(&scope.uuid).unwrap();

    expect_not_found(
        scope_register_tool_sanitize_request_guardrail(
            &scope.uuid,
            "missing-tool-sanitize",
            1,
            Box::new(|_name, args| args),
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
            Box::new(|_name, args| Ok(args)),
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
        Box::new(|_name, mut args| {
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
        Box::new(|_name, mut result| {
            result
                .as_object_mut()
                .unwrap()
                .insert("sanitized_response".into(), json!(true));
            result
        }),
    )
    .unwrap();

    let handle = tool_call(
        "tool-api",
        json!({"value": 1}),
        None,
        ToolAttributes::LOCAL,
        Some(json!({"phase": "start"})),
        Some(json!({"meta": "tool"})),
        Some("tool-call-id".to_string()),
    )
    .unwrap();
    tool_call_end(
        &handle,
        json!({"ok": true}),
        Some(json!({"phase": "end"})),
        Some(json!({"meta": "tool"})),
    )
    .unwrap();

    let captured = events.lock().unwrap().clone();
    assert_eq!(captured[0].kind(), "ToolStart");
    assert_eq!(
        captured[0].input().unwrap()["sanitized_request"],
        json!(true)
    );
    assert_eq!(captured[0].tool_call_id(), Some("tool-call-id"));
    assert_eq!(captured[1].kind(), "ToolEnd");
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
        Box::new(|_name, mut args| {
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
        Box::new(|_name, _args| Ok(Some("tool denied".into()))),
    )
    .unwrap();
    assert!(matches!(
        tool_conditional_execution("tool-api", &json!({"value": 3})),
        Err(FlowError::GuardrailRejected(reason)) if reason == "tool denied"
    ));
    assert!(matches!(
        tool_call_execute(
            "tool-api",
            json!({"value": 3}),
            noop_tool_exec(),
            None,
            ToolAttributes::empty(),
            Some(json!({"request": "rejected"})),
            None,
        )
        .await,
        Err(FlowError::GuardrailRejected(reason)) if reason == "tool denied"
    ));
    let rejection_events = events.lock().unwrap().clone();
    let mark = rejection_events.last().unwrap();
    assert_eq!(mark.kind(), "Mark");
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
            "tool-api",
            json!({"value": 4}),
            failing_tool_exec(),
            None,
            ToolAttributes::empty(),
            Some(json!({"request": "failed"})),
            None,
        )
        .await,
        Err(FlowError::Internal(message)) if message == "tool execution failed"
    ));
    let failed_events = events.lock().unwrap();
    assert_eq!(failed_events[baseline].kind(), "ToolStart");
    assert_eq!(failed_events[baseline + 1].kind(), "ToolEnd");
    assert!(failed_events[baseline + 1].output().is_none());
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
        Box::new(|mut request| {
            request.headers.insert("x-sanitized".into(), json!(true));
            request
        }),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Box::new(|mut response| {
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
        "llm-api",
        &request,
        None,
        LLMAttributes::STATELESS,
        Some(json!({"phase": "start"})),
        Some(json!({"meta": "llm"})),
        Some("test-model".to_string()),
        None,
    )
    .unwrap();
    llm_call_end(
        &handle,
        json!({"response": "ok"}),
        Some(json!({"phase": "end"})),
        Some(json!({"meta": "llm"})),
        None,
    )
    .unwrap();

    let captured = events.lock().unwrap().clone();
    assert_eq!(captured[0].kind(), "LLMStart");
    assert_eq!(
        captured[0].input().unwrap()["headers"]["x-sanitized"],
        json!(true)
    );
    assert_eq!(captured[0].model_name(), Some("test-model"));
    assert_eq!(captured[1].kind(), "LLMEnd");
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
        Box::new(|_name, mut request, annotated| {
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
        Box::new(|_request| Ok(Some("llm denied".into()))),
    )
    .unwrap();
    assert!(matches!(
        llm_conditional_execution(&make_llm_request(json!({"messages": []}))),
        Err(FlowError::GuardrailRejected(reason)) if reason == "llm denied"
    ));
    assert!(matches!(
        llm_call_execute(
            "llm-api",
            make_llm_request(json!({"messages": []})),
            noop_llm_exec(),
            None,
            LLMAttributes::empty(),
            Some(json!({"request": "rejected"})),
            None,
            Some("reject-model".to_string()),
            None,
            None,
        )
        .await,
        Err(FlowError::GuardrailRejected(reason)) if reason == "llm denied"
    ));
    let rejection_events = events.lock().unwrap().clone();
    let mark = rejection_events.last().unwrap();
    assert_eq!(mark.kind(), "Mark");
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
            "llm-api",
            make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]})),
            failing_llm_exec(),
            None,
            LLMAttributes::empty(),
            Some(json!({"request": "failed"})),
            None,
            Some("error-model".to_string()),
            None,
            None,
        )
        .await,
        Err(FlowError::Internal(message)) if message == "llm execution failed"
    ));
    let failed_events = events.lock().unwrap();
    assert_eq!(failed_events[baseline].kind(), "LLMStart");
    assert_eq!(failed_events[baseline + 1].kind(), "LLMEnd");
    assert!(failed_events[baseline + 1].output().is_none());
    assert_eq!(
        failed_events[baseline + 1].model_name(),
        Some("error-model")
    );
    drop(failed_events);

    deregister_subscriber("llm-api-events").unwrap();
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
        "llm-stream",
        make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]})),
        noop_llm_stream_exec(),
        collector,
        finalizer,
        None,
        LLMAttributes::STREAMING,
        Some(json!({"request": "stream"})),
        None,
        Some("stream-model".to_string()),
        None,
        None,
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
    assert_eq!(success_events[0].kind(), "LLMStart");
    assert_eq!(success_events.last().unwrap().kind(), "LLMEnd");
    assert_eq!(
        success_events.last().unwrap().output().unwrap(),
        &json!([{"messages": [{"role": "user", "content": "hello"}]}])
    );
    drop(success_events);

    register_llm_conditional_execution_guardrail(
        "llm-stream-reject",
        1,
        Box::new(|_request| Ok(Some("stream denied".into()))),
    )
    .unwrap();
    let reject_collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let reject_finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!(null));
    assert!(matches!(
        llm_stream_call_execute(
            "llm-stream",
            make_llm_request(json!({"messages": []})),
            noop_llm_stream_exec(),
            reject_collector,
            reject_finalizer,
            None,
            LLMAttributes::STREAMING,
            Some(json!({"request": "rejected"})),
            None,
            Some("stream-model".to_string()),
            None,
            None,
        )
        .await,
        Err(FlowError::GuardrailRejected(reason)) if reason == "stream denied"
    ));
    let rejection_events = events.lock().unwrap().clone();
    assert_eq!(rejection_events.last().unwrap().kind(), "Mark");
    deregister_llm_conditional_execution_guardrail("llm-stream-reject").unwrap();

    let error_collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let error_finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!(null));
    let baseline = events.lock().unwrap().len();
    assert!(matches!(
        llm_stream_call_execute(
            "llm-stream",
            make_llm_request(json!({"messages": []})),
            failing_llm_stream_exec(),
            error_collector,
            error_finalizer,
            None,
            LLMAttributes::STREAMING,
            Some(json!({"request": "failed"})),
            None,
            Some("stream-error-model".to_string()),
            None,
            None,
        )
        .await,
        Err(FlowError::Internal(message)) if message == "llm stream execution failed"
    ));
    let failed_events = events.lock().unwrap();
    assert_eq!(failed_events[baseline].kind(), "LLMStart");
    assert_eq!(failed_events[baseline + 1].kind(), "LLMEnd");
    assert!(failed_events[baseline + 1].output().is_none());
    assert_eq!(
        failed_events[baseline + 1].model_name(),
        Some("stream-error-model")
    );
    drop(failed_events);

    event("standalone-mark", None, Some(json!({"seen": true})), None).unwrap();
    let marked_events = events.lock().unwrap();
    assert_eq!(marked_events.last().unwrap().name(), "standalone-mark");
    assert_eq!(marked_events.last().unwrap().kind(), "Mark");
    drop(marked_events);

    deregister_subscriber("llm-stream-events").unwrap();
}
