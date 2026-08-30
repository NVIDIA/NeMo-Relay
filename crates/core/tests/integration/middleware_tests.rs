// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive middleware chain tests for the NeMo Relay core runtime.
//!
//! These tests exercise the middleware pipeline mechanics: priority ordering,
//! break_chain short-circuiting, execution intercept middleware chains (next()),
//! conditional execution guardrails, scope-local middleware lifecycle, global +
//! scope-local merging, error propagation, and concurrent mutations.

#![allow(clippy::await_holding_lock)]

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use chrono::Utc;
mod test_support;
use test_support::{ready, ready_result};

use futures::StreamExt;
use nemo_relay::api::event::{
    CategoryProfile, DataSchema, Event, EventCategory, LOG_SEVERITY_METADATA_KEY, LogSeverity,
    PendingMarkSpec, ScopeCategory,
};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_request_intercepts,
    llm_stream_call_execute,
};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::optimization::record_llm_optimization_contribution;
use nemo_relay::api::registry::{
    RuntimeRegistrationKind, RuntimeRegistrationOwnerKind,
    deregister_conditional_middleware_guardrail, deregister_llm_conditional_execution_guardrail,
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    deregister_llm_sanitize_request_guardrail, deregister_llm_sanitize_response_guardrail,
    deregister_llm_stream_execution_intercept, deregister_mark_sanitize_guardrail,
    deregister_scope_sanitize_end_guardrail, deregister_scope_sanitize_start_guardrail,
    deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
    deregister_tool_request_intercept, deregister_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_response_guardrail, list_runtime_registrations,
    register_conditional_middleware_guardrail, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
    register_llm_stream_execution_intercept, register_mark_sanitize_guardrail,
    register_scope_sanitize_end_guardrail, register_scope_sanitize_start_guardrail,
    register_tool_conditional_execution_guardrail, register_tool_execution_intercept,
    register_tool_request_intercept, register_tool_sanitize_request_guardrail,
    register_tool_sanitize_response_guardrail, scope_deregister_tool_request_intercept,
    scope_register_llm_conditional_execution_guardrail, scope_register_llm_execution_intercept,
    scope_register_llm_request_intercept, scope_register_llm_sanitize_request_guardrail,
    scope_register_llm_sanitize_response_guardrail, scope_register_llm_stream_execution_intercept,
    scope_register_mark_sanitize_guardrail, scope_register_scope_sanitize_end_guardrail,
    scope_register_tool_conditional_execution_guardrail, scope_register_tool_execution_intercept,
    scope_register_tool_request_intercept, scope_register_tool_sanitize_request_guardrail,
    scope_register_tool_sanitize_response_guardrail,
};
use nemo_relay::api::runtime::NemoRelayContextState;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, LlmStreamInner, TASK_SCOPE_STACK,
    ToolExecutionNextFn, capture_propagation_context, task_scope_top,
};
use nemo_relay::api::runtime::{create_scope_stack, current_scope_stack, set_thread_scope_stack};
use nemo_relay::api::scope::{EmitMarkEventParams, ScopeHandle, ScopeType, event};
use nemo_relay::api::scope::{pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{
    ToolExecutionInterceptOutcome, ToolExecutionResult, tool_call, tool_call_end,
    tool_call_execute, tool_conditional_execution, tool_request_intercepts,
};
use nemo_relay::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationTokenImpact,
    LlmOptimizationTokens,
};
use nemo_relay::error::FlowError;
use nemo_relay::json::Json;
use nemo_relay::observability::OpenTelemetryType;
use nemo_relay::observability::atif::{AtifAgentInfo, AtifExporter};
use nemo_relay::observability::otel::OpenTelemetrySubscriber;
use nemo_relay::plugin::{PluginRegistrationContext, rollback_registrations};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use serde_json::json;

// All tests share the global context, so we serialize them.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn assert_flush_waits_for_pending_completion(complete: impl FnOnce()) {
    let (flush_started_tx, flush_started_rx) = std::sync::mpsc::channel();
    let (flush_done_tx, flush_done_rx) = std::sync::mpsc::channel();
    let flush_thread = std::thread::spawn(move || {
        flush_started_tx.send(()).unwrap();
        flush_done_tx.send(flush_subscribers()).unwrap();
    });
    flush_started_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("subscriber flush thread did not start");
    assert!(
        matches!(
            flush_done_rx.recv_timeout(std::time::Duration::from_millis(50)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ),
        "flush must wait for pending managed completion"
    );

    complete();
    flush_done_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("subscriber flush did not complete after cancellation")
        .unwrap();
    flush_thread.join().unwrap();
}

struct CloseCallsStreamNext {
    next: Option<LlmStreamExecutionNextFn>,
    request: Option<LlmRequest>,
}

impl futures::Stream for CloseCallsStreamNext {
    type Item = Result<Json, FlowError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl LlmStreamInner for CloseCallsStreamNext {
    fn close(
        self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = Result<(), FlowError>> + Send + '_>> {
        Box::pin(async move {
            let this = self.get_mut();
            let next = this.next.take().expect("close must run only once");
            let request = this.request.take().expect("close must run only once");
            let mut downstream = next(request).await?;
            downstream.close().await
        })
    }
}

fn is_scope_event(event: &Event, scope_type: ScopeType, scope_category: ScopeCategory) -> bool {
    event.scope_type() == Some(scope_type) && event.scope_category() == Some(scope_category)
}

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

/// Helper: create a fresh scope stack on the current thread.
fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

/// Helper: create a fresh scope stack on the current thread and push a scope,
/// returning the scope handle.
fn setup_isolated_scope(name: &str) -> ScopeHandle {
    setup_isolated_thread();
    push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap()
}

fn captured_events_snapshot(events: &Arc<Mutex<Vec<Event>>>) -> Vec<Event> {
    flush_subscribers().unwrap();
    events.lock().unwrap().clone()
}

fn assert_tool_lifecycle_correlation(
    events: &[Event],
    name: &str,
    expected_tool_call_id: Option<&str>,
) {
    let lifecycle = events
        .iter()
        .filter(|event| event.name() == name && event.scope_type() == Some(ScopeType::Tool))
        .collect::<Vec<_>>();
    assert_eq!(lifecycle.len(), 2, "expected one managed Start/End pair");
    let start = lifecycle[0];
    let end = lifecycle[1];
    assert_eq!(start.scope_category(), Some(ScopeCategory::Start));
    assert_eq!(end.scope_category(), Some(ScopeCategory::End));
    assert_eq!(start.uuid(), end.uuid());
    assert_eq!(start.tool_call_id(), expected_tool_call_id);
    assert_eq!(end.tool_call_id(), expected_tool_call_id);
}

fn assert_middleware_callback_locks_are_free() {
    let scope_stack = current_scope_stack();
    assert!(
        scope_stack.try_write().is_ok(),
        "middleware callback ran while its scope stack lock was held"
    );
}

/// Queued payload sanitizers may overlap later hot-path registry reads. They
/// still must not run while holding their captured scope-stack lock.
fn assert_queued_sanitizer_scope_lock_is_free() {
    let scope_stack = current_scope_stack();
    assert!(
        scope_stack.try_write().is_ok(),
        "queued sanitizer ran while the scope stack lock was held"
    );
}

fn record_middleware_callback(callbacks: &Arc<Mutex<Vec<&'static str>>>, label: &'static str) {
    callbacks.lock().unwrap().push(label);
}

fn assert_middleware_callback_labels(
    callbacks: &Arc<Mutex<Vec<&'static str>>>,
    expected: &[&'static str],
) {
    let mut actual = callbacks.lock().unwrap().clone();
    let mut expected = expected.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

// =========================================================================
// Priority Ordering Tests
// =========================================================================

/// Register 3 tool sanitize request guardrails at priorities 1, 3, 2;
/// verify execution order is 1, 2, 3.
#[tokio::test]
async fn test_sanitize_guardrail_priority_ordering() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let order = Arc::new(Mutex::new(Vec::<i32>::new()));

    // Register at priority 1
    let o1 = order.clone();
    register_tool_sanitize_request_guardrail(
        "g_p1",
        1,
        Arc::new(move |_name, args| {
            o1.lock().unwrap().push(1);
            ready(args)
        }),
    )
    .unwrap();

    // Register at priority 3
    let o3 = order.clone();
    register_tool_sanitize_request_guardrail(
        "g_p3",
        3,
        Arc::new(move |_name, args| {
            o3.lock().unwrap().push(3);
            ready(args)
        }),
    )
    .unwrap();

    // Register at priority 2
    let o2 = order.clone();
    register_tool_sanitize_request_guardrail(
        "g_p2",
        2,
        Arc::new(move |_name, args| {
            o2.lock().unwrap().push(2);
            ready(args)
        }),
    )
    .unwrap();

    // Trigger the chain via tool_call (which runs sanitize request guardrails)
    let _handle = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("test_tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();

    let recorded = order.lock().unwrap();
    assert_eq!(
        *recorded,
        vec![1, 2, 3],
        "Guardrails should run in ascending priority order"
    );

    // Cleanup
    deregister_tool_sanitize_request_guardrail("g_p1").unwrap();
    deregister_tool_sanitize_request_guardrail("g_p2").unwrap();
    deregister_tool_sanitize_request_guardrail("g_p3").unwrap();
}

/// Register 3 tool request intercepts at priorities 1, 3, 2;
/// verify execution order is 1, 2, 3.
#[tokio::test]
async fn test_request_intercept_priority_ordering() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let order = Arc::new(Mutex::new(Vec::<i32>::new()));

    let o1 = order.clone();
    register_tool_request_intercept(
        "i_p1",
        1,
        false,
        Arc::new(move |_name, args| {
            o1.lock().unwrap().push(1);
            ready(args)
        }),
    )
    .unwrap();

    let o3 = order.clone();
    register_tool_request_intercept(
        "i_p3",
        3,
        false,
        Arc::new(move |_name, args| {
            o3.lock().unwrap().push(3);
            ready(args)
        }),
    )
    .unwrap();

    let o2 = order.clone();
    register_tool_request_intercept(
        "i_p2",
        2,
        false,
        Arc::new(move |_name, args| {
            o2.lock().unwrap().push(2);
            ready(args)
        }),
    )
    .unwrap();

    // Use the standalone intercept chain function
    let _result = tool_request_intercepts("test_tool", json!({}))
        .await
        .unwrap();

    let recorded = order.lock().unwrap();
    assert_eq!(
        *recorded,
        vec![1, 2, 3],
        "Intercepts should run in ascending priority order"
    );

    // Cleanup
    deregister_tool_request_intercept("i_p1").unwrap();
    deregister_tool_request_intercept("i_p2").unwrap();
    deregister_tool_request_intercept("i_p3").unwrap();
}

/// Verify that deregistering and re-registering at a different priority re-sorts.
#[tokio::test]
async fn test_re_registration_at_different_priority_re_sorts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    let o_a = order.clone();
    register_tool_request_intercept(
        "intercept_a",
        10,
        false,
        Arc::new(move |_name, args| {
            o_a.lock().unwrap().push("a_p10".into());
            ready(args)
        }),
    )
    .unwrap();

    let o_b = order.clone();
    register_tool_request_intercept(
        "intercept_b",
        20,
        false,
        Arc::new(move |_name, args| {
            o_b.lock().unwrap().push("b_p20".into());
            ready(args)
        }),
    )
    .unwrap();

    // First call: a runs before b
    let _ = tool_request_intercepts("test", json!({})).await.unwrap();
    {
        let recorded = order.lock().unwrap();
        assert_eq!(*recorded, vec!["a_p10", "b_p20"]);
    }

    // Deregister a and re-register at priority 30 (after b)
    deregister_tool_request_intercept("intercept_a").unwrap();
    let o_a2 = order.clone();
    register_tool_request_intercept(
        "intercept_a",
        30,
        false,
        Arc::new(move |_name, args| {
            o_a2.lock().unwrap().push("a_p30".into());
            ready(args)
        }),
    )
    .unwrap();

    // Clear and re-run
    order.lock().unwrap().clear();
    let _ = tool_request_intercepts("test", json!({})).await.unwrap();
    {
        let recorded = order.lock().unwrap();
        assert_eq!(
            *recorded,
            vec!["b_p20", "a_p30"],
            "After re-registration, b should run before a"
        );
    }

    // Cleanup
    deregister_tool_request_intercept("intercept_a").unwrap();
    deregister_tool_request_intercept("intercept_b").unwrap();
}

// =========================================================================
// Break Chain (Request Intercepts) Tests
// =========================================================================

/// Register 2 request intercepts, first with break_chain=true.
/// Verify second intercept is NOT called and the result from the first is used.
#[tokio::test]
async fn test_break_chain_stops_subsequent_intercepts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let second_called = Arc::new(AtomicBool::new(false));

    register_tool_request_intercept(
        "breaker",
        1,
        true, // break_chain = true
        Arc::new(|_name, mut args| {
            args.as_object_mut()
                .unwrap()
                .insert("breaker_ran".into(), json!(true));
            ready(args)
        }),
    )
    .unwrap();

    let sc = second_called.clone();
    register_tool_request_intercept(
        "after_breaker",
        2,
        false,
        Arc::new(move |_name, mut args| {
            sc.store(true, Ordering::SeqCst);
            args.as_object_mut()
                .unwrap()
                .insert("after_ran".into(), json!(true));
            ready(args)
        }),
    )
    .unwrap();

    let result = tool_request_intercepts("tool", json!({})).await.unwrap();

    // First intercept's transformation should be applied
    assert_eq!(result["breaker_ran"], true);
    // Second intercept should NOT have been called
    assert!(
        !second_called.load(Ordering::SeqCst),
        "Second intercept should not run after break_chain"
    );
    assert!(
        result.get("after_ran").is_none(),
        "After-breaker output should not be present"
    );

    // Cleanup
    deregister_tool_request_intercept("breaker").unwrap();
    deregister_tool_request_intercept("after_breaker").unwrap();
}

/// With break_chain=false on all intercepts, both should be called.
#[tokio::test]
async fn test_no_break_chain_runs_all_intercepts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let call_count = Arc::new(AtomicU32::new(0));

    let c1 = call_count.clone();
    register_tool_request_intercept(
        "first",
        1,
        false,
        Arc::new(move |_name, args| {
            c1.fetch_add(1, Ordering::SeqCst);
            ready(args)
        }),
    )
    .unwrap();

    let c2 = call_count.clone();
    register_tool_request_intercept(
        "second",
        2,
        false,
        Arc::new(move |_name, args| {
            c2.fetch_add(1, Ordering::SeqCst);
            ready(args)
        }),
    )
    .unwrap();

    let _ = tool_request_intercepts("tool", json!({})).await.unwrap();

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "Both intercepts should run when break_chain=false"
    );

    // Cleanup
    deregister_tool_request_intercept("first").unwrap();
    deregister_tool_request_intercept("second").unwrap();
}

// =========================================================================
// Execution Intercepts (Middleware Chain) Tests
// =========================================================================

#[tokio::test]
async fn managed_tool_call_id_correlates_success_and_absent_lifecycles() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "managed_tool_call_id_success_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("managed-correlated-tool")
            .args(json!({"value": 42}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .tool_call_id("call-managed-42")
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result, json!({"value": 42}));

    tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("managed-uncorrelated-tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    )
    .await
    .unwrap();

    let captured = captured_events_snapshot(&events);
    assert_tool_lifecycle_correlation(
        &captured,
        "managed-correlated-tool",
        Some("call-managed-42"),
    );
    assert_tool_lifecycle_correlation(&captured, "managed-uncorrelated-tool", None);

    deregister_subscriber("managed_tool_call_id_success_observer").unwrap();
}

#[tokio::test]
async fn managed_tool_call_id_correlates_callback_error_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "managed_tool_call_id_error_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let error = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("managed-error-tool")
            .args(json!({}))
            .func(Arc::new(|_args| {
                Box::pin(async { Err(FlowError::Internal("callback failed".into())) })
            }))
            .tool_call_id("call-managed-error")
            .build(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, FlowError::Internal(message) if message == "callback failed"));

    let captured = captured_events_snapshot(&events);
    assert_tool_lifecycle_correlation(&captured, "managed-error-tool", Some("call-managed-error"));

    deregister_subscriber("managed_tool_call_id_error_observer").unwrap();
}

/// Register an execution intercept that calls next().
/// Verify the original callable is invoked.
#[tokio::test]
async fn test_execution_intercept_calls_next() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let original_called = Arc::new(AtomicBool::new(false));

    // Register an execution intercept that passes through to next
    register_tool_execution_intercept(
        "passthrough",
        1,
        Arc::new(|_name, args, next| {
            Box::pin(async move {
                // Call next — this should reach the original callable
                next(args).await.map(Into::into)
            })
        }),
    )
    .unwrap();

    let oc = original_called.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        oc.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(args.into()) })
    });

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"value": 42}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert!(
        original_called.load(Ordering::SeqCst),
        "Original callable should be invoked"
    );
    assert_eq!(result.result["value"], 42);

    // Cleanup
    deregister_tool_execution_intercept("passthrough").unwrap();
}

/// Register an execution intercept that does NOT call next().
/// Verify the original callable is NOT invoked.
#[tokio::test]
async fn test_execution_intercept_skips_next() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let original_called = Arc::new(AtomicBool::new(false));

    // Register an execution intercept that short-circuits (does not call next)
    register_tool_execution_intercept(
        "short_circuit",
        1,
        Arc::new(|_name, _args, _next| {
            Box::pin(async move {
                // Return a custom result without calling next
                Ok(json!({"intercepted": true}).into())
            })
        }),
    )
    .unwrap();

    let oc = original_called.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        oc.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(args.into()) })
    });

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"value": 42}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert!(
        !original_called.load(Ordering::SeqCst),
        "Original callable should NOT be invoked"
    );
    assert_eq!(result.result["intercepted"], true);

    // Cleanup
    deregister_tool_execution_intercept("short_circuit").unwrap();
}

#[tokio::test]
async fn tool_execution_result_annotation_is_explicit_through_the_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_execution_intercept(
        "annotation",
        1,
        Arc::new(|_name, args, next| {
            Box::pin(async move {
                let mut execution_result = next(args).await?;
                assert_eq!(
                    execution_result.annotation,
                    Some(json!({"source": "producer"}))
                );
                execution_result.result["rewritten"] = json!(true);
                execution_result.annotation = Some(json!({"source": "middleware"}));
                Ok(execution_result.into())
            })
        }),
    )
    .unwrap();

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("annotated-tool")
            .args(json!({}))
            .func(Arc::new(|_args| {
                Box::pin(async {
                    Ok(ToolExecutionResult::annotated(
                        json!({"raw": true}),
                        json!({"source": "producer"}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result.result, json!({"raw": true, "rewritten": true}));
    assert_eq!(result.annotation, Some(json!({"source": "middleware"})));
    deregister_tool_execution_intercept("annotation").unwrap();
}

#[tokio::test]
async fn tool_execution_intercepts_can_preserve_remove_and_short_circuit_annotations() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_execution_intercept(
        "annotation-preserve",
        1,
        Arc::new(|_name, args, next| Box::pin(async move { next(args).await.map(Into::into) })),
    )
    .unwrap();
    let preserved = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("preserved-annotation")
            .args(json!({}))
            .func(Arc::new(|_args| {
                Box::pin(async {
                    Ok(ToolExecutionResult::annotated(
                        json!({"result": "preserved"}),
                        json!({"source": "producer"}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(preserved.annotation, Some(json!({"source": "producer"})));
    deregister_tool_execution_intercept("annotation-preserve").unwrap();

    register_tool_execution_intercept(
        "annotation-remove",
        1,
        Arc::new(|_name, args, next| {
            Box::pin(async move {
                let result = next(args).await?.without_annotation();
                Ok(result.into())
            })
        }),
    )
    .unwrap();
    let removed = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("removed-annotation")
            .args(json!({}))
            .func(Arc::new(|_args| {
                Box::pin(async {
                    Ok(ToolExecutionResult::annotated(
                        json!({"result": "removed"}),
                        json!({"source": "producer"}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(removed.result, json!({"result": "removed"}));
    assert_eq!(removed.annotation, None);
    deregister_tool_execution_intercept("annotation-remove").unwrap();

    let provider_called = Arc::new(AtomicBool::new(false));
    register_tool_execution_intercept(
        "annotation-short-circuit",
        1,
        Arc::new(|_name, _args, _next| {
            Box::pin(async {
                Ok(ToolExecutionInterceptOutcome::annotated(
                    json!({"result": "short-circuit"}),
                    json!({"source": "middleware"}),
                ))
            })
        }),
    )
    .unwrap();
    let captured_provider_called = Arc::clone(&provider_called);
    let short_circuited = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("short-circuit-annotation")
            .args(json!({}))
            .func(Arc::new(move |_args| {
                captured_provider_called.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(ToolExecutionResult::new(json!({}))) })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert!(!provider_called.load(Ordering::SeqCst));
    assert_eq!(short_circuited.result, json!({"result": "short-circuit"}));
    assert_eq!(
        short_circuited.annotation,
        Some(json!({"source": "middleware"}))
    );
    deregister_tool_execution_intercept("annotation-short-circuit").unwrap();
}

#[tokio::test]
async fn mcp_error_result_remains_successful_and_distinct_from_relay_annotation() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "mcp-tool-result-observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let mcp_result = json!({
        "content": [{"type": "text", "text": "provider-reported failure"}],
        "structuredContent": {"code": "not_found"},
        "_meta": {"provider": "mcp"},
        "isError": true,
    });
    let relay_annotation = json!({"cache": {"hit": false}});

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("mcp-error-result")
            .args(json!({}))
            .func(Arc::new({
                let mcp_result = mcp_result.clone();
                let relay_annotation = relay_annotation.clone();
                move |_args| {
                    let mcp_result = mcp_result.clone();
                    let relay_annotation = relay_annotation.clone();
                    Box::pin(async move {
                        Ok(ToolExecutionResult::annotated(mcp_result, relay_annotation))
                    })
                }
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result, mcp_result);
    assert_eq!(result.annotation, Some(relay_annotation.clone()));

    let captured = captured_events_snapshot(&events);
    let end = captured
        .iter()
        .find(|event| {
            event.name() == "mcp-error-result" && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert_eq!(end.data(), Some(&mcp_result));
    assert_eq!(end.metadata().unwrap()["otel.status_code"], "OK");
    assert_eq!(end.tool_result_annotation().unwrap(), relay_annotation);

    deregister_subscriber("mcp-tool-result-observer").unwrap();
}

#[tokio::test]
async fn tool_result_annotation_uses_the_event_sanitizer_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "tool-annotation-observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_scope_sanitize_end_guardrail(
        "tool-annotation-sanitizer",
        1,
        Arc::new(|event, mut fields| {
            Box::pin(async move {
                if event.name() == "sanitized-annotation-tool"
                    && let Some(annotation) = fields
                        .category_profile
                        .as_mut()
                        .and_then(|profile| profile.tool_result_annotation.as_mut())
                        .and_then(Json::as_object_mut)
                {
                    annotation.insert("secret".into(), json!("[redacted]"));
                }
                Ok(fields)
            })
        }),
    )
    .unwrap();
    register_tool_sanitize_response_guardrail(
        "tool-result-sanitizer",
        1,
        Arc::new(|_name, mut result| {
            result["raw"] = json!("[redacted]");
            ready(result)
        }),
    )
    .unwrap();

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("sanitized-annotation-tool")
            .args(json!({}))
            .func(Arc::new(|_args| {
                Box::pin(async {
                    Ok(ToolExecutionResult::annotated(
                        json!({"raw": true}),
                        json!({"secret": "classified"}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result, json!({"raw": true}));
    assert_eq!(result.annotation, Some(json!({"secret": "classified"})));

    let captured = captured_events_snapshot(&events);
    let end = captured
        .iter()
        .find(|event| {
            event.name() == "sanitized-annotation-tool"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert_eq!(
        end.tool_result_annotation().unwrap()["secret"],
        "[redacted]"
    );
    assert_eq!(end.data().unwrap(), &json!({"raw": "[redacted]"}));

    deregister_scope_sanitize_end_guardrail("tool-annotation-sanitizer").unwrap();
    register_scope_sanitize_end_guardrail(
        "tool-annotation-failure",
        1,
        Arc::new(|_event, _fields| {
            Box::pin(async {
                Err(FlowError::Internal(
                    "intentional annotation sanitizer failure".into(),
                ))
            })
        }),
    )
    .unwrap();
    let failed_result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("failed-annotation-tool")
            .args(json!({}))
            .func(Arc::new(|_args| {
                Box::pin(async {
                    Ok(ToolExecutionResult::annotated(
                        json!({"raw": true}),
                        json!({"secret": "still-visible-to-caller"}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(
        failed_result.annotation,
        Some(json!({"secret": "still-visible-to-caller"}))
    );
    let captured = captured_events_snapshot(&events);
    let failed_end = captured
        .iter()
        .find(|event| {
            event.name() == "failed-annotation-tool"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert!(failed_end.data().is_none());
    assert!(failed_end.category_profile().is_none());

    deregister_tool_sanitize_response_guardrail("tool-result-sanitizer").unwrap();
    deregister_scope_sanitize_end_guardrail("tool-annotation-failure").unwrap();
    deregister_subscriber("tool-annotation-observer").unwrap();
}

/// Register 2 chained execution intercepts. Verify both run in priority order
/// and the original callable is ultimately invoked.
#[tokio::test]
async fn test_execution_intercept_chain_ordering() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    // Intercept at priority 1 (runs first in the chain)
    let o1 = order.clone();
    register_tool_execution_intercept(
        "exec_p1",
        1,
        Arc::new(move |_name, args, next| {
            let o = o1.clone();
            Box::pin(async move {
                o.lock().unwrap().push("intercept_1_before".into());
                let result = next(args).await;
                o.lock().unwrap().push("intercept_1_after".into());
                result.map(Into::into)
            })
        }),
    )
    .unwrap();

    // Intercept at priority 2 (runs second, nested inside first)
    let o2 = order.clone();
    register_tool_execution_intercept(
        "exec_p2",
        2,
        Arc::new(move |_name, args, next| {
            let o = o2.clone();
            Box::pin(async move {
                o.lock().unwrap().push("intercept_2_before".into());
                let result = next(args).await;
                o.lock().unwrap().push("intercept_2_after".into());
                result.map(Into::into)
            })
        }),
    )
    .unwrap();

    let o_orig = order.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        o_orig.lock().unwrap().push("original".into());
        Box::pin(async move { Ok(args.into()) })
    });

    let _ = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    let recorded = order.lock().unwrap();
    // Middleware chain pattern: 1 wraps 2 wraps original
    assert_eq!(
        *recorded,
        vec![
            "intercept_1_before",
            "intercept_2_before",
            "original",
            "intercept_2_after",
            "intercept_1_after",
        ],
        "Execution intercepts should follow middleware chain (onion) pattern"
    );

    // Cleanup
    deregister_tool_execution_intercept("exec_p1").unwrap();
    deregister_tool_execution_intercept("exec_p2").unwrap();
}

/// Verify execution intercept can modify args before passing to next.
#[tokio::test]
async fn test_execution_intercept_modifies_args() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_execution_intercept(
        "arg_modifier",
        1,
        Arc::new(|_name, mut args, next| {
            Box::pin(async move {
                args.as_object_mut()
                    .unwrap()
                    .insert("injected".into(), json!(true));
                next(args).await.map(Into::into)
            })
        }),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"original": true}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result.result["original"], true);
    assert_eq!(result.result["injected"], true);

    // Cleanup
    deregister_tool_execution_intercept("arg_modifier").unwrap();
}

#[tokio::test]
async fn test_tool_execution_outcome_marks_follow_end_with_tool_parentage() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "tool_outcome_mark_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_mark_sanitize_guardrail(
        "tool_pending_mark_sanitizer",
        1,
        Arc::new(|_, mut fields| {
            let mut metadata = fields.metadata.unwrap_or_else(|| json!({}));
            metadata["sanitized"] = json!(true);
            fields.metadata = Some(metadata);
            ready(fields)
        }),
    )
    .unwrap();
    let mut plugin_ctx = PluginRegistrationContext::new();
    plugin_ctx
        .register_tool_execution_intercept(
            "outcome_outer",
            1,
            Arc::new(|_name, args, next| {
                Box::pin(async move {
                    let result = next(args).await?;
                    Ok(
                        ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("tool.mark.outer")
                                .data(json!({"layer": "outer"}))
                                .build(),
                        ),
                    )
                })
            }),
        )
        .unwrap();
    register_tool_execution_intercept(
        "passthrough_between_outcomes",
        2,
        Arc::new(|_name, args, next| Box::pin(async move { next(args).await.map(Into::into) })),
    )
    .unwrap();
    plugin_ctx
        .register_tool_execution_intercept(
            "outcome_inner",
            3,
            Arc::new(|_name, args, next| {
                Box::pin(async move {
                    let mut result = next(args).await?;
                    result.result["compressed"] = json!(true);
                    Ok(ToolExecutionInterceptOutcome::from(result)
                        .with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("tool.mark.invalid")
                                .metadata(json!("not an object"))
                                .severity(LogSeverity::Info)
                                .build(),
                        )
                        .with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("tool.mark.inner")
                                .category(EventCategory::custom())
                                .category_profile(
                                    CategoryProfile::builder()
                                        .subtype("example.tool.compression")
                                        .build(),
                                )
                                .data(json!({"saved_tokens": 12}))
                                .data_schema(
                                    DataSchema::builder()
                                        .name("example.tool_pending_mark")
                                        .version("1")
                                        .build(),
                                )
                                .metadata(json!({"source": "test"}))
                                .severity(LogSeverity::Error)
                                .build(),
                        ))
                })
            }),
        )
        .unwrap();

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool-outcome")
            .args(json!({"value": 42}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result, json!({"value": 42, "compressed": true}));
    assert!(result.result.get("pending_marks").is_none());

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    let start_index = captured
        .iter()
        .position(|event| {
            event.name() == "tool-outcome" && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    let end_index = captured
        .iter()
        .position(|event| {
            event.name() == "tool-outcome" && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    let inner_index = captured
        .iter()
        .position(|event| event.name() == "tool.mark.inner")
        .unwrap();
    assert!(
        captured
            .iter()
            .all(|event| event.name() != "tool.mark.invalid")
    );
    let outer_index = captured
        .iter()
        .position(|event| event.name() == "tool.mark.outer")
        .unwrap();
    assert_eq!(captured[inner_index].metadata().unwrap()["sanitized"], true);
    assert_eq!(captured[outer_index].metadata().unwrap()["sanitized"], true);
    assert!(start_index < end_index);
    assert!(end_index < inner_index);
    assert!(inner_index < outer_index);

    let start = &captured[start_index];
    let end = &captured[end_index];
    let inner = &captured[inner_index];
    let outer = &captured[outer_index];
    assert_eq!(inner.parent_uuid(), Some(start.uuid()));
    assert_eq!(outer.parent_uuid(), Some(start.uuid()));
    assert!(inner.timestamp() > end.timestamp());
    assert!(outer.timestamp() > inner.timestamp());
    assert_eq!(end.data().unwrap(), &result.result);
    assert_eq!(inner.category().map(EventCategory::as_str), Some("custom"));
    assert_eq!(
        inner
            .category_profile()
            .and_then(|profile| profile.subtype.as_deref()),
        Some("example.tool.compression")
    );
    assert_eq!(inner.data().unwrap()["saved_tokens"], 12);
    assert_eq!(inner.metadata().unwrap()["source"], "test");
    assert_eq!(
        inner.metadata().unwrap()[LOG_SEVERITY_METADATA_KEY],
        "error"
    );
    assert_eq!(
        inner.data_schema().unwrap(),
        &DataSchema::builder()
            .name("example.tool_pending_mark")
            .version("1")
            .build()
    );
    drop(captured);

    deregister_tool_execution_intercept("passthrough_between_outcomes").unwrap();
    let mut registrations = plugin_ctx.into_registrations();
    rollback_registrations(&mut registrations);
    deregister_mark_sanitize_guardrail("tool_pending_mark_sanitizer").unwrap();
    deregister_subscriber("tool_outcome_mark_observer").unwrap();
}

#[tokio::test]
async fn managed_tool_call_id_projects_to_atif_and_otel() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let agent = setup_isolated_scope("managed-tool-exporter-correlation");

    let atif = AtifExporter::new(
        "managed-tool-exporter-correlation".to_string(),
        AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    register_subscriber("managed_tool_exporter_atif", atif.subscriber()).unwrap();

    let otel_exporter = InMemorySpanExporterBuilder::new().build();
    let otel_provider = SdkTracerProvider::builder()
        .with_simple_exporter(otel_exporter.clone())
        .build();
    let otel =
        OpenTelemetrySubscriber::from_tracer_provider(otel_provider, "managed-tool-exporter-otel");
    register_subscriber("managed_tool_exporter_otel", otel.subscriber()).unwrap();

    let tool_call_id = "call-managed-exporter-42";
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("model-call")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "weather in NYC"}],
                }),
            })
            .func(Arc::new(move |_| {
                Box::pin(async move {
                    Ok(json!({
                        "model": "test-model",
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": null,
                                "tool_calls": [{
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "{\\\"city\\\":\\\"NYC\\\"}",
                                    },
                                }],
                            },
                        }],
                    }))
                })
            }))
            .build(),
    )
    .await
    .unwrap();

    tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("get_weather")
            .args(json!({"city": "NYC"}))
            .tool_call_id(tool_call_id)
            .func(Arc::new(|args| {
                Box::pin(async move { Ok(ToolExecutionResult::new(args)) })
            }))
            .build(),
    )
    .await
    .unwrap();

    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&agent.uuid)
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();

    let trajectory = atif.export().unwrap();
    let agent_step = trajectory
        .steps
        .iter()
        .find(|step| step.source == "agent")
        .expect("managed LLM event must create an agent step");
    assert_eq!(
        agent_step.tool_calls.as_ref().unwrap()[0].tool_call_id,
        tool_call_id
    );
    assert_eq!(
        agent_step.observation.as_ref().unwrap().results[0]
            .source_call_id
            .as_deref(),
        Some(tool_call_id)
    );

    otel.force_flush().unwrap();
    let tool_span = otel_exporter
        .get_finished_spans()
        .unwrap()
        .into_iter()
        .find(|span| span.name.as_ref() == "get_weather")
        .expect("managed tool span must be exported");
    assert!(tool_span.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "nemo_relay.tool_call_id"
            && attribute.value.to_string() == tool_call_id
    }));

    deregister_subscriber("managed_tool_exporter_atif").unwrap();
    deregister_subscriber("managed_tool_exporter_otel").unwrap();
}

#[tokio::test]
async fn test_managed_tool_pending_marks_project_through_trace_exporters_only() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "managed_tool_projection_events",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let atif = AtifExporter::new(
        "managed-tool-projection".to_string(),
        AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    register_subscriber("managed_tool_projection_atif", atif.subscriber()).unwrap();

    let otel_exporter = InMemorySpanExporterBuilder::new().build();
    let otel_provider = SdkTracerProvider::builder()
        .with_simple_exporter(otel_exporter.clone())
        .build();
    let otel = OpenTelemetrySubscriber::from_tracer_provider(
        otel_provider,
        "managed-tool-projection-otel",
    );
    register_subscriber("managed_tool_projection_otel", otel.subscriber()).unwrap();

    let openinference_exporter = InMemorySpanExporterBuilder::new().build();
    let openinference_provider = SdkTracerProvider::builder()
        .with_simple_exporter(openinference_exporter.clone())
        .build();
    let openinference = OpenTelemetrySubscriber::from_tracer_provider_with_type(
        openinference_provider,
        "managed-tool-projection-openinference",
        OpenTelemetryType::OpenInference,
    );
    register_subscriber(
        "managed_tool_projection_openinference",
        openinference.subscriber(),
    )
    .unwrap();

    register_tool_execution_intercept(
        "managed_tool_projection_intercept",
        1,
        Arc::new(|_name, args, next| {
            Box::pin(async move {
                let result = next(args).await?;
                Ok(
                    ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("plugin.output_compacted")
                            .category(EventCategory::custom())
                            .category_profile(
                                CategoryProfile::builder()
                                    .subtype("example.compaction")
                                    .build(),
                            )
                            .data(json!({"saved_tokens": 12}))
                            .metadata(json!({"source": "test"}))
                            .build(),
                    ),
                )
            })
        }),
    )
    .unwrap();

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("managed-tool")
            .args(json!({"value": 42}))
            .func(Arc::new(|args| {
                Box::pin(async move {
                    Ok(ToolExecutionResult::annotated(
                        args,
                        json!({"opaque": {"rank": 1}}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result, json!({"value": 42}));
    assert_eq!(result.annotation, Some(json!({"opaque": {"rank": 1}})));

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    let tool_end_index = captured
        .iter()
        .position(|event| {
            event.name() == "managed-tool" && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    let mark_index = captured
        .iter()
        .position(|event| event.name() == "plugin.output_compacted")
        .unwrap();
    assert!(tool_end_index < mark_index);
    let tool_end = &captured[tool_end_index];
    let mark = &captured[mark_index];
    assert_eq!(
        tool_end.tool_result_annotation().unwrap(),
        json!({"opaque": {"rank": 1}})
    );
    assert_eq!(mark.parent_uuid(), Some(tool_end.uuid()));
    assert!(mark.timestamp() > tool_end.timestamp());
    drop(captured);

    let trajectory = atif.export().unwrap();
    assert!(trajectory.steps.iter().any(|step| {
        step.observation.as_ref().is_some_and(|observation| {
            observation.results.iter().any(|result| {
                result.extra.as_ref().is_some_and(|extra| {
                    extra.get("tool_result_annotation") == Some(&json!({"opaque": {"rank": 1}}))
                })
            })
        })
    }));
    assert!(trajectory.steps.iter().all(|step| {
        !step.tool_calls.as_deref().is_some_and(|calls| {
            calls
                .iter()
                .any(|call| call.function_name == "plugin.output_compacted")
        })
    }));

    otel.force_flush().unwrap();
    let otel_spans = otel_exporter.get_finished_spans().unwrap();
    let otel_tool = otel_spans
        .iter()
        .find(|span| span.name.as_ref() == "managed-tool")
        .unwrap();
    let otel_mark = otel_spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.output_compacted")
        .unwrap();
    assert!(otel_tool.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "nemo_relay.tool.result.annotation"
            && attribute.value.to_string() == r#"{"opaque":{"rank":1}}"#
    }));
    assert_eq!(otel_mark.parent_span_id, otel_tool.span_context.span_id());
    assert!(otel_mark.start_time > otel_tool.end_time);
    assert!(otel_mark.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "nemo_relay.mark.orphan"
            && attribute.value == opentelemetry::Value::Bool(true)
    }));
    for (key, value) in [
        ("nemo_relay.mark.category", "custom"),
        (
            "nemo_relay.mark.category_profile.subtype",
            "example.compaction",
        ),
        ("nemo_relay.mark.metadata.source", "test"),
    ] {
        assert!(otel_mark.attributes.iter().any(|attribute| {
            attribute.key.as_str() == key && attribute.value.to_string() == value
        }));
    }

    openinference.force_flush().unwrap();
    let openinference_spans = openinference_exporter.get_finished_spans().unwrap();
    let openinference_tool = openinference_spans
        .iter()
        .find(|span| span.name.as_ref() == "managed-tool")
        .unwrap();
    let openinference_mark = openinference_spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.output_compacted")
        .unwrap();
    assert!(openinference_tool.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "nemo_relay.tool.result.annotation"
            && attribute.value.to_string() == r#"{"opaque":{"rank":1}}"#
    }));
    assert_eq!(
        openinference_mark.parent_span_id,
        openinference_tool.span_context.span_id()
    );
    assert!(openinference_mark.start_time > openinference_tool.end_time);
    assert!(openinference_mark.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "nemo_relay.mark.orphan"
            && attribute.value == opentelemetry::Value::Bool(true)
    }));
    assert!(openinference_mark.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "openinference.span.kind"
            && attribute.value.to_string() == "CHAIN"
    }));
    for (key, value) in [
        ("nemo_relay.mark.category", "custom"),
        (
            "nemo_relay.mark.category_profile.subtype",
            "example.compaction",
        ),
        ("nemo_relay.mark.metadata.source", "test"),
    ] {
        assert!(openinference_mark.attributes.iter().any(|attribute| {
            attribute.key.as_str() == key && attribute.value.to_string() == value
        }));
    }

    deregister_tool_execution_intercept("managed_tool_projection_intercept").unwrap();
    deregister_subscriber("managed_tool_projection_events").unwrap();
    deregister_subscriber("managed_tool_projection_atif").unwrap();
    deregister_subscriber("managed_tool_projection_otel").unwrap();
    deregister_subscriber("managed_tool_projection_openinference").unwrap();
}

#[tokio::test]
async fn test_tool_execution_error_discards_downstream_pending_marks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "tool_outcome_error_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    register_tool_execution_intercept(
        "error_after_outcome",
        1,
        Arc::new(|_name, args, next| {
            Box::pin(async move {
                let _ = next(args).await?;
                Err(FlowError::Internal("outer failure".into()))
            })
        }),
    )
    .unwrap();
    let mut plugin_ctx = PluginRegistrationContext::new();
    plugin_ctx
        .register_tool_execution_intercept(
            "outcome_before_error",
            2,
            Arc::new(|_name, args, next| {
                Box::pin(async move {
                    let result = next(args).await?;
                    Ok(
                        ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("tool.mark.must_not_emit")
                                .build(),
                        ),
                    )
                })
            }),
        )
        .unwrap();

    let error = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool-outcome-error")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("outer failure"));

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    assert!(
        captured
            .iter()
            .all(|event| event.name() != "tool.mark.must_not_emit")
    );
    let error_end = captured
        .iter()
        .find(|event| {
            event.name() == "tool-outcome-error"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert!(error_end.category_profile().is_none_or(|profile| {
        profile
            .tool_result_annotation
            .as_ref()
            .is_none_or(|value| value.is_null())
    }));
    drop(captured);

    deregister_tool_execution_intercept("error_after_outcome").unwrap();
    let mut registrations = plugin_ctx.into_registrations();
    rollback_registrations(&mut registrations);
    deregister_subscriber("tool_outcome_error_observer").unwrap();
}

#[tokio::test]
async fn test_managed_tool_reuses_start_subscriber_snapshot_for_end_and_marks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let original_events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_original = original_events.clone();
    register_subscriber(
        "tool_lifecycle_original",
        Arc::new(move |event| captured_original.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let replacement_events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_replacement = replacement_events.clone();
    let mut plugin_ctx = PluginRegistrationContext::new();
    plugin_ctx
        .register_tool_execution_intercept(
            "mutate_tool_subscribers",
            1,
            Arc::new(move |_name, args, next| {
                let captured_replacement = captured_replacement.clone();
                Box::pin(async move {
                    assert!(deregister_subscriber("tool_lifecycle_original").unwrap());
                    register_subscriber(
                        "tool_lifecycle_replacement",
                        Arc::new(move |event| {
                            captured_replacement.lock().unwrap().push(event.clone());
                        }),
                    )
                    .unwrap();
                    let result = next(args).await?;
                    Ok(
                        ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("tool.snapshot.mark")
                                .build(),
                        ),
                    )
                })
            }),
        )
        .unwrap();

    tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool-subscriber-snapshot")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    )
    .await
    .unwrap();
    flush_subscribers().unwrap();

    let original_events = original_events.lock().unwrap();
    assert!(original_events.iter().any(|event| {
        event.name() == "tool-subscriber-snapshot"
            && event.scope_category() == Some(ScopeCategory::Start)
    }));
    assert!(original_events.iter().any(|event| {
        event.name() == "tool-subscriber-snapshot"
            && event.scope_category() == Some(ScopeCategory::End)
    }));
    assert!(
        original_events
            .iter()
            .any(|event| event.name() == "tool.snapshot.mark")
    );
    drop(original_events);
    assert!(replacement_events.lock().unwrap().is_empty());

    assert!(deregister_subscriber("tool_lifecycle_replacement").unwrap());
    let mut registrations = plugin_ctx.into_registrations();
    rollback_registrations(&mut registrations);
}

#[tokio::test]
async fn test_managed_tool_reuses_start_subscriber_snapshot_for_error_end() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let original_events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_original = original_events.clone();
    register_subscriber(
        "tool_error_original",
        Arc::new(move |event| captured_original.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let replacement_events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_replacement = replacement_events.clone();
    register_tool_execution_intercept(
        "mutate_tool_error_subscribers",
        1,
        Arc::new(move |_name, _args, _next| {
            let captured_replacement = captured_replacement.clone();
            Box::pin(async move {
                assert!(deregister_subscriber("tool_error_original").unwrap());
                register_subscriber(
                    "tool_error_replacement",
                    Arc::new(move |event| {
                        captured_replacement.lock().unwrap().push(event.clone());
                    }),
                )
                .unwrap();
                Err(FlowError::Internal("managed tool failure".into()))
            })
        }),
    )
    .unwrap();

    let error = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool-error-subscriber-snapshot")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("managed tool failure"));
    flush_subscribers().unwrap();

    let original_events = original_events.lock().unwrap();
    assert!(original_events.iter().any(|event| {
        event.name() == "tool-error-subscriber-snapshot"
            && event.scope_category() == Some(ScopeCategory::Start)
    }));
    assert!(original_events.iter().any(|event| {
        event.name() == "tool-error-subscriber-snapshot"
            && event.scope_category() == Some(ScopeCategory::End)
    }));
    drop(original_events);
    assert!(replacement_events.lock().unwrap().is_empty());

    deregister_tool_execution_intercept("mutate_tool_error_subscribers").unwrap();
    assert!(deregister_subscriber("tool_error_replacement").unwrap());
}

#[tokio::test]
async fn test_repeated_next_marks_follow_invocation_order_not_completion_order() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_events = events.clone();
    register_subscriber(
        "tool_concurrent_next_observer",
        Arc::new(move |event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    register_tool_execution_intercept(
        "concurrent_next",
        1,
        Arc::new(|_name, _args, next| {
            Box::pin(async move {
                let first = next(json!({"branch": "first", "delay_ms": 40}));
                let second = next(json!({"branch": "second", "delay_ms": 1}));
                let (first, second) = tokio::join!(first, second);
                let first = first?;
                let second = second?;
                assert_eq!(first.annotation, Some(json!({"branch": "first"})));
                assert_eq!(second.annotation, Some(json!({"branch": "second"})));
                Ok(ToolExecutionInterceptOutcome::annotated(
                    json!({
                        "first": first.result,
                        "second": second.result,
                    }),
                    json!({"combined": true}),
                ))
            })
        }),
    )
    .unwrap();

    let completion_order = Arc::new(Mutex::new(Vec::<String>::new()));
    let captured_completion_order = completion_order.clone();
    let mut plugin_ctx = PluginRegistrationContext::new();
    plugin_ctx
        .register_tool_execution_intercept(
            "delayed_outcomes",
            2,
            Arc::new(move |_name, args, next| {
                let captured_completion_order = captured_completion_order.clone();
                Box::pin(async move {
                    let branch = args["branch"].as_str().unwrap().to_string();
                    let delay_ms = args["delay_ms"].as_u64().unwrap();
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    let result = next(args).await?;
                    captured_completion_order
                        .lock()
                        .unwrap()
                        .push(branch.clone());
                    Ok(
                        ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                            PendingMarkSpec::builder()
                                .name(format!("tool.concurrent.{branch}"))
                                .build(),
                        ),
                    )
                })
            }),
        )
        .unwrap();

    let provider_barrier = Arc::new(tokio::sync::Barrier::new(2));
    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool-concurrent-next")
            .args(json!({}))
            .func(Arc::new(move |args| {
                let provider_barrier = Arc::clone(&provider_barrier);
                Box::pin(async move {
                    let branch = args["branch"].as_str().unwrap().to_string();
                    let scope_name = if branch == "first" {
                        "tool-concurrent-next-first"
                    } else {
                        "tool-concurrent-next-second"
                    };
                    let scope = push_scope(
                        nemo_relay::api::scope::PushScopeParams::builder()
                            .name(scope_name)
                            .scope_type(ScopeType::Tool)
                            .build(),
                    )?;
                    provider_barrier.wait().await;
                    if branch == "first" {
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    }
                    pop_scope(
                        nemo_relay::api::scope::PopScopeParams::builder()
                            .handle_uuid(&scope.uuid)
                            .build(),
                    )?;
                    Ok(ToolExecutionResult::annotated(
                        args,
                        json!({"branch": branch}),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result["first"]["branch"], "first");
    assert_eq!(result.result["second"]["branch"], "second");
    assert_eq!(result.annotation, Some(json!({"combined": true})));
    flush_subscribers().unwrap();

    assert_eq!(
        *completion_order.lock().unwrap(),
        vec!["second".to_string(), "first".to_string()]
    );
    let events = events.lock().unwrap();
    let marks = events
        .iter()
        .filter(|event| event.name().starts_with("tool.concurrent."))
        .collect::<Vec<_>>();
    assert_eq!(
        marks.iter().map(|event| event.name()).collect::<Vec<_>>(),
        ["tool.concurrent.first", "tool.concurrent.second"]
    );
    assert!(marks[0].timestamp() < marks[1].timestamp());
    drop(events);

    deregister_tool_execution_intercept("concurrent_next").unwrap();
    let mut registrations = plugin_ctx.into_registrations();
    rollback_registrations(&mut registrations);
    deregister_subscriber("tool_concurrent_next_observer").unwrap();
}

#[tokio::test]
async fn execution_next_is_revoked_after_each_interceptor_settles() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let tool_next = Arc::new(Mutex::new(None::<ToolExecutionNextFn>));
    let captured_tool_next = Arc::clone(&tool_next);
    register_tool_execution_intercept(
        "late_tool_next",
        1,
        Arc::new(move |_name, _args, next| {
            *captured_tool_next.lock().unwrap() = Some(next);
            Box::pin(async {
                Ok(ToolExecutionInterceptOutcome::new(
                    json!({"source": "tool-intercept"}),
                ))
            })
        }),
    )
    .unwrap();
    let tool_provider_calls = Arc::new(AtomicU32::new(0));
    let captured_tool_provider_calls = Arc::clone(&tool_provider_calls);
    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("late-tool-next")
            .args(json!({}))
            .func(Arc::new(move |args| {
                captured_tool_provider_calls.fetch_add(1, Ordering::AcqRel);
                ready_result(Ok(args.into()))
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result, json!({"source": "tool-intercept"}));
    let late_tool_next = tool_next.lock().unwrap().take().unwrap();
    let error = late_tool_next(json!({"late": true})).await.unwrap_err();
    assert!(matches!(
        error,
        FlowError::InvalidArgument(message)
            if message == "execution continuation is no longer active"
    ));
    assert_eq!(tool_provider_calls.load(Ordering::Acquire), 0);
    deregister_tool_execution_intercept("late_tool_next").unwrap();

    let llm_next = Arc::new(Mutex::new(None::<LlmExecutionNextFn>));
    let captured_llm_next = Arc::clone(&llm_next);
    register_llm_execution_intercept(
        "late_llm_next",
        1,
        Arc::new(move |_name, _request, next| {
            *captured_llm_next.lock().unwrap() = Some(next);
            ready_result(Ok(json!({"source": "llm-intercept"})))
        }),
    )
    .unwrap();
    let llm_provider_calls = Arc::new(AtomicU32::new(0));
    let captured_llm_provider_calls = Arc::clone(&llm_provider_calls);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "late"}),
    };
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("late-llm-next")
            .request(request.clone())
            .func(Arc::new(move |_request| {
                captured_llm_provider_calls.fetch_add(1, Ordering::AcqRel);
                ready_result(Ok(json!({"source": "provider"})))
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result, json!({"source": "llm-intercept"}));
    let late_llm_next = llm_next.lock().unwrap().take().unwrap();
    let error = late_llm_next(request.clone()).await.unwrap_err();
    assert!(matches!(
        error,
        FlowError::InvalidArgument(message)
            if message == "execution continuation is no longer active"
    ));
    assert_eq!(llm_provider_calls.load(Ordering::Acquire), 0);
    deregister_llm_execution_intercept("late_llm_next").unwrap();

    let stream_next = Arc::new(Mutex::new(None::<LlmStreamExecutionNextFn>));
    let captured_stream_next = Arc::clone(&stream_next);
    register_llm_stream_execution_intercept(
        "late_llm_stream_next",
        1,
        Arc::new(move |_name, _request, next| {
            *captured_stream_next.lock().unwrap() = Some(next);
            Box::pin(async {
                Ok(LlmJsonStream::new(futures::stream::iter(vec![Ok(
                    json!({"source": "stream-intercept"}),
                )])))
            })
        }),
    )
    .unwrap();
    let stream_provider_calls = Arc::new(AtomicU32::new(0));
    let captured_stream_provider_calls = Arc::clone(&stream_provider_calls);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("late-llm-stream-next")
            .request(request.clone())
            .func(Arc::new(move |_request| {
                captured_stream_provider_calls.fetch_add(1, Ordering::AcqRel);
                Box::pin(async {
                    Ok(LlmJsonStream::new(futures::stream::iter(vec![Ok(
                        json!({"source": "provider"}),
                    )])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"complete": true})))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        json!({"source": "stream-intercept"})
    );
    assert!(stream.next().await.is_none());
    let late_stream_next = stream_next.lock().unwrap().take().unwrap();
    let error = match late_stream_next(request).await {
        Ok(_) => panic!("late stream continuation should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        FlowError::InvalidArgument(message)
            if message == "execution continuation is no longer active"
    ));
    assert_eq!(stream_provider_calls.load(Ordering::Acquire), 0);
    deregister_llm_stream_execution_intercept("late_llm_stream_next").unwrap();
}

#[tokio::test]
async fn stream_next_is_revoked_when_the_managed_stream_terminalizes_with_an_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "terminal-error"}),
    };
    let upstream_error_next = Arc::new(Mutex::new(None::<LlmStreamExecutionNextFn>));
    let captured_upstream_error_next = Arc::clone(&upstream_error_next);
    register_llm_stream_execution_intercept(
        "upstream_error_stream_next",
        1,
        Arc::new(move |_name, request, next| {
            *captured_upstream_error_next.lock().unwrap() = Some(next.clone());
            next(request)
        }),
    )
    .unwrap();
    let upstream_provider_calls = Arc::new(AtomicU32::new(0));
    let captured_upstream_provider_calls = Arc::clone(&upstream_provider_calls);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("upstream-error-stream-next")
            .request(request.clone())
            .func(Arc::new(move |_request| {
                captured_upstream_provider_calls.fetch_add(1, Ordering::AcqRel);
                Box::pin(async {
                    Ok(LlmJsonStream::new(futures::stream::iter(vec![Err(
                        FlowError::Internal("provider stream failed".into()),
                    )])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({})))
            .build(),
    )
    .await
    .unwrap();
    assert!(stream.next().await.unwrap().is_err());
    let late_next = upstream_error_next.lock().unwrap().take().unwrap();
    let error = match late_next(request.clone()).await {
        Ok(_) => panic!("terminal upstream error must revoke the stream continuation"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        FlowError::InvalidArgument(message)
            if message == "execution continuation is no longer active"
    ));
    assert_eq!(upstream_provider_calls.load(Ordering::Acquire), 1);
    deregister_llm_stream_execution_intercept("upstream_error_stream_next").unwrap();

    let collector_error_next = Arc::new(Mutex::new(None::<LlmStreamExecutionNextFn>));
    let captured_collector_error_next = Arc::clone(&collector_error_next);
    register_llm_stream_execution_intercept(
        "collector_error_stream_next",
        1,
        Arc::new(move |_name, request, next| {
            *captured_collector_error_next.lock().unwrap() = Some(next.clone());
            next(request)
        }),
    )
    .unwrap();
    let collector_provider_calls = Arc::new(AtomicU32::new(0));
    let captured_collector_provider_calls = Arc::clone(&collector_provider_calls);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("collector-error-stream-next")
            .request(request.clone())
            .func(Arc::new(move |_request| {
                captured_collector_provider_calls.fetch_add(1, Ordering::AcqRel);
                Box::pin(async {
                    Ok(LlmJsonStream::new(futures::stream::iter(vec![Ok(
                        json!({"chunk": true}),
                    )])))
                })
            }))
            .collector(Box::new(|_| {
                Err(FlowError::Internal("collector failed".into()))
            }))
            .finalizer(Box::new(|| json!({})))
            .build(),
    )
    .await
    .unwrap();
    assert!(stream.next().await.unwrap().is_err());
    let late_next = collector_error_next.lock().unwrap().take().unwrap();
    let error = match late_next(request).await {
        Ok(_) => panic!("terminal collector error must revoke the stream continuation"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        FlowError::InvalidArgument(message)
            if message == "execution continuation is no longer active"
    ));
    assert_eq!(collector_provider_calls.load(Ordering::Acquire), 1);
    deregister_llm_stream_execution_intercept("collector_error_stream_next").unwrap();
}

#[tokio::test]
async fn spawned_rust_next_preserves_the_full_managed_context() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_events = Arc::clone(&events);
    register_subscriber(
        "spawned_rust_next_context",
        Arc::new(move |event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_tool_execution_intercept(
        "spawned_rust_next",
        1,
        Arc::new(|_name, args, next| {
            Box::pin(async move {
                tokio::spawn(async move { next(args).await })
                    .await
                    .map_err(|error| FlowError::Internal(error.to_string()))?
                    .map(Into::into)
            })
        }),
    )
    .unwrap();

    let task_stack = create_scope_stack();
    let (provider_parent, owner) = TASK_SCOPE_STACK
        .scope(task_stack, async {
            let owner = push_scope(
                nemo_relay::api::scope::PushScopeParams::builder()
                    .name("spawned-rust-next-owner")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .unwrap();
            let result = tool_call_execute(
                nemo_relay::api::tool::ToolCallExecuteParams::builder()
                    .name("spawned-rust-next")
                    .args(json!({}))
                    .func(Arc::new(|_args| {
                        Box::pin(async {
                            Ok(json!({
                                "parent_uuid": capture_propagation_context()?.parent_uuid.to_string(),
                                "scope_uuid": task_scope_top().uuid.to_string(),
                            })
                            .into())
                        })
                    }))
                    .build(),
            )
            .await
            .unwrap();
            pop_scope(
                nemo_relay::api::scope::PopScopeParams::builder()
                    .handle_uuid(&owner.uuid)
                    .build(),
            )
            .unwrap();
            (result, owner)
        })
        .await;
    flush_subscribers().unwrap();

    let start_uuid = events
        .lock()
        .unwrap()
        .iter()
        .find(|event| {
            event.name() == "spawned-rust-next"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap()
        .uuid()
        .to_string();
    assert_eq!(
        provider_parent.result,
        json!({
            "parent_uuid": start_uuid,
            "scope_uuid": owner.uuid.to_string(),
        })
    );

    deregister_tool_execution_intercept("spawned_rust_next").unwrap();
    deregister_subscriber("spawned_rust_next_context").unwrap();
}

#[tokio::test]
async fn default_lazy_stream_preserves_the_full_managed_context() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured_events = Arc::clone(&events);
    register_subscriber(
        "default_lazy_stream_context",
        Arc::new(move |event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let owner = setup_isolated_scope("default-lazy-stream-owner");
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("default-lazy-stream-context")
            .request(LlmRequest {
                headers: Default::default(),
                content: json!({}),
            })
            .func(Arc::new(|_| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(futures::stream::once(async {
                        Ok(json!({
                            "parent_uuid": capture_propagation_context()?.parent_uuid.to_string(),
                            "scope_uuid": task_scope_top().uuid.to_string(),
                        }))
                    })))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({})))
            .build(),
    )
    .await
    .unwrap();

    let provider_context = stream.next().await.unwrap().unwrap();
    assert!(stream.next().await.is_none());
    flush_subscribers().unwrap();
    let start_uuid = events
        .lock()
        .unwrap()
        .iter()
        .find(|event| {
            event.name() == "default-lazy-stream-context"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap()
        .uuid()
        .to_string();
    assert_eq!(
        provider_context,
        json!({
            "parent_uuid": start_uuid,
            "scope_uuid": owner.uuid.to_string(),
        })
    );

    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&owner.uuid)
            .build(),
    )
    .unwrap();
    deregister_subscriber("default_lazy_stream_context").unwrap();
}

#[tokio::test]
async fn stream_next_preserves_each_invocation_scope_while_polling() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let first_stack = create_scope_stack();
    let first_scope_uuid = first_stack.read().unwrap().top().uuid;
    let second_stack = create_scope_stack();
    let second_scope_uuid = second_stack.read().unwrap().top().uuid;
    register_llm_stream_execution_intercept(
        "scoped_stream_next",
        1,
        Arc::new(move |_name, request, next| {
            let first_stack = first_stack.clone();
            let second_stack = second_stack.clone();
            Box::pin(async move {
                let first_request = LlmRequest {
                    headers: request.headers.clone(),
                    content: json!({"branch": "first"}),
                };
                let second_request = LlmRequest {
                    headers: request.headers,
                    content: json!({"branch": "second"}),
                };
                let first_next = {
                    let next = next.clone();
                    TASK_SCOPE_STACK.scope(first_stack, async move { next(first_request).await })
                };
                let second_next =
                    TASK_SCOPE_STACK.scope(second_stack, async move { next(second_request).await });
                let (first_stream, second_stream) = tokio::join!(first_next, second_next);
                let first_stream = first_stream?;
                let second_stream = second_stream?;
                Ok(LlmJsonStream::new(first_stream.chain(second_stream)))
            })
        }),
    )
    .unwrap();

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("scoped-stream-next")
            .request(LlmRequest {
                headers: Default::default(),
                content: json!({}),
            })
            .func(Arc::new(|request| {
                Box::pin(async move {
                    Ok(LlmJsonStream::new(futures::stream::once(async move {
                        Ok(json!({
                            "branch": request.content["branch"],
                            "scope_uuid": task_scope_top().uuid.to_string(),
                        }))
                    })))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({})))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        json!({
            "branch": "first",
            "scope_uuid": first_scope_uuid.to_string(),
        })
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        json!({
            "branch": "second",
            "scope_uuid": second_scope_uuid.to_string(),
        })
    );
    assert!(stream.next().await.is_none());
    deregister_llm_stream_execution_intercept("scoped_stream_next").unwrap();
}

#[tokio::test]
async fn stream_next_remains_active_during_interceptor_stream_close() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_llm_stream_execution_intercept(
        "close_calls_stream_next",
        1,
        Arc::new(move |_name, request, next| {
            Box::pin(async move {
                Ok(LlmJsonStream::from_closeable(CloseCallsStreamNext {
                    next: Some(next),
                    request: Some(request),
                }))
            })
        }),
    )
    .unwrap();
    let provider_calls = Arc::new(AtomicU32::new(0));
    let captured_provider_calls = Arc::clone(&provider_calls);
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("close-calls-stream-next")
            .request(LlmRequest {
                headers: Default::default(),
                content: json!({}),
            })
            .func(Arc::new(move |_request| {
                captured_provider_calls.fetch_add(1, Ordering::AcqRel);
                Box::pin(async { Ok(LlmJsonStream::new(futures::stream::empty())) })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({})))
            .build(),
    )
    .await
    .unwrap();

    stream.close().await.unwrap();
    assert_eq!(provider_calls.load(Ordering::Acquire), 1);
    deregister_llm_stream_execution_intercept("close_calls_stream_next").unwrap();
}

#[tokio::test]
async fn dropping_pending_tool_execution_closes_the_managed_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "cancelled_tool_lifecycle",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    register_tool_execution_intercept(
        "pending_tool_execution",
        1,
        Arc::new(move |_name, _args, _next| {
            if let Some(sender) = entered_tx.lock().unwrap().take() {
                let _ = sender.send(());
            }
            Box::pin(std::future::pending())
        }),
    )
    .unwrap();

    let mut execution = Box::pin(tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("cancelled-tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .tool_call_id("call-managed-cancelled")
            .build(),
    ));
    tokio::select! {
        result = &mut execution => panic!("execution unexpectedly completed: {result:?}"),
        result = entered_rx => result.unwrap(),
    }

    assert_flush_waits_for_pending_completion(|| drop(execution));

    let captured = captured_events_snapshot(&events);
    assert_tool_lifecycle_correlation(&captured, "cancelled-tool", Some("call-managed-cancelled"));

    let lifecycle = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == "cancelled-tool")
        .filter_map(Event::scope_category)
        .collect::<Vec<_>>();
    assert_eq!(lifecycle, [ScopeCategory::Start, ScopeCategory::End]);
    let event_log = events.lock().unwrap();
    let cancelled_end = event_log
        .iter()
        .find(|event| {
            event.name() == "cancelled-tool" && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert!(cancelled_end.category_profile().is_none_or(|profile| {
        profile
            .tool_result_annotation
            .as_ref()
            .is_none_or(|value| value.is_null())
    }));
    drop(event_log);

    deregister_tool_execution_intercept("pending_tool_execution").unwrap();
    deregister_subscriber("cancelled_tool_lifecycle").unwrap();
}

#[tokio::test]
async fn cancelled_tool_end_uses_the_originating_scope_sanitizer() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let originating_stack = current_scope_stack();
    let owner = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("cancelled-tool-sanitizer-owner")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope_register_scope_sanitize_end_guardrail(
        &owner.uuid,
        "cancelled-tool-end-sanitizer",
        1,
        Arc::new(|event, mut fields| {
            if event.name() == "cancelled-tool-cross-scope" {
                fields.data = Some(json!({"secret": "[redacted]"}));
            }
            ready(fields)
        }),
    )
    .unwrap();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "cancelled_tool_cross_scope_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    register_tool_execution_intercept(
        "pending_tool_cross_scope",
        1,
        Arc::new(move |_name, _args, _next| {
            if let Some(sender) = entered_tx.lock().unwrap().take() {
                let _ = sender.send(());
            }
            Box::pin(std::future::pending())
        }),
    )
    .unwrap();

    let mut execution = Box::pin(tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("cancelled-tool-cross-scope")
            .args(json!({}))
            .data(json!({"secret": "classified"}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    ));
    tokio::select! {
        result = &mut execution => panic!("execution unexpectedly completed: {result:?}"),
        result = entered_rx => result.unwrap(),
    }

    set_thread_scope_stack(create_scope_stack());
    drop(execution);
    flush_subscribers().unwrap();

    let end = captured_events_snapshot(&events)
        .into_iter()
        .find(|event| {
            event.name() == "cancelled-tool-cross-scope"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .expect("cancelled tool should emit an end event");
    assert_eq!(end.data(), Some(&json!({"secret": "[redacted]"})));
    assert!(end.category_profile().is_none_or(|profile| {
        profile
            .tool_result_annotation
            .as_ref()
            .is_none_or(|value| value.is_null())
    }));

    set_thread_scope_stack(originating_stack);
    deregister_tool_execution_intercept("pending_tool_cross_scope").unwrap();
    deregister_subscriber("cancelled_tool_cross_scope_observer").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&owner.uuid)
            .build(),
    )
    .unwrap();
}

#[tokio::test]
async fn dropping_pending_conditional_closes_the_guardrail_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "cancelled_guardrail_lifecycle",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    register_tool_conditional_execution_guardrail(
        "pending_conditional",
        1,
        Arc::new(move |_name, _args| {
            if let Some(sender) = entered_tx.lock().unwrap().take() {
                let _ = sender.send(());
            }
            Box::pin(std::future::pending())
        }),
    )
    .unwrap();

    let mut execution = Box::pin(tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("cancelled-conditional-tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
            .build(),
    ));
    tokio::select! {
        result = &mut execution => panic!("execution unexpectedly completed: {result:?}"),
        result = entered_rx => result.unwrap(),
    }
    assert_flush_waits_for_pending_completion(|| drop(execution));

    let lifecycle = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == "pending_conditional")
        .filter_map(Event::scope_category)
        .collect::<Vec<_>>();
    assert_eq!(lifecycle, [ScopeCategory::Start, ScopeCategory::End]);

    deregister_tool_conditional_execution_guardrail("pending_conditional").unwrap();
    deregister_subscriber("cancelled_guardrail_lifecycle").unwrap();
}

#[tokio::test]
async fn cancelled_guardrail_end_uses_the_originating_scope_sanitizer() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let originating_stack = current_scope_stack();
    let owner = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("cancelled-guardrail-sanitizer-owner")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope_register_scope_sanitize_end_guardrail(
        &owner.uuid,
        "cancelled-guardrail-end-sanitizer",
        1,
        Arc::new(|event, mut fields| {
            if event.name() == "pending-conditional-cross-scope" {
                fields.metadata = Some(json!({"sanitized_by": "originating-scope"}));
            }
            ready(fields)
        }),
    )
    .unwrap();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "cancelled_guardrail_cross_scope_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    register_tool_conditional_execution_guardrail(
        "pending-conditional-cross-scope",
        1,
        Arc::new(move |_name, _args| {
            if let Some(sender) = entered_tx.lock().unwrap().take() {
                let _ = sender.send(());
            }
            Box::pin(std::future::pending())
        }),
    )
    .unwrap();

    let conditional_args = json!({});
    let mut evaluation = Box::pin(tool_conditional_execution(
        "cancelled-conditional-cross-scope",
        &conditional_args,
    ));
    tokio::select! {
        result = &mut evaluation => panic!("evaluation unexpectedly completed: {result:?}"),
        result = entered_rx => result.unwrap(),
    }

    set_thread_scope_stack(create_scope_stack());
    drop(evaluation);
    flush_subscribers().unwrap();

    let end = captured_events_snapshot(&events)
        .into_iter()
        .find(|event| {
            event.name() == "pending-conditional-cross-scope"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .expect("cancelled guardrail should emit an end event");
    assert_eq!(
        end.metadata(),
        Some(&json!({"sanitized_by": "originating-scope"}))
    );

    set_thread_scope_stack(originating_stack);
    deregister_tool_conditional_execution_guardrail("pending-conditional-cross-scope").unwrap();
    deregister_subscriber("cancelled_guardrail_cross_scope_observer").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&owner.uuid)
            .build(),
    )
    .unwrap();
}

#[tokio::test]
async fn dropping_pending_llm_execution_closes_the_managed_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "cancelled_llm_lifecycle",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    register_llm_execution_intercept(
        "pending_llm_execution",
        1,
        Arc::new(move |_name, _request, _next| {
            if let Some(sender) = entered_tx.lock().unwrap().take() {
                let _ = sender.send(());
            }
            Box::pin(std::future::pending())
        }),
    )
    .unwrap();

    let mut execution = Box::pin(llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("cancelled-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "test"}),
            })
            .data(json!({"fallback": true}))
            .func(Arc::new(|_request| Box::pin(async { Ok(json!({})) })))
            .build(),
    ));
    tokio::select! {
        result = &mut execution => panic!("execution unexpectedly completed: {result:?}"),
        result = entered_rx => result.unwrap(),
    }
    assert_flush_waits_for_pending_completion(|| drop(execution));

    let lifecycle = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == "cancelled-llm")
        .filter_map(Event::scope_category)
        .collect::<Vec<_>>();
    assert_eq!(lifecycle, [ScopeCategory::Start, ScopeCategory::End]);

    deregister_llm_execution_intercept("pending_llm_execution").unwrap();
    deregister_subscriber("cancelled_llm_lifecycle").unwrap();
}

// =========================================================================
// Guardrail Conditional Execution Tests
// =========================================================================

/// Register a conditional guardrail that rejects (returns Some).
/// Verify tool_call_execute returns GuardrailRejected error.
#[tokio::test]
async fn test_conditional_guardrail_rejects() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "managed_tool_call_id_rejection_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    register_tool_conditional_execution_guardrail(
        "rejector",
        1,
        Arc::new(|_name, _args| Box::pin(async { Ok(Some("not allowed".to_string())) })),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .tool_call_id("call-managed-rejected")
            .build(),
    )
    .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FlowError::GuardrailRejected(reason) => {
            assert_eq!(reason, "not allowed");
        }
        other => panic!("Expected GuardrailRejected, got: {:?}", other),
    }

    let captured = captured_events_snapshot(&events);
    assert!(
        captured
            .iter()
            .all(|event| { event.name() != "tool" || event.scope_type() != Some(ScopeType::Tool) })
    );

    // Cleanup
    deregister_tool_conditional_execution_guardrail("rejector").unwrap();
    deregister_subscriber("managed_tool_call_id_rejection_observer").unwrap();
}

/// Register a conditional guardrail that allows (returns None). Execution proceeds.
#[tokio::test]
async fn test_conditional_guardrail_allows() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_conditional_execution_guardrail(
        "allower",
        1,
        Arc::new(|_name, _args| Box::pin(async { Ok(None) })),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"input": "data"}))
            .func(func)
            .build(),
    )
    .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().result["input"], "data");

    // Cleanup
    deregister_tool_conditional_execution_guardrail("allower").unwrap();
}

/// Conditional tool guardrails emit Guardrail scope start/end pairs for allow
/// and reject decisions.
#[tokio::test]
async fn test_tool_conditional_guardrail_emits_guardrail_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "tool_guardrail_scope_capture",
        Arc::new(move |event| {
            captured.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();

    register_tool_conditional_execution_guardrail(
        "tool_scope_allow",
        1,
        Arc::new(|_, _| ready(None)),
    )
    .unwrap();
    register_tool_conditional_execution_guardrail(
        "tool_scope_reject",
        2,
        Arc::new(|_, _| ready(Some("blocked by tool guardrail".to_string()))),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));
    let allowed = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"safe": true}))
            .func(func.clone())
            .build(),
    )
    .await;
    assert!(allowed.is_err(), "second guardrail should reject");

    deregister_tool_conditional_execution_guardrail("tool_scope_reject").unwrap();
    let allowed = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"safe": true}))
            .func(func)
            .build(),
    )
    .await;
    assert!(allowed.is_ok());

    deregister_tool_conditional_execution_guardrail("tool_scope_allow").unwrap();
    deregister_subscriber("tool_guardrail_scope_capture").unwrap();

    let events = captured_events_snapshot(&events);
    let guardrail_events = events
        .iter()
        .filter(|event| event.scope_type() == Some(ScopeType::Guardrail))
        .collect::<Vec<_>>();
    assert_eq!(
        guardrail_events
            .iter()
            .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
            .count(),
        3
    );
    assert_eq!(
        guardrail_events
            .iter()
            .filter(|event| event.scope_category() == Some(ScopeCategory::End))
            .count(),
        3
    );
    assert!(guardrail_events.iter().all(|event| {
        event.scope_category() != Some(ScopeCategory::Start)
            || event.data().and_then(|data| data.get("input")).is_none()
    }));
    assert!(guardrail_events.iter().any(|event| {
        event.name() == "tool_scope_allow"
            && event.scope_category() == Some(ScopeCategory::End)
            && event
                .data()
                .and_then(|data| data.get("allowed"))
                .and_then(|value| value.as_bool())
                == Some(true)
    }));
    assert!(guardrail_events.iter().any(|event| {
        event.name() == "tool_scope_reject"
            && event.scope_category() == Some(ScopeCategory::End)
            && event
                .data()
                .and_then(|data| data.get("rejection_reason"))
                .and_then(|value| value.as_str())
                == Some("blocked by tool guardrail")
    }));
}

/// Multiple conditional guardrails: first allows, second rejects.
/// The second one should reject (first rejection wins).
#[tokio::test]
async fn test_conditional_guardrail_first_rejection_wins() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_conditional_execution_guardrail(
        "allows",
        1,
        Arc::new(|_name, _args| Box::pin(async { Ok(None) })),
    )
    .unwrap();

    register_tool_conditional_execution_guardrail(
        "rejects",
        2,
        Arc::new(|_name, _args| Box::pin(async { Ok(Some("blocked by second".to_string())) })),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .build(),
    )
    .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FlowError::GuardrailRejected(reason) => {
            assert!(reason.contains("blocked by second"));
        }
        other => panic!("Expected GuardrailRejected, got: {:?}", other),
    }

    // Cleanup
    deregister_tool_conditional_execution_guardrail("allows").unwrap();
    deregister_tool_conditional_execution_guardrail("rejects").unwrap();
}

/// Conditional guardrail that only rejects specific tool names.
#[tokio::test]
async fn test_conditional_guardrail_tool_name_filtering() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_conditional_execution_guardrail(
        "name_filter",
        1,
        Arc::new(|name, _args| {
            if name == "dangerous_tool" {
                ready(Some("dangerous_tool is forbidden".to_string()))
            } else {
                ready(None)
            }
        }),
    )
    .unwrap();

    // Dangerous tool is rejected
    let func1: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));
    let err = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("dangerous_tool")
            .args(json!({}))
            .func(func1)
            .build(),
    )
    .await;
    assert!(err.is_err());

    // Safe tool is allowed
    let func2: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));
    let ok = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({}))
            .func(func2)
            .build(),
    )
    .await;
    assert!(ok.is_ok());

    // Cleanup
    deregister_tool_conditional_execution_guardrail("name_filter").unwrap();
}

// =========================================================================
// Scope-Local Middleware Tests
// =========================================================================

/// Push scope, register scope-local guardrail, verify it applies,
/// pop scope, verify it no longer applies.
#[tokio::test]
async fn test_scope_local_guardrail_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("lifecycle_scope");

    let call_count = Arc::new(AtomicU32::new(0));

    // Register a scope-local sanitize request guardrail
    let cc = call_count.clone();
    scope_register_tool_sanitize_request_guardrail(
        &handle.uuid,
        "scoped_guardrail",
        1,
        Arc::new(move |_name, args| {
            cc.fetch_add(1, Ordering::SeqCst);
            ready(args)
        }),
    )
    .unwrap();

    // Invoke tool call -- guardrail should fire
    let _tool = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "Scope-local guardrail should run"
    );

    // Pop scope -- guardrail should be cleaned up
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();

    // Invoke tool call again -- guardrail should NOT fire
    let _tool2 = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "After scope pop, guardrail should not run"
    );
}

/// Scope-local execution intercept is cleaned up on scope pop.
#[tokio::test]
async fn test_scope_local_execution_intercept_cleanup() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("exec_scope");

    let intercept_called = Arc::new(AtomicU32::new(0));

    let ic = intercept_called.clone();
    scope_register_tool_execution_intercept(
        &handle.uuid,
        "scoped_exec",
        1,
        Arc::new(move |_name, args, next| {
            ic.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { next(args).await.map(Into::into) })
        }),
    )
    .unwrap();

    // Execute -- intercept should fire
    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));
    let _ = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(intercept_called.load(Ordering::SeqCst), 1);

    // Pop scope
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();

    // Execute again -- intercept should NOT fire
    let func2: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));
    let _ = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func2)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(
        intercept_called.load(Ordering::SeqCst),
        1,
        "Scope-local execution intercept should not run after pop"
    );
}

// =========================================================================
// Scope-Local + Global Merging Tests
// =========================================================================

/// Register global guardrail at priority 5, scope-local guardrail at priority 3.
/// Verify scope-local runs first (lower priority number = higher priority).
/// Verify both are applied.
#[tokio::test]
async fn test_scope_local_and_global_guardrail_merge_priority() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("merge_scope");

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    // Global guardrail at priority 5
    let og = order.clone();
    register_tool_sanitize_request_guardrail(
        "global_g",
        5,
        Arc::new(move |_name, mut args| {
            og.lock().unwrap().push("global".into());
            args.as_object_mut()
                .unwrap()
                .insert("global".into(), json!(true));
            ready(args)
        }),
    )
    .unwrap();

    // Scope-local guardrail at priority 3
    let ol = order.clone();
    scope_register_tool_sanitize_request_guardrail(
        &handle.uuid,
        "local_g",
        3,
        Arc::new(move |_name, mut args| {
            ol.lock().unwrap().push("local".into());
            args.as_object_mut()
                .unwrap()
                .insert("local".into(), json!(true));
            ready(args)
        }),
    )
    .unwrap();

    // Capture via events
    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "merge_observer",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let _tool = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();

    // Verify order: local (priority 3) runs before global (priority 5)
    let recorded = order.lock().unwrap();
    assert_eq!(
        *recorded,
        vec!["local", "global"],
        "Lower priority should run first"
    );

    // Verify both guardrails applied their transformations
    let captured = captured_events_snapshot(&events);
    let start_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Tool, ScopeCategory::Start))
        .unwrap();
    let input = start_event.input().unwrap();
    assert_eq!(input["global"], true);
    assert_eq!(input["local"], true);

    // Cleanup
    deregister_tool_sanitize_request_guardrail("global_g").unwrap();
    deregister_subscriber("merge_observer").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();
}

/// Global and scope-local execution intercepts merge in priority order.
#[tokio::test]
async fn test_scope_local_and_global_execution_intercept_merge() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("exec_merge");

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    // Global execution intercept at priority 10
    let og = order.clone();
    register_tool_execution_intercept(
        "global_exec",
        10,
        Arc::new(move |_name, args, next| {
            let o = og.clone();
            Box::pin(async move {
                o.lock().unwrap().push("global_before".into());
                let r = next(args).await;
                o.lock().unwrap().push("global_after".into());
                r.map(Into::into)
            })
        }),
    )
    .unwrap();

    // Scope-local execution intercept at priority 5 (runs first)
    let ol = order.clone();
    scope_register_tool_execution_intercept(
        &handle.uuid,
        "local_exec",
        5,
        Arc::new(move |_name, args, next| {
            let o = ol.clone();
            Box::pin(async move {
                o.lock().unwrap().push("local_before".into());
                let r = next(args).await;
                o.lock().unwrap().push("local_after".into());
                r.map(Into::into)
            })
        }),
    )
    .unwrap();

    let oo = order.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        oo.lock().unwrap().push("original".into());
        Box::pin(async move { Ok(args.into()) })
    });

    let _ = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    let recorded = order.lock().unwrap();
    assert_eq!(
        *recorded,
        vec![
            "local_before",
            "global_before",
            "original",
            "global_after",
            "local_after",
        ],
        "Scope-local at lower priority should wrap the global intercept"
    );

    // Cleanup
    deregister_tool_execution_intercept("global_exec").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();
}

// =========================================================================
// Error Propagation Tests
// =========================================================================

/// Conditional guardrail that rejects prevents request intercepts from running.
#[tokio::test]
async fn test_conditional_rejection_prevents_intercepts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let intercept_called = Arc::new(AtomicBool::new(false));

    // Register a conditional guardrail that rejects
    register_tool_conditional_execution_guardrail(
        "gate",
        1,
        Arc::new(|_name, _args| Box::pin(async { Ok(Some("blocked".to_string())) })),
    )
    .unwrap();

    // Register a request intercept -- should NOT run because conditional rejects first
    let ic = intercept_called.clone();
    register_tool_request_intercept(
        "should_not_run",
        1,
        false,
        Arc::new(move |_name, args| {
            ic.store(true, Ordering::SeqCst);
            ready(args)
        }),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));
    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .build(),
    )
    .await;

    assert!(result.is_err());
    // In the pipeline, conditional guardrails run *before* request intercepts
    assert!(
        !intercept_called.load(Ordering::SeqCst),
        "Request intercepts should not run when conditional guardrail rejects"
    );

    // Cleanup
    deregister_tool_conditional_execution_guardrail("gate").unwrap();
    deregister_tool_request_intercept("should_not_run").unwrap();
}

/// Conditional guardrail rejection prevents execution intercepts from running.
#[tokio::test]
async fn test_conditional_rejection_prevents_execution() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let exec_called = Arc::new(AtomicBool::new(false));

    register_tool_conditional_execution_guardrail(
        "gate2",
        1,
        Arc::new(|_name, _args| Box::pin(async { Ok(Some("no execution".to_string())) })),
    )
    .unwrap();

    let ec = exec_called.clone();
    register_tool_execution_intercept(
        "should_not_execute",
        1,
        Arc::new(move |_name, args, next| {
            ec.store(true, Ordering::SeqCst);
            Box::pin(async move { next(args).await.map(Into::into) })
        }),
    )
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let oc = original_called.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        oc.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(args.into()) })
    });

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({}))
            .func(func)
            .build(),
    )
    .await;

    assert!(result.is_err());
    assert!(
        !exec_called.load(Ordering::SeqCst),
        "Execution intercept should not run when conditional rejects"
    );
    assert!(
        !original_called.load(Ordering::SeqCst),
        "Original callable should not run when conditional rejects"
    );

    // Cleanup
    deregister_tool_conditional_execution_guardrail("gate2").unwrap();
    deregister_tool_execution_intercept("should_not_execute").unwrap();
}

// =========================================================================
// Sanitize Guardrail Chain Tests
// =========================================================================

/// Sanitize guardrails pipe data through sequentially.
#[tokio::test]
async fn test_sanitize_guardrails_pipe_data() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // First guardrail adds field_a
    register_tool_sanitize_request_guardrail(
        "add_a",
        1,
        Arc::new(|_name, mut args| {
            args.as_object_mut()
                .unwrap()
                .insert("field_a".into(), json!(true));
            ready(args)
        }),
    )
    .unwrap();

    // Second guardrail reads field_a and adds field_b
    register_tool_sanitize_request_guardrail(
        "add_b",
        2,
        Arc::new(|_name, mut args| {
            // Verify field_a was added by the previous guardrail
            let has_a = args.get("field_a").is_some();
            args.as_object_mut()
                .unwrap()
                .insert("field_b".into(), json!(has_a));
            ready(args)
        }),
    )
    .unwrap();

    // Capture the sanitized args via events
    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "pipe_observer",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let _tool = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let start = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Tool, ScopeCategory::Start))
        .unwrap();
    let input = start.input().unwrap();
    assert_eq!(input["field_a"], true, "First guardrail should add field_a");
    assert_eq!(
        input["field_b"], true,
        "Second guardrail should see field_a and add field_b=true"
    );

    // Cleanup
    deregister_tool_sanitize_request_guardrail("add_a").unwrap();
    deregister_tool_sanitize_request_guardrail("add_b").unwrap();
    deregister_subscriber("pipe_observer").unwrap();
}

/// Response sanitize guardrails also pipe through.
#[tokio::test]
async fn test_response_sanitize_guardrails_pipe() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_sanitize_response_guardrail(
        "resp_g1",
        1,
        Arc::new(|_name, mut result| {
            result
                .as_object_mut()
                .unwrap()
                .insert("sanitized".into(), json!(true));
            ready(result)
        }),
    )
    .unwrap();

    // Capture events
    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "resp_observer",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let tool_handle = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();

    tool_call_end(
        nemo_relay::api::tool::ToolCallEndParams::builder()
            .handle(&tool_handle)
            .execution_result(json!({"raw": true}).into())
            .build(),
    )
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let end = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Tool, ScopeCategory::End))
        .unwrap();
    let output = end.output().unwrap();
    assert_eq!(output["sanitized"], true);
    assert_eq!(output["raw"], true);

    // Cleanup
    deregister_tool_sanitize_response_guardrail("resp_g1").unwrap();
    deregister_subscriber("resp_observer").unwrap();
}

// =========================================================================
// Concurrent Mutations Tests
// =========================================================================

/// Use multiple threads to register/deregister guardrails concurrently.
/// Verify no panics or data races.
#[tokio::test]
async fn test_concurrent_register_deregister() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let barrier = Arc::new(std::sync::Barrier::new(8));

    let handles: Vec<_> = (0..8i32)
        .map(|i| {
            let b = barrier.clone();
            std::thread::spawn(move || {
                let name = format!("concurrent_guardrail_{i}");
                b.wait(); // synchronize thread start

                // Register
                let res = register_tool_sanitize_request_guardrail(
                    &name,
                    i,
                    Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
                );
                assert!(res.is_ok(), "Registration should succeed for {name}");

                // Brief pause to let other threads interleave
                std::thread::yield_now();

                // Deregister
                let res = deregister_tool_sanitize_request_guardrail(&name);
                assert!(res.is_ok());
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread should not panic");
    }

    for i in 0..10i32 {
        let name = format!("concurrent_guardrail_{i}");
        assert!(
            !deregister_tool_sanitize_request_guardrail(&name).unwrap(),
            "{name} should already be deregistered"
        );
    }
}

/// Concurrent register/deregister of intercepts across multiple threads.
#[tokio::test]
async fn test_concurrent_intercept_mutations() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let barrier = Arc::new(std::sync::Barrier::new(10));

    let handles: Vec<_> = (0..10i32)
        .map(|i| {
            let b = barrier.clone();
            std::thread::spawn(move || {
                let name = format!("concurrent_intercept_{i}");
                b.wait();

                let res = register_tool_request_intercept(
                    &name,
                    i,
                    false,
                    Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
                );
                assert!(res.is_ok());

                std::thread::yield_now();

                let res = deregister_tool_request_intercept(&name);
                assert!(res.is_ok());
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread should not panic");
    }

    for i in 0..10i32 {
        let name = format!("concurrent_intercept_{i}");
        assert!(
            !deregister_tool_request_intercept(&name).unwrap(),
            "{name} should already be deregistered"
        );
    }
}

/// Interleaved register and tool call execution from multiple threads.
#[tokio::test]
async fn test_concurrent_register_and_read() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Pre-register some guardrails
    for i in 0..4 {
        register_tool_sanitize_request_guardrail(
            &format!("stable_{i}"),
            i,
            Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
        )
        .unwrap();
    }

    let barrier = Arc::new(std::sync::Barrier::new(8));

    let handles: Vec<_> = (0..8i32)
        .map(|i| {
            let b = barrier.clone();
            std::thread::spawn(move || {
                b.wait();

                if i < 4 {
                    // Writer threads: register then deregister
                    let name = format!("dynamic_{i}");
                    let _ = register_tool_sanitize_request_guardrail(
                        &name,
                        100 + i,
                        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
                    );
                    std::thread::yield_now();
                    let _ = deregister_tool_sanitize_request_guardrail(&name);
                } else {
                    // Reader threads: set up scope stack and do tool calls
                    let stack = create_scope_stack();
                    set_thread_scope_stack(stack);
                    let _ = tool_call(
                        nemo_relay::api::tool::ToolCallParams::builder()
                            .name("tool")
                            .args(json!({}))
                            .build(),
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("Thread should not panic during concurrent read/write");
    }

    // Clean up stable guardrails
    for i in 0..4 {
        deregister_tool_sanitize_request_guardrail(&format!("stable_{i}")).unwrap();
    }
}

// =========================================================================
// Lock Regression Tests
// =========================================================================

#[tokio::test]
async fn test_tool_request_intercept_registry_mutations_apply_to_later_calls() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let callbacks = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let late_registered = Arc::new(AtomicBool::new(false));

    let tracked = callbacks.clone();
    let registered = late_registered.clone();
    register_tool_request_intercept(
        "snapshot_tool_request_initial",
        1,
        false,
        Arc::new(move |_, args| {
            record_middleware_callback(&tracked, "tool_request_initial");
            assert_middleware_callback_locks_are_free();

            if !registered.swap(true, Ordering::SeqCst) {
                let tracked = tracked.clone();
                register_tool_request_intercept(
                    "snapshot_tool_request_late",
                    2,
                    false,
                    Arc::new(move |_, args| {
                        record_middleware_callback(&tracked, "tool_request_late");
                        assert_middleware_callback_locks_are_free();
                        ready(args)
                    }),
                )
                .unwrap();
            }

            ready(args)
        }),
    )
    .unwrap();

    let args = tool_request_intercepts("tool", json!({"round": 1}))
        .await
        .unwrap();
    assert_eq!(args["round"], 1);
    assert_middleware_callback_labels(&callbacks, &["tool_request_initial"]);

    callbacks.lock().unwrap().clear();
    let args = tool_request_intercepts("tool", json!({"round": 2}))
        .await
        .unwrap();
    assert_eq!(args["round"], 2);
    assert_middleware_callback_labels(&callbacks, &["tool_request_initial", "tool_request_late"]);

    deregister_tool_request_intercept("snapshot_tool_request_initial").unwrap();
    deregister_tool_request_intercept("snapshot_tool_request_late").unwrap();
}

#[tokio::test]
async fn test_llm_request_intercept_registry_mutations_apply_to_later_calls() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let callbacks = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let late_registered = Arc::new(AtomicBool::new(false));

    let tracked = callbacks.clone();
    let registered = late_registered.clone();
    register_llm_request_intercept(
        "snapshot_llm_request_initial",
        1,
        false,
        Arc::new(move |_, request, annotated| {
            record_middleware_callback(&tracked, "llm_request_initial");
            assert_middleware_callback_locks_are_free();

            if !registered.swap(true, Ordering::SeqCst) {
                let tracked = tracked.clone();
                register_llm_request_intercept(
                    "snapshot_llm_request_late",
                    2,
                    false,
                    Arc::new(move |_, request, annotated| {
                        record_middleware_callback(&tracked, "llm_request_late");
                        assert_middleware_callback_locks_are_free();
                        ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                            request, annotated,
                        ))
                    }),
                )
                .unwrap();
            }

            ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                request, annotated,
            ))
        }),
    )
    .unwrap();
    let request = llm_request_intercepts(
        "llm",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"round": 1}),
        },
    )
    .await
    .unwrap();
    assert_eq!(request.request.content["round"], 1);
    assert_middleware_callback_labels(&callbacks, &["llm_request_initial"]);

    callbacks.lock().unwrap().clear();
    let request = llm_request_intercepts(
        "llm",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"round": 2}),
        },
    )
    .await
    .unwrap();
    assert_eq!(request.request.content["round"], 2);
    assert_middleware_callback_labels(&callbacks, &["llm_request_initial", "llm_request_late"]);

    deregister_llm_request_intercept("snapshot_llm_request_initial").unwrap();
    deregister_llm_request_intercept("snapshot_llm_request_late").unwrap();
}

#[tokio::test]
async fn test_tool_middleware_callbacks_run_without_registry_or_scope_locks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let scope = setup_isolated_scope("tool_lock_regression");
    let callbacks = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let tracked = callbacks.clone();
    register_tool_conditional_execution_guardrail(
        "lock_global_tool_conditional",
        1,
        Arc::new(move |_, _| {
            record_middleware_callback(&tracked, "tool_conditional_global");
            assert_middleware_callback_locks_are_free();
            ready(None)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_tool_conditional_execution_guardrail(
        &scope.uuid,
        "lock_scope_tool_conditional",
        2,
        Arc::new(move |_, _| {
            record_middleware_callback(&tracked, "tool_conditional_scope");
            assert_middleware_callback_locks_are_free();
            ready(None)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_tool_request_intercept(
        "lock_global_tool_request",
        1,
        false,
        Arc::new(move |_, args| {
            record_middleware_callback(&tracked, "tool_request_global");
            assert_middleware_callback_locks_are_free();
            ready(args)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_tool_request_intercept(
        &scope.uuid,
        "lock_scope_tool_request",
        2,
        false,
        Arc::new(move |_, args| {
            record_middleware_callback(&tracked, "tool_request_scope");
            assert_middleware_callback_locks_are_free();
            ready(args)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_tool_sanitize_request_guardrail(
        "lock_global_tool_sanitize_request",
        1,
        Arc::new(move |_, args| {
            record_middleware_callback(&tracked, "tool_sanitize_request_global");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(args)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_tool_sanitize_request_guardrail(
        &scope.uuid,
        "lock_scope_tool_sanitize_request",
        2,
        Arc::new(move |_, args| {
            record_middleware_callback(&tracked, "tool_sanitize_request_scope");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(args)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_tool_execution_intercept(
        "lock_global_tool_execution",
        1,
        Arc::new(move |_, args, next| {
            record_middleware_callback(&tracked, "tool_execution_global");
            assert_middleware_callback_locks_are_free();
            Box::pin(async move { next(args).await.map(Into::into) })
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_tool_execution_intercept(
        &scope.uuid,
        "lock_scope_tool_execution",
        2,
        Arc::new(move |_, args, next| {
            record_middleware_callback(&tracked, "tool_execution_scope");
            assert_middleware_callback_locks_are_free();
            Box::pin(async move { next(args).await.map(Into::into) })
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_tool_sanitize_response_guardrail(
        "lock_global_tool_sanitize_response",
        1,
        Arc::new(move |_, result| {
            record_middleware_callback(&tracked, "tool_sanitize_response_global");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(result)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_tool_sanitize_response_guardrail(
        &scope.uuid,
        "lock_scope_tool_sanitize_response",
        2,
        Arc::new(move |_, result| {
            record_middleware_callback(&tracked, "tool_sanitize_response_scope");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(result)
        }),
    )
    .unwrap();

    let tracked = callbacks.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        record_middleware_callback(&tracked, "tool_func");
        assert_middleware_callback_locks_are_free();
        Box::pin(async move { Ok(args.into()) })
    });
    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"ok": true}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result.result["ok"], true);
    flush_subscribers().unwrap();
    assert_middleware_callback_labels(
        &callbacks,
        &[
            "tool_conditional_global",
            "tool_conditional_scope",
            "tool_request_global",
            "tool_request_scope",
            "tool_sanitize_request_global",
            "tool_sanitize_request_scope",
            "tool_execution_global",
            "tool_execution_scope",
            "tool_func",
            "tool_sanitize_response_global",
            "tool_sanitize_response_scope",
        ],
    );

    deregister_tool_conditional_execution_guardrail("lock_global_tool_conditional").unwrap();
    deregister_tool_request_intercept("lock_global_tool_request").unwrap();
    deregister_tool_sanitize_request_guardrail("lock_global_tool_sanitize_request").unwrap();
    deregister_tool_execution_intercept("lock_global_tool_execution").unwrap();
    deregister_tool_sanitize_response_guardrail("lock_global_tool_sanitize_response").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope.uuid)
            .build(),
    )
    .unwrap();
}

#[tokio::test]
async fn managed_llm_injects_runtime_owned_traceparent() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();
    let captured = Arc::new(Mutex::new(None::<LlmRequest>));
    let captured_request = captured.clone();
    let request = LlmRequest {
        headers: serde_json::Map::from_iter([
            ("TraceParent".to_string(), json!("user-value")),
            ("TRACEPARENT".to_string(), json!("duplicate")),
        ]),
        content: json!({"prompt": "hello"}),
    };
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("traceparent-test")
            .request(request)
            .func(Arc::new(move |request| {
                *captured_request.lock().unwrap() = Some(request);
                Box::pin(async { Ok(json!({"ok": true})) })
            }))
            .build(),
    )
    .await
    .unwrap();
    let request = captured.lock().unwrap().take().unwrap();
    assert_eq!(request.headers.len(), 1);
    let traceparent = request
        .headers
        .get("traceparent")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(traceparent.starts_with("00-"));
    assert!(traceparent.ends_with("-01"));
    assert_eq!(traceparent.len(), 55);
}

#[tokio::test]
async fn test_llm_middleware_callbacks_run_without_registry_or_scope_locks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let scope = setup_isolated_scope("llm_lock_regression");
    let callbacks = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let tracked = callbacks.clone();
    register_llm_conditional_execution_guardrail(
        "lock_global_llm_conditional",
        1,
        Arc::new(move |_| {
            record_middleware_callback(&tracked, "llm_conditional_global");
            assert_middleware_callback_locks_are_free();
            ready(None)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_conditional_execution_guardrail(
        &scope.uuid,
        "lock_scope_llm_conditional",
        2,
        Arc::new(move |_| {
            record_middleware_callback(&tracked, "llm_conditional_scope");
            assert_middleware_callback_locks_are_free();
            ready(None)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_llm_request_intercept(
        "lock_global_llm_request",
        1,
        false,
        Arc::new(move |_, request, annotated| {
            record_middleware_callback(&tracked, "llm_request_global");
            assert_middleware_callback_locks_are_free();
            ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                request, annotated,
            ))
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_request_intercept(
        &scope.uuid,
        "lock_scope_llm_request",
        2,
        false,
        Arc::new(move |_, request, annotated| {
            record_middleware_callback(&tracked, "llm_request_scope");
            assert_middleware_callback_locks_are_free();
            ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                request, annotated,
            ))
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_llm_sanitize_request_guardrail(
        "lock_global_llm_sanitize_request",
        1,
        Arc::new(move |request, _context| {
            record_middleware_callback(&tracked, "llm_sanitize_request_global");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(Some(request))
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_sanitize_request_guardrail(
        &scope.uuid,
        "lock_scope_llm_sanitize_request",
        2,
        Arc::new(move |request, _context| {
            record_middleware_callback(&tracked, "llm_sanitize_request_scope");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(Some(request))
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_llm_execution_intercept(
        "lock_global_llm_execution",
        1,
        Arc::new(move |_, request, next| {
            record_middleware_callback(&tracked, "llm_execution_global");
            assert_middleware_callback_locks_are_free();
            Box::pin(async move { next(request).await })
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_execution_intercept(
        &scope.uuid,
        "lock_scope_llm_execution",
        2,
        Arc::new(move |_, request, next| {
            record_middleware_callback(&tracked, "llm_execution_scope");
            assert_middleware_callback_locks_are_free();
            Box::pin(async move { next(request).await })
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_llm_stream_execution_intercept(
        "lock_global_llm_stream_execution",
        1,
        Arc::new(move |_, request, next| {
            record_middleware_callback(&tracked, "llm_stream_execution_global");
            assert_middleware_callback_locks_are_free();
            Box::pin(async move { next(request).await })
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_stream_execution_intercept(
        &scope.uuid,
        "lock_scope_llm_stream_execution",
        2,
        Arc::new(move |_, request, next| {
            record_middleware_callback(&tracked, "llm_stream_execution_scope");
            assert_middleware_callback_locks_are_free();
            Box::pin(async move { next(request).await })
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_llm_sanitize_response_guardrail(
        "lock_global_llm_sanitize_response",
        1,
        Arc::new(move |response, _context| {
            record_middleware_callback(&tracked, "llm_sanitize_response_global");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(Some(response))
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_sanitize_response_guardrail(
        &scope.uuid,
        "lock_scope_llm_sanitize_response",
        2,
        Arc::new(move |response, _context| {
            record_middleware_callback(&tracked, "llm_sanitize_response_scope");
            assert_queued_sanitizer_scope_lock_is_free();
            ready(Some(response))
        }),
    )
    .unwrap();

    let tracked = callbacks.clone();
    let func: LlmExecutionNextFn = Arc::new(move |_| {
        record_middleware_callback(&tracked, "llm_func");
        assert_middleware_callback_locks_are_free();
        Box::pin(async move { Ok(json!({"ok": true})) })
    });
    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": []}),
            })
            .func(func)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(response["ok"], true);

    let tracked = callbacks.clone();
    let stream_func: LlmStreamExecutionNextFn = Arc::new(move |_| {
        record_middleware_callback(&tracked, "llm_stream_func");
        assert_middleware_callback_locks_are_free();
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": true}))]);
            Ok(LlmJsonStream::new(stream))
        })
    });
    let tracked = callbacks.clone();
    let collector = Box::new(move |_| {
        record_middleware_callback(&tracked, "llm_collector");
        assert_middleware_callback_locks_are_free();
        Ok(())
    });
    let tracked = callbacks.clone();
    let finalizer = Box::new(move || {
        record_middleware_callback(&tracked, "llm_finalizer");
        assert_middleware_callback_locks_are_free();
        json!({"stream": true})
    });
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("llm-stream")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": []}),
            })
            .func(stream_func)
            .collector(collector)
            .finalizer(finalizer)
            .build(),
    )
    .await
    .unwrap();
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
    stream.close().await.unwrap();
    flush_subscribers().unwrap();
    assert_middleware_callback_labels(
        &callbacks,
        &[
            "llm_conditional_global",
            "llm_conditional_global",
            "llm_conditional_scope",
            "llm_conditional_scope",
            "llm_request_global",
            "llm_request_global",
            "llm_request_scope",
            "llm_request_scope",
            "llm_sanitize_request_global",
            "llm_sanitize_request_global",
            "llm_sanitize_request_scope",
            "llm_sanitize_request_scope",
            "llm_execution_global",
            "llm_execution_scope",
            "llm_func",
            "llm_stream_execution_global",
            "llm_stream_execution_scope",
            "llm_stream_func",
            "llm_collector",
            "llm_finalizer",
            "llm_sanitize_response_global",
            "llm_sanitize_response_global",
            "llm_sanitize_response_scope",
            "llm_sanitize_response_scope",
        ],
    );

    deregister_llm_conditional_execution_guardrail("lock_global_llm_conditional").unwrap();
    deregister_llm_request_intercept("lock_global_llm_request").unwrap();
    deregister_llm_sanitize_request_guardrail("lock_global_llm_sanitize_request").unwrap();
    deregister_llm_execution_intercept("lock_global_llm_execution").unwrap();
    deregister_llm_stream_execution_intercept("lock_global_llm_stream_execution").unwrap();
    deregister_llm_sanitize_response_guardrail("lock_global_llm_sanitize_response").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope.uuid)
            .build(),
    )
    .unwrap();
}

// =========================================================================
// Full Pipeline Integration Test
// =========================================================================

/// End-to-end test: request intercepts, sanitize guardrails, conditional
/// guardrails, execution intercepts, sanitize response
/// guardrails -- all in one tool_call_execute call.
#[tokio::test]
async fn test_full_pipeline_integration() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    // Request intercept
    let o1 = order.clone();
    register_tool_request_intercept(
        "req_intercept",
        1,
        false,
        Arc::new(move |_name, mut args| {
            o1.lock().unwrap().push("request_intercept".into());
            args.as_object_mut()
                .unwrap()
                .insert("intercepted".into(), json!(true));
            ready(args)
        }),
    )
    .unwrap();

    // Sanitize request guardrail
    let o2 = order.clone();
    register_tool_sanitize_request_guardrail(
        "sanitize_req",
        1,
        Arc::new(move |_name, args| {
            o2.lock().unwrap().push("sanitize_request".into());
            ready(args)
        }),
    )
    .unwrap();

    // Conditional guardrail (allows)
    let o3 = order.clone();
    register_tool_conditional_execution_guardrail(
        "conditional",
        1,
        Arc::new(move |_name, _args| {
            o3.lock().unwrap().push("conditional".into());
            ready(None) // Allow
        }),
    )
    .unwrap();

    // Execution intercept
    let o4 = order.clone();
    register_tool_execution_intercept(
        "exec_intercept",
        1,
        Arc::new(move |_name, args, next| {
            let o = o4.clone();
            Box::pin(async move {
                o.lock().unwrap().push("execution_intercept".into());
                next(args).await.map(Into::into)
            })
        }),
    )
    .unwrap();

    // Sanitize response guardrail
    let o5 = order.clone();
    register_tool_sanitize_response_guardrail(
        "sanitize_resp",
        1,
        Arc::new(move |_name, result| {
            o5.lock().unwrap().push("sanitize_response".into());
            ready(result)
        }),
    )
    .unwrap();

    let o_orig = order.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        o_orig.lock().unwrap().push("original_execution".into());
        Box::pin(async move { Ok(args.into()) })
    });

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"data": "test"}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    flush_subscribers().unwrap();

    // Application middleware remains ordered on the managed path. Payload
    // sanitizers run on the publication path, where request still precedes
    // response but may race with tool execution.
    let recorded = order.lock().unwrap();
    let index = |name: &str| recorded.iter().position(|entry| entry == name).unwrap();
    assert!(index("conditional") < index("request_intercept"));
    assert!(index("request_intercept") < index("execution_intercept"));
    assert!(index("execution_intercept") < index("original_execution"));
    assert!(index("request_intercept") < index("sanitize_request"));
    assert!(index("sanitize_request") < index("sanitize_response"));
    assert!(index("original_execution") < index("sanitize_response"));

    // Verify the request intercept's modification persists through the pipeline
    assert_eq!(result.result["intercepted"], true);
    assert_eq!(result.result["data"], "test");

    // Cleanup
    deregister_tool_request_intercept("req_intercept").unwrap();
    deregister_tool_sanitize_request_guardrail("sanitize_req").unwrap();
    deregister_tool_conditional_execution_guardrail("conditional").unwrap();
    deregister_tool_execution_intercept("exec_intercept").unwrap();
    deregister_tool_sanitize_response_guardrail("sanitize_resp").unwrap();
}

// =========================================================================
// Duplicate Registration Tests
// =========================================================================

/// Attempting to register a guardrail with the same name returns AlreadyExists.
#[tokio::test]
async fn test_duplicate_guardrail_registration_returns_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    register_tool_sanitize_request_guardrail(
        "duplicate",
        1,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();

    let err = register_tool_sanitize_request_guardrail(
        "duplicate",
        2,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    );

    assert!(err.is_err());
    match err.unwrap_err() {
        FlowError::AlreadyExists(msg) => {
            assert!(msg.contains("duplicate"));
        }
        other => panic!("Expected AlreadyExists, got: {:?}", other),
    }

    // Cleanup
    deregister_tool_sanitize_request_guardrail("duplicate").unwrap();
}

/// Attempting to register an intercept with the same name returns AlreadyExists.
#[tokio::test]
async fn test_duplicate_intercept_registration_returns_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    register_tool_request_intercept(
        "dup_intercept",
        1,
        false,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();

    let err = register_tool_request_intercept(
        "dup_intercept",
        2,
        false,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    );

    assert!(err.is_err());
    match err.unwrap_err() {
        FlowError::AlreadyExists(msg) => {
            assert!(msg.contains("dup_intercept"));
        }
        other => panic!("Expected AlreadyExists, got: {:?}", other),
    }

    // Cleanup
    deregister_tool_request_intercept("dup_intercept").unwrap();
}

// =========================================================================
// Deregistration Tests
// =========================================================================

/// Deregistering a non-existent guardrail returns false.
#[tokio::test]
async fn test_deregister_nonexistent_returns_false() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let result = deregister_tool_sanitize_request_guardrail("nonexistent").unwrap();
    assert!(
        !result,
        "Deregistering non-existent entry should return false"
    );
}

/// Deregistering removes the guardrail from the chain.
#[tokio::test]
async fn test_deregister_removes_from_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let call_count = Arc::new(AtomicU32::new(0));

    let cc = call_count.clone();
    register_tool_sanitize_request_guardrail(
        "removable",
        1,
        Arc::new(move |_name, args| {
            cc.fetch_add(1, Ordering::SeqCst);
            ready(args)
        }),
    )
    .unwrap();

    // First call -- guardrail runs
    let _ = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Deregister
    let removed = deregister_tool_sanitize_request_guardrail("removable").unwrap();
    assert!(removed, "Should return true for existing entry");

    // Second call -- guardrail should NOT run
    let _ = tool_call(
        nemo_relay::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "Guardrail should not run after deregistration"
    );
}

// =========================================================================
// LLM Middleware Chain Tests
// =========================================================================

/// LLM conditional guardrail rejection returns GuardrailRejected.
#[tokio::test]
async fn test_llm_conditional_guardrail_rejects() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_llm_conditional_execution_guardrail(
        "llm_gate",
        1,
        Arc::new(|_req| ready(Some("LLM call rejected".to_string()))),
    )
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(|_req| Box::pin(async move { Ok(json!({"response": "ok"})) }));

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(func)
            .build(),
    )
    .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FlowError::GuardrailRejected(reason) => {
            assert!(reason.contains("LLM call rejected"));
        }
        other => panic!("Expected GuardrailRejected, got: {:?}", other),
    }

    // Cleanup
    deregister_llm_conditional_execution_guardrail("llm_gate").unwrap();
}

/// Conditional LLM guardrails emit Guardrail scope start/end pairs for allow
/// and reject decisions.
#[tokio::test]
async fn test_llm_conditional_guardrail_emits_guardrail_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "llm_guardrail_scope_capture",
        Arc::new(move |event| {
            captured.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();

    register_llm_conditional_execution_guardrail("llm_scope_allow", 1, Arc::new(|_| ready(None)))
        .unwrap();
    register_llm_conditional_execution_guardrail(
        "llm_scope_reject",
        2,
        Arc::new(|_| ready(Some("blocked by llm guardrail".to_string()))),
    )
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(|_req| Box::pin(async move { Ok(json!({"response": "ok"})) }));
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let rejected = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request.clone())
            .func(func.clone())
            .build(),
    )
    .await;
    assert!(rejected.is_err());

    deregister_llm_conditional_execution_guardrail("llm_scope_reject").unwrap();
    let allowed = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(func)
            .build(),
    )
    .await;
    assert!(allowed.is_ok());

    deregister_llm_conditional_execution_guardrail("llm_scope_allow").unwrap();
    deregister_subscriber("llm_guardrail_scope_capture").unwrap();

    let events = captured_events_snapshot(&events);
    let guardrail_events = events
        .iter()
        .filter(|event| event.scope_type() == Some(ScopeType::Guardrail))
        .collect::<Vec<_>>();
    assert_eq!(
        guardrail_events
            .iter()
            .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
            .count(),
        3
    );
    assert_eq!(
        guardrail_events
            .iter()
            .filter(|event| event.scope_category() == Some(ScopeCategory::End))
            .count(),
        3
    );
    assert!(guardrail_events.iter().all(|event| {
        event.scope_category() != Some(ScopeCategory::Start)
            || event.data().and_then(|data| data.get("input")).is_none()
    }));
    assert!(guardrail_events.iter().any(|event| {
        event.name() == "llm_scope_allow"
            && event.scope_category() == Some(ScopeCategory::End)
            && event
                .data()
                .and_then(|data| data.get("allowed"))
                .and_then(|value| value.as_bool())
                == Some(true)
    }));
    assert!(guardrail_events.iter().any(|event| {
        event.name() == "llm_scope_reject"
            && event.scope_category() == Some(ScopeCategory::End)
            && event
                .data()
                .and_then(|data| data.get("rejection_reason"))
                .and_then(|value| value.as_str())
                == Some("blocked by llm guardrail")
    }));
}

/// LLM request intercept transforms the request.
#[tokio::test]
async fn test_llm_request_intercept_transforms() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_llm_request_intercept(
        "llm_req_i",
        1,
        false,
        Arc::new(|_name: String, mut req: LlmRequest, annotated| {
            req.headers.insert("x-intercepted".into(), json!(true));
            ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
        }),
    )
    .unwrap();

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let result = llm_request_intercepts("test_llm", request).await.unwrap();
    assert_eq!(result.request.headers["x-intercepted"], true);

    // Cleanup
    deregister_llm_request_intercept("llm_req_i").unwrap();
}

#[tokio::test]
async fn test_llm_request_intercept_pending_marks_preserve_order_and_break_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    for (name, priority, break_chain, mark_name) in [
        ("pending_first", 1, false, "first"),
        ("pending_break", 2, true, "second"),
        ("pending_skipped", 3, false, "skipped"),
    ] {
        register_llm_request_intercept(
            name,
            priority,
            break_chain,
            Arc::new(move |_name, request, annotated| {
                ready(
                    LlmRequestInterceptOutcome::new(request, annotated)
                        .with_pending_mark(PendingMarkSpec::builder().name(mark_name).build()),
                )
            }),
        )
        .unwrap();
    }

    let outcome = llm_request_intercepts(
        "llm",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"prompt": "hello"}),
        },
    )
    .await
    .unwrap();

    assert_eq!(
        outcome
            .pending_marks
            .iter()
            .map(|mark| mark.name.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(outcome.request.content["prompt"], "hello");

    for name in ["pending_first", "pending_break", "pending_skipped"] {
        deregister_llm_request_intercept(name).unwrap();
    }
}

#[tokio::test]
async fn test_managed_llm_event_sanitizers_run_off_execution_path_in_fifo_order() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let sanitizer_started = Arc::new(tokio::sync::Notify::new());
    let sanitizer_release = Arc::new(tokio::sync::Notify::new());
    register_scope_sanitize_start_guardrail(
        "managed_async_publication_sanitizer",
        1,
        Arc::new({
            let sanitizer_started = Arc::clone(&sanitizer_started);
            let sanitizer_release = Arc::clone(&sanitizer_release);
            move |_event, fields| {
                let sanitizer_started = Arc::clone(&sanitizer_started);
                let sanitizer_release = Arc::clone(&sanitizer_release);
                Box::pin(async move {
                    sanitizer_started.notify_one();
                    sanitizer_release.notified().await;
                    Ok(fields)
                })
            }
        }),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "managed_async_publication_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let call = tokio::spawn(async {
        llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("managed-async-publication")
                .request(LlmRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"prompt": "hello"}),
                })
                .func(Arc::new(|_| {
                    Box::pin(async { Ok(json!({"response": "done"})) })
                }))
                .build(),
        )
        .await
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        sanitizer_started.notified(),
    )
    .await
    .expect("managed start sanitizer did not run on the dispatcher");
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), call)
        .await
        .expect("event sanitizer blocked managed provider execution")
        .expect("managed call task should join")
        .expect("managed call should succeed");
    assert_eq!(result, json!({"response": "done"}));

    sanitizer_release.notify_one();
    flush_subscribers().unwrap();
    let events = events.lock().unwrap();
    let lifecycle = events
        .iter()
        .filter(|event| event.name() == "managed-async-publication")
        .map(|event| event.scope_category())
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        [Some(ScopeCategory::Start), Some(ScopeCategory::End),]
    );
    drop(events);

    deregister_scope_sanitize_start_guardrail("managed_async_publication_sanitizer").unwrap();
    deregister_subscriber("managed_async_publication_observer").unwrap();
}

#[tokio::test]
async fn test_managed_llm_payload_sanitizers_are_queued_off_execution_path() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let request_started = Arc::new(tokio::sync::Notify::new());
    let request_release = Arc::new(tokio::sync::Notify::new());
    register_llm_sanitize_request_guardrail(
        "managed_queued_llm_request",
        1,
        Arc::new({
            let request_started = Arc::clone(&request_started);
            let request_release = Arc::clone(&request_release);
            move |request, _context| {
                let request_started = Arc::clone(&request_started);
                let request_release = Arc::clone(&request_release);
                Box::pin(async move {
                    request_started.notify_one();
                    request_release.notified().await;
                    let mut request = request;
                    request.content["sanitized"] = json!(true);
                    Ok(Some(request))
                })
            }
        }),
    )
    .unwrap();
    let response_started = Arc::new(tokio::sync::Notify::new());
    let response_release = Arc::new(tokio::sync::Notify::new());
    register_llm_sanitize_response_guardrail(
        "managed_queued_llm_response",
        1,
        Arc::new({
            let response_started = Arc::clone(&response_started);
            let response_release = Arc::clone(&response_release);
            move |response, _context| {
                let response_started = Arc::clone(&response_started);
                let response_release = Arc::clone(&response_release);
                Box::pin(async move {
                    response_started.notify_one();
                    response_release.notified().await;
                    let mut response = response;
                    response["sanitized"] = json!(true);
                    Ok(Some(response))
                })
            }
        }),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    register_subscriber(
        "managed_queued_llm_observer",
        Arc::new({
            let events = Arc::clone(&events);
            move |event| events.lock().unwrap().push(event.clone())
        }),
    )
    .unwrap();

    let call = tokio::spawn(async {
        llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("managed-queued-llm")
                .request(LlmRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"prompt": "hello"}),
                })
                .func(Arc::new(|_| {
                    Box::pin(async { Ok(json!({"response": "done"})) })
                }))
                .build(),
        )
        .await
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        request_started.notified(),
    )
    .await
    .expect("managed request sanitizer did not start");
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), call)
        .await
        .expect("request sanitizer blocked managed provider execution")
        .expect("managed call task should join")
        .expect("managed call should succeed");
    let completed_at = Utc::now();
    assert_eq!(result, json!({"response": "done"}));

    request_release.notify_one();
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        response_started.notified(),
    )
    .await
    .expect("managed response sanitizer did not start");
    assert_flush_waits_for_pending_completion(|| response_release.notify_one());
    let events = events.lock().unwrap();
    let start = events
        .iter()
        .find(|event| {
            event.name() == "managed-queued-llm"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .expect("managed LLM START event should be published after flush");
    assert_eq!(start.input().unwrap()["content"]["sanitized"], true);
    let end = events
        .iter()
        .find(|event| {
            event.name() == "managed-queued-llm"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .expect("managed LLM END event should be published after flush");
    assert_eq!(end.data().unwrap()["sanitized"], true);
    assert!(*end.timestamp() <= completed_at);
    drop(events);
    deregister_llm_sanitize_request_guardrail("managed_queued_llm_request").unwrap();
    deregister_llm_sanitize_response_guardrail("managed_queued_llm_response").unwrap();
    deregister_subscriber("managed_queued_llm_observer").unwrap();
}

#[tokio::test]
async fn test_managed_tool_payload_sanitizers_are_queued_off_execution_path() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let request_started = Arc::new(tokio::sync::Notify::new());
    let request_release = Arc::new(tokio::sync::Notify::new());
    register_tool_sanitize_request_guardrail(
        "managed_queued_tool_request",
        1,
        Arc::new({
            let request_started = Arc::clone(&request_started);
            let request_release = Arc::clone(&request_release);
            move |_name, args| {
                let request_started = Arc::clone(&request_started);
                let request_release = Arc::clone(&request_release);
                Box::pin(async move {
                    request_started.notify_one();
                    request_release.notified().await;
                    Ok(args)
                })
            }
        }),
    )
    .unwrap();
    let response_started = Arc::new(tokio::sync::Notify::new());
    let response_release = Arc::new(tokio::sync::Notify::new());
    register_tool_sanitize_response_guardrail(
        "managed_queued_tool_response",
        1,
        Arc::new({
            let response_started = Arc::clone(&response_started);
            let response_release = Arc::clone(&response_release);
            move |_name, mut result| {
                let response_started = Arc::clone(&response_started);
                let response_release = Arc::clone(&response_release);
                Box::pin(async move {
                    response_started.notify_one();
                    response_release.notified().await;
                    result["sanitized"] = json!(true);
                    Ok(result)
                })
            }
        }),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    register_subscriber(
        "managed_queued_tool_observer",
        Arc::new({
            let events = Arc::clone(&events);
            move |event| events.lock().unwrap().push(event.clone())
        }),
    )
    .unwrap();

    let call = tokio::spawn(async {
        tool_call_execute(
            nemo_relay::api::tool::ToolCallExecuteParams::builder()
                .name("managed-queued-tool")
                .args(json!({"input": true}))
                .func(Arc::new(|args| {
                    Box::pin(async move {
                        Ok(ToolExecutionResult::annotated(
                            args,
                            json!({"opaque": "caller-visible"}),
                        ))
                    })
                }))
                .build(),
        )
        .await
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        request_started.notified(),
    )
    .await
    .expect("managed tool request sanitizer did not start");
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), call)
        .await
        .expect("request sanitizer blocked managed tool execution")
        .expect("managed tool task should join")
        .expect("managed tool call should succeed");
    assert_eq!(result.result, json!({"input": true}));
    assert_eq!(result.annotation, Some(json!({"opaque": "caller-visible"})));

    request_release.notify_one();
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        response_started.notified(),
    )
    .await
    .expect("managed tool response sanitizer did not start");
    assert_flush_waits_for_pending_completion(|| response_release.notify_one());
    let events = events.lock().unwrap();
    let end = events
        .iter()
        .find(|event| {
            event.name() == "managed-queued-tool"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .expect("managed tool END event should be published after flush");
    assert_eq!(end.data().unwrap()["sanitized"], true);
    assert_eq!(
        end.tool_result_annotation().unwrap(),
        json!({"opaque": "caller-visible"})
    );
    drop(events);
    deregister_tool_sanitize_request_guardrail("managed_queued_tool_request").unwrap();
    deregister_tool_sanitize_response_guardrail("managed_queued_tool_response").unwrap();
    deregister_subscriber("managed_queued_tool_observer").unwrap();
}

#[tokio::test]
async fn test_stream_termination_does_not_await_response_sanitization() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let sanitizer_started = Arc::new(tokio::sync::Notify::new());
    let sanitizer_release = Arc::new(tokio::sync::Notify::new());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    register_llm_sanitize_response_guardrail(
        "managed_queued_stream_response",
        1,
        Arc::new({
            let sanitizer_started = Arc::clone(&sanitizer_started);
            let sanitizer_release = Arc::clone(&sanitizer_release);
            move |response, _context| {
                let sanitizer_started = Arc::clone(&sanitizer_started);
                let sanitizer_release = Arc::clone(&sanitizer_release);
                Box::pin(async move {
                    sanitizer_started.notify_one();
                    sanitizer_release.notified().await;
                    Ok(Some(response))
                })
            }
        }),
    )
    .unwrap();
    register_subscriber(
        "managed_queued_stream_observer",
        Arc::new({
            let events = Arc::clone(&events);
            move |event| events.lock().unwrap().push(event.clone())
        }),
    )
    .unwrap();

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("managed-queued-stream")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(|_| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                        "chunk": "done"
                    }))])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"response": "done"})))
            .build(),
    )
    .await
    .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
    })
    .await
    .expect("response sanitizer blocked stream termination");
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        sanitizer_started.notified(),
    )
    .await
    .expect("stream response sanitizer did not start");
    let terminal_timestamp = Utc::now();

    assert_flush_waits_for_pending_completion(|| sanitizer_release.notify_one());
    let end = events
        .lock()
        .unwrap()
        .iter()
        .find(|event| {
            event.name() == "managed-queued-stream"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .cloned()
        .expect("stream END event should be published after flush");
    assert!(*end.timestamp() <= terminal_timestamp);

    deregister_llm_sanitize_response_guardrail("managed_queued_stream_response").unwrap();
    deregister_subscriber("managed_queued_stream_observer").unwrap();
}

#[tokio::test]
async fn test_stream_response_sanitizer_nested_mark_precedes_end_event() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "stream_nested_publication_observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "stream_nested_publication_sanitizer",
        1,
        Arc::new(|response, _context| {
            Box::pin(async move {
                tokio::task::yield_now().await;
                event(
                    EmitMarkEventParams::builder()
                        .name("stream-sanitizer-nested-mark")
                        .build(),
                )
                .unwrap();
                Ok(Some(response))
            })
        }),
    )
    .unwrap();

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("stream-nested-publication")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(|_| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                        "chunk": "done"
                    }))])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"response": "done"})))
            .build(),
    )
    .await
    .unwrap();
    while let Some(item) = stream.next().await {
        item.unwrap();
    }
    stream.close().await.unwrap();
    flush_subscribers().unwrap();

    let names = events
        .lock()
        .unwrap()
        .iter()
        .map(|event| {
            (
                event.name().to_string(),
                event.scope_category(),
                event.parent_uuid(),
            )
        })
        .collect::<Vec<_>>();
    let mark_index = names
        .iter()
        .position(|(name, _, _)| name == "stream-sanitizer-nested-mark")
        .unwrap();
    let end_index = names
        .iter()
        .position(|(name, category, _)| {
            name == "stream-nested-publication" && *category == Some(ScopeCategory::End)
        })
        .unwrap();
    assert!(mark_index < end_index);

    deregister_llm_sanitize_response_guardrail("stream_nested_publication_sanitizer").unwrap();
    deregister_subscriber("stream_nested_publication_observer").unwrap();
}

#[tokio::test]
async fn test_managed_llm_emits_pending_marks_under_started_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "pending_mark_observer",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_mark_sanitize_guardrail(
        "llm_pending_mark_sanitizer",
        1,
        Arc::new(|event, mut fields| {
            let mut metadata = fields.metadata.unwrap_or_else(|| json!({}));
            metadata["sanitized_mark"] = json!(event.name());
            fields.metadata = Some(metadata);
            ready(fields)
        }),
    )
    .unwrap();
    register_llm_request_intercept(
        "pending_managed",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            ready(
                LlmRequestInterceptOutcome::new(request, annotated)
                    .with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("request.optimized.invalid")
                            .metadata(json!("not an object"))
                            .severity(LogSeverity::Info)
                            .build(),
                    )
                    .with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("request.optimized")
                            .category(EventCategory::custom())
                            .category_profile(
                                CategoryProfile::builder()
                                    .subtype("optimizer.saved_tokens")
                                    .build(),
                            )
                            .data(json!({"saved_tokens": 12}))
                            .data_schema(
                                DataSchema::builder()
                                    .name("example.llm_pending_mark")
                                    .version("1")
                                    .build(),
                            )
                            .metadata(json!({(LOG_SEVERITY_METADATA_KEY): "debug"}))
                            .severity(LogSeverity::Warn)
                            .build(),
                    )
                    .with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("request.optimized.second")
                            .build(),
                    ),
            )
        }),
    )
    .unwrap();

    let provider_request = Arc::new(Mutex::new(None::<LlmRequest>));
    let captured_request = provider_request.clone();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("pending-managed-llm")
            .request(LlmRequest {
                headers: serde_json::Map::from_iter([(
                    "x-pending-mark-test".into(),
                    json!("preserved"),
                )]),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(move |request| {
                *captured_request.lock().unwrap() = Some(request);
                Box::pin(async { Ok(json!({"response": "done"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    let provider_request = provider_request.lock().unwrap().clone().unwrap();
    assert_eq!(
        provider_request.headers.get("x-pending-mark-test"),
        Some(&json!("preserved"))
    );
    assert_eq!(provider_request.content["prompt"], "hello");
    assert!(provider_request.content.get("pending_marks").is_none());
    assert!(provider_request.content.get("annotated_request").is_none());

    let captured = captured_events_snapshot(&events);
    let start = captured
        .iter()
        .find(|event| {
            event.name() == "pending-managed-llm"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    let mark = captured
        .iter()
        .find(|event| event.name() == "request.optimized")
        .unwrap();
    let second_mark = captured
        .iter()
        .find(|event| event.name() == "request.optimized.second")
        .unwrap();
    assert!(
        captured
            .iter()
            .all(|event| event.name() != "request.optimized.invalid")
    );
    let end = captured
        .iter()
        .find(|event| {
            event.name() == "pending-managed-llm"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert_eq!(mark.parent_uuid(), Some(start.uuid()));
    assert_eq!(second_mark.parent_uuid(), Some(start.uuid()));
    assert!(mark.timestamp() > start.timestamp());
    assert_eq!(mark.timestamp(), second_mark.timestamp());
    assert!(end.timestamp() >= mark.timestamp());
    assert_eq!(mark.data().unwrap()["saved_tokens"], 12);
    assert_eq!(
        mark.data_schema().unwrap(),
        &DataSchema::builder()
            .name("example.llm_pending_mark")
            .version("1")
            .build()
    );
    assert_eq!(mark.metadata().unwrap()[LOG_SEVERITY_METADATA_KEY], "warn");
    assert_eq!(
        mark.metadata().unwrap()["sanitized_mark"],
        "request.optimized"
    );
    assert_eq!(
        second_mark.metadata().unwrap()["sanitized_mark"],
        "request.optimized.second"
    );

    deregister_llm_request_intercept("pending_managed").unwrap();
    deregister_mark_sanitize_guardrail("llm_pending_mark_sanitizer").unwrap();
    deregister_subscriber("pending_mark_observer").unwrap();
}

#[tokio::test]
async fn test_managed_llm_materializes_optimization_mark_and_end_summary() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "optimization_observer",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_mark_sanitize_guardrail(
        "optimization_sanitizer",
        1,
        Arc::new(|event, mut fields| {
            if event.name() == "nemo_relay.llm.optimization"
                && let Some(data) = fields.data.as_mut().and_then(Json::as_object_mut)
            {
                data.insert("payload".to_string(), json!({"secret": "[redacted]"}));
                data.remove("future_secret");
            }
            ready(fields)
        }),
    )
    .unwrap();
    register_scope_sanitize_end_guardrail(
        "optimization_end_sanitizer",
        1,
        Arc::new(|event, mut fields| {
            if event.name() == "optimized-managed-llm"
                && let Some(profile) = fields.category_profile.as_mut()
                && let Some(response) = profile.annotated_response.as_mut()
                && let Some(summary) = Arc::make_mut(response).optimization_summary.as_mut()
                && let Some(contribution) = summary.contributions.first_mut()
            {
                contribution.payload = Some(json!({"secret": "[scope-end-redacted]"}));
                contribution.extra.remove("future_secret");
            }
            ready(fields)
        }),
    )
    .unwrap();

    register_llm_request_intercept(
        "optimization_contributor",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            let mut contribution =
                LlmOptimizationContribution::new("test.optimizer", "test_custom_kind");
            contribution.token_impact = Some(LlmOptimizationTokenImpact {
                saved: Some(LlmOptimizationTokens::saved_prompt(12)),
                quality: Some(LlmOptimizationEvidenceQuality::Estimated),
                estimation_method: Some("test-counter".to_string()),
                ..LlmOptimizationTokenImpact::default()
            });
            contribution.payload_schema = Some(DataSchema {
                name: "test.optimizer_evidence".to_string(),
                version: "1".to_string(),
            });
            contribution.payload = Some(json!({"secret": "classified"}));
            contribution
                .extra
                .insert("future_secret".to_string(), json!("classified"));
            ready(
                LlmRequestInterceptOutcome::new(request, annotated)
                    .with_optimization_contribution(contribution),
            )
        }),
    )
    .unwrap();
    register_llm_execution_intercept(
        "optimization_execution_contributor",
        1,
        Arc::new(|_name, request, next| {
            let contribution =
                LlmOptimizationContribution::new("test.execution", "test_execution_kind");
            assert!(record_llm_optimization_contribution(contribution));
            Box::pin(async move { next(request).await })
        }),
    )
    .unwrap();

    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("optimized-managed-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(|_| {
                Box::pin(async { Ok(json!({"response": "done"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let start = captured
        .iter()
        .find(|event| {
            event.name() == "optimized-managed-llm"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    let marks = captured
        .iter()
        .filter(|event| event.name() == "nemo_relay.llm.optimization")
        .collect::<Vec<_>>();
    assert_eq!(marks.len(), 2);
    assert!(
        marks
            .iter()
            .all(|mark| mark.parent_uuid() == Some(start.uuid()))
    );
    assert!(marks.iter().all(|mark| {
        mark.data_schema().unwrap().name == "nemo.relay.llm_optimization_contribution"
    }));
    assert_eq!(
        marks[0].data().unwrap()["token_impact"]["saved"]["prompt_tokens"],
        12
    );
    assert_eq!(marks[0].data().unwrap()["sequence"], 0);
    assert_eq!(marks[1].data().unwrap()["sequence"], 1);
    assert_eq!(marks[0].data().unwrap()["payload"]["secret"], "[redacted]");
    assert!(marks[0].data().unwrap().get("future_secret").is_none());

    let end = captured
        .iter()
        .find(|event| {
            event.name() == "optimized-managed-llm"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert!(marks[0].timestamp() > start.timestamp());
    assert!(marks[0].timestamp() <= marks[1].timestamp());
    assert!(marks[1].timestamp() <= end.timestamp());
    let summary = end
        .annotated_response()
        .unwrap()
        .optimization_summary
        .as_ref()
        .unwrap();
    assert_eq!(summary.tokens_saved.prompt_tokens, Some(12));
    assert_eq!(summary.contributions.len(), 2);
    assert_eq!(summary.contributions[0].producer, "test.optimizer");
    assert_eq!(
        summary.contributions[0].payload.as_ref().unwrap()["secret"],
        "[scope-end-redacted]"
    );
    assert!(!summary.contributions[0].extra.contains_key("future_secret"));
    assert_eq!(summary.contributions[1].producer, "test.execution");

    deregister_llm_request_intercept("optimization_contributor").unwrap();
    deregister_llm_execution_intercept("optimization_execution_contributor").unwrap();
    deregister_mark_sanitize_guardrail("optimization_sanitizer").unwrap();
    deregister_scope_sanitize_end_guardrail("optimization_end_sanitizer").unwrap();
    deregister_subscriber("optimization_observer").unwrap();
}

#[tokio::test]
async fn execution_optimization_mark_keeps_decision_commit_timestamp_order() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "optimization_timestamp_observer",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_execution_intercept(
        "optimization_timestamp_contributor",
        1,
        Arc::new(|_name, request, next| {
            Box::pin(async move {
                event(
                    EmitMarkEventParams::builder()
                        .name("test.router.requested")
                        .build(),
                )?;
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                event(
                    EmitMarkEventParams::builder()
                        .name("test.router.decision")
                        .build(),
                )?;
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                assert!(record_llm_optimization_contribution(
                    LlmOptimizationContribution::new("test.execution.timestamp", "model_routing")
                ));
                next(request).await
            })
        }),
    )
    .unwrap();

    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("optimized-timestamp-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(|_| {
                Box::pin(async { Ok(json!({"response": "done"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let requested = captured
        .iter()
        .find(|event| event.name() == "test.router.requested")
        .unwrap();
    let decision = captured
        .iter()
        .find(|event| event.name() == "test.router.decision")
        .unwrap();
    let contribution = captured
        .iter()
        .find(|event| {
            event.name() == "nemo_relay.llm.optimization"
                && event.data().and_then(|data| data["producer"].as_str())
                    == Some("test.execution.timestamp")
        })
        .unwrap();
    let end = captured
        .iter()
        .find(|event| {
            event.name() == "optimized-timestamp-llm"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();

    assert!(requested.timestamp() < decision.timestamp());
    assert!(decision.timestamp() < contribution.timestamp());
    assert!(contribution.timestamp() <= end.timestamp());

    deregister_llm_execution_intercept("optimization_timestamp_contributor").unwrap();
    deregister_subscriber("optimization_timestamp_observer").unwrap();
}

#[tokio::test]
async fn test_stream_optimization_mark_uses_the_llm_captured_sanitizer_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let original_stack = current_scope_stack();
    let owner = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("optimization-sanitizer-owner")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope_register_mark_sanitize_guardrail(
        &owner.uuid,
        "stream-optimization-sanitizer",
        1,
        Arc::new(|event, mut fields| {
            if event.name() == "nemo_relay.llm.optimization"
                && let Some(data) = fields.data.as_mut().and_then(Json::as_object_mut)
            {
                data.insert("payload".to_string(), json!({"secret": "[redacted]"}));
            }
            ready(fields)
        }),
    )
    .unwrap();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "stream_optimization_sanitizer_observer",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_stream_execution_intercept(
        "stream_optimization_sanitizer_contributor",
        1,
        Arc::new(|_name, request, next| {
            let mut contribution = LlmOptimizationContribution::new("test.stream", "stream_test");
            contribution.payload_schema = Some(DataSchema {
                name: "test.stream_evidence".to_string(),
                version: "1".to_string(),
            });
            contribution.payload = Some(json!({"secret": "classified"}));
            assert!(record_llm_optimization_contribution(contribution));
            Box::pin(async move { next(request).await })
        }),
    )
    .unwrap();

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("stream-optimization-sanitized")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(|_| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                        "chunk": "done"
                    }))])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"response": "done"})))
            .build(),
    )
    .await
    .unwrap();

    // Poll under a different ambient stack. The wrapper must continue using
    // the scope captured with the LLM handle, where the sanitizer is registered.
    set_thread_scope_stack(create_scope_stack());
    while let Some(item) = stream.next().await {
        item.unwrap();
    }
    stream.close().await.unwrap();
    set_thread_scope_stack(original_stack);

    let captured = captured_events_snapshot(&events);
    let mark = captured
        .iter()
        .find(|event| event.name() == "nemo_relay.llm.optimization")
        .expect("expected a canonical optimization mark");
    assert_eq!(mark.data().unwrap()["payload"]["secret"], "[redacted]");

    deregister_llm_stream_execution_intercept("stream_optimization_sanitizer_contributor").unwrap();
    deregister_subscriber("stream_optimization_sanitizer_observer").unwrap();
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&owner.uuid)
            .build(),
    )
    .unwrap();
}

#[tokio::test]
async fn test_concurrent_managed_llm_calls_keep_optimization_evidence_isolated() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "concurrent_optimization_observer",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_request_intercept(
        "concurrent_optimization_contributor",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            let call = request.content["call"].as_str().unwrap().to_string();
            let saved_tokens = if call == "a" { 11 } else { 22 };
            let mut contribution =
                LlmOptimizationContribution::new(format!("test.{call}"), "concurrency_test");
            contribution.token_impact = Some(LlmOptimizationTokenImpact {
                saved: Some(LlmOptimizationTokens::saved_prompt(saved_tokens)),
                ..LlmOptimizationTokenImpact::default()
            });
            ready(
                LlmRequestInterceptOutcome::new(request, annotated)
                    .with_optimization_contribution(contribution),
            )
        }),
    )
    .unwrap();

    let call = |name: &'static str, label: &'static str| {
        llm_call_execute(
            LlmCallExecuteParams::builder()
                .name(name)
                .request(LlmRequest {
                    headers: serde_json::Map::new(),
                    content: json!({"call": label}),
                })
                .func(Arc::new(move |_| {
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        Ok(json!({"call": label}))
                    })
                }))
                .build(),
        )
    };
    let (a, b) = tokio::join!(call("concurrent-a", "a"), call("concurrent-b", "b"));
    a.unwrap();
    b.unwrap();

    let captured = captured_events_snapshot(&events);
    for (name, expected_producer, expected_tokens) in [
        ("concurrent-a", "test.a", 11),
        ("concurrent-b", "test.b", 22),
    ] {
        let end = captured
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::End)
            })
            .unwrap_or_else(|| panic!("missing end event for {name}"));
        let contributions = &end
            .annotated_response()
            .and_then(|response| response.optimization_summary.as_ref())
            .unwrap_or_else(|| panic!("missing optimization summary for {name}"))
            .contributions;
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].producer, expected_producer);
        assert_eq!(
            end.annotated_response()
                .unwrap()
                .optimization_summary
                .as_ref()
                .unwrap()
                .tokens_saved
                .prompt_tokens,
            Some(expected_tokens)
        );

        let mark = captured
            .iter()
            .find(|event| {
                event.name() == "nemo_relay.llm.optimization"
                    && event.parent_uuid() == Some(end.uuid())
            })
            .unwrap_or_else(|| panic!("missing optimization mark for {name}"));
        assert_eq!(mark.data().unwrap()["producer"], expected_producer);
        assert_eq!(
            mark.data().unwrap()["token_impact"]["saved"]["prompt_tokens"],
            expected_tokens
        );
    }

    deregister_llm_request_intercept("concurrent_optimization_contributor").unwrap();
    deregister_subscriber("concurrent_optimization_observer").unwrap();
}

#[tokio::test]
async fn test_failed_request_intercept_does_not_emit_pending_marks_or_start_scope() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "failed_pending_mark_observer",
        Arc::new(move |event: &Event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_request_intercept(
        "pending_before_failure",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            ready(
                LlmRequestInterceptOutcome::new(request, annotated)
                    .with_pending_mark(PendingMarkSpec::builder().name("must.not.emit").build()),
            )
        }),
    )
    .unwrap();
    register_llm_request_intercept(
        "pending_failure",
        2,
        false,
        Arc::new(|_name, _request, _annotated| {
            ready_result(Err(FlowError::Internal("request intercept failed".into())))
        }),
    )
    .unwrap();

    let provider_called = Arc::new(AtomicBool::new(false));
    let called = provider_called.clone();
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("failed-pending-llm")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"prompt": "hello"}),
            })
            .func(Arc::new(move |_request| {
                called.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(json!({"response": "unexpected"})) })
            }))
            .build(),
    )
    .await;

    assert!(result.is_err());
    assert!(!provider_called.load(Ordering::SeqCst));
    assert!(captured_events_snapshot(&events).is_empty());

    deregister_llm_request_intercept("pending_before_failure").unwrap();
    deregister_llm_request_intercept("pending_failure").unwrap();
    deregister_subscriber("failed_pending_mark_observer").unwrap();
}

/// LLM execution intercept middleware chain with next().
#[tokio::test]
async fn test_llm_execution_intercept_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    let o1 = order.clone();
    register_llm_execution_intercept(
        "llm_exec_1",
        1,
        Arc::new(move |_name, req, next| {
            let o = o1.clone();
            Box::pin(async move {
                o.lock().unwrap().push("intercept_before".into());
                let r = next(req).await;
                o.lock().unwrap().push("intercept_after".into());
                r
            })
        }),
    )
    .unwrap();

    let oo = order.clone();
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        oo.lock().unwrap().push("original".into());
        Box::pin(async move { Ok(json!({"response": "done"})) })
    });

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({}),
    };

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("llm")
            .request(request)
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    let recorded = order.lock().unwrap();
    assert_eq!(
        *recorded,
        vec!["intercept_before", "original", "intercept_after"]
    );
    assert_eq!(result["response"], "done");

    // Cleanup
    deregister_llm_execution_intercept("llm_exec_1").unwrap();
}

/// LLM start is queued after request intercepts and before execution intercepts,
/// even when an execution intercept replaces the callback.
#[tokio::test]
async fn test_llm_start_emits_before_short_circuit_execution_intercept() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let ec = events.clone();
    register_subscriber(
        "llm_short_circuit_start_observer",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    register_llm_request_intercept(
        "llm_short_circuit_request",
        1,
        false,
        Arc::new(|_name, mut req, annotated| {
            req.content
                .as_object_mut()
                .unwrap()
                .insert("phase".into(), json!("request"));
            ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
        }),
    )
    .unwrap();

    register_llm_execution_intercept(
        "llm_short_circuit_exec",
        1,
        Arc::new(move |_name, mut req, _next| {
            Box::pin(async move {
                req.content
                    .as_object_mut()
                    .unwrap()
                    .insert("phase".into(), json!("execution"));
                Ok(json!({"response": "short-circuited"}))
            })
        }),
    )
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let oc = original_called.clone();
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        oc.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(json!({"response": "original"})) })
    });

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("llm")
            .request(request)
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result["response"], "short-circuited");
    assert!(
        !original_called.load(Ordering::SeqCst),
        "Original callable should not be invoked"
    );

    let captured = captured_events_snapshot(&events);
    let llm_events = captured
        .iter()
        .filter(|e| e.scope_type() == Some(ScopeType::Llm))
        .collect::<Vec<_>>();
    assert_eq!(llm_events.len(), 2);
    assert_eq!(llm_events[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(
        llm_events[0].input().unwrap()["content"]["phase"],
        json!("request")
    );
    assert_eq!(llm_events[1].scope_category(), Some(ScopeCategory::End));
    deregister_llm_execution_intercept("llm_short_circuit_exec").unwrap();
    deregister_llm_request_intercept("llm_short_circuit_request").unwrap();
    deregister_subscriber("llm_short_circuit_start_observer").unwrap();
}

/// Streaming LLM start follows the same pre-execution ordering as non-streaming
/// calls when a stream execution intercept replaces the callback.
#[tokio::test]
async fn test_llm_stream_start_emits_before_short_circuit_execution_intercept() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let ec = events.clone();
    register_subscriber(
        "llm_stream_short_circuit_start_observer",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    register_llm_request_intercept(
        "llm_stream_short_circuit_request",
        1,
        false,
        Arc::new(|_name, mut req, annotated| {
            req.content
                .as_object_mut()
                .unwrap()
                .insert("phase".into(), json!("request"));
            ready(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
        }),
    )
    .unwrap();

    register_llm_stream_execution_intercept(
        "llm_stream_short_circuit_exec",
        1,
        Arc::new(move |_name, mut req, _next| {
            Box::pin(async move {
                req.content
                    .as_object_mut()
                    .unwrap()
                    .insert("phase".into(), json!("execution"));
                let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "short-circuited"}))]);
                Ok(LlmJsonStream::new(stream))
            })
        }),
    )
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let oc = original_called.clone();
    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        oc.store(true, Ordering::SeqCst);
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(LlmJsonStream::new(stream))
        })
    });

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("llm-stream")
            .request(request)
            .func(func)
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"response": "stream-complete"})))
            .build(),
    )
    .await
    .unwrap();

    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
    stream.close().await.unwrap();

    assert!(
        !original_called.load(Ordering::SeqCst),
        "Original stream callable should not be invoked"
    );

    let captured = captured_events_snapshot(&events);
    let llm_events = captured
        .iter()
        .filter(|e| e.scope_type() == Some(ScopeType::Llm))
        .collect::<Vec<_>>();
    assert_eq!(llm_events.len(), 2);
    assert_eq!(llm_events[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(
        llm_events[0].input().unwrap()["content"]["phase"],
        json!("request")
    );
    assert_eq!(llm_events[1].scope_category(), Some(ScopeCategory::End));
    deregister_llm_stream_execution_intercept("llm_stream_short_circuit_exec").unwrap();
    deregister_llm_request_intercept("llm_stream_short_circuit_request").unwrap();
    deregister_subscriber("llm_stream_short_circuit_start_observer").unwrap();
}

// =========================================================================
// Standalone Chain API Tests
// =========================================================================

/// tool_conditional_execution returns Ok(()) when no guardrails reject.
#[tokio::test]
async fn test_standalone_conditional_execution_passes() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let result = tool_conditional_execution("tool", &json!({})).await;
    assert!(result.is_ok(), "No guardrails means no rejection");
}

/// tool_conditional_execution returns GuardrailRejected when a guardrail rejects.
#[tokio::test]
async fn test_standalone_conditional_execution_rejects() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_conditional_execution_guardrail(
        "standalone_gate",
        1,
        Arc::new(|_name, _args| Box::pin(async { Ok(Some("rejected by standalone".to_string())) })),
    )
    .unwrap();

    let result = tool_conditional_execution("tool", &json!({})).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        FlowError::GuardrailRejected(reason) => {
            assert!(reason.contains("rejected by standalone"));
        }
        other => panic!("Expected GuardrailRejected, got: {:?}", other),
    }

    // Cleanup
    deregister_tool_conditional_execution_guardrail("standalone_gate").unwrap();
}

// =========================================================================
// Empty Chain Tests
// =========================================================================

/// With no guardrails or intercepts registered, the pipeline passes through cleanly.
#[tokio::test]
async fn test_empty_chain_passthrough() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args.into()) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"value": "unchanged"}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(
        result.result["value"], "unchanged",
        "Data should pass through unmodified"
    );
}

/// Standalone intercept chain with no registrations returns input unchanged.
#[tokio::test]
async fn test_empty_request_intercept_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let result = tool_request_intercepts("tool", json!({"key": "val"}))
        .await
        .unwrap();
    assert_eq!(result["key"], "val");
}

#[tokio::test]
async fn test_conditional_middleware_guardrail_toggles_only_global_target() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_request_intercept(
        "conditional_target",
        1,
        false,
        Arc::new(|_name, mut args| {
            args["global"] = json!(true);
            ready(args)
        }),
    )
    .unwrap();

    let scope = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("conditional-local-scope")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope_register_tool_request_intercept(
        &scope.uuid,
        "conditional_target",
        2,
        false,
        Arc::new(|_name, mut args| {
            args["local"] = json!(true);
            ready(args)
        }),
    )
    .unwrap();

    let kinds = BTreeSet::from([RuntimeRegistrationKind::ToolRequestIntercept]);
    let paused = Arc::new(AtomicBool::new(true));
    let captured_paused = Arc::clone(&paused);
    register_conditional_middleware_guardrail(
        "conditional_gate",
        kinds.clone(),
        "conditional_target",
        Arc::new(move |received_kinds, received_name| {
            assert_eq!(received_kinds, &kinds);
            assert_eq!(received_name, "conditional_target");
            captured_paused
                .load(Ordering::Acquire)
                .then(|| "temporarily disabled".to_string())
        }),
    )
    .unwrap();

    let disabled = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(disabled, json!({"local": true}));

    paused.store(false, Ordering::Release);
    let enabled_by_shared_state = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(
        enabled_by_shared_state,
        json!({"global": true, "local": true})
    );

    paused.store(true, Ordering::Release);
    assert!(deregister_conditional_middleware_guardrail("conditional_gate").unwrap());
    let enabled = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(enabled, json!({"global": true, "local": true}));

    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope.uuid)
            .build(),
    )
    .unwrap();
    deregister_tool_request_intercept("conditional_target").unwrap();
}

#[tokio::test]
async fn test_conditional_middleware_guardrail_callback_runs_without_global_runtime_lock() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_request_intercept(
        "self_deregistering_target",
        0,
        false,
        Arc::new(|_name, mut args| {
            args["intercepted"] = json!(true);
            ready(args)
        }),
    )
    .unwrap();
    register_conditional_middleware_guardrail(
        "self_deregistering_gate",
        BTreeSet::from([RuntimeRegistrationKind::ToolRequestIntercept]),
        "self_deregistering_target",
        Arc::new(|_, _| {
            assert!(deregister_tool_request_intercept("self_deregistering_target").unwrap());
            None
        }),
    )
    .unwrap();

    let current_snapshot = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(current_snapshot, json!({"intercepted": true}));
    let next_snapshot = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(next_snapshot, json!({}));

    assert!(deregister_conditional_middleware_guardrail("self_deregistering_gate").unwrap());
}

#[tokio::test]
async fn test_conditional_middleware_guardrail_callback_runs_without_scope_stack_lock() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_request_intercept(
        "scope_mutating_target",
        0,
        false,
        Arc::new(|_name, mut args| {
            args["global"] = json!(true);
            ready(args)
        }),
    )
    .unwrap();
    let scope = push_scope(
        nemo_relay::api::scope::PushScopeParams::builder()
            .name("scope-mutating-gate")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope_register_tool_request_intercept(
        &scope.uuid,
        "scope_owned_target",
        1,
        false,
        Arc::new(|_name, mut args| {
            args["local"] = json!(true);
            ready(args)
        }),
    )
    .unwrap();
    let scope_uuid = scope.uuid;
    register_conditional_middleware_guardrail(
        "scope_mutating_gate",
        BTreeSet::from([RuntimeRegistrationKind::ToolRequestIntercept]),
        "scope_mutating_target",
        Arc::new(move |_, _| {
            let _ =
                scope_deregister_tool_request_intercept(&scope_uuid, "scope_owned_target").unwrap();
            None
        }),
    )
    .unwrap();

    let current_snapshot = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(current_snapshot, json!({"global": true, "local": true}));
    let next_snapshot = tool_request_intercepts("tool", json!({})).await.unwrap();
    assert_eq!(next_snapshot, json!({"global": true}));

    assert!(deregister_conditional_middleware_guardrail("scope_mutating_gate").unwrap());
    pop_scope(
        nemo_relay::api::scope::PopScopeParams::builder()
            .handle_uuid(&scope.uuid)
            .build(),
    )
    .unwrap();
    deregister_tool_request_intercept("scope_mutating_target").unwrap();
}

#[tokio::test]
async fn test_conditional_middleware_guardrails_are_ordered_and_fail_open() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_request_intercept(
        "ordered_target",
        0,
        false,
        Arc::new(|_name, mut args| {
            args["intercepted"] = json!(true);
            ready(args)
        }),
    )
    .unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    for (gate_name, decision) in [
        ("a_panics", None),
        ("b_enables", None),
        ("c_disables", Some("disabled")),
        ("d_not_reached", Some("also disabled")),
    ] {
        let calls = Arc::clone(&calls);
        register_conditional_middleware_guardrail(
            gate_name,
            BTreeSet::from([RuntimeRegistrationKind::ToolRequestIntercept]),
            "ordered_target",
            Arc::new(move |_, _| {
                calls.lock().unwrap().push(gate_name);
                assert_ne!(gate_name, "a_panics", "expected fail-open panic");
                decision.map(str::to_string)
            }),
        )
        .unwrap();
    }

    assert_eq!(
        tool_request_intercepts("tool", json!({})).await.unwrap(),
        json!({})
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec!["a_panics", "b_enables", "c_disables"]
    );

    for gate_name in ["a_panics", "b_enables", "c_disables", "d_not_reached"] {
        assert!(deregister_conditional_middleware_guardrail(gate_name).unwrap());
    }
    assert!(deregister_tool_request_intercept("ordered_target").unwrap());
}

#[tokio::test]
async fn test_runtime_registration_discovery_and_cross_owner_subscriber_gate() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let mut context =
        PluginRegistrationContext::with_namespace("nemo-relay-plugin.v1.observability:1:");
    let deliveries = Arc::new(AtomicU32::new(0));
    let captured_deliveries = Arc::clone(&deliveries);
    context
        .register_subscriber(
            "opentelemetry",
            Arc::new(move |_| {
                captured_deliveries.fetch_add(1, Ordering::AcqRel);
            }),
        )
        .unwrap();

    let kinds = BTreeSet::from([RuntimeRegistrationKind::Subscriber]);
    let registrations = list_runtime_registrations(Some(&kinds)).unwrap();
    let registration = registrations
        .iter()
        .find(|registration| registration.local_name == "opentelemetry")
        .unwrap();
    assert_eq!(registration.kind, RuntimeRegistrationKind::Subscriber);
    assert_eq!(
        registration.effective_name,
        "nemo-relay-plugin.v1.observability:1:opentelemetry"
    );
    assert_eq!(
        registration.owner.kind,
        RuntimeRegistrationOwnerKind::Plugin
    );
    assert_eq!(
        registration.owner.plugin_kind.as_deref(),
        Some("observability")
    );
    assert_eq!(registration.owner.component_ordinal, Some(1));

    register_conditional_middleware_guardrail(
        "external_observability_gate",
        kinds,
        &registration.effective_name,
        Arc::new(|_, _| Some("paused by another owner".into())),
    )
    .unwrap();
    event(
        EmitMarkEventParams::builder()
            .name("cross-owner-gated")
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert_eq!(deliveries.load(Ordering::Acquire), 0);

    assert!(deregister_conditional_middleware_guardrail("external_observability_gate").unwrap());
    event(
        EmitMarkEventParams::builder()
            .name("cross-owner-enabled")
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert_eq!(deliveries.load(Ordering::Acquire), 1);

    let mut registrations = context.into_registrations();
    rollback_registrations(&mut registrations);
}
