// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive middleware chain tests for the NeMo Relay core runtime.
//!
//! These tests exercise the middleware pipeline mechanics: priority ordering,
//! break_chain short-circuiting, execution intercept middleware chains (next()),
//! conditional execution guardrails, scope-local middleware lifecycle, global +
//! scope-local merging, error propagation, and concurrent mutations.

#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use nemo_relay::api::event::{
    CategoryProfile, DataSchema, Event, EventCategory, PendingMarkSpec, ScopeCategory,
};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_request_intercepts,
    llm_stream_call_execute,
};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::optimization::record_llm_optimization_contribution;
use nemo_relay::api::registry::{
    deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
    deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_response_guardrail, deregister_llm_stream_execution_intercept,
    deregister_mark_sanitize_guardrail, deregister_scope_sanitize_end_guardrail,
    deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
    deregister_tool_request_intercept, deregister_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_response_guardrail, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
    register_llm_stream_execution_intercept, register_mark_sanitize_guardrail,
    register_scope_sanitize_end_guardrail, register_tool_conditional_execution_guardrail,
    register_tool_execution_intercept, register_tool_request_intercept,
    register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
    scope_register_llm_conditional_execution_guardrail, scope_register_llm_execution_intercept,
    scope_register_llm_request_intercept, scope_register_llm_sanitize_request_guardrail,
    scope_register_llm_sanitize_response_guardrail, scope_register_llm_stream_execution_intercept,
    scope_register_mark_sanitize_guardrail, scope_register_tool_conditional_execution_guardrail,
    scope_register_tool_execution_intercept, scope_register_tool_request_intercept,
    scope_register_tool_sanitize_request_guardrail,
    scope_register_tool_sanitize_response_guardrail,
};
use nemo_relay::api::runtime::NemoRelayContextState;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, ToolExecutionNextFn,
};
use nemo_relay::api::runtime::{create_scope_stack, current_scope_stack, set_thread_scope_stack};
use nemo_relay::api::scope::{EmitMarkEventParams, ScopeHandle, ScopeType, event};
use nemo_relay::api::scope::{pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{
    ToolExecutionInterceptOutcome, tool_call, tool_call_end, tool_call_execute,
    tool_conditional_execution, tool_request_intercepts,
};
use nemo_relay::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationTokenImpact,
    LlmOptimizationTokens,
};
use nemo_relay::error::FlowError;
use nemo_relay::json::Json;
#[cfg(all(feature = "otel", feature = "openinference"))]
use nemo_relay::observability::MarkProjection;
#[cfg(all(feature = "otel", feature = "openinference"))]
use nemo_relay::observability::atif::{AtifAgentInfo, AtifExporter};
#[cfg(all(feature = "otel", feature = "openinference"))]
use nemo_relay::observability::openinference::OpenInferenceSubscriber;
#[cfg(all(feature = "otel", feature = "openinference"))]
use nemo_relay::observability::otel::OpenTelemetrySubscriber;
use nemo_relay::plugin::{PluginRegistrationContext, rollback_registrations};
#[cfg(all(feature = "otel", feature = "openinference"))]
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use serde_json::json;

// All tests share the global context, so we serialize them.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

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

fn assert_middleware_callback_locks_are_free() {
    let context = global_context();
    assert!(
        context.try_write().is_ok(),
        "middleware callback ran while the global registry lock was held"
    );

    let scope_stack = current_scope_stack();
    assert!(
        scope_stack.try_write().is_ok(),
        "middleware callback ran while the scope stack lock was held"
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
#[test]
fn test_sanitize_guardrail_priority_ordering() {
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
            args
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
            args
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
            args
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
#[test]
fn test_request_intercept_priority_ordering() {
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
            Ok(args)
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
            Ok(args)
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
            Ok(args)
        }),
    )
    .unwrap();

    // Use the standalone intercept chain function
    let _result = tool_request_intercepts("test_tool", json!({})).unwrap();

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
#[test]
fn test_re_registration_at_different_priority_re_sorts() {
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
            Ok(args)
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
            Ok(args)
        }),
    )
    .unwrap();

    // First call: a runs before b
    let _ = tool_request_intercepts("test", json!({})).unwrap();
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
            Ok(args)
        }),
    )
    .unwrap();

    // Clear and re-run
    order.lock().unwrap().clear();
    let _ = tool_request_intercepts("test", json!({})).unwrap();
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
#[test]
fn test_break_chain_stops_subsequent_intercepts() {
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
            Ok(args)
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
            Ok(args)
        }),
    )
    .unwrap();

    let result = tool_request_intercepts("tool", json!({})).unwrap();

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
#[test]
fn test_no_break_chain_runs_all_intercepts() {
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
            Ok(args)
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
            Ok(args)
        }),
    )
    .unwrap();

    let _ = tool_request_intercepts("tool", json!({})).unwrap();

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
        Box::pin(async move { Ok(args) })
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
    assert_eq!(result["value"], 42);

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
        Box::pin(async move { Ok(args) })
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
    assert_eq!(result["intercepted"], true);

    // Cleanup
    deregister_tool_execution_intercept("short_circuit").unwrap();
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
        Box::pin(async move { Ok(args) })
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

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"original": true}))
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result["original"], true);
    assert_eq!(result["injected"], true);

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
            fields
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
                        ToolExecutionInterceptOutcome::new(result).with_pending_mark(
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
                    result["compressed"] = json!(true);
                    Ok(
                        ToolExecutionInterceptOutcome::new(result).with_pending_mark(
                            PendingMarkSpec::builder()
                                .name("tool.mark.inner")
                                .category(EventCategory::custom())
                                .category_profile(
                                    CategoryProfile::builder()
                                        .subtype("example.tool.compression")
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
            .name("tool-outcome")
            .args(json!({"value": 42}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result, json!({"value": 42, "compressed": true}));
    assert!(result.get("pending_marks").is_none());

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
    assert_eq!(end.data().unwrap(), &result);
    assert_eq!(inner.category().map(EventCategory::as_str), Some("custom"));
    assert_eq!(
        inner
            .category_profile()
            .and_then(|profile| profile.subtype.as_deref()),
        Some("example.tool.compression")
    );
    assert_eq!(inner.data().unwrap()["saved_tokens"], 12);
    assert_eq!(inner.metadata().unwrap()["source"], "test");
    drop(captured);

    deregister_tool_execution_intercept("passthrough_between_outcomes").unwrap();
    let mut registrations = plugin_ctx.into_registrations();
    rollback_registrations(&mut registrations);
    deregister_mark_sanitize_guardrail("tool_pending_mark_sanitizer").unwrap();
    deregister_subscriber("tool_outcome_mark_observer").unwrap();
}

#[cfg(all(feature = "otel", feature = "openinference"))]
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
    let otel = OpenTelemetrySubscriber::from_tracer_provider_with_mark_projection(
        otel_provider,
        "managed-tool-projection-otel",
        MarkProjection::Tool,
    );
    register_subscriber("managed_tool_projection_otel", otel.subscriber()).unwrap();

    let openinference_exporter = InMemorySpanExporterBuilder::new().build();
    let openinference_provider = SdkTracerProvider::builder()
        .with_simple_exporter(openinference_exporter.clone())
        .build();
    let openinference = OpenInferenceSubscriber::from_tracer_provider_with_mark_projection(
        openinference_provider,
        "managed-tool-projection-openinference",
        MarkProjection::Tool,
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
                    ToolExecutionInterceptOutcome::new(result).with_pending_mark(
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result, json!({"value": 42}));

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
    assert_eq!(mark.parent_uuid(), Some(tool_end.uuid()));
    assert!(mark.timestamp() > tool_end.timestamp());
    drop(captured);

    let trajectory = atif.export().unwrap();
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
    assert_eq!(otel_mark.parent_span_id, otel_tool.span_context.span_id());
    assert!(otel_mark.start_time > otel_tool.end_time);
    assert!(otel_mark.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "nemo_relay.mark.projection"
            && attribute.value.to_string() == "tool"
    }));
    for (key, value) in [
        ("nemo_relay.mark.category", "custom"),
        (
            "nemo_relay.mark.category_profile_json",
            "{\"subtype\":\"example.compaction\"}",
        ),
        ("nemo_relay.mark.metadata_json", "{\"source\":\"test\"}"),
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
    assert_eq!(
        openinference_mark.parent_span_id,
        openinference_tool.span_context.span_id()
    );
    assert!(openinference_mark.start_time > openinference_tool.end_time);
    assert!(openinference_mark.attributes.iter().any(|attribute| {
        attribute.key.as_str() == "openinference.span.kind" && attribute.value.to_string() == "TOOL"
    }));
    for (key, value) in [
        ("nemo_relay.mark.category", "custom"),
        (
            "nemo_relay.mark.category_profile_json",
            "{\"subtype\":\"example.compaction\"}",
        ),
        ("nemo_relay.mark.metadata_json", "{\"source\":\"test\"}"),
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
                        ToolExecutionInterceptOutcome::new(result).with_pending_mark(
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
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
    assert!(captured.iter().any(|event| {
        event.name() == "tool-outcome-error" && event.scope_category() == Some(ScopeCategory::End)
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
                        ToolExecutionInterceptOutcome::new(result).with_pending_mark(
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
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
                Ok(json!({
                    "first": first?,
                    "second": second?,
                })
                .into())
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
                        ToolExecutionInterceptOutcome::new(result).with_pending_mark(
                            PendingMarkSpec::builder()
                                .name(format!("tool.concurrent.{branch}"))
                                .build(),
                        ),
                    )
                })
            }),
        )
        .unwrap();

    tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool-concurrent-next")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
            .build(),
    )
    .await
    .unwrap();
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

    register_tool_conditional_execution_guardrail(
        "rejector",
        1,
        Arc::new(|_name, _args| Ok(Some("not allowed".to_string()))),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

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
            assert_eq!(reason, "not allowed");
        }
        other => panic!("Expected GuardrailRejected, got: {:?}", other),
    }

    // Cleanup
    deregister_tool_conditional_execution_guardrail("rejector").unwrap();
}

/// Register a conditional guardrail that allows (returns None). Execution proceeds.
#[tokio::test]
async fn test_conditional_guardrail_allows() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_conditional_execution_guardrail("allower", 1, Arc::new(|_name, _args| Ok(None)))
        .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

    let result = tool_call_execute(
        nemo_relay::api::tool::ToolCallExecuteParams::builder()
            .name("tool")
            .args(json!({"input": "data"}))
            .func(func)
            .build(),
    )
    .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap()["input"], "data");

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

    register_tool_conditional_execution_guardrail("tool_scope_allow", 1, Arc::new(|_, _| Ok(None)))
        .unwrap();
    register_tool_conditional_execution_guardrail(
        "tool_scope_reject",
        2,
        Arc::new(|_, _| Ok(Some("blocked by tool guardrail".to_string()))),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
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

    register_tool_conditional_execution_guardrail("allows", 1, Arc::new(|_name, _args| Ok(None)))
        .unwrap();

    register_tool_conditional_execution_guardrail(
        "rejects",
        2,
        Arc::new(|_name, _args| Ok(Some("blocked by second".to_string()))),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

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
                Ok(Some("dangerous_tool is forbidden".to_string()))
            } else {
                Ok(None)
            }
        }),
    )
    .unwrap();

    // Dangerous tool is rejected
    let func1: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
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
    let func2: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
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
#[test]
fn test_scope_local_guardrail_lifecycle() {
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
            args
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
    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
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
    let func2: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
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
#[test]
fn test_scope_local_and_global_guardrail_merge_priority() {
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
            args
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
            args
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
        Box::pin(async move { Ok(args) })
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
        Arc::new(|_name, _args| Ok(Some("blocked".to_string()))),
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
            Ok(args)
        }),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
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
        Arc::new(|_name, _args| Ok(Some("no execution".to_string()))),
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
        Box::pin(async move { Ok(args) })
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
#[test]
fn test_sanitize_guardrails_pipe_data() {
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
            args
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
            args
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
#[test]
fn test_response_sanitize_guardrails_pipe() {
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
            result
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
            .result(json!({"raw": true}))
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
#[test]
fn test_concurrent_register_deregister() {
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
                    Arc::new(|_name, args| args),
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
#[test]
fn test_concurrent_intercept_mutations() {
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
                    Arc::new(|_name, args| Ok(args)),
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
#[test]
fn test_concurrent_register_and_read() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Pre-register some guardrails
    for i in 0..4 {
        register_tool_sanitize_request_guardrail(
            &format!("stable_{i}"),
            i,
            Arc::new(|_name, args| args),
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
                        Arc::new(|_name, args| args),
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

#[test]
fn test_tool_request_intercept_registry_mutations_apply_to_later_calls() {
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
                        Ok(args)
                    }),
                )
                .unwrap();
            }

            Ok(args)
        }),
    )
    .unwrap();

    let args = tool_request_intercepts("tool", json!({"round": 1})).unwrap();
    assert_eq!(args["round"], 1);
    assert_middleware_callback_labels(&callbacks, &["tool_request_initial"]);

    callbacks.lock().unwrap().clear();
    let args = tool_request_intercepts("tool", json!({"round": 2})).unwrap();
    assert_eq!(args["round"], 2);
    assert_middleware_callback_labels(&callbacks, &["tool_request_initial", "tool_request_late"]);

    deregister_tool_request_intercept("snapshot_tool_request_initial").unwrap();
    deregister_tool_request_intercept("snapshot_tool_request_late").unwrap();
}

#[test]
fn test_llm_request_intercept_registry_mutations_apply_to_later_calls() {
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
                        Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                            request, annotated,
                        ))
                    }),
                )
                .unwrap();
            }

            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
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
            Ok(None)
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
            Ok(None)
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
            Ok(args)
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
            Ok(args)
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_tool_sanitize_request_guardrail(
        "lock_global_tool_sanitize_request",
        1,
        Arc::new(move |_, args| {
            record_middleware_callback(&tracked, "tool_sanitize_request_global");
            assert_middleware_callback_locks_are_free();
            args
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
            assert_middleware_callback_locks_are_free();
            args
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
            assert_middleware_callback_locks_are_free();
            result
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
            assert_middleware_callback_locks_are_free();
            result
        }),
    )
    .unwrap();

    let tracked = callbacks.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        record_middleware_callback(&tracked, "tool_func");
        assert_middleware_callback_locks_are_free();
        Box::pin(async move { Ok(args) })
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
    assert_eq!(result["ok"], true);
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
            Ok(None)
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
            Ok(None)
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                request, annotated,
            ))
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    register_llm_sanitize_request_guardrail(
        "lock_global_llm_sanitize_request",
        1,
        Arc::new(move |request| {
            record_middleware_callback(&tracked, "llm_sanitize_request_global");
            assert_middleware_callback_locks_are_free();
            request
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_sanitize_request_guardrail(
        &scope.uuid,
        "lock_scope_llm_sanitize_request",
        2,
        Arc::new(move |request| {
            record_middleware_callback(&tracked, "llm_sanitize_request_scope");
            assert_middleware_callback_locks_are_free();
            request
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
        Arc::new(move |response| {
            record_middleware_callback(&tracked, "llm_sanitize_response_global");
            assert_middleware_callback_locks_are_free();
            response
        }),
    )
    .unwrap();
    let tracked = callbacks.clone();
    scope_register_llm_sanitize_response_guardrail(
        &scope.uuid,
        "lock_scope_llm_sanitize_response",
        2,
        Arc::new(move |response| {
            record_middleware_callback(&tracked, "llm_sanitize_response_scope");
            assert_middleware_callback_locks_are_free();
            response
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
            Ok(Box::pin(stream) as LlmJsonStream)
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
            Ok(args)
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
            args
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
            Ok(None) // Allow
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
            result
        }),
    )
    .unwrap();

    let o_orig = order.clone();
    let func: ToolExecutionNextFn = Arc::new(move |args| {
        o_orig.lock().unwrap().push("original_execution".into());
        Box::pin(async move { Ok(args) })
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

    // Verify the pipeline order:
    // 1. conditional (runs on raw args, before intercepts)
    // 2. request_intercept (transforms args)
    // 3. sanitize_request (inside tool_call)
    // 4. execution_intercept -> original_execution
    // 5. sanitize_response (inside tool_call_end)
    let recorded = order.lock().unwrap();
    assert_eq!(
        *recorded,
        vec![
            "conditional",
            "request_intercept",
            "sanitize_request",
            "execution_intercept",
            "original_execution",
            "sanitize_response",
        ],
        "Full pipeline should execute in the correct order"
    );

    // Verify the request intercept's modification persists through the pipeline
    assert_eq!(result["intercepted"], true);
    assert_eq!(result["data"], "test");

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
#[test]
fn test_duplicate_guardrail_registration_returns_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    register_tool_sanitize_request_guardrail("duplicate", 1, Arc::new(|_name, args| args)).unwrap();

    let err =
        register_tool_sanitize_request_guardrail("duplicate", 2, Arc::new(|_name, args| args));

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
#[test]
fn test_duplicate_intercept_registration_returns_error() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    register_tool_request_intercept("dup_intercept", 1, false, Arc::new(|_name, args| Ok(args)))
        .unwrap();

    let err = register_tool_request_intercept(
        "dup_intercept",
        2,
        false,
        Arc::new(|_name, args| Ok(args)),
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
#[test]
fn test_deregister_nonexistent_returns_false() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let result = deregister_tool_sanitize_request_guardrail("nonexistent").unwrap();
    assert!(
        !result,
        "Deregistering non-existent entry should return false"
    );
}

/// Deregistering removes the guardrail from the chain.
#[test]
fn test_deregister_removes_from_chain() {
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
            args
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
        Arc::new(|_req| Ok(Some("LLM call rejected".to_string()))),
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

    register_llm_conditional_execution_guardrail("llm_scope_allow", 1, Arc::new(|_| Ok(None)))
        .unwrap();
    register_llm_conditional_execution_guardrail(
        "llm_scope_reject",
        2,
        Arc::new(|_| Ok(Some("blocked by llm guardrail".to_string()))),
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
        Arc::new(|_name: &str, mut req: LlmRequest, annotated| {
            req.headers.insert("x-intercepted".into(), json!(true));
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
        }),
    )
    .unwrap();

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let result = llm_request_intercepts("test_llm", request).unwrap();
    assert_eq!(result.request.headers["x-intercepted"], true);

    // Cleanup
    deregister_llm_request_intercept("llm_req_i").unwrap();
}

#[test]
fn test_llm_request_intercept_pending_marks_preserve_order_and_break_chain() {
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
                Ok(LlmRequestInterceptOutcome::new(request, annotated)
                    .with_pending_mark(PendingMarkSpec::builder().name(mark_name).build()))
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
            fields.metadata = Some(json!({"sanitized_mark": event.name()}));
            fields
        }),
    )
    .unwrap();
    register_llm_request_intercept(
        "pending_managed",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Ok(LlmRequestInterceptOutcome::new(request, annotated)
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
                        .build(),
                )
                .with_pending_mark(
                    PendingMarkSpec::builder()
                        .name("request.optimized.second")
                        .build(),
                ))
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
            fields
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
            fields
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
            Ok(LlmRequestInterceptOutcome::new(request, annotated)
                .with_optimization_contribution(contribution))
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
            fields
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
                    Ok(
                        Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": "done"}))]))
                            as LlmJsonStream,
                    )
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
            Ok(LlmRequestInterceptOutcome::new(request, annotated)
                .with_optimization_contribution(contribution))
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
            Ok(LlmRequestInterceptOutcome::new(request, annotated)
                .with_pending_mark(PendingMarkSpec::builder().name("must.not.emit").build()))
        }),
    )
    .unwrap();
    register_llm_request_intercept(
        "pending_failure",
        2,
        false,
        Arc::new(|_name, _request, _annotated| {
            Err(FlowError::Internal("request intercept failed".into()))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
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
                Ok(Box::pin(stream) as LlmJsonStream)
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
            Ok(Box::pin(stream) as LlmJsonStream)
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
#[test]
fn test_standalone_conditional_execution_passes() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let result = tool_conditional_execution("tool", &json!({}));
    assert!(result.is_ok(), "No guardrails means no rejection");
}

/// tool_conditional_execution returns GuardrailRejected when a guardrail rejects.
#[test]
fn test_standalone_conditional_execution_rejects() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_tool_conditional_execution_guardrail(
        "standalone_gate",
        1,
        Arc::new(|_name, _args| Ok(Some("rejected by standalone".to_string()))),
    )
    .unwrap();

    let result = tool_conditional_execution("tool", &json!({}));
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

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

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
        result["value"], "unchanged",
        "Data should pass through unmodified"
    );
}

/// Standalone intercept chain with no registrations returns input unchanged.
#[test]
fn test_empty_request_intercept_chain() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let result = tool_request_intercepts("tool", json!({"key": "val"})).unwrap();
    assert_eq!(result["key"], "val");
}
