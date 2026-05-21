// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive middleware chain tests for the NeMo Flow core runtime.
//!
//! These tests exercise the middleware pipeline mechanics: priority ordering,
//! break_chain short-circuiting, execution intercept middleware chains (next()),
//! conditional execution guardrails, scope-local middleware lifecycle, global +
//! scope-local merging, error propagation, and concurrent mutations.

#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nemo_flow::api::event::{Event, ScopeCategory};
use nemo_flow::api::llm::LlmRequest;
use nemo_flow::api::llm::{LlmCallExecuteParams, llm_call_execute, llm_request_intercepts};
use nemo_flow::api::registry::{
    deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
    deregister_llm_request_intercept, deregister_tool_conditional_execution_guardrail,
    deregister_tool_execution_intercept, deregister_tool_request_intercept,
    deregister_tool_sanitize_request_guardrail, deregister_tool_sanitize_response_guardrail,
    register_llm_conditional_execution_guardrail, register_llm_execution_intercept,
    register_llm_request_intercept, register_tool_conditional_execution_guardrail,
    register_tool_execution_intercept, register_tool_request_intercept,
    register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
    scope_register_tool_execution_intercept, scope_register_tool_sanitize_request_guardrail,
};
use nemo_flow::api::runtime::NemoFlowContextState;
use nemo_flow::api::runtime::global_context;
use nemo_flow::api::runtime::{LlmExecutionNextFn, ToolExecutionNextFn};
use nemo_flow::api::runtime::{create_scope_stack, set_thread_scope_stack};
use nemo_flow::api::scope::{ScopeHandle, ScopeType};
use nemo_flow::api::scope::{pop_scope, push_scope};
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_flow::api::tool::{
    tool_call, tool_call_end, tool_call_execute, tool_conditional_execution,
    tool_request_intercepts,
};
use nemo_flow::error::FlowError;
use serde_json::json;

// All tests share the global context, so we serialize them.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn is_scope_event(event: &Event, scope_type: ScopeType, scope_category: ScopeCategory) -> bool {
    event.scope_type() == Some(scope_type) && event.scope_category() == Some(scope_category)
}

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoFlowContextState::new();
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
        nemo_flow::api::scope::PushScopeParams::builder()
            .name(name)
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap()
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
            o2.lock().unwrap().push(2);
            args
        }),
    )
    .unwrap();

    // Trigger the chain via tool_call (which runs sanitize request guardrails)
    let _handle = tool_call(
        nemo_flow::api::tool::ToolCallParams::builder()
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
        Box::new(|_name, mut args| {
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
        Box::new(move |_name, mut args| {
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
        Box::new(move |_name, args| {
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
        Box::new(move |_name, args| {
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
                next(args).await
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
                Ok(json!({"intercepted": true}))
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
                result
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
                result
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
                next(args).await
            })
        }),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

    let result = tool_call_execute(
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
    assert!(global_context().read().unwrap().extensions.is_empty());

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let allowed = tool_call_execute(
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"safe": true}))
            .func(func.clone())
            .build(),
    )
    .await;
    assert!(allowed.is_err(), "second guardrail should reject");

    deregister_tool_conditional_execution_guardrail("tool_scope_reject").unwrap();
    let allowed = tool_call_execute(
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"safe": true}))
            .func(func)
            .build(),
    )
    .await;
    assert!(allowed.is_ok());

    deregister_tool_conditional_execution_guardrail("tool_scope_allow").unwrap();
    deregister_subscriber("tool_guardrail_scope_capture").unwrap();

    let events = events.lock().unwrap();
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        Box::new(move |_name, args| {
            cc.fetch_add(1, Ordering::SeqCst);
            args
        }),
    )
    .unwrap();

    // Invoke tool call -- guardrail should fire
    let _tool = tool_call(
        nemo_flow::api::tool::ToolCallParams::builder()
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
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();

    // Invoke tool call again -- guardrail should NOT fire
    let _tool2 = tool_call(
        nemo_flow::api::tool::ToolCallParams::builder()
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
            Box::pin(async move { next(args).await })
        }),
    )
    .unwrap();

    // Execute -- intercept should fire
    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let _ = tool_call_execute(
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();

    // Execute again -- intercept should NOT fire
    let func2: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let _ = tool_call_execute(
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        Box::new(move |_name, mut args| {
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
        Box::new(move |_name, mut args| {
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
        nemo_flow::api::tool::ToolCallParams::builder()
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
    let captured = events.lock().unwrap();
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
        nemo_flow::api::scope::PopScopeParams::builder()
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
                r
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
                r
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        nemo_flow::api::scope::PopScopeParams::builder()
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
        Box::new(move |_name, args| {
            ic.store(true, Ordering::SeqCst);
            Ok(args)
        }),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = tool_call_execute(
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
            Box::pin(async move { next(args).await })
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
        Box::new(|_name, mut args| {
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
        Box::new(|_name, mut args| {
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
        nemo_flow::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();

    let captured = events.lock().unwrap();
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
        Box::new(|_name, mut result| {
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
        nemo_flow::api::tool::ToolCallParams::builder()
            .name("tool")
            .args(json!({}))
            .build(),
    )
    .unwrap();

    tool_call_end(
        nemo_flow::api::tool::ToolCallEndParams::builder()
            .handle(&tool_handle)
            .result(json!({"raw": true}))
            .build(),
    )
    .unwrap();

    let captured = events.lock().unwrap();
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
                    Box::new(|_name, args| args),
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

    // Verify the context is still in a consistent state
    let ctx = global_context();
    let state = ctx.read().unwrap();
    assert!(
        state
            .tool_sanitize_request_guardrails
            .sorted_values()
            .is_empty(),
        "All guardrails should be deregistered"
    );
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
                    Box::new(|_name, args| Ok(args)),
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

    let ctx = global_context();
    let state = ctx.read().unwrap();
    assert!(
        state.tool_request_intercepts.sorted_values().is_empty(),
        "All intercepts should be deregistered"
    );
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
            Box::new(|_name, args| args),
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
                        Box::new(|_name, args| args),
                    );
                    std::thread::yield_now();
                    let _ = deregister_tool_sanitize_request_guardrail(&name);
                } else {
                    // Reader threads: set up scope stack and do tool calls
                    let stack = create_scope_stack();
                    set_thread_scope_stack(stack);
                    let _ = tool_call(
                        nemo_flow::api::tool::ToolCallParams::builder()
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
        Box::new(move |_name, mut args| {
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
        Box::new(move |_name, args| {
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
                next(args).await
            })
        }),
    )
    .unwrap();

    // Sanitize response guardrail
    let o5 = order.clone();
    register_tool_sanitize_response_guardrail(
        "sanitize_resp",
        1,
        Box::new(move |_name, result| {
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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

    register_tool_sanitize_request_guardrail("duplicate", 1, Box::new(|_name, args| args)).unwrap();

    let err =
        register_tool_sanitize_request_guardrail("duplicate", 2, Box::new(|_name, args| args));

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

    register_tool_request_intercept("dup_intercept", 1, false, Box::new(|_name, args| Ok(args)))
        .unwrap();

    let err = register_tool_request_intercept(
        "dup_intercept",
        2,
        false,
        Box::new(|_name, args| Ok(args)),
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
        Box::new(move |_name, args| {
            cc.fetch_add(1, Ordering::SeqCst);
            args
        }),
    )
    .unwrap();

    // First call -- guardrail runs
    let _ = tool_call(
        nemo_flow::api::tool::ToolCallParams::builder()
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
        nemo_flow::api::tool::ToolCallParams::builder()
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

    let events = events.lock().unwrap();
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
        Box::new(|_name: &str, mut req: LlmRequest, annotated| {
            req.headers.insert("x-intercepted".into(), json!(true));
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"prompt": "hello"}),
    };

    let result = llm_request_intercepts("test_llm", request).unwrap();
    assert_eq!(result.headers["x-intercepted"], true);

    // Cleanup
    deregister_llm_request_intercept("llm_req_i").unwrap();
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
        nemo_flow::api::tool::ToolCallExecuteParams::builder()
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
