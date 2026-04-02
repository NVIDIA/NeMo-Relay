// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the scope-local middleware registry feature.
//!
//! These tests verify that guardrails, intercepts, and subscribers registered on
//! specific scopes execute correctly, are cleaned up on scope pop, merge properly
//! with global registrations, and remain isolated across concurrent scope stacks.

#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nvidia_nat_nexus_core::context::*;
use nvidia_nat_nexus_core::error::NexusError;
use nvidia_nat_nexus_core::types::*;
use nvidia_nat_nexus_core::*;
use serde_json::json;

// All tests share the global context, so we serialize them.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NatNexusContextState::new();
}

/// Helper: create a fresh scope stack on the current thread and push a scope,
/// returning the scope handle.
fn setup_isolated_scope(name: &str) -> ScopeHandle {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
    nat_nexus_push_scope(
        name,
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap()
}

// -----------------------------------------------------------------------
// 1. Scope-local guardrail registration and execution
//
// Registers a scope-local tool sanitize request guardrail and verifies it
// runs during nat_nexus_tool_call within that scope by inspecting the
// event's `input` field (sanitize guardrails transform what is recorded
// in events, not the execution-pipeline args).
// -----------------------------------------------------------------------

#[test]
fn test_scope_local_guardrail_registration_and_execution() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("scope_guardrail");

    // Register a scope-local tool sanitize request guardrail that adds a marker.
    nat_nexus_scope_register_tool_sanitize_request_guardrail(
        &handle.uuid,
        "local_sanitizer",
        10,
        Box::new(|_name, mut args| {
            args.as_object_mut()
                .unwrap()
                .insert("scope_sanitized".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Capture events via a global subscriber to inspect the input field.
    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    nat_nexus_register_subscriber(
        "sanitize_observer",
        Box::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    // Invoke nat_nexus_tool_call — the sanitize guardrail runs inside.
    let tool_handle = nat_nexus_tool_call(
        "test_tool",
        json!({"input": "data"}),
        None,
        ToolAttributes::empty(),
        None,
        None,
        None,
    )
    .unwrap();

    // The Start event's input should contain the sanitized args.
    {
        let captured = events.lock().unwrap();
        let start_event = &captured[0];
        let input = start_event.input.as_ref().unwrap();
        assert_eq!(input["scope_sanitized"], true);
        assert_eq!(input["input"], "data");
    }

    nat_nexus_tool_call_end(&tool_handle, json!("ok"), None, None).unwrap();

    // Cleanup
    nat_nexus_deregister_subscriber("sanitize_observer").unwrap();
    nat_nexus_pop_scope(&handle.uuid).unwrap();
}

// -----------------------------------------------------------------------
// 2. Auto-cleanup on scope pop
//
// Registers a scope-local request intercept (which transforms execution
// args), pops the scope, and verifies the intercept no longer runs.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_auto_cleanup_on_scope_pop() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);

    let handle = nat_nexus_push_scope(
        "ephemeral",
        ScopeType::Function,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();

    // Register a scope-local request intercept that appends a field.
    nat_nexus_scope_register_tool_request_intercept(
        &handle.uuid,
        "ephemeral_intercept",
        1,
        false,
        Box::new(|_name, mut args| {
            args.as_object_mut()
                .unwrap()
                .insert("ephemeral".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Verify it runs before pop.
    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = nat_nexus_tool_call_execute(
        "tool",
        json!({"v": 1}),
        func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(result["ephemeral"], true);

    // Pop the scope — middleware should be cleaned up.
    nat_nexus_pop_scope(&handle.uuid).unwrap();

    // Now execute again — the field should NOT appear.
    let func2: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result2 = nat_nexus_tool_call_execute(
        "tool",
        json!({"v": 2}),
        func2,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert!(result2.get("ephemeral").is_none());
    assert_eq!(result2["v"], 2);
}

// -----------------------------------------------------------------------
// 3. Priority merge across global + scope-local
//
// Registers global request intercepts at priorities 10 and 30, and a
// scope-local request intercept at priority 20. Verifies they execute
// in ascending priority order: 10, 20, 30.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_priority_merge_global_and_scope_local() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("merge_test");

    let order = Arc::new(Mutex::new(Vec::<i32>::new()));

    // Global intercept at priority 10
    let o1 = order.clone();
    nat_nexus_register_tool_request_intercept(
        "global_p10",
        10,
        false,
        Box::new(move |_name, mut args| {
            o1.lock().unwrap().push(10);
            args.as_object_mut()
                .unwrap()
                .insert("p10".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Global intercept at priority 30
    let o3 = order.clone();
    nat_nexus_register_tool_request_intercept(
        "global_p30",
        30,
        false,
        Box::new(move |_name, mut args| {
            o3.lock().unwrap().push(30);
            args.as_object_mut()
                .unwrap()
                .insert("p30".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Scope-local intercept at priority 20
    let o2 = order.clone();
    nat_nexus_scope_register_tool_request_intercept(
        &handle.uuid,
        "local_p20",
        20,
        false,
        Box::new(move |_name, mut args| {
            o2.lock().unwrap().push(20);
            args.as_object_mut()
                .unwrap()
                .insert("p20".into(), json!(true));
            args
        }),
    )
    .unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = nat_nexus_tool_call_execute(
        "tool",
        json!({}),
        func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();

    // All three intercepts ran.
    assert_eq!(result["p10"], true);
    assert_eq!(result["p20"], true);
    assert_eq!(result["p30"], true);

    // Verify execution order: 10, 20, 30.
    let recorded = order.lock().unwrap();
    assert_eq!(*recorded, vec![10, 20, 30]);

    // Cleanup
    nat_nexus_deregister_tool_request_intercept("global_p10").unwrap();
    nat_nexus_deregister_tool_request_intercept("global_p30").unwrap();
    nat_nexus_pop_scope(&handle.uuid).unwrap();
}

// -----------------------------------------------------------------------
// 4. Name coexistence — same name in global and scope-local
//
// Registers a sanitize guardrail with the same name in both global and
// scope-local registries. Verifies both run (names are namespaced by
// registry, so no collision).
// -----------------------------------------------------------------------

#[test]
fn test_name_coexistence_global_and_scope_local() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("coexist_test");

    let count = Arc::new(AtomicU32::new(0));

    // Global guardrail named "shared_name"
    let c1 = count.clone();
    nat_nexus_register_tool_sanitize_request_guardrail(
        "shared_name",
        1,
        Box::new(move |_name, args| {
            c1.fetch_add(1, Ordering::SeqCst);
            args
        }),
    )
    .unwrap();

    // Scope-local guardrail also named "shared_name"
    let c2 = count.clone();
    nat_nexus_scope_register_tool_sanitize_request_guardrail(
        &handle.uuid,
        "shared_name",
        2,
        Box::new(move |_name, args| {
            c2.fetch_add(1, Ordering::SeqCst);
            args
        }),
    )
    .unwrap();

    // Use nat_nexus_tool_call which exercises sanitize guardrails.
    let _tool_handle = nat_nexus_tool_call(
        "tool",
        json!({}),
        None,
        ToolAttributes::empty(),
        None,
        None,
        None,
    )
    .unwrap();

    // Both guardrails with the same name ran.
    assert_eq!(count.load(Ordering::SeqCst), 2);

    // Cleanup
    nat_nexus_deregister_tool_sanitize_request_guardrail("shared_name").unwrap();
    nat_nexus_pop_scope(&handle.uuid).unwrap();
}

// -----------------------------------------------------------------------
// 5. Scope isolation — two concurrent scope stacks
//
// Two separate scope stacks each with different scope-local request
// intercepts. Verifies no cross-contamination between stacks.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_scope_isolation_between_stacks() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let stack_a = create_scope_stack();
    let stack_b = create_scope_stack();

    // Set up stack A with a scope-local intercept that adds "agent_a"
    let scope_a = {
        set_thread_scope_stack(stack_a.clone());
        let s = nat_nexus_push_scope(
            "agent_a",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        nat_nexus_scope_register_tool_request_intercept(
            &s.uuid,
            "a_intercept",
            1,
            false,
            Box::new(|_name, mut args| {
                args.as_object_mut()
                    .unwrap()
                    .insert("agent".into(), json!("a"));
                args
            }),
        )
        .unwrap();
        s
    };

    // Set up stack B with a scope-local intercept that adds "agent_b"
    let scope_b = {
        set_thread_scope_stack(stack_b.clone());
        let s = nat_nexus_push_scope(
            "agent_b",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        nat_nexus_scope_register_tool_request_intercept(
            &s.uuid,
            "b_intercept",
            1,
            false,
            Box::new(|_name, mut args| {
                args.as_object_mut()
                    .unwrap()
                    .insert("agent".into(), json!("b"));
                args
            }),
        )
        .unwrap();
        s
    };

    // Execute on stack A — should see agent_a's intercept only
    set_thread_scope_stack(stack_a.clone());
    let func_a: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result_a = nat_nexus_tool_call_execute(
        "tool",
        json!({}),
        func_a,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(result_a["agent"], "a");

    // Execute on stack B — should see agent_b's intercept only
    set_thread_scope_stack(stack_b.clone());
    let func_b: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result_b = nat_nexus_tool_call_execute(
        "tool",
        json!({}),
        func_b,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(result_b["agent"], "b");

    // Cleanup
    set_thread_scope_stack(stack_a);
    nat_nexus_pop_scope(&scope_a.uuid).unwrap();
    set_thread_scope_stack(stack_b);
    nat_nexus_pop_scope(&scope_b.uuid).unwrap();
}

// -----------------------------------------------------------------------
// 6. Nested scope inheritance
//
// Pushes scope A with middleware, then child scope B with its own
// middleware. Verifies a call within B sees both A's and B's scope-local
// middleware plus global.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_nested_scope_inheritance() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);

    let order = Arc::new(Mutex::new(Vec::<String>::new()));

    // Global request intercept
    let og = order.clone();
    nat_nexus_register_tool_request_intercept(
        "global_intercept",
        1,
        false,
        Box::new(move |_name, mut args| {
            og.lock().unwrap().push("global".into());
            args.as_object_mut()
                .unwrap()
                .insert("global".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Push scope A with its own request intercept
    let scope_a = nat_nexus_push_scope(
        "scope_a",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();
    let oa = order.clone();
    nat_nexus_scope_register_tool_request_intercept(
        &scope_a.uuid,
        "a_intercept",
        5,
        false,
        Box::new(move |_name, mut args| {
            oa.lock().unwrap().push("scope_a".into());
            args.as_object_mut()
                .unwrap()
                .insert("scope_a".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Push child scope B with its own request intercept
    let scope_b = nat_nexus_push_scope(
        "scope_b",
        ScopeType::Function,
        Some(&scope_a),
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();
    let ob = order.clone();
    nat_nexus_scope_register_tool_request_intercept(
        &scope_b.uuid,
        "b_intercept",
        10,
        false,
        Box::new(move |_name, mut args| {
            ob.lock().unwrap().push("scope_b".into());
            args.as_object_mut()
                .unwrap()
                .insert("scope_b".into(), json!(true));
            args
        }),
    )
    .unwrap();

    // Execute within scope B — should see global + scope_a + scope_b
    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = nat_nexus_tool_call_execute(
        "tool",
        json!({}),
        func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(result["global"], true);
    assert_eq!(result["scope_a"], true);
    assert_eq!(result["scope_b"], true);

    // Verify all three ran in priority order: 1 (global), 5 (a), 10 (b)
    let recorded = order.lock().unwrap();
    assert_eq!(*recorded, vec!["global", "scope_a", "scope_b"]);

    // Cleanup
    nat_nexus_pop_scope(&scope_b.uuid).unwrap();
    nat_nexus_pop_scope(&scope_a.uuid).unwrap();
    nat_nexus_deregister_tool_request_intercept("global_intercept").unwrap();
}

// -----------------------------------------------------------------------
// 7. Scope-local subscriber
//
// Registers a scope-local event subscriber, verifies it receives events
// for operations within that scope, and stops receiving after scope pop.
// -----------------------------------------------------------------------

#[test]
fn test_scope_local_subscriber() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("sub_scope");

    let events = Arc::new(Mutex::new(Vec::<EventType>::new()));
    let ec = events.clone();
    nat_nexus_scope_register_subscriber(
        &handle.uuid,
        "local_sub",
        Box::new(move |e: &Event| {
            ec.lock().unwrap().push(e.event_type);
        }),
    )
    .unwrap();

    // Push a child scope — this emits a Start event
    let child = nat_nexus_push_scope(
        "child",
        ScopeType::Function,
        Some(&handle),
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();

    // Pop the child — emits End event
    nat_nexus_pop_scope(&child.uuid).unwrap();

    {
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], EventType::Start);
        assert_eq!(captured[1], EventType::End);
    }

    // Pop the scope that owns the subscriber.
    // The End event for this scope is emitted *before* removal, so the
    // scope-local subscriber sees its own scope's End event as well.
    nat_nexus_pop_scope(&handle.uuid).unwrap();

    {
        let captured = events.lock().unwrap();
        // 3 events: Start(child), End(child), End(handle)
        assert_eq!(captured.len(), 3);
        assert_eq!(captured[2], EventType::End);
    }

    // After pop, push another scope — the subscriber should NOT fire
    let another = nat_nexus_push_scope(
        "after_pop",
        ScopeType::Function,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();
    nat_nexus_pop_scope(&another.uuid).unwrap();

    let captured2 = events.lock().unwrap();
    // Still only 3 events (the subscriber was cleaned up with the scope)
    assert_eq!(captured2.len(), 3);
}

// -----------------------------------------------------------------------
// 8. Scope-local conditional execution guardrail
//
// Registers a scope-local conditional guardrail that rejects calls to
// a specific tool. Verifies rejection works and other tools are allowed.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_scope_local_conditional_execution_guardrail() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    let handle = setup_isolated_scope("cond_scope");

    // Register a scope-local conditional guardrail that rejects "banned_tool"
    nat_nexus_scope_register_tool_conditional_execution_guardrail(
        &handle.uuid,
        "tool_blocker",
        1,
        Box::new(|name, _args| {
            if name == "banned_tool" {
                Some("banned_tool is not allowed in this scope".to_string())
            } else {
                None
            }
        }),
    )
    .unwrap();

    // Call to banned_tool should be rejected
    let func_banned: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let err = nat_nexus_tool_call_execute(
        "banned_tool",
        json!({"input": 1}),
        func_banned,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await;

    assert!(err.is_err());
    match err.unwrap_err() {
        NexusError::GuardrailRejected(reason) => {
            assert!(reason.contains("banned_tool is not allowed"));
        }
        other => panic!("Expected GuardrailRejected, got: {:?}", other),
    }

    // Call to a different tool should succeed
    let func_ok: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let result = nat_nexus_tool_call_execute(
        "allowed_tool",
        json!({"input": 2}),
        func_ok,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(result["input"], 2);

    nat_nexus_pop_scope(&handle.uuid).unwrap();
}
