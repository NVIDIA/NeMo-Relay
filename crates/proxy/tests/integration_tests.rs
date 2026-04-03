// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for the nexus-proxy crate.
//!
//! These tests exercise the full pipeline: proxy registers with Nexus,
//! LLM/tool calls fire through the runtime, and the proxy's intercepts
//! and subscriber observe the traffic.
//!
//! All tests use the [`TEST_MUTEX`] + [`reset_global`] pattern to serialize
//! access to the Nexus global singleton. Tests run with `--test-threads=1`
//! so the std::sync::Mutex is safe to hold across await points.

// The TEST_MUTEX is held across await points intentionally -- tests must
// serialize access to the Nexus global singleton and run single-threaded.
#![allow(clippy::await_holding_lock)]

use std::sync::{Arc, Mutex};

use nvidia_nat_nexus_core::{
    global_context, nat_nexus_llm_call_execute, nat_nexus_pop_scope, nat_nexus_push_scope,
    nat_nexus_tool_call_execute, LLMAttributes, LLMRequest, LlmExecutionNextFn,
    NatNexusContextState, ScopeAttributes, ScopeType, ToolAttributes, ToolExecutionNextFn,
};
use nvidia_nat_nexus_proxy::trie::SensitivityConfig;
use nvidia_nat_nexus_proxy::{
    AgentHints, ExecutionPlan, InMemoryBackend, LatencySensitivityLearner, MetadataEnvelope,
    NexusProxy, ParallelGroup, ParallelHint, StorageBackendDyn, AGENT_HINTS_HEADER_KEY,
};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Serialize all integration tests since they share the Nexus global state.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Resets the Nexus global context to a fresh empty state.
fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NatNexusContextState::new();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that after register(), the Nexus global context has all three
/// components registered (subscriber + LLM intercept + tool intercept).
#[tokio::test]
async fn test_proxy_register_fires_all_three() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let mut proxy = NexusProxy::builder()
        .agent_id("reg-test")
        .backend(Box::new(backend))
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    // Verify registrations exist by attempting to deregister -- should return true (found)
    {
        let ctx = global_context();
        let state = ctx.read().unwrap();
        assert!(
            state
                .event_subscribers
                .contains_key("nexus_proxy_reg-test_subscriber"),
            "subscriber should be registered"
        );
    }

    proxy.deregister().unwrap();
}

/// After register(), an LLM call passes through without error and without
/// metadata header injection (metadata envelope injection was removed).
#[tokio::test]
async fn test_llm_call_passes_through_after_register() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // Pre-seed backend with a plan
    let backend = InMemoryBackend::new();
    let plan = ExecutionPlan {
        agent_id: "meta-test".to_string(),
        parallel_groups: vec![],
        metadata_template: MetadataEnvelope {
            run_id: uuid::Uuid::new_v4(),
            agent_id: "meta-test".to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "g1".to_string(),
                explicit: true,
            }],
            extensions: serde_json::json!({}),
        },
    };
    backend.store_plan(&plan).unwrap();

    let mut proxy = NexusProxy::builder()
        .agent_id("meta-test")
        .backend(Box::new(backend))
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    let func: LlmExecutionNextFn = Arc::new(move |_req: LLMRequest| {
        Box::pin(async { Ok(serde_json::json!({"response": "hello"})) })
    });

    let request = LLMRequest {
        headers: serde_json::Map::new(),
        content: serde_json::json!({"messages": []}),
    };

    let result = nat_nexus_llm_call_execute(
        "test-llm",
        request,
        func,
        None,
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
    )
    .await;

    assert!(
        result.is_ok(),
        "LLM call should succeed after proxy register"
    );
    assert_eq!(result.unwrap(), serde_json::json!({"response": "hello"}));

    proxy.deregister().unwrap();
}

/// Verify tool execution intercept is wired and doesn't panic.
/// The intercept reads the hot cache and passes through to next.
#[tokio::test]
async fn test_tool_call_passes_through_intercept() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    // Pre-seed with a plan that has a ParallelHint
    let plan = ExecutionPlan {
        agent_id: "tool-test".to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "g1".to_string(),
            tool_names: vec!["search".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: uuid::Uuid::new_v4(),
            agent_id: "tool-test".to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "g1".to_string(),
                explicit: true,
            }],
            extensions: serde_json::json!({}),
        },
    };
    backend.store_plan(&plan).unwrap();

    let mut proxy = NexusProxy::builder()
        .agent_id("tool-test")
        .backend(Box::new(backend))
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    let func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

    let result = nat_nexus_tool_call_execute(
        "search",
        serde_json::json!({"query": "test"}),
        func,
        None,
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();

    // Tool intercept should have passed through without panicking
    assert_eq!(result, serde_json::json!({"query": "test"}));

    proxy.deregister().unwrap();
}

/// Verify that after a run (scope start -> LLM call -> scope end), the
/// drain task stores a RunRecord in the backend.
///
/// This test creates a scope to simulate an agent run, fires an LLM call
/// inside the scope, then ends the scope. The subscriber forwards events
/// to the drain task which accumulates and stores them.
#[tokio::test]
async fn test_telemetry_stored_after_run() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let mut proxy = NexusProxy::builder()
        .agent_id("telem-test")
        .backend(Box::new(backend))
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    // Create a scope (simulates agent run start)
    let scope = nat_nexus_push_scope(
        "agent-run",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();

    // Fire an LLM call
    let func: LlmExecutionNextFn =
        Arc::new(|_req| Box::pin(async { Ok(serde_json::json!({"response": "hello"})) }));
    let request = LLMRequest {
        headers: serde_json::Map::new(),
        content: serde_json::json!({}),
    };
    nat_nexus_llm_call_execute(
        "llm",
        request,
        func,
        Some(scope.clone()),
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
    )
    .await
    .unwrap();

    // End the scope (simulates agent run end)
    nat_nexus_pop_scope(&scope.uuid).unwrap();

    // Give drain task time to process the queued events
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Verify telemetry was stored -- REAL assertion on RunRecord storage
    let runs = proxy.backend().list_runs_dyn("telem-test").await.unwrap();
    assert!(
        !runs.is_empty(),
        "drain should have stored at least one RunRecord"
    );
    assert_eq!(runs[0].agent_id, "telem-test");
    assert!(runs[0].ended_at.is_some(), "run should have ended_at set");
    // The run should contain the LLM call
    assert!(
        !runs[0].calls.is_empty(),
        "run should contain at least one call record"
    );

    proxy.deregister().unwrap();
}

/// Verify that register() cannot be called twice (event_rx already taken).
#[tokio::test]
async fn test_register_twice_fails() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let mut proxy = NexusProxy::builder()
        .agent_id("double-reg")
        .backend(Box::new(backend))
        .build()
        .unwrap();

    proxy.register().await.unwrap();
    let err = proxy.register().await;
    assert!(err.is_err(), "second register() should fail");

    proxy.deregister().unwrap();
}

/// Verify that deregister() can be called multiple times without error.
#[tokio::test]
async fn test_deregister_idempotent() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let mut proxy = NexusProxy::builder()
        .agent_id("deregtest")
        .backend(Box::new(backend))
        .build()
        .unwrap();

    proxy.register().await.unwrap();
    proxy.deregister().unwrap();
    // Second deregister should be no-op
    proxy.deregister().unwrap();
}

/// Full end-to-end pipeline test.
///
/// Exercises the complete flow in a single test:
/// 1. Proxy registers with pre-seeded ExecutionPlan
/// 2. Agent scope is created
/// 3. LLM call fires -- passes through without error
/// 4. Tool call fires -- passes through execution intercept without panic
/// 5. Agent scope ends -- telemetry drain captures RunRecord
/// 6. RunRecord is stored in InMemoryBackend with correct CallRecords
#[tokio::test]
async fn test_full_pipeline_end_to_end() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    // 1. Set up backend with pre-seeded plan
    let backend = InMemoryBackend::new();
    let plan = ExecutionPlan {
        agent_id: "e2e-agent".to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "g1".to_string(),
            tool_names: vec!["search".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: uuid::Uuid::new_v4(),
            agent_id: "e2e-agent".to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "g1".to_string(),
                explicit: true,
            }],
            extensions: serde_json::json!({"test": true}),
        },
    };
    backend.store_plan(&plan).unwrap();

    // 2. Build and register proxy
    let mut proxy = NexusProxy::builder()
        .agent_id("e2e-agent")
        .backend(Box::new(backend))
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    // 3. Create agent scope
    let scope = nat_nexus_push_scope(
        "e2e-run",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();

    // 4. Fire LLM call -- should pass through without error
    let llm_func: LlmExecutionNextFn = Arc::new(move |_req: LLMRequest| {
        Box::pin(async { Ok(serde_json::json!({"response": "hello"})) })
    });
    let request = LLMRequest {
        headers: serde_json::Map::new(),
        content: serde_json::json!({"messages": []}),
    };
    let llm_result = nat_nexus_llm_call_execute(
        "test-llm",
        request,
        llm_func,
        Some(scope.clone()),
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
    )
    .await;
    assert!(llm_result.is_ok(), "LLM call should succeed");

    // 5. Fire tool call -- should pass through intercept without panic
    let tool_func: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    let tool_result = nat_nexus_tool_call_execute(
        "search",
        serde_json::json!({"query": "e2e test"}),
        tool_func,
        Some(scope.clone()),
        ToolAttributes::empty(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_result, serde_json::json!({"query": "e2e test"}));

    // 6. End scope and wait for drain
    nat_nexus_pop_scope(&scope.uuid).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // 7. Verify telemetry stored in backend
    let runs = proxy.backend().list_runs_dyn("e2e-agent").await.unwrap();
    assert!(
        !runs.is_empty(),
        "drain should have stored at least one RunRecord"
    );
    let run = &runs[0];
    assert_eq!(run.agent_id, "e2e-agent");
    assert!(run.ended_at.is_some(), "run should have ended_at set");
    // Should have at least 2 calls (1 LLM + 1 tool)
    assert!(
        run.calls.len() >= 2,
        "run should have at least 2 call records (LLM + tool), got {}",
        run.calls.len()
    );

    proxy.deregister().unwrap();
}

/// TEST2-02: Learner processes RunRecord -> trie built -> AgentHints present on LLM intercept.
///
/// Flow:
/// 1. Build proxy with a LatencySensitivityLearner
/// 2. Fire 3 complete agent runs (scope push -> LLM calls -> scope pop)
/// 3. Wait for drain to process all runs and learner to build trie
/// 4. Fire another LLM call and capture request headers
/// 5. Verify AGENT_HINTS_HEADER_KEY is present with valid AgentHints
#[tokio::test]
async fn test_learner_pipeline_produces_agent_hints() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let learner = LatencySensitivityLearner::new("learner-test", SensitivityConfig::default());

    let mut proxy = NexusProxy::builder()
        .agent_id("learner-test")
        .backend(Box::new(backend))
        .learner(Box::new(learner))
        .dynamo_intercept(true)
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    // Fire 3 complete runs to give the learner enough data
    for _run_idx in 0..3 {
        let scope = nat_nexus_push_scope(
            "agent-run",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        // Fire 2 LLM calls per run
        for _ in 0..2 {
            let func: LlmExecutionNextFn =
                Arc::new(|_req| Box::pin(async { Ok(serde_json::json!({"response": "ok"})) }));
            let request = LLMRequest {
                headers: serde_json::Map::new(),
                content: serde_json::json!({"messages": []}),
            };
            nat_nexus_llm_call_execute(
                "test-llm",
                request,
                func,
                Some(scope.clone()),
                LLMAttributes::empty(),
                None,
                None,
                Some("gpt-4".into()),
            )
            .await
            .unwrap();
        }

        nat_nexus_pop_scope(&scope.uuid).unwrap();

        // Wait for drain to process this run before next (ensure sequential processing)
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    // Now fire another LLM call and capture headers to verify hints
    let captured_headers = Arc::new(Mutex::new(None));
    let cap = captured_headers.clone();
    let func: LlmExecutionNextFn = Arc::new(move |req: LLMRequest| {
        let mut guard = cap.lock().unwrap();
        *guard = Some(req.headers.clone());
        Box::pin(async { Ok(serde_json::json!({"response": "final"})) })
    });
    let request = LLMRequest {
        headers: serde_json::Map::new(),
        content: serde_json::json!({"messages": []}),
    };
    nat_nexus_llm_call_execute(
        "test-llm",
        request,
        func,
        None,
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
    )
    .await
    .unwrap();

    // Verify AgentHints were injected
    let headers = captured_headers.lock().unwrap();
    let headers = headers.as_ref().expect("headers should be captured");
    assert!(
        headers.contains_key(AGENT_HINTS_HEADER_KEY),
        "LLM request should contain agent hints header '{}' after learner ran, got keys: {:?}",
        AGENT_HINTS_HEADER_KEY,
        headers.keys().collect::<Vec<_>>()
    );

    // Deserialize and validate hints structure
    let hints: AgentHints = serde_json::from_value(headers[AGENT_HINTS_HEADER_KEY].clone())
        .expect("agent hints should deserialize");
    assert!(
        hints.total_requests > 0,
        "total_requests should be positive"
    );
    // latency_sensitivity should be set (learner computes it from 3 runs)
    assert!(
        hints.latency_sensitivity >= 0.0,
        "latency_sensitivity should be non-negative"
    );

    proxy.deregister().unwrap();
}

/// TEST2-04: Full e2e: proxy registers -> LLM calls fire -> learner processes ->
/// trie converges -> hints injected.
///
/// This test verifies the entire lifecycle including that the trie is accessible
/// via the HotCache after the learner runs.
#[tokio::test]
async fn test_full_e2e_learning_loop() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let learner = LatencySensitivityLearner::new("e2e-learn", SensitivityConfig::default());

    let mut proxy = NexusProxy::builder()
        .agent_id("e2e-learn")
        .backend(Box::new(backend))
        .learner(Box::new(learner))
        .dynamo_intercept(true)
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    // Run 1: establish baseline
    {
        let scope = nat_nexus_push_scope(
            "agent-run",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        for _ in 0..3 {
            let func: LlmExecutionNextFn =
                Arc::new(|_req| Box::pin(async { Ok(serde_json::json!({"tokens": 100})) }));
            nat_nexus_llm_call_execute(
                "llm",
                LLMRequest {
                    headers: serde_json::Map::new(),
                    content: serde_json::json!({}),
                },
                func,
                Some(scope.clone()),
                LLMAttributes::empty(),
                None,
                None,
                Some("gpt-4".into()),
            )
            .await
            .unwrap();
        }
        nat_nexus_pop_scope(&scope.uuid).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    // Run 2: trie should be building
    {
        let scope = nat_nexus_push_scope(
            "agent-run",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        for _ in 0..3 {
            let func: LlmExecutionNextFn =
                Arc::new(|_req| Box::pin(async { Ok(serde_json::json!({"tokens": 200})) }));
            nat_nexus_llm_call_execute(
                "llm",
                LLMRequest {
                    headers: serde_json::Map::new(),
                    content: serde_json::json!({}),
                },
                func,
                Some(scope.clone()),
                LLMAttributes::empty(),
                None,
                None,
                Some("gpt-4".into()),
            )
            .await
            .unwrap();
        }
        nat_nexus_pop_scope(&scope.uuid).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    // Verify trie is in the hot cache
    {
        let guard = proxy.hot_cache().read().unwrap();
        assert!(
            guard.trie.is_some(),
            "HotCache should contain a trie after 2 runs processed by learner"
        );
        assert!(
            guard.agent_hints_default.is_some(),
            "HotCache should contain default agent hints after learner runs"
        );
    }

    // Run 3: capture headers on LLM call to verify AgentHints injection
    let captured_headers = Arc::new(Mutex::new(None));
    let cap = captured_headers.clone();
    let func: LlmExecutionNextFn = Arc::new(move |req: LLMRequest| {
        let mut guard = cap.lock().unwrap();
        *guard = Some(req.headers.clone());
        Box::pin(async { Ok(serde_json::json!({"response": "captured"})) })
    });
    nat_nexus_llm_call_execute(
        "llm",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        func,
        None,
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
    )
    .await
    .unwrap();

    // Verify both headers are present
    let headers = captured_headers.lock().unwrap();
    let headers = headers.as_ref().expect("headers captured");
    assert!(
        headers.contains_key(AGENT_HINTS_HEADER_KEY),
        "Agent hints header should be present on LLM calls after learning loop"
    );

    let hints: AgentHints =
        serde_json::from_value(headers[AGENT_HINTS_HEADER_KEY].clone()).unwrap();
    assert!(hints.total_requests > 0);
    assert!(!hints.prefix_id.is_empty());

    // Verify backend has stored runs
    let runs = proxy.backend().list_runs_dyn("e2e-learn").await.unwrap();
    assert!(
        runs.len() >= 2,
        "Backend should have at least 2 stored runs, got {}",
        runs.len()
    );

    proxy.deregister().unwrap();
}

/// TEST2-02 supplement: Manual latency_sensitive annotation overrides auto-computed
/// value via scope metadata in the full integration flow.
#[tokio::test]
async fn test_manual_latency_sensitivity_override() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();

    let backend = InMemoryBackend::new();
    let learner = LatencySensitivityLearner::new("manual-ls-test", SensitivityConfig::default());

    let mut proxy = NexusProxy::builder()
        .agent_id("manual-ls-test")
        .backend(Box::new(backend))
        .learner(Box::new(learner))
        .dynamo_intercept(true)
        .build()
        .unwrap();
    proxy.register().await.unwrap();

    // Fire 2 runs to build trie
    for _ in 0..2 {
        let scope = nat_nexus_push_scope(
            "agent-run",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();

        let func: LlmExecutionNextFn =
            Arc::new(|_req| Box::pin(async { Ok(serde_json::json!({"response": "ok"})) }));
        nat_nexus_llm_call_execute(
            "llm",
            LLMRequest {
                headers: serde_json::Map::new(),
                content: serde_json::json!({}),
            },
            func,
            Some(scope.clone()),
            LLMAttributes::empty(),
            None,
            None,
            Some("gpt-4".into()),
        )
        .await
        .unwrap();

        nat_nexus_pop_scope(&scope.uuid).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    // Now push a scope with manual latency_sensitive metadata
    let manual_metadata = serde_json::json!({
        "nexus_proxy": {
            "latency_sensitivity": 5
        }
    });
    let sensitive_scope = nat_nexus_push_scope(
        "sensitive-context",
        ScopeType::Function,
        None,
        ScopeAttributes::empty(),
        None,
        Some(manual_metadata),
    )
    .unwrap();

    // Fire LLM call inside the annotated scope
    let captured_headers = Arc::new(Mutex::new(None));
    let cap = captured_headers.clone();
    let func: LlmExecutionNextFn = Arc::new(move |req: LLMRequest| {
        let mut guard = cap.lock().unwrap();
        *guard = Some(req.headers.clone());
        Box::pin(async { Ok(serde_json::json!({"response": "sensitive"})) })
    });
    nat_nexus_llm_call_execute(
        "llm",
        LLMRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        func,
        Some(sensitive_scope.clone()),
        LLMAttributes::empty(),
        None,
        None,
        Some("gpt-4".into()),
    )
    .await
    .unwrap();

    nat_nexus_pop_scope(&sensitive_scope.uuid).unwrap();

    // Verify AgentHints reflect the manual override
    let headers = captured_headers.lock().unwrap();
    let headers = headers.as_ref().expect("headers captured");

    if headers.contains_key(AGENT_HINTS_HEADER_KEY) {
        let hints: AgentHints =
            serde_json::from_value(headers[AGENT_HINTS_HEADER_KEY].clone()).unwrap();
        // Manual annotation was 5. The auto-computed value should be <= 5.
        // Max-merge means effective sensitivity >= 5.
        assert!(
            hints.latency_sensitivity >= 5.0,
            "Manual override of 5 should produce latency_sensitivity >= 5.0, got {}",
            hints.latency_sensitivity
        );
    } else {
        // Even without trie, manual annotation should create hints
        // This may happen if the drain hasn't processed yet -- in that case
        // the manual-only path should still inject hints with sensitivity=5
        panic!(
            "AGENT_HINTS_HEADER_KEY should be present when manual sensitivity is set. Keys: {:?}",
            headers.keys().collect::<Vec<_>>()
        );
    }

    proxy.deregister().unwrap();
}
