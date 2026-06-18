// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for intercepts in the NeMo Relay adaptive crate.

use super::*;
use crate::acg::stability::StabilityAnalysisResult;
use crate::types::cache::HotCache;
use crate::types::metadata::{MetadataEnvelope, ParallelHint};
use crate::types::plan::{ExecutionPlan, ParallelGroup};
use nemo_relay::api::runtime::{create_scope_stack, set_thread_scope_stack};
use nemo_relay::api::scope::{ScopeHandle, ScopeType};
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Builds a test [`ExecutionPlan`] with one parallel hint.
fn make_test_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "pg-1".to_string(),
            tool_names: vec!["search".to_string(), "fetch".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::nil(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "pg-1".to_string(),
                explicit: true,
            }],
            extensions: json!({"version": 1}),
        },
    }
}

fn make_hot_cache(
    plan: Option<ExecutionPlan>,
    stable_prefix_length: usize,
    observation_count: u32,
) -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(StabilityAnalysisResult {
            scores: vec![],
            stable_prefix_length,
            total_observations: observation_count,
            converged: false,
        }),
        acg_observation_count: observation_count,
    }))
}

fn install_scope_stack() -> (Uuid, Uuid) {
    let stack = create_scope_stack();
    let mut guard = stack.write().unwrap();
    let root_uuid = guard.root_uuid();
    let agent = ScopeHandle::builder()
        .name("agent".to_string())
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let agent_uuid = agent.uuid;
    let function = ScopeHandle::builder()
        .name("workflow".to_string())
        .scope_type(ScopeType::Function)
        .parent_uuid(agent_uuid)
        .build();
    guard.push(agent);
    guard.push(function);
    drop(guard);
    set_thread_scope_stack(stack);
    (root_uuid, agent_uuid)
}

fn reset_scope_stack() {
    set_thread_scope_stack(create_scope_stack());
}

// ---- Tool execution intercept tests ----

#[tokio::test]
async fn test_tool_intercept_calls_next() {
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let intercept = create_tool_execution_intercept(hot_cache);

    let next: ToolExecutionNextFn =
        Arc::new(|_args| Box::pin(async move { Ok(json!({"result": "ok"})) }));

    let result = intercept("test", json!({"input": 1}), next).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), json!({"result": "ok"}));
}

#[tokio::test]
async fn test_tool_intercept_with_populated_cache() {
    let plan = make_test_plan("test-agent");
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: Some(plan),
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let intercept = create_tool_execution_intercept(hot_cache);

    let next: ToolExecutionNextFn =
        Arc::new(|_args| Box::pin(async move { Ok(json!({"from_next": true})) }));

    // Should not panic and should return next's result
    let result = intercept("test", json!({"tool_input": "data"}), next).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), json!({"from_next": true}));
}

#[tokio::test]
async fn test_tool_intercept_passes_args_to_next() {
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let intercept = create_tool_execution_intercept(hot_cache);

    // next captures and returns the args it received, proving pass-through
    let next: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

    let input = json!({"tool_arg": "value", "count": 42});
    let result = intercept("test", input.clone(), next).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), input);
}

#[test]
fn test_warm_first_eligibility_evaluates_expected_thresholds() {
    let eligible = WarmFirstEligibility::evaluate(3, 10, 4);
    assert_eq!(eligible.follower_count, 2);
    assert_eq!(eligible.confidence_units, 4);
    assert!(eligible.is_eligible());

    let ineligible = WarmFirstEligibility::evaluate(2, 1, 1);
    assert_eq!(ineligible.follower_count, 1);
    assert!(!ineligible.is_eligible());
}

#[tokio::test]
async fn test_cohort_gate_release_unblocks_waiters() {
    let gate = Arc::new(CohortGate::new());
    let waiter = {
        let gate = gate.clone();
        tokio::spawn(async move {
            gate.wait_for_release().await;
        })
    };

    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(10), waiter)
            .await
            .is_err()
    );
    gate.release();

    let waiter = {
        let gate = gate.clone();
        tokio::spawn(async move {
            gate.wait_for_release().await;
        })
    };
    waiter.await.unwrap();
    assert!(gate.is_released());
    assert!(!gate.primer_active());
}

#[test]
fn test_resolve_warm_first_cohort_key_requires_schedule_mode_and_eligible_context() {
    reset_scope_stack();
    let hot_cache = make_hot_cache(Some(make_test_plan("test-agent")), 8, 4);
    assert!(resolve_warm_first_cohort_key("search", "observe_only", &hot_cache).is_none());

    let (root_uuid, shared_parent_uuid) = install_scope_stack();
    let cohort = resolve_warm_first_cohort_key("search", "schedule", &hot_cache).unwrap();
    assert_eq!(cohort.root_uuid, root_uuid);
    assert_eq!(cohort.shared_parent_uuid, shared_parent_uuid);
    assert_eq!(cohort.group_id, "pg-1");

    reset_scope_stack();
}

#[test]
fn test_resolve_warm_first_cohort_key_rejects_singleton_or_low_signal_groups() {
    let _ = install_scope_stack();
    let singleton_plan = ExecutionPlan {
        agent_id: "agent".to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "pg-1".to_string(),
            tool_names: vec!["search".to_string()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::nil(),
            agent_id: "agent".to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "pg-1".to_string(),
                explicit: true,
            }],
            extensions: json!({}),
        },
    };

    assert!(
        resolve_warm_first_cohort_key(
            "search",
            "schedule",
            &make_hot_cache(Some(singleton_plan), 8, 4)
        )
        .is_none()
    );
    assert!(
        resolve_warm_first_cohort_key(
            "search",
            "schedule",
            &make_hot_cache(Some(make_test_plan("agent")), 0, 4)
        )
        .is_none()
    );
    assert!(
        resolve_warm_first_cohort_key(
            "search",
            "schedule",
            &make_hot_cache(Some(make_test_plan("agent")), 8, 1)
        )
        .is_none()
    );
    assert!(
        resolve_warm_first_cohort_key(
            "missing",
            "schedule",
            &make_hot_cache(Some(make_test_plan("agent")), 8, 4)
        )
        .is_none()
    );

    reset_scope_stack();
}

#[tokio::test]
async fn test_resolve_warm_first_role_and_cleanup_reuse_registry_entries() {
    let registry: Arc<Mutex<std::collections::HashMap<CohortKey, Arc<CohortGate>>>> =
        Arc::new(Mutex::new(std::collections::HashMap::new()));
    let cohort_key = CohortKey {
        root_uuid: Uuid::new_v4(),
        shared_parent_uuid: Uuid::new_v4(),
        group_id: "pg-1".to_string(),
    };

    let primer_gate = match resolve_warm_first_role(&registry, cohort_key.clone()).await {
        WarmFirstRole::Primer(gate) => gate,
        WarmFirstRole::Follower(_) => panic!("first caller should become primer"),
    };
    match resolve_warm_first_role(&registry, cohort_key.clone()).await {
        WarmFirstRole::Follower(gate) => assert!(Arc::ptr_eq(&gate, &primer_gate)),
        WarmFirstRole::Primer(_) => panic!("second caller should become follower"),
    }

    primer_gate.release();
    cleanup_cohort_gate(&registry, &cohort_key, &primer_gate).await;
    assert!(registry.lock().await.is_empty());

    match resolve_warm_first_role(&registry, cohort_key).await {
        WarmFirstRole::Primer(_) => {}
        WarmFirstRole::Follower(_) => panic!("released cohort should allocate a fresh primer"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn test_schedule_mode_intercept_waits_for_primer_before_running_follower() {
    let _ = install_scope_stack();
    let hot_cache = make_hot_cache(Some(make_test_plan("agent")), 8, 4);
    let intercept = create_tool_execution_intercept_with_mode(hot_cache, "schedule".to_string());
    let next_order = Arc::new(AtomicUsize::new(0));
    let primer_released = Arc::new(AtomicBool::new(false));

    let next: ToolExecutionNextFn = {
        let next_order = next_order.clone();
        let primer_released = primer_released.clone();
        Arc::new(move |args| {
            let next_order = next_order.clone();
            let primer_released = primer_released.clone();
            Box::pin(async move {
                let order = next_order.fetch_add(1, Ordering::SeqCst);
                if order == 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                    primer_released.store(true, Ordering::SeqCst);
                } else {
                    assert!(
                        primer_released.load(Ordering::SeqCst),
                        "follower should not call next until the primer has released the cohort"
                    );
                }
                Ok(args)
            })
        })
    };

    let primer = tokio::spawn(intercept("search", json!({"call": 1}), next.clone()));
    tokio::task::yield_now().await;
    let follower = tokio::spawn(intercept("search", json!({"call": 2}), next.clone()));

    assert_eq!(primer.await.unwrap().unwrap(), json!({"call": 1}));
    assert_eq!(follower.await.unwrap().unwrap(), json!({"call": 2}));
    assert_eq!(next_order.load(Ordering::SeqCst), 2);

    reset_scope_stack();
}
