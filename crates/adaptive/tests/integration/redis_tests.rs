// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis integration tests for [`RedisBackend`].
//!
//! These tests require a running Redis instance at `redis://127.0.0.1/`
//! and only run when `NEMO_RELAY_RUN_REDIS_TESTS=1` is set.

#![cfg(feature = "redis-backend")]

use std::sync::{Arc, Once, RwLock};

use chrono::Utc;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay_adaptive::acg::{StabilityThresholds, analyze_stability, build_prompt_ir};
use nemo_relay_adaptive::acg_learner::AcgLearner;
use nemo_relay_adaptive::cache_diagnostics::{CacheDiagnosticsTracker, build_cache_request_facts};
use nemo_relay_adaptive::learner::traits::Learner;
use uuid::Uuid;

use nemo_relay_adaptive::redis::RedisBackend;
use nemo_relay_adaptive::storage::traits::{StorageBackend, StorageBackendDyn};
use nemo_relay_adaptive::trie::accumulator::{AccumulatorState, NodeAccumulators, RunningStats};
use nemo_relay_adaptive::trie::data_models::PredictionTrieNode;
use nemo_relay_adaptive::trie::serialization::TrieEnvelope;
use nemo_relay_adaptive::types::cache::HotCache;
use nemo_relay_adaptive::types::metadata::MetadataEnvelope;
use nemo_relay_adaptive::types::plan::ExecutionPlan;
use nemo_relay_adaptive::types::records::{CallKind, CallRecord, RunRecord};

const REDIS_TEST_ENV: &str = "NEMO_RELAY_RUN_REDIS_TESTS";
static TEST_LOGGING: Once = Once::new();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn env_value_is_truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some(value) if !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    )
}

fn enable_operational_logs() {
    TEST_LOGGING.call_once(|| {
        let runtime =
            nemo_relay::logging::init_logging(&nemo_relay::logging::LoggingConfig::default())
                .expect("test logging should initialize");
        Box::leak(Box::new(runtime));
    });
}

/// Attempt to connect to a local Redis instance. Returns `None` (skip) if
/// Redis tests were not explicitly enabled or Redis is unavailable.
async fn get_test_redis() -> Option<RedisBackend> {
    enable_operational_logs();
    let redis_test_env =
        std::env::var_os(REDIS_TEST_ENV).map(|value| value.to_string_lossy().into_owned());
    if !env_value_is_truthy(redis_test_env.as_deref()) {
        eprintln!(
            "SKIP: set {REDIS_TEST_ENV} to a truthy value (for example, {REDIS_TEST_ENV}=1) to run Redis-backed tests"
        );
        return None;
    }

    // Use unique prefix per test run to avoid key collisions
    let prefix = format!("test:{}:", Uuid::now_v7());
    match RedisBackend::new("redis://127.0.0.1/", prefix).await {
        Ok(backend) => Some(backend),
        Err(_) => {
            eprintln!("SKIP: Redis not available at 127.0.0.1:6379");
            None
        }
    }
}

async fn get_test_redis_with_prefix() -> Option<(RedisBackend, String)> {
    enable_operational_logs();
    let prefix = format!("test:{}:", Uuid::now_v7());
    match RedisBackend::new("redis://127.0.0.1/", prefix.clone()).await {
        Ok(backend) => Some((backend, prefix)),
        Err(_) => {
            eprintln!("SKIP: Redis not available at 127.0.0.1:6379");
            None
        }
    }
}

async fn load_raw_json(key: &str) -> Option<String> {
    let client = redis::Client::open("redis://127.0.0.1/").ok()?;
    let mut conn = client.get_multiplexed_async_connection().await.ok()?;
    redis::cmd("GET").arg(key).query_async(&mut conn).await.ok()
}

#[test]
fn redis_test_env_truthy_parsing() {
    assert!(!env_value_is_truthy(None));
    assert!(!env_value_is_truthy(Some("")));
    assert!(!env_value_is_truthy(Some("   ")));
    assert!(!env_value_is_truthy(Some("0")));
    assert!(!env_value_is_truthy(Some(" false ")));
    assert!(!env_value_is_truthy(Some("FALSE")));
    assert!(env_value_is_truthy(Some("1")));
    assert!(env_value_is_truthy(Some("true")));
    assert!(env_value_is_truthy(Some("yes")));
}

fn make_test_run(agent_id: &str) -> RunRecord {
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![],
        started_at: Utc::now(),
        ended_at: None,
    }
}

fn sample_annotated_request(model: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are a careful planner".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Summarize the latest findings".to_string()),
                name: None,
            },
        ],
        model: Some(model.to_string()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    }
}

fn empty_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

fn sample_run_with_request(agent_id: &str, annotated_request: AnnotatedLlmRequest) -> RunRecord {
    let started_at = Utc::now();

    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "planner".to_string(),
            started_at,
            ended_at: Some(started_at),
            metadata_snapshot: None,
            output_tokens: Some(128),
            prompt_tokens: Some(32),
            total_tokens: Some(160),
            model_name: Some("claude-3-5-sonnet".to_string()),
            tool_call_count: None,
            annotated_request: Some(Arc::new(annotated_request)),
            annotated_response: None,
        }],
        started_at,
        ended_at: Some(started_at),
    }
}

fn make_test_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::now_v7(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![],
            extensions: serde_json::json!({}),
        },
    }
}

fn make_test_trie_envelope(workflow_name: &str) -> TrieEnvelope {
    let mut root = PredictionTrieNode::new("root");
    let child = PredictionTrieNode::new("child_agent");
    root.children.insert("child_agent".to_string(), child);
    TrieEnvelope::new(root, workflow_name)
}

fn make_test_accumulator_state() -> AccumulatorState {
    let mut state = AccumulatorState::default();
    let mut node_acc = NodeAccumulators::default();

    // Per-index stats
    let mut stats = RunningStats::new();
    stats.add_sample(100.0);
    stats.add_sample(200.0);
    stats.add_sample(300.0);
    node_acc.remaining_calls.insert(1, stats);

    // Aggregate stats -- must have samples so the TDigest is non-empty and
    // survives JSON round-trip (empty TDigest contains NaN internals).
    node_acc.all_remaining_calls.add_sample(100.0);
    node_acc.all_remaining_calls.add_sample(200.0);
    node_acc.all_remaining_calls.add_sample(300.0);
    node_acc.all_interarrival_ms.add_sample(50.0);
    node_acc.all_output_tokens.add_sample(256.0);
    node_acc.all_sensitivity.add_sample(0.8);

    state.nodes.insert("workflow/agent".to_string(), node_acc);
    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_redis_store_load_run() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let record = make_test_run("agent-redis-run");
    let record_id = record.id;
    backend.store_run(&record).await.unwrap();
    let runs = backend.list_runs("agent-redis-run").await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].id, record_id);
    assert_eq!(runs[0].agent_id, "agent-redis-run");
}

#[tokio::test]
async fn redis_store_load_plan() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let plan = make_test_plan("agent-redis-plan");
    backend.store_plan(&plan).unwrap();

    let loaded = backend.load_plan("agent-redis-plan").await.unwrap();
    assert!(
        loaded.is_some(),
        "stored plan should be returned by load_plan"
    );
    let loaded = loaded.unwrap();
    assert_eq!(loaded.agent_id, "agent-redis-plan");

    let loaded_dyn = backend.load_plan_dyn("agent-redis-plan").await.unwrap();
    assert!(
        loaded_dyn.is_some(),
        "stored plan should be returned by load_plan_dyn"
    );
    let loaded_dyn = loaded_dyn.unwrap();
    assert_eq!(loaded_dyn.agent_id, "agent-redis-plan");
}

#[tokio::test]
async fn test_redis_trie_atomic_roundtrip() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let envelope = make_test_trie_envelope("redis-trie-workflow");

    backend
        .store_trie("agent-redis-trie", &envelope)
        .await
        .unwrap();
    let loaded = backend.load_trie("agent-redis-trie").await.unwrap();

    assert!(loaded.is_some(), "stored trie should be loadable");
    let loaded = loaded.unwrap();
    assert_eq!(loaded.workflow_name, "redis-trie-workflow");
    assert_eq!(loaded.root.name, "root");
    assert!(
        loaded.root.children.contains_key("child_agent"),
        "child node should survive round-trip"
    );
    assert_eq!(loaded.version, "1.0");
}

#[tokio::test]
async fn test_redis_accumulators_roundtrip() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let state = make_test_accumulator_state();

    backend
        .store_accumulators("agent-redis-acc", &state)
        .await
        .unwrap();
    let loaded = backend.load_accumulators("agent-redis-acc").await.unwrap();

    assert!(loaded.is_some(), "stored accumulators should be loadable");
    let loaded = loaded.unwrap();
    assert!(
        loaded.nodes.contains_key("workflow/agent"),
        "path key should survive round-trip"
    );
    let node_acc = &loaded.nodes["workflow/agent"];
    assert!(
        node_acc.remaining_calls.contains_key(&1),
        "call index 1 should exist"
    );
    let stats = &node_acc.remaining_calls[&1];
    assert_eq!(stats.count, 3, "should have 3 samples");
    assert!(
        (stats.mean - 200.0).abs() < 1e-6,
        "mean should be 200.0, got {}",
        stats.mean
    );
}

#[tokio::test]
async fn test_redis_load_nonexistent_trie() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let loaded = backend.load_trie("agent-does-not-exist").await.unwrap();
    assert!(
        loaded.is_none(),
        "load_trie for nonexistent agent should return None"
    );
}

#[tokio::test]
async fn test_redis_load_nonexistent_accumulators() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let loaded = backend
        .load_accumulators("agent-does-not-exist")
        .await
        .unwrap();
    assert!(
        loaded.is_none(),
        "load_accumulators for nonexistent agent should return None"
    );
}

#[tokio::test]
async fn test_redis_overwrite_trie() {
    let Some(backend) = get_test_redis().await else {
        return;
    };
    let first = make_test_trie_envelope("first-workflow");
    let second = make_test_trie_envelope("second-workflow");

    backend
        .store_trie("agent-redis-overwrite", &first)
        .await
        .unwrap();
    backend
        .store_trie("agent-redis-overwrite", &second)
        .await
        .unwrap();

    let loaded = backend.load_trie("agent-redis-overwrite").await.unwrap();

    assert!(loaded.is_some(), "overwritten trie should be loadable");
    let loaded = loaded.unwrap();
    assert_eq!(
        loaded.workflow_name, "second-workflow",
        "load should return the second (overwritten) trie, not the first"
    );
}

#[tokio::test]
async fn redis_integration_round_trips_canonical_acg_payloads_under_literal_keys() {
    let Some((backend, prefix)) = get_test_redis_with_prefix().await else {
        return;
    };
    let agent_id = "agent-redis-acg";
    let request = sample_annotated_request("claude-3-5-sonnet");
    let prompt_ir = build_prompt_ir(&request).expect("request should build canonical PromptIR");
    let stability = analyze_stability(
        std::slice::from_ref(&prompt_ir),
        &StabilityThresholds::default(),
    );

    backend
        .store_observations(agent_id, std::slice::from_ref(&prompt_ir))
        .await
        .expect("canonical PromptIR should store in Redis");
    backend
        .store_stability(agent_id, &stability)
        .await
        .expect("canonical stability should store in Redis");

    let observations_key = format!("{prefix}acg_observations:{agent_id}");
    let stability_key = format!("{prefix}acg_stability:{agent_id}");
    let raw_observations = load_raw_json(&observations_key)
        .await
        .expect("acg_observations key should exist");
    let raw_stability = load_raw_json(&stability_key)
        .await
        .expect("acg_stability key should exist");

    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&raw_observations).unwrap(),
        serde_json::to_value(vec![prompt_ir.clone()]).unwrap()
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&raw_stability).unwrap(),
        serde_json::to_value(stability.clone()).unwrap()
    );

    assert_eq!(
        backend.load_observations(agent_id).await.unwrap(),
        Some(vec![prompt_ir])
    );
    assert_eq!(
        backend.load_stability(agent_id).await.unwrap(),
        Some(stability)
    );
}

#[tokio::test]
async fn redis_integration_persists_runtime_seed_entries_and_manifest_cleanup() {
    let Some((backend, prefix)) = get_test_redis_with_prefix().await else {
        return;
    };
    let agent_id = "agent-redis-runtime-seed";
    let hot_cache = empty_hot_cache();
    let request = sample_annotated_request("claude-3-5-sonnet");
    let learner = AcgLearner::new(agent_id, 8, StabilityThresholds::default());

    learner
        .process_run(
            &sample_run_with_request(agent_id, request.clone()),
            &backend,
            &hot_cache,
        )
        .await
        .expect("ACG learner should persist Redis runtime seed state");

    let observations_key = format!("{prefix}acg_observations:{agent_id}");
    let stability_key = format!("{prefix}acg_stability:{agent_id}");
    assert!(
        load_raw_json(&observations_key).await.is_some(),
        "aggregate agent observations should persist under the literal acg_observations key"
    );
    assert!(
        load_raw_json(&stability_key).await.is_some(),
        "aggregate agent stability should persist under the literal acg_stability key"
    );

    let seeded_hot_cache = empty_hot_cache();
    let loaded_stability = backend.load_stability(agent_id).await.unwrap();
    let loaded_observation_count = backend
        .load_observations(agent_id)
        .await
        .unwrap()
        .map(|observations| observations.len() as u32)
        .unwrap_or(0);
    {
        let mut guard = seeded_hot_cache.write().unwrap();
        guard.acg_stability = loaded_stability;
        guard.acg_observation_count = loaded_observation_count;
    }

    let tracker = Arc::new(RwLock::new(CacheDiagnosticsTracker::default()));
    let facts = build_cache_request_facts(
        agent_id,
        "passthrough",
        &request,
        &seeded_hot_cache,
        &tracker,
    )
    .expect("runtime seed entries should hydrate cache diagnostics");
    assert_ne!(facts.stable_prefix_length, 0);
    assert!(
        !facts
            .missing_facts
            .contains(&"acg_stability_unavailable".to_string())
    );

    let manifest = std::fs::read_to_string(format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR")))
        .expect("adaptive manifest should be readable");
    assert!(
        !manifest.contains("nemo-relay-acg"),
        "adaptive manifest should not depend directly on the compatibility shim"
    );
}
