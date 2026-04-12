// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::learner::latency::{LatencySensitivityLearner, compute_default_hints};
use crate::learner::traits::Learner;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackendDyn;
use crate::trie::builder::SensitivityConfig;
use crate::trie::data_models::{LlmCallPrediction, PredictionMetrics, PredictionTrieNode};
use crate::types::cache::HotCache;
use crate::types::records::{CallKind, CallRecord, RunRecord};

fn sample_run(agent_id: &str) -> RunRecord {
    let started_at = Utc::now();
    let llm_end = started_at + Duration::milliseconds(500);
    let tool_start = llm_end + Duration::milliseconds(100);
    let tool_end = tool_start + Duration::milliseconds(50);

    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![
            CallRecord {
                kind: CallKind::Llm,
                name: "planner".to_string(),
                started_at,
                ended_at: Some(llm_end),
                metadata_snapshot: None,
                output_tokens: Some(120),
                prompt_tokens: Some(32),
                total_tokens: Some(152),
                model_name: Some("gpt-test".to_string()),
                tool_call_count: Some(1),
            },
            CallRecord {
                kind: CallKind::Tool,
                name: "search".to_string(),
                started_at: tool_start,
                ended_at: Some(tool_end),
                metadata_snapshot: None,
                output_tokens: None,
                prompt_tokens: None,
                total_tokens: None,
                model_name: None,
                tool_call_count: None,
            },
        ],
        started_at,
        ended_at: Some(tool_end),
    }
}

#[test]
fn compute_default_hints_maps_prediction_metrics_to_agent_hints() {
    let trie_root = PredictionTrieNode {
        name: "root".to_string(),
        children: HashMap::new(),
        predictions_by_call_index: HashMap::new(),
        predictions_any_index: Some(LlmCallPrediction {
            remaining_calls: PredictionMetrics {
                sample_count: 3,
                mean: 2.0,
                p50: 2.0,
                p90: 4.0,
                p95: 4.5,
            },
            interarrival_ms: PredictionMetrics {
                sample_count: 3,
                mean: 75.0,
                p50: 60.0,
                p90: 90.0,
                p95: 95.0,
            },
            output_tokens: PredictionMetrics {
                sample_count: 3,
                mean: 128.0,
                p50: 128.0,
                p90: 256.0,
                p95: 300.0,
            },
            latency_sensitivity: Some(2),
        }),
    };

    let hints = compute_default_hints(&trie_root, 5).unwrap();
    assert_eq!(hints.osl, 256);
    assert_eq!(hints.iat, 75);
    assert_eq!(hints.priority, 3);
    assert_eq!(hints.latency_sensitivity, 2.0);
    assert_eq!(hints.prefix_id, "default");
    assert_eq!(hints.total_requests, 3);
}

#[tokio::test(flavor = "current_thread")]
async fn latency_learner_persists_trie_accumulators_and_hot_cache() {
    let backend = InMemoryBackend::new();
    let learner = LatencySensitivityLearner::new("agent-c", SensitivityConfig::default());
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
    }));

    learner
        .process_run(&sample_run("agent-c"), &backend, &hot_cache)
        .await
        .unwrap();

    let trie = backend.load_trie("agent-c").await.unwrap();
    let accumulators = backend.load_accumulators("agent-c").await.unwrap();
    let cache = hot_cache.read().unwrap();

    assert!(trie.is_some());
    assert!(accumulators.is_some());
    assert!(cache.trie.is_some());
    assert!(cache.agent_hints_default.is_some());
}
