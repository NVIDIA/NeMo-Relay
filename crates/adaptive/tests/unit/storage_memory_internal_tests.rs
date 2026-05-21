// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for storage memory internal in the NeMo Relay adaptive crate.

use super::*;

use chrono::Utc;
use uuid::Uuid;

use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
};
use crate::acg::stability::StabilityAnalysisResult;
use crate::trie::data_models::PredictionTrieNode;
use crate::types::metadata::MetadataEnvelope;
use crate::types::records::{CallKind, CallRecord};

fn sample_run_record(agent_id: &str) -> RunRecord {
    let now = Utc::now();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: agent_id.to_string(),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "planner".to_string(),
            started_at: now,
            ended_at: Some(now),
            metadata_snapshot: None,
            output_tokens: None,
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
            annotated_request: None,
            annotated_response: None,
        }],
        started_at: now,
        ended_at: Some(now),
    }
}

fn sample_plan_record(agent_id: &str) -> ExecutionPlan {
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

fn sample_prompt_ir_record() -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![PromptBlock {
            span_id: SpanId("span-0".to_string()),
            sequence_index: 0,
            role: PromptRole::System,
            content: "hello".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn sample_stability_record() -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: vec![BlockStabilityScore {
            span_id: SpanId("span-0".to_string()),
            classification: StabilityClass::Stable,
            score: 1.0,
            confidence: 1.0,
            observation_count: 1,
        }],
        stable_prefix_length: 1,
        total_observations: 1,
    }
}

fn poison_lock<T>(lock: &std::sync::RwLock<T>) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = lock.write().unwrap();
        panic!("poison");
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn in_memory_backend_reports_lock_poisoning_across_all_storage_maps() {
    let run_backend = InMemoryBackend::new();
    poison_lock(&run_backend.runs);
    assert!(
        run_backend
            .store_run(&sample_run_record("a"))
            .await
            .is_err()
    );
    assert!(run_backend.list_runs("a").await.is_err());

    let plan_backend = InMemoryBackend::new();
    poison_lock(&plan_backend.plans);
    assert!(plan_backend.load_plan("a").await.is_err());
    assert!(plan_backend.store_plan(&sample_plan_record("a")).is_err());

    let trie_backend = InMemoryBackend::new();
    poison_lock(&trie_backend.tries);
    let envelope = TrieEnvelope::new(PredictionTrieNode::new("root"), "a");
    assert!(trie_backend.store_trie("a", &envelope).await.is_err());
    assert!(trie_backend.load_trie("a").await.is_err());

    let accumulator_backend = InMemoryBackend::new();
    poison_lock(&accumulator_backend.accumulators);
    assert!(
        accumulator_backend
            .store_accumulators("a", &AccumulatorState::default())
            .await
            .is_err()
    );
    assert!(accumulator_backend.load_accumulators("a").await.is_err());

    let observation_backend = InMemoryBackend::new();
    poison_lock(&observation_backend.observations);
    let observations = vec![sample_prompt_ir_record()];
    assert!(
        observation_backend
            .store_observations("a", &observations)
            .await
            .is_err()
    );
    assert!(observation_backend.load_observations("a").await.is_err());

    let stability_backend = InMemoryBackend::new();
    poison_lock(&stability_backend.stability);
    let stability = sample_stability_record();
    assert!(
        stability_backend
            .store_stability("a", &stability)
            .await
            .is_err()
    );
    assert!(stability_backend.load_stability("a").await.is_err());
}
