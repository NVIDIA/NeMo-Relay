// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for types in the NeMo Relay adaptive crate.

use std::collections::HashMap;

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::SpanId;
use crate::acg::stability::StabilityAnalysisResult;
use crate::trie::data_models::PredictionTrieNode;
use crate::types::cache::HotCache;
use crate::types::metadata::{AgentHints, MetadataEnvelope, ParallelHint};
use crate::types::plan::{ExecutionPlan, ParallelGroup};
use crate::types::records::{CallKind, CallRecord, RunRecord};

fn sample_metadata() -> MetadataEnvelope {
    MetadataEnvelope {
        run_id: Uuid::now_v7(),
        agent_id: "agent-1".to_string(),
        parallel_hints: vec![ParallelHint {
            tool_name: "search".to_string(),
            group_id: "g1".to_string(),
            explicit: true,
        }],
        extensions: json!({"flag": true}),
    }
}

fn sample_stability_result() -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: vec![BlockStabilityScore {
            span_id: SpanId("span-0".to_string()),
            classification: StabilityClass::Stable,
            score: 1.0,
            confidence: 1.0,
            observation_count: 4,
        }],
        stable_prefix_length: 1,
        total_observations: 4,
        converged: false,
    }
}

#[test]
fn metadata_and_plan_round_trip_through_serde() {
    let metadata = sample_metadata();
    let plan = ExecutionPlan {
        agent_id: metadata.agent_id.clone(),
        parallel_groups: vec![ParallelGroup {
            group_id: "g1".to_string(),
            tool_names: vec!["search".to_string(), "summarize".to_string()],
        }],
        metadata_template: metadata.clone(),
    };

    let encoded = serde_json::to_value(&plan).unwrap();
    assert_eq!(encoded["agent_id"], json!("agent-1"));
    assert_eq!(
        encoded["parallel_groups"][0]["tool_names"][1],
        json!("summarize")
    );

    let decoded: ExecutionPlan = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.agent_id, "agent-1");
    assert_eq!(decoded.parallel_groups.len(), 1);
    assert_eq!(decoded.metadata_template.parallel_hints.len(), 1);
    assert_eq!(decoded.metadata_template.extensions, json!({"flag": true}));
}

#[test]
fn run_record_serializes_call_kind_and_optional_fields() {
    let now = Utc::now();
    let record = RunRecord {
        id: Uuid::now_v7(),
        agent_id: "agent-1".to_string(),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "planner".to_string(),
            started_at: now,
            ended_at: Some(now),
            metadata_snapshot: Some(sample_metadata()),
            output_tokens: Some(128),
            prompt_tokens: Some(32),
            total_tokens: Some(160),
            model_name: Some("gpt-test".to_string()),
            tool_call_count: Some(2),
            annotated_request: None,
            annotated_response: None,
        }],
        started_at: now,
        ended_at: Some(now),
    };

    let encoded = serde_json::to_string(&record).unwrap();
    assert!(encoded.contains("\"kind\":\"llm\""));
    assert!(encoded.contains("\"output_tokens\":128"));

    let decoded: RunRecord = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.calls.len(), 1);
    assert!(matches!(decoded.calls[0].kind, CallKind::Llm));
    assert_eq!(decoded.calls[0].model_name.as_deref(), Some("gpt-test"));
}

#[test]
fn hot_cache_round_trip_preserves_optional_sections() {
    let cache = HotCache {
        plan: Some(ExecutionPlan {
            agent_id: "agent-1".to_string(),
            parallel_groups: vec![],
            metadata_template: sample_metadata(),
        }),
        trie: Some(PredictionTrieNode::new("root")),
        agent_hints_default: Some(AgentHints {
            osl: 256,
            iat: 75,
            priority: 3,
            latency_sensitivity: 2.0,
            prefix_id: "default".to_string(),
            total_requests: 4,
        }),
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    };

    let encoded = serde_json::to_value(&cache).unwrap();
    let decoded: HotCache = serde_json::from_value(encoded).unwrap();

    assert_eq!(decoded.plan.as_ref().unwrap().agent_id, "agent-1");
    assert_eq!(decoded.trie.as_ref().unwrap().name, "root");
    assert_eq!(decoded.agent_hints_default.as_ref().unwrap().osl, 256);
}

#[test]
fn acg_storage_and_hot_cache_sources_use_canonical_acg_types() {
    let canonical_sources: [(&str, &str, &[&str]); 2] = [
        (
            "src/storage/traits.rs",
            include_str!("../../src/storage/traits.rs"),
            &[
                "type PromptIrList = Vec<crate::acg::prompt_ir::PromptIR>;",
                "type StabilityResult = crate::acg::stability::StabilityAnalysisResult;",
            ],
        ),
        (
            "src/types/cache.rs",
            include_str!("../../src/types/cache.rs"),
            &[
                "pub acg_profiles: HashMap<String, crate::acg::stability::StabilityAnalysisResult>",
                "pub acg_stability: Option<crate::acg::stability::StabilityAnalysisResult>",
            ],
        ),
    ];

    for (path, source, canonical_patterns) in canonical_sources {
        assert!(
            !source.contains("nemo_relay_acg::"),
            "{path} should not point back at the shim-owned namespace",
        );
        for canonical_pattern in canonical_patterns {
            assert!(
                source.contains(canonical_pattern),
                "{path} should keep canonical ACG ownership via `{canonical_pattern}`",
            );
        }
    }
}

#[test]
fn hot_cache_serialization_keeps_acg_field_names_stable() {
    let stability = sample_stability_result();
    let cache = HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: HashMap::from([("profile-a".to_string(), stability.clone())]),
        acg_profile_observation_counts: HashMap::from([("profile-a".to_string(), 4)]),
        acg_stability: Some(stability),
        acg_observation_count: 4,
    };

    let encoded = serde_json::to_value(&cache).unwrap();
    assert!(encoded.get("acg_profiles").is_some());
    assert!(encoded.get("acg_profile_observation_counts").is_some());
    assert!(encoded.get("acg_stability").is_some());
    assert_eq!(encoded["acg_observation_count"], json!(4));

    let decoded: HotCache = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.acg_profiles["profile-a"].stable_prefix_length, 1);
    assert_eq!(
        decoded.acg_stability.as_ref().unwrap().total_observations,
        4
    );
}
