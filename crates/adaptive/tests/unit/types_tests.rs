// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::Utc;
use serde_json::json;
use uuid::Uuid;

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
    };

    let encoded = serde_json::to_value(&cache).unwrap();
    let decoded: HotCache = serde_json::from_value(encoded).unwrap();

    assert_eq!(decoded.plan.as_ref().unwrap().agent_id, "agent-1");
    assert_eq!(decoded.trie.as_ref().unwrap().name, "root");
    assert_eq!(decoded.agent_hints_default.as_ref().unwrap().osl, 256);
}
