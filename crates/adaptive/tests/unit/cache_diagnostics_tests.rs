// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for cache diagnostics in the NeMo Relay adaptive crate.

use std::sync::{Arc, RwLock};

use crate::acg::canonicalize::sha256_hex;
use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
    TokenizationMetadata,
};
use crate::acg::stability::StabilityAnalysisResult;
use chrono::{Duration, TimeZone, Utc};
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use serde_json::Map;
use uuid::Uuid;

use super::{
    CacheDiagnosticsTracker, CacheFactsBuildInput, build_cache_request_facts,
    build_cache_request_facts_from_prompt_ir,
};
use crate::types::cache::HotCache;

fn sample_timestamp() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0)
        .single()
        .expect("valid timestamp")
}

fn make_prompt_ir(blocks: Vec<(&str, &str, Option<u32>)>) -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: blocks
            .into_iter()
            .enumerate()
            .map(|(index, (span_id, content, token_count))| PromptBlock {
                span_id: SpanId(span_id.to_string()),
                sequence_index: index as u32,
                role: if index == 0 {
                    PromptRole::System
                } else {
                    PromptRole::User
                },
                content: content.to_string(),
                content_type: BlockContentType::Text,
                provenance: if index == 0 {
                    ProvenanceLabel::System
                } else {
                    ProvenanceLabel::User
                },
                sensitivity: SensitivityLabel::Public,
                token_metadata: token_count.map(|token_count| TokenizationMetadata {
                    model_family: "claude".to_string(),
                    token_count,
                }),
            })
            .collect(),
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: sample_timestamp(),
    }
}

fn make_hot_cache(stable_prefix_length: Option<usize>) -> HotCache {
    HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: stable_prefix_length.map(|stable_prefix_length| StabilityAnalysisResult {
            scores: (0..stable_prefix_length)
                .map(|index| BlockStabilityScore {
                    span_id: SpanId(format!("span-{index}")),
                    classification: StabilityClass::Stable,
                    score: 1.0,
                    confidence: 1.0,
                    observation_count: 4,
                })
                .collect(),
            stable_prefix_length,
            total_observations: 4,
        }),
        acg_observation_count: 4,
    }
}

fn sample_request(model: Option<&str>) -> AnnotatedLlmRequest {
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
        model: model.map(str::to_string),
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
        extra: Map::new(),
    }
}

fn short_hash_prefix(content: &str) -> String {
    let full = sha256_hex(content);
    format!("sha256:{}", &full["sha256:".len()..][..12])
}

#[test]
fn cache_request_facts_reports_first_mismatch_against_last_exemplar() {
    let hot_cache = make_hot_cache(Some(2));
    let mut tracker = CacheDiagnosticsTracker::default();
    let baseline = make_prompt_ir(vec![
        ("system-0", "You are a careful planner", Some(700)),
        ("user-1", "Find sources about caching", Some(500)),
        ("user-2", "volatile suffix", Some(30)),
    ]);
    let changed = make_prompt_ir(vec![
        ("system-0", "You are a careful planner", Some(700)),
        ("user-1", "Find reports about caching", Some(500)),
        ("user-2", "volatile suffix", Some(30)),
    ]);

    let first = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "openai",
            model: Some("gpt-4o"),
            prompt_ir: &baseline,
            hot_cache: &hot_cache,
            profile_key: "test-profile",
            now: sample_timestamp(),
        },
        &mut tracker,
    );
    assert_eq!(first.first_mismatch_span_id, None);

    let second = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "openai",
            model: Some("gpt-4o"),
            prompt_ir: &changed,
            hot_cache: &hot_cache,
            profile_key: "test-profile",
            now: sample_timestamp() + Duration::seconds(30),
        },
        &mut tracker,
    );

    assert_eq!(second.first_mismatch_span_id, Some("user-1".to_string()));
    assert_eq!(second.first_mismatch_sequence_index, Some(1));
    assert_eq!(
        second.expected_hash_prefix,
        Some(short_hash_prefix("Find sources about caching"))
    );
    assert_eq!(
        second.actual_hash_prefix,
        Some(short_hash_prefix("Find reports about caching"))
    );
}

#[test]
fn cache_request_facts_keeps_missing_facts_bounded_when_inputs_are_unavailable() {
    let tracker = Arc::new(RwLock::new(CacheDiagnosticsTracker::default()));
    let hot_cache = Arc::new(RwLock::new(make_hot_cache(None)));

    let facts = build_cache_request_facts(
        "agent-1",
        "anthropic",
        &sample_request(Some("claude-sonnet-4")),
        &hot_cache,
        &tracker,
    )
    .expect("facts should still be returned when stability is missing");

    assert_eq!(
        facts.missing_facts,
        vec!["acg_stability_unavailable".to_string()]
    );
    assert_eq!(facts.stable_prefix_tokens, None);
    assert_eq!(facts.first_mismatch_span_id, None);
    assert_eq!(facts.expected_hash_prefix, None);

    let hot_cache = make_hot_cache(Some(2));
    let mut tracker = CacheDiagnosticsTracker::default();
    let prompt_ir = make_prompt_ir(vec![
        ("system-0", "You are a careful planner", Some(700)),
        ("user-1", "Summarize the latest findings", None),
    ]);

    let facts = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "openai",
            model: Some("gpt-4o"),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: "test-profile",
            now: sample_timestamp(),
        },
        &mut tracker,
    );

    assert!(
        facts
            .missing_facts
            .contains(&"stable_prefix_tokens_unavailable".to_string())
    );
    assert_eq!(facts.stable_prefix_tokens, None);
}

#[test]
fn cache_request_facts_populates_provider_thresholds_and_retention_defaults() {
    let hot_cache = make_hot_cache(Some(2));
    let prompt_ir = make_prompt_ir(vec![
        ("system-0", "You are a careful planner", Some(700)),
        ("user-1", "Summarize the latest findings", Some(500)),
    ]);

    let anthropic = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "anthropic",
            model: Some("claude-sonnet-4"),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: "anthropic-profile",
            now: sample_timestamp(),
        },
        &mut CacheDiagnosticsTracker::default(),
    );
    assert_eq!(anthropic.required_min_tokens, Some(1024));
    assert_eq!(anthropic.retention_window_secs, Some(300.0));

    let openai = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "openai",
            model: Some("gpt-4o"),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: "openai-profile",
            now: sample_timestamp(),
        },
        &mut CacheDiagnosticsTracker::default(),
    );
    assert_eq!(openai.required_min_tokens, Some(1024));
    assert_eq!(openai.retention_window_secs, None);
}

#[test]
fn cache_request_facts_tracks_observed_gap_for_repeated_prefixes() {
    let hot_cache = make_hot_cache(Some(2));
    let prompt_ir = make_prompt_ir(vec![
        ("system-0", "You are a careful planner", Some(700)),
        ("user-1", "Summarize the latest findings", Some(500)),
    ]);
    let mut tracker = CacheDiagnosticsTracker::default();

    let first = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "anthropic",
            model: Some("claude-sonnet-4"),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: "test-profile",
            now: sample_timestamp(),
        },
        &mut tracker,
    );
    assert_eq!(first.observed_gap_secs, None);

    let second = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "anthropic",
            model: Some("claude-sonnet-4"),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: "test-profile",
            now: sample_timestamp() + Duration::seconds(42),
        },
        &mut tracker,
    );
    assert_eq!(second.observed_gap_secs, Some(42.0));
}

#[test]
fn cache_request_facts_below_minimum_no_write_uses_prefix_matched_anthropic_thresholds() {
    let hot_cache = make_hot_cache(Some(2));
    let prompt_ir = make_prompt_ir(vec![
        ("system-0", "You are a careful planner", Some(700)),
        ("user-1", "Summarize the latest findings", Some(900)),
    ]);

    let facts = build_cache_request_facts_from_prompt_ir(
        CacheFactsBuildInput {
            agent_id: "agent-1",
            provider: "anthropic",
            model: Some("claude-sonnet-4.6-20260101"),
            prompt_ir: &prompt_ir,
            hot_cache: &hot_cache,
            profile_key: "anthropic-profile",
            now: sample_timestamp(),
        },
        &mut CacheDiagnosticsTracker::default(),
    );

    assert_eq!(facts.stable_prefix_tokens, Some(1600));
    assert_eq!(
        facts.required_min_tokens,
        Some(2048),
        "planner-driven no-write diagnosis must use the canonical prefix-matched Anthropic threshold",
    );
    assert_eq!(facts.retention_window_secs, Some(300.0));
}
