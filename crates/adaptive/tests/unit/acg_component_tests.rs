// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for acg component in the NeMo Flow adaptive crate.

use super::*;

use std::future::Future;
use std::pin::Pin;

use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::SpanId;
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel,
};
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackendDyn;
use nemo_flow::api::llm::LlmRequest;
use nemo_flow::api::runtime::LlmExecutionNextFn;
use nemo_flow::api::runtime::LlmStreamExecutionNextFn;
use nemo_flow::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use serde_json::{Value, json};
use tokio_stream::StreamExt;

fn sample_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(StabilityAnalysisResult {
            scores: vec![
                BlockStabilityScore {
                    span_id: SpanId("block-0".to_string()),
                    classification: StabilityClass::Stable,
                    score: 0.99,
                    confidence: 1.0,
                    observation_count: 8,
                },
                BlockStabilityScore {
                    span_id: SpanId("block-1".to_string()),
                    classification: StabilityClass::Stable,
                    score: 0.98,
                    confidence: 1.0,
                    observation_count: 8,
                },
            ],
            stable_prefix_length: 2,
            total_observations: 8,
        }),
        acg_observation_count: 8,
    }))
}

fn sample_openai_chat_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_result", "data": {"z": 1, "a": 2}}
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"z": 1, "a": 2}
                    }
                }
            ]
        }),
    }
}

fn sample_annotated_request(model: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Hello".to_string()),
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

fn sample_openai_responses_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4.1",
            "instructions": "You are helpful.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_result", "data": {"z": 1, "a": 2}}
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"z": 1, "a": 2}
                    }
                }
            ]
        }),
    }
}

fn sample_anthropic_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4-20250514",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }),
    }
}

fn long_text(token_count: usize) -> String {
    "x".repeat(token_count * 4)
}

fn sample_layered_anthropic_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-sonnet-4-20250514",
            "system": long_text(1400),
            "messages": [
                {"role": "user", "content": long_text(1500)},
                {"role": "user", "content": long_text(1600)}
            ]
        }),
    }
}

fn marker_positions(req: &LlmRequest) -> Vec<(String, usize)> {
    let mut positions = Vec::new();
    append_system_marker_positions(req, &mut positions);
    append_message_marker_positions(req, &mut positions);
    positions
}

fn append_system_marker_positions(req: &LlmRequest, positions: &mut Vec<(String, usize)>) {
    if let Some(system) = req.content.get("system").and_then(|value| value.as_array()) {
        append_block_marker_positions("system", system, positions);
    }
}

fn append_message_marker_positions(req: &LlmRequest, positions: &mut Vec<(String, usize)>) {
    if let Some(messages) = req
        .content
        .get("messages")
        .and_then(|value| value.as_array())
    {
        for (message_index, message) in messages.iter().enumerate() {
            if let Some(content) = message.get("content").and_then(|value| value.as_array()) {
                append_block_marker_positions(
                    &format!("messages[{message_index}].content"),
                    content,
                    positions,
                );
            }
        }
    }
}

fn append_block_marker_positions(
    source: &str,
    blocks: &[Value],
    positions: &mut Vec<(String, usize)>,
) {
    positions.extend(
        blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.get("cache_control").is_some())
            .map(|(index, _)| (source.to_string(), index)),
    );
}

// Helper - yields a StabilityAnalysisResult with N stable prefix blocks.
fn stability_with_prefix(prefix_len: u32, observations: u32) -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: (0..prefix_len)
            .map(|index| BlockStabilityScore {
                span_id: SpanId(format!("layer-{index}")),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.95 - (index as f64 * 0.05),
                observation_count: observations,
            })
            .collect(),
        stable_prefix_length: prefix_len as usize,
        total_observations: observations,
    }
}

fn layered_stability_result(observation_count: u32) -> StabilityAnalysisResult {
    StabilityAnalysisResult {
        scores: vec![
            BlockStabilityScore {
                span_id: SpanId("block-0".to_string()),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.95,
                observation_count,
            },
            BlockStabilityScore {
                span_id: SpanId("block-1".to_string()),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.9,
                observation_count,
            },
            BlockStabilityScore {
                span_id: SpanId("block-2".to_string()),
                classification: StabilityClass::Stable,
                score: 0.99,
                confidence: 0.85,
                observation_count,
            },
        ],
        stable_prefix_length: 3,
        total_observations: observation_count,
    }
}

fn sample_prompt_ir(span: &str) -> PromptIR {
    PromptIR {
        ir_id: Uuid::new_v4(),
        blocks: vec![PromptBlock {
            span_id: SpanId(span.to_string()),
            sequence_index: 0,
            role: PromptRole::System,
            content: "You are helpful.".to_string(),
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

struct FailingStabilityBackend;

impl StorageBackendDyn for FailingStabilityBackend {
    fn store_run_dyn<'a>(
        &'a self,
        _record: &'a crate::types::records::RunRecord,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_plan_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = crate::error::Result<Option<crate::types::plan::ExecutionPlan>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(None) })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = crate::error::Result<Vec<crate::types::records::RunRecord>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_trie<'a>(
        &'a self,
        _agent_id: &'a str,
        _envelope: &'a crate::trie::serialization::TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_trie<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = crate::error::Result<Option<crate::trie::serialization::TrieEnvelope>>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(None) })
    }

    fn store_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
        _state: &'a crate::trie::accumulator::AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = crate::error::Result<
                        Option<crate::trie::accumulator::AccumulatorState>,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async { Ok(None) })
    }

    fn load_stability<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<
        Box<dyn Future<Output = crate::error::Result<Option<StabilityAnalysisResult>>> + Send + 'a>,
    > {
        Box::pin(async {
            Err(crate::error::AdaptiveError::Storage(
                "stability failed".into(),
            ))
        })
    }
}

#[test]
fn acg_component_translate_request_degrades_when_provider_semantics_do_not_match_request_surface() {
    let request = sample_openai_chat_request();
    let hot_cache = sample_hot_cache();
    let plugin = build_provider_plugin("anthropic").expect("anthropic plugin should build");

    let translated = translate_request(
        &request,
        "agent-1",
        "anthropic",
        plugin.as_ref(),
        &hot_cache,
    );

    assert!(
        translated.is_none(),
        "runtime should pass through when semantic provider and request surface diverge",
    );
}

#[test]
fn acg_component_translate_request_applies_openai_semantics_on_resolved_request_surface() {
    let request = sample_openai_responses_request();
    let hot_cache = sample_hot_cache();
    let plugin = build_provider_plugin("openai").expect("openai plugin should build");

    let translated = translate_request(&request, "agent-1", "openai", plugin.as_ref(), &hot_cache)
        .expect("openai request should translate through the resolved responses surface");

    assert!(translated.content.get("input").is_some());
    assert_eq!(
        translated.content["tools"][0]["function"]["parameters"],
        json!({"a": 2, "z": 1}),
    );
}

#[test]
fn acg_component_translate_request_passes_through_when_planner_finds_no_profitable_breakpoints() {
    let request = sample_anthropic_request();
    let hot_cache = sample_hot_cache();
    let plugin = build_provider_plugin("anthropic").expect("anthropic plugin should build");

    let translated = translate_request(
        &request,
        "agent-1",
        "anthropic",
        plugin.as_ref(),
        &hot_cache,
    );

    assert!(
        translated.is_none(),
        "non-profitable anthropic prefixes should leave the request unchanged",
    );
}

#[test]
fn acg_component_translate_request_applies_multiple_ordered_anthropic_breakpoints_from_planner() {
    let request = sample_layered_anthropic_request();
    let plugin = build_provider_plugin("anthropic").expect("anthropic plugin should build");
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(layered_stability_result(6)),
        acg_observation_count: 6,
    }));

    let translated = translate_request(
        &request,
        "agent-1",
        "anthropic",
        plugin.as_ref(),
        &hot_cache,
    )
    .expect("profitable layered anthropic request should translate");

    assert!(translated.content["system"][0]["cache_control"].is_object());
    assert!(translated.content["messages"][0]["content"][0]["cache_control"].is_object());
    assert!(translated.content["messages"][1]["content"][0]["cache_control"].is_object());
}

#[test]
fn rewrite_request_with_hot_cache_passes_through_when_no_adaptive_state() {
    let request = sample_anthropic_request();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let translated = rewrite_request_with_hot_cache(&request, hot_cache, "agent-1", "anthropic")
        .expect("stateless rewrite should succeed");

    assert_eq!(translated.content, request.content);
}

#[test]
fn rewrite_request_with_hot_cache_adaptive_placement_differs_by_state() {
    let request = sample_layered_anthropic_request();

    let hot_cache_short = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(stability_with_prefix(1, 8)),
        acg_observation_count: 8,
    }));
    let translated_short =
        rewrite_request_with_hot_cache(&request, hot_cache_short, "agent-1", "anthropic")
            .expect("short-prefix rewrite should succeed");

    let hot_cache_long = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(stability_with_prefix(3, 8)),
        acg_observation_count: 8,
    }));
    let translated_long =
        rewrite_request_with_hot_cache(&request, hot_cache_long, "agent-1", "anthropic")
            .expect("long-prefix rewrite should succeed");

    let markers_short = marker_positions(&translated_short);
    let markers_long = marker_positions(&translated_long);

    assert_ne!(
        markers_short, markers_long,
        "different hot-cache stability should change marker placement"
    );
}

#[test]
fn acg_component_translate_request_uses_learning_key_for_profile_lookup() {
    let request = sample_layered_anthropic_request();
    let semantic_request_view =
        build_semantic_request_view(&request).expect("anthropic request should decode");
    let learning_key = crate::acg_profile::derive_acg_learning_key(
        "agent-1",
        &semantic_request_view.annotated_request,
    );
    let diagnostic_key = crate::acg_profile::derive_acg_profile_key(
        "agent-1",
        &semantic_request_view.annotated_request,
    );
    assert_ne!(
        learning_key, diagnostic_key,
        "the lookup key should stay stable even when the diagnostic key keeps the exact request shape",
    );

    let plugin = build_provider_plugin("anthropic").expect("anthropic plugin should build");
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::from([(
            learning_key.clone(),
            layered_stability_result(6),
        )]),
        acg_profile_observation_counts: std::collections::HashMap::from([(learning_key, 6)]),
        acg_stability: None,
        acg_observation_count: 0,
    }));

    let translated = translate_request(
        &request,
        "agent-1",
        "anthropic",
        plugin.as_ref(),
        &hot_cache,
    )
    .expect("learning-keyed hot cache entries should still translate live requests");

    assert!(translated.content["system"][0]["cache_control"].is_object());
    assert!(translated.content["messages"][0]["content"][0]["cache_control"].is_object());
    assert!(translated.content["messages"][1]["content"][0]["cache_control"].is_object());
}

#[tokio::test]
async fn acg_component_stream_execution_intercept_rewrites_streaming_requests() {
    let request = sample_layered_anthropic_request();
    let plugin = build_provider_plugin("anthropic").expect("anthropic plugin should build");
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(layered_stability_result(6)),
        acg_observation_count: 6,
    }));

    let intercept = create_acg_llm_stream_execution_intercept(
        hot_cache,
        "agent-1".to_string(),
        "anthropic".to_string(),
        plugin,
    );
    let next: LlmStreamExecutionNextFn = Arc::new(|req| {
        Box::pin(async move {
            Ok(Box::pin(tokio_stream::iter(vec![Ok(req.content)]))
                as Pin<
                    Box<dyn tokio_stream::Stream<Item = nemo_flow::error::Result<Json>> + Send>,
                >)
        })
    });

    let mut stream = intercept("anthropic", request, next)
        .await
        .expect("stream intercept should succeed");
    let first = stream
        .next()
        .await
        .expect("stream should yield one item")
        .expect("stream item should be ok");

    assert!(first["system"][0]["cache_control"].is_object());
    assert!(first["messages"][0]["content"][0]["cache_control"].is_object());
    assert!(first["messages"][1]["content"][0]["cache_control"].is_object());
}

#[test]
fn acg_component_build_intent_bundle_requires_at_least_two_observations() {
    let request = sample_annotated_request("claude-sonnet-4-20250514");
    let prompt_ir = crate::acg::ir_builder::build_prompt_ir(&request)
        .expect("annotated request should build prompt ir");
    let plugin = build_provider_plugin("anthropic").expect("anthropic plugin should build");
    let stability = StabilityAnalysisResult {
        scores: vec![
            BlockStabilityScore {
                span_id: SpanId("system-0".to_string()),
                classification: StabilityClass::Stable,
                score: 1.0,
                confidence: 0.02,
                observation_count: 1,
            },
            BlockStabilityScore {
                span_id: SpanId("user-1".to_string()),
                classification: StabilityClass::Stable,
                score: 1.0,
                confidence: 0.02,
                observation_count: 1,
            },
        ],
        stable_prefix_length: 2,
        total_observations: 1,
    };

    let intent_bundle = build_intent_bundle(
        "agent-1",
        "anthropic",
        plugin.as_ref(),
        RequestSurface::AnthropicMessages,
        &request,
        &prompt_ir,
        &stability,
        1,
    );
    assert!(
        intent_bundle.is_none(),
        "single-observation stability should not emit cache intents"
    );

    let intent_bundle = build_intent_bundle(
        "agent-1",
        "anthropic",
        plugin.as_ref(),
        RequestSurface::AnthropicMessages,
        &request,
        &prompt_ir,
        &stability,
        2,
    );
    assert!(
        intent_bundle.is_none(),
        "anthropic planning should still fail open when the prompt cannot clear the economics gate"
    );
}

#[tokio::test]
async fn acg_component_load_persisted_state_prefers_stability_and_falls_back_to_observations() {
    let backend = InMemoryBackend::new();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let stability = layered_stability_result(5);

    backend
        .store_stability("agent-stable", &stability)
        .await
        .unwrap();
    load_persisted_acg_state("agent-stable", &backend, &hot_cache)
        .await
        .unwrap();
    {
        let guard = hot_cache.read().unwrap();
        assert_eq!(guard.acg_stability.as_ref().unwrap().total_observations, 5);
        assert_eq!(guard.acg_observation_count, 5);
    }

    backend
        .store_observations(
            "agent-observed",
            &[sample_prompt_ir("system-0"), sample_prompt_ir("system-1")],
        )
        .await
        .unwrap();
    load_persisted_acg_state("agent-observed", &backend, &hot_cache)
        .await
        .unwrap();

    let guard = hot_cache.read().unwrap();
    assert!(guard.acg_stability.is_none());
    assert_eq!(guard.acg_observation_count, 2);
}

#[tokio::test]
async fn acg_component_load_persisted_state_handles_empty_backend_and_poisoned_cache() {
    let backend = InMemoryBackend::new();
    let hot_cache = sample_hot_cache();

    load_persisted_acg_state("missing-agent", &backend, &hot_cache)
        .await
        .unwrap();
    assert_eq!(hot_cache.read().unwrap().acg_observation_count, 8);

    backend
        .store_observations("poisoned-agent", &[sample_prompt_ir("system-0")])
        .await
        .unwrap();
    let poisoned_cache = sample_hot_cache();
    let poisoned = poisoned_cache.clone();
    let _ = std::panic::catch_unwind(move || {
        let _guard = poisoned.write().unwrap();
        panic!("poison acg hot cache");
    });

    let error = load_persisted_acg_state("poisoned-agent", &backend, &poisoned_cache)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AdaptiveError::Internal(message) if message.contains("hot cache lock poisoned"))
    );
}

#[test]
fn acg_component_build_provider_plugin_supports_passthrough_and_rejects_unknown() {
    assert_eq!(
        build_provider_plugin("passthrough").unwrap().plugin_id(),
        "passthrough"
    );
    assert!(matches!(
        build_provider_plugin("unknown"),
        Err(AdaptiveError::InvalidConfig(message)) if message.contains("unsupported acg provider")
    ));
}

#[test]
fn acg_component_build_semantic_request_view_decodes_supported_surfaces_and_rejects_unknown_shape()
{
    let anthropic = build_semantic_request_view(&sample_anthropic_request()).unwrap();
    assert_eq!(anthropic.request_surface, RequestSurface::AnthropicMessages);
    assert_eq!(
        anthropic.annotated_request.model.as_deref(),
        Some("claude-sonnet-4-20250514")
    );

    let chat = build_semantic_request_view(&sample_openai_chat_request()).unwrap();
    assert_eq!(chat.request_surface, RequestSurface::OpenAIChat);
    assert_eq!(chat.annotated_request.model.as_deref(), Some("gpt-4o"));

    let responses = build_semantic_request_view(&sample_openai_responses_request()).unwrap();
    assert_eq!(responses.request_surface, RequestSurface::OpenAIResponses);
    assert_eq!(
        responses.annotated_request.model.as_deref(),
        Some("gpt-4.1")
    );

    let invalid = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "unknown"}),
    };
    assert!(matches!(
        build_semantic_request_view(&invalid),
        Err(AdaptiveError::Internal(message)) if message.contains("unable to resolve request surface")
    ));
}

#[test]
fn acg_component_build_cache_stability_intent_uses_lowest_prefix_score_and_confidence() {
    let stability = layered_stability_result(6);

    let intent = build_cache_stability_intent(&stability, 3, SharingScope::Session)
        .expect("stable prefix should emit an intent");

    match intent {
        OptimizationIntent::CacheStability(intent) => {
            assert_eq!(intent.stability_score, 0.99);
            assert_eq!(intent.confidence, 0.85);
            assert_eq!(intent.evidence_count, 6);
            assert_eq!(intent.stable_prefix_end, 3);
        }
        other => panic!("unexpected intent: {other:?}"),
    }

    assert!(build_cache_stability_intent(&stability, 0, SharingScope::Session).is_none());
}

#[test]
fn acg_component_build_intent_bundle_supports_openai_and_rejects_unknown_provider() {
    let request = sample_annotated_request("gpt-4o");
    let prompt_ir = crate::acg::ir_builder::build_prompt_ir(&request).unwrap();
    let plugin = build_provider_plugin("openai").unwrap();
    let stability = layered_stability_result(4);

    let bundle = build_intent_bundle(
        "agent-openai",
        "openai",
        plugin.as_ref(),
        RequestSurface::OpenAIChat,
        &request,
        &prompt_ir,
        &stability,
        4,
    )
    .expect("openai surface should emit one cache stability intent");
    assert_eq!(bundle.intents.len(), 1);
    assert_eq!(bundle.agent_identity.toolset_hash, "tool-count-0");

    let passthrough = build_provider_plugin("passthrough").unwrap();
    assert!(
        build_intent_bundle(
            "agent-openai",
            "unsupported",
            passthrough.as_ref(),
            RequestSurface::OpenAIChat,
            &request,
            &prompt_ir,
            &stability,
            4,
        )
        .is_none()
    );
}

#[test]
fn acg_component_anthropic_cache_intents_fail_open_when_surface_or_model_do_not_match() {
    let plugin = build_provider_plugin("anthropic").unwrap();
    let annotated_request = sample_annotated_request("unknown-model");
    let prompt_ir = crate::acg::ir_builder::build_prompt_ir(&annotated_request).unwrap();
    let stability = layered_stability_result(6);

    assert!(
        build_anthropic_cache_intents(
            plugin.as_ref(),
            RequestSurface::OpenAIChat,
            &annotated_request,
            &prompt_ir,
            &stability,
            6,
        )
        .is_none()
    );
    assert!(
        build_anthropic_cache_intents(
            plugin.as_ref(),
            RequestSurface::AnthropicMessages,
            &annotated_request,
            &prompt_ir,
            &stability,
            6,
        )
        .is_none()
    );
}

#[test]
fn acg_component_resolve_model_family_capabilities_prefers_longest_prefix() {
    let plugin = build_provider_plugin("anthropic").unwrap();
    let backend = plugin.capabilities();

    let exact = resolve_model_family_capabilities(&backend, "claude-sonnet-4").unwrap();
    assert_eq!(exact.model_family, "claude-sonnet-4");

    let prefixed = resolve_model_family_capabilities(&backend, "claude-sonnet-4-preview").unwrap();
    assert_eq!(prefixed.model_family, "claude-sonnet-4");

    assert!(resolve_model_family_capabilities(&backend, "missing-model").is_none());
}

#[test]
fn acg_component_build_hint_translation_and_apply_hint_translation_cover_passthrough_and_errors() {
    let request = sample_openai_chat_request();
    let semantic_view = build_semantic_request_view(&request).unwrap();
    let prompt_ir =
        crate::acg::ir_builder::build_prompt_ir(&semantic_view.annotated_request).unwrap();
    let plugin = build_provider_plugin("openai").unwrap();
    let stability = layered_stability_result(4);
    let bundle = build_intent_bundle(
        "agent-openai",
        "openai",
        plugin.as_ref(),
        RequestSurface::OpenAIChat,
        &semantic_view.annotated_request,
        &prompt_ir,
        &stability,
        4,
    )
    .unwrap();
    let input = PluginInput {
        original_request: &request,
        rewritten_request: &request,
        prompt_ir: &prompt_ir,
        intent_bundle: &bundle,
        agent_identity: &bundle.agent_identity,
    };

    let translation = build_hint_translation("passthrough", &input).unwrap();
    assert_eq!(translation.hint_plan.provider, "passthrough");
    assert!(translation.hint_plan.directives.is_empty());
    assert!(matches!(
        build_hint_translation("unknown", &input),
        Err(crate::acg::AcgError::Internal(message)) if message.contains("unsupported semantic provider")
    ));
    assert!(matches!(
        apply_hint_translation(
            &request,
            "anthropic",
            RequestSurface::OpenAIChat,
            &prompt_ir,
            &translation.hint_plan,
        ),
        Err(crate::acg::AcgError::Internal(message)) if message.contains("incompatible")
    ));
}

#[test]
fn acg_component_translate_request_uses_profile_specific_stability_and_fails_open_without_any_stability()
 {
    let request = sample_openai_responses_request();
    let semantic_view = build_semantic_request_view(&request).unwrap();
    let learning_key = crate::acg_profile::derive_acg_learning_key(
        "agent-profile",
        &semantic_view.annotated_request,
    );
    let plugin = build_provider_plugin("openai").unwrap();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::from([(
            learning_key.clone(),
            layered_stability_result(6),
        )]),
        acg_profile_observation_counts: std::collections::HashMap::from([(learning_key, 6)]),
        acg_stability: None,
        acg_observation_count: 0,
    }));

    let translated = translate_request(
        &request,
        "agent-profile",
        "openai",
        plugin.as_ref(),
        &hot_cache,
    );
    assert!(translated.is_some());

    let empty_hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    assert!(
        translate_request(
            &request,
            "agent-profile",
            "openai",
            plugin.as_ref(),
            &empty_hot_cache
        )
        .is_none()
    );
}

#[tokio::test]
async fn acg_component_execution_intercept_rewrites_non_streaming_requests() {
    let request = sample_layered_anthropic_request();
    let plugin = build_provider_plugin("anthropic").unwrap();
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: Some(layered_stability_result(6)),
        acg_observation_count: 6,
    }));

    let intercept = create_acg_llm_execution_intercept(
        hot_cache,
        "agent-1".to_string(),
        "anthropic".to_string(),
        plugin,
    );
    let next: LlmExecutionNextFn = Arc::new(|req| Box::pin(async move { Ok(req.content) }));

    let result = intercept("anthropic", request, next)
        .await
        .expect("execution intercept should succeed");

    assert!(result["system"][0]["cache_control"].is_object());
    assert!(result["messages"][0]["content"][0]["cache_control"].is_object());
}

#[tokio::test]
async fn acg_component_register_fails_open_when_stability_backend_errors() {
    let hot_cache = sample_hot_cache();
    let error = load_persisted_acg_state("agent-1", &FailingStabilityBackend, &hot_cache)
        .await
        .unwrap_err();

    assert!(
        matches!(error, AdaptiveError::Storage(message) if message.contains("stability failed"))
    );
}

#[test]
fn acg_component_decode_request_for_surface_reports_codec_specific_errors() {
    let invalid_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!("not-an-object"),
    };

    assert!(matches!(
        decode_request_for_surface(RequestSurface::AnthropicMessages, &invalid_request),
        Err(AdaptiveError::Internal(message)) if message.contains("failed to decode anthropic request")
    ));
    assert!(matches!(
        decode_request_for_surface(RequestSurface::OpenAIChat, &invalid_request),
        Err(AdaptiveError::Internal(message)) if message.contains("failed to decode openai chat request")
    ));
    assert!(matches!(
        decode_request_for_surface(RequestSurface::OpenAIResponses, &invalid_request),
        Err(AdaptiveError::Internal(message)) if message.contains("failed to decode openai responses request")
    ));
}

#[test]
fn acg_component_build_hint_translation_succeeds_for_openai_and_anthropic() {
    let openai_request = sample_openai_chat_request();
    let openai_view = build_semantic_request_view(&openai_request).unwrap();
    let openai_prompt_ir =
        crate::acg::ir_builder::build_prompt_ir(&openai_view.annotated_request).unwrap();
    let openai_plugin = build_provider_plugin("openai").unwrap();
    let openai_bundle = build_intent_bundle(
        "agent-openai",
        "openai",
        openai_plugin.as_ref(),
        RequestSurface::OpenAIChat,
        &openai_view.annotated_request,
        &openai_prompt_ir,
        &layered_stability_result(4),
        4,
    )
    .unwrap();
    let openai_input = PluginInput {
        original_request: &openai_request,
        rewritten_request: &openai_request,
        prompt_ir: &openai_prompt_ir,
        intent_bundle: &openai_bundle,
        agent_identity: &openai_bundle.agent_identity,
    };

    let openai_translation = build_hint_translation("openai", &openai_input).unwrap();
    assert_eq!(openai_translation.hint_plan.provider, "openai");
    assert!(!openai_translation.hint_plan.directives.is_empty());

    let openai_applied = apply_hint_translation(
        &openai_request,
        "openai",
        RequestSurface::OpenAIChat,
        &openai_prompt_ir,
        &openai_translation.hint_plan,
    )
    .unwrap();
    assert_eq!(
        openai_applied.content["tools"][0]["function"]["parameters"],
        json!({"a": 2, "z": 1}),
    );

    let anthropic_request = sample_layered_anthropic_request();
    let anthropic_view = build_semantic_request_view(&anthropic_request).unwrap();
    let anthropic_prompt_ir =
        crate::acg::ir_builder::build_prompt_ir(&anthropic_view.annotated_request).unwrap();
    let anthropic_plugin = build_provider_plugin("anthropic").unwrap();
    let anthropic_bundle = build_intent_bundle(
        "agent-anthropic",
        "anthropic",
        anthropic_plugin.as_ref(),
        RequestSurface::AnthropicMessages,
        &anthropic_view.annotated_request,
        &anthropic_prompt_ir,
        &layered_stability_result(6),
        6,
    )
    .unwrap();
    let anthropic_input = PluginInput {
        original_request: &anthropic_request,
        rewritten_request: &anthropic_request,
        prompt_ir: &anthropic_prompt_ir,
        intent_bundle: &anthropic_bundle,
        agent_identity: &anthropic_bundle.agent_identity,
    };

    let anthropic_translation = build_hint_translation("anthropic", &anthropic_input).unwrap();
    assert!(anthropic_translation.hint_plan.has_anthropic_breakpoint());

    let anthropic_applied = apply_hint_translation(
        &anthropic_request,
        "anthropic",
        RequestSurface::AnthropicMessages,
        &anthropic_prompt_ir,
        &anthropic_translation.hint_plan,
    )
    .unwrap();
    assert!(anthropic_applied.content["system"][0]["cache_control"].is_object());
}

#[test]
fn acg_component_anthropic_cache_intents_require_model_name() {
    let plugin = build_provider_plugin("anthropic").unwrap();
    let mut annotated_request = sample_annotated_request("claude-sonnet-4-20250514");
    annotated_request.model = None;
    let prompt_ir = crate::acg::ir_builder::build_prompt_ir(&annotated_request).unwrap();

    assert!(
        build_anthropic_cache_intents(
            plugin.as_ref(),
            RequestSurface::AnthropicMessages,
            &annotated_request,
            &prompt_ir,
            &layered_stability_result(6),
            6,
        )
        .is_none()
    );
}

#[test]
fn acg_component_translate_request_fails_open_for_invalid_requests() {
    let invalid_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "unknown"}),
    };
    let plugin = build_provider_plugin("anthropic").unwrap();

    assert!(
        translate_request(
            &invalid_request,
            "agent-invalid",
            "anthropic",
            plugin.as_ref(),
            &sample_hot_cache(),
        )
        .is_none()
    );
}

#[tokio::test]
async fn acg_component_execution_intercept_passes_original_request_when_translation_is_skipped() {
    let invalid_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "unknown"}),
    };
    let plugin = build_provider_plugin("anthropic").unwrap();
    let intercept = create_acg_llm_execution_intercept(
        sample_hot_cache(),
        "agent-1".to_string(),
        "anthropic".to_string(),
        plugin,
    );
    let next: LlmExecutionNextFn = Arc::new(|req| Box::pin(async move { Ok(req.content) }));

    let result = intercept("anthropic", invalid_request.clone(), next)
        .await
        .expect("execution intercept should succeed");

    assert_eq!(result, invalid_request.content);
}
