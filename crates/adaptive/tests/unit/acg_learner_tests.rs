// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for acg learner in the NeMo Relay adaptive crate.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use chrono::Utc;
use nemo_relay::codec::request::{
    AnnotatedLlmRequest, FunctionDefinition, Message, MessageContent, ToolDefinition,
};
use uuid::Uuid;

use super::*;

use crate::acg::profile::{BlockStabilityScore, StabilityClass};
use crate::acg::prompt_ir::SpanId;
use crate::acg_profile::derive_acg_learning_key;
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::plan::ExecutionPlan;
use crate::types::records::{CallRecord, RunRecord};

fn sample_request(model: &str, system: &str, user: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text(system.to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text(user.to_string()),
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

fn layered_agent_request(work_item: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are a repo coding agent.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Apply the repository review checklist.".to_string()),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Acknowledged. I will review with that checklist.".to_string(),
                )),
                tool_calls: None,
                name: None,
            },
            Message::User {
                content: MessageContent::Text(work_item.to_string()),
                name: None,
            },
        ],
        model: Some("gpt-4o".to_string()),
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

fn layered_agent_request_with_extra_suffix(work_item: &str) -> AnnotatedLlmRequest {
    let mut request = layered_agent_request(work_item);
    request.messages.push(Message::Assistant {
        content: Some(MessageContent::Text(
            "I need one more repository fact before final review.".to_string(),
        )),
        tool_calls: None,
        name: None,
    });
    request.messages.push(Message::User {
        content: MessageContent::Text("Additional volatile suffix context".to_string()),
        name: None,
    });
    request
}

fn pipe_tool_definition() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "policy_lookup".to_string(),
            description: Some("Look up policy guidance for a moderation item.".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "policy_area": {"type": "string"}
                },
                "required": ["policy_area"]
            })),
        },
    }
}

fn pipe_response_format(include_severity: bool) -> serde_json::Value {
    let mut properties = serde_json::json!({
        "decision": {"type": "string"},
        "reason": {"type": "string"}
    });
    let mut required = vec![
        serde_json::Value::String("decision".to_string()),
        serde_json::Value::String("reason".to_string()),
    ];
    if include_severity {
        properties["severity"] = serde_json::json!({"type": "string"});
        required.push(serde_json::Value::String("severity".to_string()));
    }

    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "moderation_decision",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    })
}

fn layered_agent_pipe_request(work_item: &str, include_severity: bool) -> AnnotatedLlmRequest {
    let mut request = layered_agent_request(work_item);
    request.messages[0] = Message::System {
        content: MessageContent::Text("Apply the moderation policy exactly.".to_string()),
        name: None,
    };
    request.messages[1] = Message::User {
        content: MessageContent::Text(
            "Use the reusable moderation workflow before judging the item.".to_string(),
        ),
        name: None,
    };
    request.messages[2] = Message::Assistant {
        content: Some(MessageContent::Text(
            "I will return only the required moderation decision object.".to_string(),
        )),
        tool_calls: None,
        name: None,
    };
    request.tools = Some(vec![pipe_tool_definition()]);
    request.extra.insert(
        "response_format".to_string(),
        pipe_response_format(include_severity),
    );
    request
}

fn sample_run(requests: Vec<AnnotatedLlmRequest>) -> RunRecord {
    let now = Utc::now();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: "agent-a".to_string(),
        calls: requests
            .into_iter()
            .map(|request| CallRecord {
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
                annotated_request: Some(request.into()),
                annotated_response: None,
            })
            .collect(),
        started_at: now,
        ended_at: Some(now),
    }
}

fn stable_score(index: usize) -> BlockStabilityScore {
    BlockStabilityScore {
        span_id: SpanId(format!("span-{index}")),
        classification: StabilityClass::Stable,
        score: 1.0,
        confidence: 1.0,
        observation_count: 3,
    }
}

fn variable_score(index: usize) -> BlockStabilityScore {
    BlockStabilityScore {
        span_id: SpanId(format!("span-{index}")),
        classification: StabilityClass::Variable,
        score: 0.0,
        confidence: 1.0,
        observation_count: 3,
    }
}

#[test]
fn acg_convergence_features_ignore_variable_suffix_shape() {
    let short_suffix = crate::acg::stability::StabilityAnalysisResult {
        scores: vec![stable_score(0), stable_score(1), variable_score(2)],
        stable_prefix_length: 2,
        stable_prefix_fingerprint: None,
        total_observations: 3,
        converged: false,
    };
    let long_suffix = crate::acg::stability::StabilityAnalysisResult {
        scores: vec![
            stable_score(0),
            stable_score(1),
            variable_score(2),
            variable_score(3),
            variable_score(4),
        ],
        stable_prefix_length: 2,
        stable_prefix_fingerprint: None,
        total_observations: 3,
        converged: false,
    };

    let (short_betti, short_drift, short_error) =
        AcgLearner::stability_to_convergence_features(&short_suffix);
    let (long_betti, long_drift, long_error) =
        AcgLearner::stability_to_convergence_features(&long_suffix);

    assert_eq!(short_betti, BettiNumbers::new(2, 0));
    assert_eq!(short_betti, long_betti);
    assert!((short_drift - long_drift).abs() < f64::EPSILON);
    assert!((short_error - long_error).abs() < f64::EPSILON);
}

fn empty_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

struct SeedObservationBackend {
    observations: std::sync::RwLock<HashMap<String, Vec<PromptIR>>>,
    stability: std::sync::RwLock<HashMap<String, crate::acg::stability::StabilityAnalysisResult>>,
    fail_observation_store: AtomicBool,
    fail_stability_store: AtomicBool,
    load_observation_count: AtomicUsize,
}

impl SeedObservationBackend {
    fn empty() -> Self {
        Self {
            observations: std::sync::RwLock::new(HashMap::new()),
            stability: std::sync::RwLock::new(HashMap::new()),
            fail_observation_store: AtomicBool::new(false),
            fail_stability_store: AtomicBool::new(false),
            load_observation_count: AtomicUsize::new(0),
        }
    }

    fn new(seed_key: &str, observations: Vec<PromptIR>) -> Self {
        let backend = Self::empty();
        backend
            .observations
            .write()
            .unwrap()
            .insert(seed_key.to_string(), observations);
        backend
    }

    fn fail_observation_stores(&self) {
        self.fail_observation_store.store(true, Ordering::SeqCst);
    }

    fn allow_observation_stores(&self) {
        self.fail_observation_store.store(false, Ordering::SeqCst);
    }

    fn fail_stability_stores(&self) {
        self.fail_stability_store.store(true, Ordering::SeqCst);
    }

    fn allow_stability_stores(&self) {
        self.fail_stability_store.store(false, Ordering::SeqCst);
    }

    fn seed_stability(
        &self,
        agent_id: &str,
        stability: crate::acg::stability::StabilityAnalysisResult,
    ) {
        self.stability
            .write()
            .unwrap()
            .insert(agent_id.to_string(), stability);
    }

    fn load_observation_count(&self) -> usize {
        self.load_observation_count.load(Ordering::SeqCst)
    }
}

impl StorageBackendDyn for SeedObservationBackend {
    fn store_run_dyn<'a>(
        &'a self,
        _record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_plan_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_trie<'a>(
        &'a self,
        _agent_id: &'a str,
        _envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_trie<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }

    fn store_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
        _state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn load_accumulators<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }

    fn store_observations<'a>(
        &'a self,
        agent_id: &'a str,
        observations: &'a [PromptIR],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let observations = observations.to_vec();
        Box::pin(async move {
            if self.fail_observation_store.load(Ordering::SeqCst) {
                return Err(AdaptiveError::Storage(
                    "forced observation storage failure".to_string(),
                ));
            }
            self.observations
                .write()
                .unwrap()
                .insert(agent_id.to_string(), observations);
            Ok(())
        })
    }

    fn load_observations<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Vec<PromptIR>>>> + Send + 'a>> {
        self.load_observation_count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(self.observations.read().unwrap().get(agent_id).cloned()) })
    }

    fn store_stability<'a>(
        &'a self,
        agent_id: &'a str,
        result: &'a crate::acg::stability::StabilityAnalysisResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let result = result.clone();
        Box::pin(async move {
            if self.fail_stability_store.load(Ordering::SeqCst) {
                return Err(AdaptiveError::Storage(
                    "forced stability storage failure".to_string(),
                ));
            }
            self.stability
                .write()
                .unwrap()
                .insert(agent_id.to_string(), result);
            Ok(())
        })
    }

    fn load_stability<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<crate::acg::stability::StabilityAnalysisResult>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move { Ok(self.stability.read().unwrap().get(agent_id).cloned()) })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_returns_early_without_llm_requests() {
    let learner = AcgLearner::new("agent-a", 2, StabilityThresholds::default());
    let run = RunRecord {
        id: Uuid::now_v7(),
        agent_id: "agent-a".to_string(),
        calls: vec![],
        started_at: Utc::now(),
        ended_at: None,
    };
    let backend = crate::storage::memory::InMemoryBackend::new();

    learner
        .process_run(&run, &backend, &empty_cache())
        .await
        .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_trims_observation_windows_and_updates_agent_seed() {
    let learner = AcgLearner::new("agent-a", 2, StabilityThresholds::default());
    let new_request = sample_request("gpt-4o", "System B", "Prompt B");
    let learning_key = derive_acg_learning_key("agent-a", &new_request);
    let old_ir = build_prompt_ir(&sample_request("gpt-4o", "System A", "Prompt A")).unwrap();
    let older_ir = build_prompt_ir(&sample_request("gpt-4o", "System OLD", "Prompt OLD")).unwrap();
    let backend = SeedObservationBackend::new(&learning_key, vec![older_ir, old_ir]);
    let hot_cache = empty_cache();

    learner
        .process_run(&sample_run(vec![new_request]), &backend, &hot_cache)
        .await
        .unwrap();

    let stored = backend
        .load_observations(&learning_key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.len(), 2);
    assert!(stored.iter().all(|ir| ir.blocks[0].content != "System OLD"));
    assert!(
        backend
            .load_observations("agent-a")
            .await
            .unwrap()
            .is_some()
    );

    let guard = hot_cache.read().unwrap();
    assert_eq!(guard.acg_profiles.len(), 1);
    assert!(guard.acg_stability.is_some());
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_prefers_profile_with_longer_stable_prefix_and_handles_poisoned_cache() {
    let learner = AcgLearner::new(
        "agent-a",
        4,
        StabilityThresholds {
            stable_threshold: 0.99,
            semi_stable_threshold: 0.5,
            min_observations_for_full_confidence: 1,
        },
    );
    let run = sample_run(vec![
        sample_request("gpt-4o", "Stable system", "Stable prompt"),
        sample_request("gpt-4o-mini", "Variable system", "Variable prompt"),
    ]);
    let hot_cache = empty_cache();
    let backend = crate::storage::memory::InMemoryBackend::new();

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();
    {
        let guard = hot_cache.read().unwrap();
        assert_eq!(guard.acg_profiles.len(), 2);
        assert!(guard.acg_observation_count >= 1);
    }

    let poisoned_cache = empty_cache();
    let poisoned = poisoned_cache.clone();
    let _ = std::panic::catch_unwind(move || {
        let _guard = poisoned.write().unwrap();
        panic!("poison acg learner cache");
    });
    let error = learner
        .process_run(&run, &backend, &poisoned_cache)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AdaptiveError::Internal(message) if message.contains("hot cache lock poisoned"))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_does_not_persist_converged_stability_when_observation_store_fails() {
    let stability_window = 3;
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let backend = SeedObservationBackend::empty();
    let hot_cache = empty_cache();

    for _ in 0..stability_window - 1 {
        learner
            .process_run(&sample_run(vec![request.clone()]), &backend, &hot_cache)
            .await
            .unwrap();
    }

    let before_failure = backend
        .load_stability(&learning_key)
        .await
        .unwrap()
        .expect("stability should be stored before the failing epoch");
    assert!(!before_failure.converged);

    backend.fail_observation_stores();
    let error = learner
        .process_run(&sample_run(vec![request]), &backend, &hot_cache)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AdaptiveError::Storage(message) if message.contains("forced observation storage failure"))
    );

    let after_failure = backend
        .load_stability(&learning_key)
        .await
        .unwrap()
        .expect("previous non-converged stability should remain stored");
    assert!(
        !after_failure.converged,
        "converged stability must not be persisted before observations are stored"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_does_not_advance_convergence_epoch_when_observation_store_fails() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let backend = SeedObservationBackend::empty();
    let hot_cache = empty_cache();

    learner
        .process_run(&sample_run(vec![request.clone()]), &backend, &hot_cache)
        .await
        .unwrap();

    backend.fail_observation_stores();
    let error = learner
        .process_run(&sample_run(vec![request.clone()]), &backend, &hot_cache)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AdaptiveError::Storage(message) if message.contains("forced observation storage failure"))
    );

    backend.allow_observation_stores();
    learner
        .process_run(&sample_run(vec![request]), &backend, &hot_cache)
        .await
        .unwrap();

    let recovered_stability = backend
        .load_stability(&learning_key)
        .await
        .unwrap()
        .expect("stability should be stored after recovery");
    assert!(
        !recovered_stability.converged,
        "failed observation storage must not advance the in-memory convergence epoch"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_repairs_converged_stability_without_observations() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let seed_observation = build_prompt_ir(&request).unwrap();
    let mut stale_stability = analyze_stability(
        std::slice::from_ref(&seed_observation),
        &StabilityThresholds::default(),
    );
    stale_stability.converged = true;

    let backend = SeedObservationBackend::empty();
    backend.seed_stability(&learning_key, stale_stability);

    learner
        .process_run(&sample_run(vec![request]), &backend, &empty_cache())
        .await
        .unwrap();

    let repaired_observations = backend
        .load_observations(&learning_key)
        .await
        .unwrap()
        .expect("missing observations should be repaired instead of trusting converged stability");
    assert_eq!(repaired_observations.len(), 1);

    let repaired_stability = backend
        .load_stability(&learning_key)
        .await
        .unwrap()
        .expect("repaired stability should be stored");
    assert!(
        !repaired_stability.converged,
        "a repaired single-observation profile should re-enter normal convergence"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_repairs_converged_stability_with_empty_observations() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let seed_observation = build_prompt_ir(&request).unwrap();
    let mut stale_stability = analyze_stability(
        std::slice::from_ref(&seed_observation),
        &StabilityThresholds::default(),
    );
    stale_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, Vec::new());
    backend.seed_stability(&learning_key, stale_stability);

    learner
        .process_run(&sample_run(vec![request]), &backend, &empty_cache())
        .await
        .unwrap();

    let repaired_observations = backend
        .load_observations(&learning_key)
        .await
        .unwrap()
        .expect("empty observations should be repaired instead of trusting converged stability");
    assert_eq!(repaired_observations.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_reuses_converged_stability_without_loading_observations() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let seed_observation = build_prompt_ir(&request).unwrap();
    let observations = vec![
        seed_observation.clone(),
        seed_observation.clone(),
        seed_observation.clone(),
    ];
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);
    let hot_cache = empty_cache();

    learner
        .process_run(&sample_run(vec![request]), &backend, &hot_cache)
        .await
        .unwrap();

    assert_eq!(
        backend.load_observation_count(),
        0,
        "converged profiles should reuse cached stability without reading the observation window"
    );
    let guard = hot_cache.read().unwrap();
    assert_eq!(guard.acg_profiles.len(), 1);
    assert_eq!(guard.acg_observation_count, 3);
    assert!(guard.acg_stability.as_ref().unwrap().converged);
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_persists_cached_winner_as_agent_stability() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let seed_observation = build_prompt_ir(&request).unwrap();
    let observations = vec![
        seed_observation.clone(),
        seed_observation.clone(),
        seed_observation.clone(),
    ];
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability.clone());

    learner
        .process_run(&sample_run(vec![request]), &backend, &empty_cache())
        .await
        .unwrap();

    let agent_stability = backend
        .load_stability("agent-a")
        .await
        .unwrap()
        .expect("cached aggregate winner should be persisted under the base agent id");
    assert_eq!(agent_stability, converged_stability);
}

#[test]
fn acg_learner_keeps_stronger_cached_aggregate_over_weaker_normal_candidate() {
    let cached = crate::acg::stability::StabilityAnalysisResult {
        scores: vec![stable_score(0), stable_score(1), stable_score(2)],
        stable_prefix_length: 3,
        stable_prefix_fingerprint: None,
        total_observations: 3,
        converged: true,
    };
    let weaker_normal = crate::acg::stability::StabilityAnalysisResult {
        scores: vec![stable_score(0)],
        stable_prefix_length: 1,
        stable_prefix_fingerprint: None,
        total_observations: 20,
        converged: false,
    };

    assert!(
        !AcgLearner::should_replace_aggregate(&weaker_normal, Some(&cached)),
        "normal candidates should compare against the current cached aggregate winner"
    );
}

#[test]
fn acg_learner_keeps_converged_aggregate_when_prefix_ties_non_converged_candidate() {
    let current = crate::acg::stability::StabilityAnalysisResult {
        scores: vec![stable_score(0), stable_score(1), stable_score(2)],
        stable_prefix_length: 3,
        stable_prefix_fingerprint: None,
        total_observations: 3,
        converged: true,
    };
    let candidate = crate::acg::stability::StabilityAnalysisResult {
        scores: vec![stable_score(0), stable_score(1), stable_score(2)],
        stable_prefix_length: 3,
        stable_prefix_fingerprint: None,
        total_observations: 20,
        converged: false,
    };

    assert!(
        !AcgLearner::should_replace_aggregate(&candidate, Some(&current)),
        "a converged aggregate should not regress to a non-converged candidate when the stable prefix ties"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_does_not_reuse_converged_stability_when_convergence_disabled() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: false,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let seed_observation = build_prompt_ir(&request).unwrap();
    let observations = vec![
        seed_observation.clone(),
        seed_observation.clone(),
        seed_observation.clone(),
    ];
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);

    learner
        .process_run(&sample_run(vec![request]), &backend, &empty_cache())
        .await
        .unwrap();

    assert!(
        backend.load_observation_count() > 0,
        "disabled convergence must not reuse cached converged stability"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_resets_convergence_detector_when_cached_profile_reopens() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let backend = SeedObservationBackend::empty();
    let hot_cache = empty_cache();

    for _ in 0..4 {
        learner
            .process_run(&sample_run(vec![request.clone()]), &backend, &hot_cache)
            .await
            .unwrap();
    }

    let mut stale_stability = backend
        .load_stability(&learning_key)
        .await
        .unwrap()
        .expect("profile should have stored stability");
    assert!(stale_stability.converged);
    stale_stability.stable_prefix_fingerprint = Some("stale-prefix-fingerprint".to_string());
    backend.seed_stability(&learning_key, stale_stability);

    learner
        .process_run(&sample_run(vec![request]), &backend, &hot_cache)
        .await
        .unwrap();

    let reopened_stability = backend
        .load_stability(&learning_key)
        .await
        .unwrap()
        .expect("reopened profile should store recomputed stability");
    assert!(
        !reopened_stability.converged,
        "reopened learning should require a fresh stability window"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_rolls_back_convergence_detector_when_stability_store_fails() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let learning_key = derive_acg_learning_key("agent-a", &request);
    let backend = SeedObservationBackend::empty();
    let hot_cache = empty_cache();

    learner
        .process_run(&sample_run(vec![request.clone()]), &backend, &hot_cache)
        .await
        .unwrap();

    let epoch_before_failure = learner
        .convergence_detectors
        .read()
        .unwrap()
        .get(&learning_key)
        .expect("first successful run should create a detector")
        .epoch();
    assert_eq!(epoch_before_failure, 1);

    backend.fail_stability_stores();
    let error = learner
        .process_run(&sample_run(vec![request]), &backend, &hot_cache)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AdaptiveError::Storage(message) if message.contains("forced stability storage failure"))
    );
    backend.allow_stability_stores();

    let epoch_after_failure = learner
        .convergence_detectors
        .read()
        .unwrap()
        .get(&learning_key)
        .expect("failed stability persistence should restore the previous detector")
        .epoch();
    assert_eq!(
        epoch_after_failure, epoch_before_failure,
        "failed stability persistence must not leave the in-memory detector ahead of storage"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_reuses_converged_profile_when_suffix_topology_changes() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let base = layered_agent_request("Review changed bundle #1");
    let grown = layered_agent_request_with_extra_suffix("Review changed bundle #2");

    let learning_key = derive_acg_learning_key("agent-a", &base);
    assert_eq!(learning_key, derive_acg_learning_key("agent-a", &grown));

    let observations = ["#1", "#2", "#3"]
        .into_iter()
        .map(|suffix| {
            build_prompt_ir(&layered_agent_request(&format!(
                "Review changed bundle {suffix}"
            )))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    assert_eq!(converged_stability.stable_prefix_length, 3);
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);

    learner
        .process_run(&sample_run(vec![grown]), &backend, &empty_cache())
        .await
        .unwrap();

    assert_eq!(
        backend.load_observation_count(),
        0,
        "suffix-only topology changes should reuse the cacheable stable prefix"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_reopens_converged_profile_when_stable_prefix_topology_changes() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let base = layered_agent_request("Review changed bundle #1");
    let mut prefix_changed = layered_agent_request("Review changed bundle #2");
    prefix_changed.messages[2] = Message::User {
        content: MessageContent::Text("Inserted user context before the work item.".to_string()),
        name: None,
    };

    let learning_key = derive_acg_learning_key("agent-a", &base);
    assert_eq!(
        learning_key,
        derive_acg_learning_key("agent-a", &prefix_changed)
    );

    let observations = ["#1", "#2", "#3"]
        .into_iter()
        .map(|suffix| {
            build_prompt_ir(&layered_agent_request(&format!(
                "Review changed bundle {suffix}"
            )))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    assert_eq!(converged_stability.stable_prefix_length, 3);
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);

    learner
        .process_run(&sample_run(vec![prefix_changed]), &backend, &empty_cache())
        .await
        .unwrap();

    assert!(
        backend.load_observation_count() > 0,
        "stable-prefix topology changes must inspect observations instead of reusing convergence"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_reopens_converged_profile_when_stable_prefix_content_changes() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let base = layered_agent_request("Review changed bundle #1");
    let mut prefix_changed = layered_agent_request("Review changed bundle #2");
    prefix_changed.messages[2] = Message::Assistant {
        content: Some(MessageContent::Text(
            "Acknowledged. I will use the updated stable review lens.".to_string(),
        )),
        tool_calls: None,
        name: None,
    };

    let learning_key = derive_acg_learning_key("agent-a", &base);
    assert_eq!(
        learning_key,
        derive_acg_learning_key("agent-a", &prefix_changed)
    );

    let observations = ["#1", "#2", "#3"]
        .into_iter()
        .map(|suffix| {
            build_prompt_ir(&layered_agent_request(&format!(
                "Review changed bundle {suffix}"
            )))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    assert_eq!(converged_stability.stable_prefix_length, 3);
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);

    learner
        .process_run(&sample_run(vec![prefix_changed]), &backend, &empty_cache())
        .await
        .unwrap();

    assert!(
        backend.load_observation_count() > 0,
        "stable-prefix content changes must inspect observations instead of reusing convergence"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_reuses_converged_agent_pipe_when_only_task_suffix_changes() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let base = layered_agent_pipe_request("Review forum post #1", false);
    let next_task = layered_agent_pipe_request("Review forum post #2", false);

    let learning_key = derive_acg_learning_key("agent-a", &base);
    assert_eq!(learning_key, derive_acg_learning_key("agent-a", &next_task));

    let observations = ["#1", "#2", "#3"]
        .into_iter()
        .map(|suffix| {
            build_prompt_ir(&layered_agent_pipe_request(
                &format!("Review forum post {suffix}"),
                false,
            ))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    assert_eq!(
        converged_stability.stable_prefix_length, 5,
        "system policy, tool schema, output contract, workflow scaffold, and output-contract acknowledgement should be the reusable pipe"
    );
    assert!(converged_stability.stable_prefix_fingerprint.is_some());
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);

    learner
        .process_run(&sample_run(vec![next_task]), &backend, &empty_cache())
        .await
        .unwrap();

    assert_eq!(
        backend.load_observation_count(),
        0,
        "task-specific suffix content should not invalidate the reusable agent pipe"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_reopens_converged_agent_pipe_when_output_contract_changes() {
    let learner = AcgLearner::new_with_convergence(
        "agent-a",
        20,
        StabilityThresholds::default(),
        Some(ConvergenceConfig {
            enabled: true,
            epsilon: 0.001,
            stability_window: 3,
        }),
    );
    let base = layered_agent_pipe_request("Review forum post #1", false);
    let changed_contract = layered_agent_pipe_request("Review forum post #2", true);

    let learning_key = derive_acg_learning_key("agent-a", &base);
    assert_eq!(
        learning_key,
        derive_acg_learning_key("agent-a", &changed_contract)
    );

    let observations = ["#1", "#2", "#3"]
        .into_iter()
        .map(|suffix| {
            build_prompt_ir(&layered_agent_pipe_request(
                &format!("Review forum post {suffix}"),
                false,
            ))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut converged_stability = analyze_stability(&observations, &StabilityThresholds::default());
    assert_eq!(converged_stability.stable_prefix_length, 5);
    assert!(converged_stability.stable_prefix_fingerprint.is_some());
    converged_stability.converged = true;

    let backend = SeedObservationBackend::new(&learning_key, observations);
    backend.seed_stability(&learning_key, converged_stability);

    learner
        .process_run(
            &sample_run(vec![changed_contract]),
            &backend,
            &empty_cache(),
        )
        .await
        .unwrap();

    assert!(
        backend.load_observation_count() > 0,
        "output contract changes must reopen learning instead of reusing stale convergence"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_seeds_agent_cache_from_profile_with_more_observations_when_prefixes_tie() {
    let learner = AcgLearner::new("agent-a", 4, StabilityThresholds::default());
    let preferred_request = sample_request("gpt-4o", "Stable system", "Stable prompt");
    let preferred_key = derive_acg_learning_key("agent-a", &preferred_request);
    let preferred_seed = build_prompt_ir(&preferred_request).unwrap();
    let backend = SeedObservationBackend::new(&preferred_key, vec![preferred_seed]);
    let hot_cache = empty_cache();

    learner
        .process_run(
            &sample_run(vec![
                preferred_request,
                sample_request("gpt-4o-mini", "Other system", "Other prompt"),
            ]),
            &backend,
            &hot_cache,
        )
        .await
        .unwrap();

    let aggregate = backend.load_observations("agent-a").await.unwrap().unwrap();
    assert_eq!(aggregate.len(), 2);
    assert!(
        aggregate
            .iter()
            .all(|ir| ir.blocks[0].content == "Stable system")
    );

    let guard = hot_cache.read().unwrap();
    assert_eq!(guard.acg_observation_count, 2);
    assert!(
        guard
            .acg_profile_observation_counts
            .values()
            .any(|count| *count == 2)
    );
    assert!(
        guard
            .acg_profile_observation_counts
            .values()
            .any(|count| *count == 1)
    );
}
