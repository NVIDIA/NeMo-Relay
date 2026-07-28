// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for acg learner in the NeMo Relay adaptive crate.

use std::future::Future;
use std::pin::Pin;

use chrono::Utc;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use uuid::Uuid;

use super::*;

use crate::acg_profile::derive_acg_learning_key;
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::plan::ExecutionPlan;
use crate::types::records::{CallRecord, RunRecord};

fn sample_request(model: &str, system: &str, user: &str) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        instructions: None,
        api_specific: None,
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
}

impl SeedObservationBackend {
    fn new(seed_key: &str, observations: Vec<PromptIR>) -> Self {
        Self {
            observations: std::sync::RwLock::new(HashMap::from([(
                seed_key.to_string(),
                observations,
            )])),
            stability: std::sync::RwLock::new(HashMap::new()),
        }
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
        Box::pin(async move { Ok(self.observations.read().unwrap().get(agent_id).cloned()) })
    }

    fn store_stability<'a>(
        &'a self,
        agent_id: &'a str,
        result: &'a crate::acg::stability::StabilityAnalysisResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let result = result.clone();
        Box::pin(async move {
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
async fn acg_learner_accumulates_same_scaffold_calls_from_one_run_in_one_profile() {
    let first = sample_request("gpt-4o", "Stable system", "task alpha");
    let second = sample_request("gpt-4o", "Stable system", "task beta");
    let learning_key = derive_acg_learning_key("agent-a", &first);
    assert_eq!(learning_key, derive_acg_learning_key("agent-a", &second));

    let learner = AcgLearner::new("agent-a", 8, StabilityThresholds::default());
    let backend = crate::storage::memory::InMemoryBackend::new();
    let hot_cache = empty_cache();
    learner
        .process_run(&sample_run(vec![first, second]), &backend, &hot_cache)
        .await
        .unwrap();

    assert_eq!(
        backend
            .load_observations(&learning_key)
            .await
            .unwrap()
            .unwrap()
            .len(),
        2
    );
    let guard = hot_cache.read().unwrap();
    assert_eq!(guard.acg_profiles.len(), 1);
    assert_eq!(guard.acg_profile_observation_counts[&learning_key], 2);
    let stored = &guard.acg_profiles[&learning_key];
    assert_eq!(
        stored.stable_prefix_fingerprint,
        crate::acg::stability::profile_prefix_fingerprint(
            &build_prompt_ir(&sample_request("gpt-4o", "Stable system", "task alpha")).unwrap(),
            stored.stable_prefix_length,
            &learning_key,
        )
    );
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

/// Backend that yields between the load and store halves of every learner
/// read-modify-write so concurrent `process_run` futures interleave.
struct InterleavingBackend {
    inner: crate::storage::memory::InMemoryBackend,
}

impl InterleavingBackend {
    /// Wrap a fresh in-memory backend with yields around observation and stability writes.
    fn new() -> Self {
        Self {
            inner: crate::storage::memory::InMemoryBackend::new(),
        }
    }
}

impl StorageBackendDyn for InterleavingBackend {
    fn store_run_dyn<'a>(
        &'a self,
        record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        self.inner.store_run_dyn(record)
    }

    fn load_plan_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>> {
        self.inner.load_plan_dyn(agent_id)
    }

    fn list_runs_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>> {
        self.inner.list_runs_dyn(agent_id)
    }

    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        self.inner.store_trie(agent_id, envelope)
    }

    fn load_trie<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>> {
        self.inner.load_trie(agent_id)
    }

    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        self.inner.store_accumulators(agent_id, state)
    }

    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>> {
        self.inner.load_accumulators(agent_id)
    }

    fn store_observations<'a>(
        &'a self,
        agent_id: &'a str,
        observations: &'a [PromptIR],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            tokio::task::yield_now().await;
            self.inner.store_observations(agent_id, observations).await
        })
    }

    fn load_observations<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Vec<PromptIR>>>> + Send + 'a>> {
        Box::pin(async move {
            tokio::task::yield_now().await;
            self.inner.load_observations(agent_id).await
        })
    }

    fn store_stability<'a>(
        &'a self,
        agent_id: &'a str,
        result: &'a crate::acg::stability::StabilityAnalysisResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            tokio::task::yield_now().await;
            self.inner.store_stability(agent_id, result).await
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
        self.inner.load_stability(agent_id)
    }
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_keeps_persisted_fingerprints_consistent_under_interleaved_runs() {
    let learner = AcgLearner::new("agent-a", 8, StabilityThresholds::default());
    let backend = InterleavingBackend::new();
    let hot_cache = empty_cache();

    let alpha = sample_request("gpt-4o", "Alpha system", "alpha task");
    let beta = sample_request("gpt-4o", "Beta system", "beta task");
    let alpha_key = derive_acg_learning_key("agent-a", &alpha);
    let beta_key = derive_acg_learning_key("agent-a", &beta);
    assert_ne!(alpha_key, beta_key);

    let alpha_run = sample_run(vec![alpha.clone()]);
    let beta_run = sample_run(vec![beta.clone()]);
    let alpha_replay = sample_run(vec![alpha.clone()]);
    let (first, second, third) = tokio::join!(
        learner.process_run(&alpha_run, &backend, &hot_cache),
        learner.process_run(&beta_run, &backend, &hot_cache),
        learner.process_run(&alpha_replay, &backend, &hot_cache),
    );
    first.unwrap();
    second.unwrap();
    third.unwrap();

    // The learner's load/store cycle is not atomic, so interleaved runs can drop
    // an observation from the window. That is a pre-existing throughput limit, not
    // a safety one: what must hold is that every persisted stability record stays
    // reproducible from the observations persisted under the same learning key, so
    // a fingerprint is never paired with a foreign window.
    assert_eq!(
        backend
            .load_observations(&alpha_key)
            .await
            .unwrap()
            .unwrap()
            .len(),
        1,
    );

    for key in [&alpha_key, &beta_key] {
        let observations = backend.load_observations(key).await.unwrap().unwrap();
        let stability = backend.load_stability(key).await.unwrap().unwrap();
        assert_eq!(
            stability.stable_prefix_fingerprint,
            crate::acg::stability::dominant_profile_prefix_fingerprint(
                &observations,
                stability.stable_prefix_length,
                key,
            ),
            "fingerprint for {key} is not reproducible from its own observation window",
        );
    }

    let guard = hot_cache.read().unwrap();
    let alpha_fingerprint = guard.acg_profiles[&alpha_key]
        .stable_prefix_fingerprint
        .clone();
    let beta_fingerprint = guard.acg_profiles[&beta_key]
        .stable_prefix_fingerprint
        .clone();
    assert_ne!(alpha_fingerprint, beta_fingerprint);

    // The agent-level seed adopts exactly one profile's fingerprint; it never
    // blends the two, so the reuse gate rejects the other profile after restart.
    let seed = guard.acg_stability.as_ref().unwrap();
    assert!(
        seed.stable_prefix_fingerprint == alpha_fingerprint
            || seed.stable_prefix_fingerprint == beta_fingerprint
    );
}

#[tokio::test(flavor = "current_thread")]
async fn acg_learner_binds_shared_scaffold_profiles_to_one_workflow_beyond_the_scaffold() {
    // Two workflows share an identical system prompt and an identical first user
    // turn, so they collapse into one learning key and the stable prefix extends
    // past the leading scaffold.
    let mut requests = Vec::new();
    for tail in ["tail one", "tail two", "tail three", "tail four"] {
        let mut request = sample_request("gpt-4o", "Shared system", "Shared anchor turn");
        request.messages.push(Message::Assistant {
            content: Some(MessageContent::Text(tail.to_string())),
            tool_calls: None,
            name: None,
        });
        requests.push(request);
    }

    let learning_key = derive_acg_learning_key("agent-a", &requests[0]);
    assert!(
        requests
            .iter()
            .all(|request| derive_acg_learning_key("agent-a", request) == learning_key)
    );

    let learner = AcgLearner::new("agent-a", 8, StabilityThresholds::default());
    let backend = crate::storage::memory::InMemoryBackend::new();
    let hot_cache = empty_cache();
    learner
        .process_run(&sample_run(requests.clone()), &backend, &hot_cache)
        .await
        .unwrap();

    let guard = hot_cache.read().unwrap();
    assert_eq!(guard.acg_profiles.len(), 1);
    let stability = &guard.acg_profiles[&learning_key];
    assert!(
        stability.stable_prefix_length > 1,
        "shared system plus shared anchor turn should extend the stable prefix past the scaffold",
    );

    // Beyond the scaffold the fingerprint binds to one complete request, so only
    // that workflow is admitted and the sibling tails fail closed.
    let admitted = requests
        .iter()
        .filter(|request| {
            crate::acg::stability::profile_prefix_fingerprint(
                &build_prompt_ir(request).unwrap(),
                stability.stable_prefix_length,
                &learning_key,
            ) == stability.stable_prefix_fingerprint
        })
        .count();
    assert_eq!(admitted, 1);
}
