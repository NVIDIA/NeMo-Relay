// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learner pipeline for online trie construction.
//!
//! The [`Learner`] trait defines a pluggable unit that processes a completed
//! [`RunRecord`] and updates the prediction trie and hot cache. Multiple
//! learners can be composed into a `Vec<Box<dyn Learner>>` pipeline.
//!
//! [`LatencySensitivityLearner`] is the primary implementation: it loads
//! accumulated statistics from the storage backend, feeds the new run through
//! [`PredictionTrieBuilder`](crate::trie::PredictionTrieBuilder), stores
//! updated state back, and refreshes the [`HotCache`].

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use crate::error::{OptimizerError, Result};
use crate::storage::StorageBackendDyn;
use crate::trie::serialization::TrieEnvelope;
use crate::trie::{PredictionTrieBuilder, PredictionTrieNode, SensitivityConfig};
use crate::types::{AgentHints, HotCache, RunRecord};

/// A pluggable learning unit that processes a completed run.
///
/// Designed for dynamic dispatch (`Vec<Box<dyn Learner>>`) via
/// `Pin<Box<dyn Future>>` returns (matching [`StorageBackendDyn`] pattern).
pub trait Learner: Send + Sync + 'static {
    /// Process a completed [`RunRecord`], updating the backend and hot cache.
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

/// Learner that computes 4-signal latency sensitivity scores.
///
/// Implements the load-merge-store cycle:
/// 1. Load existing [`AccumulatorState`](crate::trie::AccumulatorState) from backend
/// 2. Seed a [`PredictionTrieBuilder`] with those accumulators
/// 3. Add the new run and build the trie
/// 4. Store updated accumulators and trie back to the backend
/// 5. Update the [`HotCache`] with the new trie and default hints
pub struct LatencySensitivityLearner {
    config: SensitivityConfig,
    agent_id: String,
}

impl LatencySensitivityLearner {
    /// Creates a new learner for the given agent with the specified sensitivity config.
    pub fn new(agent_id: impl Into<String>, config: SensitivityConfig) -> Self {
        Self {
            config,
            agent_id: agent_id.into(),
        }
    }
}

impl Learner for LatencySensitivityLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // 1. Load existing accumulators (None on first run is OK)
            let existing = backend.load_accumulators(&self.agent_id).await?;

            // 2. Seed builder with existing state or empty default
            let mut builder = PredictionTrieBuilder::with_accumulators(
                existing.unwrap_or_default(),
                Some(self.config.clone()),
            );

            // 3. Add the new run and build the trie
            builder.add_run(run);
            let trie_root = builder.build();

            // 4. Store updated accumulators and trie
            backend
                .store_accumulators(&self.agent_id, builder.accumulators())
                .await?;
            let envelope = TrieEnvelope::new(trie_root.clone(), &self.agent_id);
            backend.store_trie(&self.agent_id, &envelope).await?;

            // 5. Update hot cache atomically (no .await inside lock)
            {
                let mut guard = hot_cache.write().map_err(|e| {
                    OptimizerError::Internal(format!("hot cache lock poisoned: {e}"))
                })?;
                guard.agent_hints_default =
                    compute_default_hints(&trie_root, self.config.sensitivity_scale);
                guard.trie = Some(trie_root);
            }

            Ok(())
        })
    }
}

/// Computes default [`AgentHints`] from the root node of a prediction trie.
///
/// Maps `predictions_any_index` fields to the hint struct. Returns `None` if
/// the root has no aggregated predictions.
///
/// Field mapping:
/// - `osl` = `output_tokens.p90` (Output Sequence Length)
/// - `iat` = `interarrival_ms.mean` (Inter-Arrival Time)
/// - `priority` = `sensitivity_scale - latency_sensitivity` (higher sensitivity = lower priority)
/// - `latency_sensitivity` = sensitivity score as f64
/// - `prefix_id` = "default" (Phase 7 fills in agent-specific value)
/// - `total_requests` = `remaining_calls.mean` + 1 (current call)
pub fn compute_default_hints(
    trie_root: &PredictionTrieNode,
    sensitivity_scale: u32,
) -> Option<AgentHints> {
    let pred = trie_root.predictions_any_index.as_ref()?;

    let ls = pred.latency_sensitivity.unwrap_or(1);
    let priority = (sensitivity_scale as i32 - ls as i32).max(0);

    Some(AgentHints {
        osl: pred.output_tokens.p90.round() as u32,
        iat: pred.interarrival_ms.mean.round() as u32,
        priority,
        latency_sensitivity: if pred.latency_sensitivity.is_some() {
            ls as f64
        } else {
            0.0
        },
        prefix_id: "default".to_string(),
        total_requests: pred.remaining_calls.mean.round() as u32 + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    use crate::storage::InMemoryBackend;
    use crate::trie::data_models::{LlmCallPrediction, PredictionMetrics};
    use crate::types::{CallKind, CallRecord, RunRecord};

    /// Helper: create a RunRecord with `llm_count` LLM calls and `tool_count` tool calls
    /// interleaved. Each call is 1s long with a 100ms gap between calls.
    fn make_test_run(llm_count: usize, tool_count: usize) -> RunRecord {
        let base = Utc::now();
        let mut calls = Vec::new();
        let mut offset_ms: i64 = 0;

        let total = llm_count + tool_count;
        let mut llm_placed = 0;
        let mut tool_placed = 0;

        for _ in 0..total {
            let (kind, name, tokens) = if llm_placed < llm_count
                && (tool_placed >= tool_count || llm_placed <= tool_placed)
            {
                llm_placed += 1;
                let tokens = if llm_placed % 2 == 0 {
                    Some(100 * llm_placed as u32)
                } else {
                    None
                };
                (CallKind::Llm, "gpt-4".to_string(), tokens)
            } else {
                tool_placed += 1;
                (CallKind::Tool, "search".to_string(), None)
            };

            let start = base + Duration::milliseconds(offset_ms);
            let end = start + Duration::seconds(1);
            calls.push(CallRecord {
                kind,
                name,
                started_at: start,
                ended_at: Some(end),
                metadata_snapshot: None,
                output_tokens: tokens,
            });
            offset_ms += 1100;
        }

        let run_end = calls.last().map(|c| c.ended_at.unwrap()).unwrap_or(base);
        RunRecord {
            id: Uuid::new_v4(),
            agent_id: "test-agent".to_string(),
            calls,
            started_at: base,
            ended_at: Some(run_end),
        }
    }

    fn make_hot_cache() -> Arc<RwLock<HotCache>> {
        Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        }))
    }

    // -----------------------------------------------------------------------
    // Learner trait object safety
    // -----------------------------------------------------------------------

    #[test]
    fn test_learner_trait_is_object_safe() {
        // Compile-time proof: &dyn Learner can be constructed
        fn assert_dyn(_l: &dyn Learner) {}
        let learner = LatencySensitivityLearner::new("agent-1", SensitivityConfig::default());
        assert_dyn(&learner);
    }

    // -----------------------------------------------------------------------
    // Sensitivity scores flow through
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_learner_produces_sensitivity_scores() {
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let learner = LatencySensitivityLearner::new("agent-sens", SensitivityConfig::default());
        let run = make_test_run(3, 1);

        learner
            .process_run(&run, &backend, &hot_cache)
            .await
            .unwrap();

        let guard = hot_cache.read().unwrap();
        let trie = guard
            .trie
            .as_ref()
            .expect("trie should be set after process_run");
        let root_any = trie
            .predictions_any_index
            .as_ref()
            .expect("root should have predictions_any_index");
        assert!(
            root_any.latency_sensitivity.is_some(),
            "Root predictions should have latency_sensitivity after learning with default config"
        );
    }

    // -----------------------------------------------------------------------
    // Incremental equals batch
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_incremental_equals_batch() {
        let config = SensitivityConfig::default();
        let run1 = make_test_run(3, 1);
        let run2 = make_test_run(2, 0);
        let run3 = make_test_run(4, 1);

        // Batch: process all 3 runs at once
        let mut batch_builder = PredictionTrieBuilder::new(Some(config.clone()));
        batch_builder.add_run(&run1);
        batch_builder.add_run(&run2);
        batch_builder.add_run(&run3);
        let batch_trie = batch_builder.build();

        // Incremental: process one run at a time through learner (store/load cycle)
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let learner = LatencySensitivityLearner::new("agent-inc", config);

        learner
            .process_run(&run1, &backend, &hot_cache)
            .await
            .unwrap();
        learner
            .process_run(&run2, &backend, &hot_cache)
            .await
            .unwrap();
        learner
            .process_run(&run3, &backend, &hot_cache)
            .await
            .unwrap();

        let guard = hot_cache.read().unwrap();
        let inc_trie = guard.trie.as_ref().expect("trie should be set");

        // Compare root sample counts (must be equal)
        let batch_root_any = batch_trie.predictions_any_index.as_ref().unwrap();
        let inc_root_any = inc_trie.predictions_any_index.as_ref().unwrap();

        assert_eq!(
            batch_root_any.remaining_calls.sample_count, inc_root_any.remaining_calls.sample_count,
            "Incremental and batch root sample counts must match"
        );

        // Compare means (within tolerance)
        let batch_mean = batch_root_any.remaining_calls.mean;
        let inc_mean = inc_root_any.remaining_calls.mean;
        assert!(
            (batch_mean - inc_mean).abs() < 0.001,
            "Incremental mean ({inc_mean}) should be within 0.001 of batch mean ({batch_mean})"
        );
    }

    // -----------------------------------------------------------------------
    // Custom sensitivity config
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_custom_sensitivity_config() {
        let config = SensitivityConfig {
            sensitivity_scale: 5,
            w_critical: 1.0,
            w_fanout: 0.0,
            w_position: 0.0,
            w_parallel: 0.0,
        };
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let learner = LatencySensitivityLearner::new("agent-custom", config);
        let run = make_test_run(3, 0);

        learner
            .process_run(&run, &backend, &hot_cache)
            .await
            .unwrap();

        let guard = hot_cache.read().unwrap();
        let trie = guard.trie.as_ref().unwrap();
        let root_any = trie.predictions_any_index.as_ref().unwrap();
        assert!(
            root_any.latency_sensitivity.is_some(),
            "Custom config should still produce sensitivity scores"
        );
    }

    // -----------------------------------------------------------------------
    // Pipeline composition
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_pipeline_composition() {
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let run = make_test_run(2, 0);

        let learner1 = LatencySensitivityLearner::new("agent-pipe-1", SensitivityConfig::default());
        let learner2 = LatencySensitivityLearner::new("agent-pipe-2", SensitivityConfig::default());

        // Compile-time proof: Vec<Box<dyn Learner>> works
        let learners: Vec<Box<dyn Learner>> = vec![Box::new(learner1), Box::new(learner2)];

        for learner in &learners {
            learner
                .process_run(&run, &backend, &hot_cache)
                .await
                .unwrap();
        }

        // Verify both agents have data in the backend
        let trie1 = backend.load_trie("agent-pipe-1").await.unwrap();
        let trie2 = backend.load_trie("agent-pipe-2").await.unwrap();
        assert!(
            trie1.is_some(),
            "agent-pipe-1 should have a trie in backend"
        );
        assert!(
            trie2.is_some(),
            "agent-pipe-2 should have a trie in backend"
        );
    }

    // -----------------------------------------------------------------------
    // compute_default_hints tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_default_hints_from_root() {
        let mut root = PredictionTrieNode::new("root");
        root.predictions_any_index = Some(LlmCallPrediction {
            remaining_calls: PredictionMetrics {
                sample_count: 10,
                mean: 3.0,
                p50: 3.0,
                p90: 5.0,
                p95: 6.0,
            },
            interarrival_ms: PredictionMetrics {
                sample_count: 10,
                mean: 150.0,
                p50: 140.0,
                p90: 200.0,
                p95: 220.0,
            },
            output_tokens: PredictionMetrics {
                sample_count: 10,
                mean: 256.0,
                p50: 240.0,
                p90: 400.0,
                p95: 450.0,
            },
            latency_sensitivity: Some(3),
        });

        let hints = compute_default_hints(&root, 5).expect("should produce hints");
        assert_eq!(hints.osl, 400, "osl = output_tokens.p90 = 400");
        assert_eq!(hints.iat, 150, "iat = interarrival_ms.mean = 150");
        assert_eq!(hints.priority, 2, "priority = 5 - 3 = 2");
        assert!((hints.latency_sensitivity - 3.0).abs() < f64::EPSILON);
        assert_eq!(hints.prefix_id, "default");
        assert_eq!(hints.total_requests, 4, "total_requests = mean(3) + 1 = 4");
    }

    #[test]
    fn test_compute_default_hints_none_when_no_predictions() {
        let root = PredictionTrieNode::new("root");
        let hints = compute_default_hints(&root, 5);
        assert!(hints.is_none(), "No predictions_any_index -> None hints");
    }

    // -----------------------------------------------------------------------
    // Backend persistence tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_run_stores_trie_in_backend() {
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let learner =
            LatencySensitivityLearner::new("agent-trie-store", SensitivityConfig::default());
        let run = make_test_run(2, 0);

        learner
            .process_run(&run, &backend, &hot_cache)
            .await
            .unwrap();

        let envelope = backend
            .load_trie("agent-trie-store")
            .await
            .unwrap()
            .expect("trie should be stored after process_run");
        assert_eq!(
            envelope.workflow_name, "agent-trie-store",
            "Trie envelope workflow_name should be the agent_id"
        );
    }

    #[tokio::test]
    async fn test_process_run_stores_accumulators_in_backend() {
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let learner =
            LatencySensitivityLearner::new("agent-acc-store", SensitivityConfig::default());
        let run = make_test_run(2, 0);

        learner
            .process_run(&run, &backend, &hot_cache)
            .await
            .unwrap();

        let accs = backend
            .load_accumulators("agent-acc-store")
            .await
            .unwrap()
            .expect("accumulators should be stored after process_run");
        assert!(
            !accs.nodes.is_empty(),
            "Stored accumulators should have non-empty nodes"
        );
    }

    #[tokio::test]
    async fn test_process_run_first_run_no_prior_accumulators() {
        let backend = InMemoryBackend::new();
        let hot_cache = make_hot_cache();
        let learner =
            LatencySensitivityLearner::new("agent-first-run", SensitivityConfig::default());
        let run = make_test_run(2, 0);

        // First run ever: load_accumulators returns None. Should succeed.
        let result = learner.process_run(&run, &backend, &hot_cache).await;
        assert!(
            result.is_ok(),
            "First run with no prior accumulators should succeed: {:?}",
            result.err()
        );

        // Trie should be in the hot cache
        let guard = hot_cache.read().unwrap();
        assert!(guard.trie.is_some(), "Trie should be set after first run");
        assert!(
            guard.agent_hints_default.is_some(),
            "agent_hints_default should be set after first run"
        );
    }
}
