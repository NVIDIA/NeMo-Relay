// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive Cache Governor (ACG) learner for the adaptive telemetry pipeline.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use crate::acg::ir_builder::build_prompt_ir;
use crate::acg::prompt_ir::PromptIR;
use crate::acg::stability::{StabilityThresholds, analyze_stability};

use crate::acg_profile::derive_acg_learning_key;
use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{CallKind, RunRecord};

/// Learner that derives prompt stability state for ACG.
///
/// This learner groups annotated LLM requests by derived ACG profile key,
/// builds prompt IR observations, persists a bounded observation window, and
/// updates the hot cache with the latest stability results.
pub struct AcgLearner {
    agent_id: String,
    observation_window: usize,
    thresholds: StabilityThresholds,
}

impl AcgLearner {
    /// Create a new ACG learner.
    ///
    /// # Parameters
    /// - `agent_id`: Agent identifier whose observations should be updated.
    /// - `observation_window`: Maximum number of observations to retain per
    ///   profile.
    /// - `thresholds`: Stability thresholds used during analysis.
    ///
    /// # Returns
    /// A configured [`AcgLearner`].
    pub fn new(
        agent_id: impl Into<String>,
        observation_window: usize,
        thresholds: StabilityThresholds,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            observation_window,
            thresholds,
        }
    }
}

impl Learner for AcgLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut grouped_observations: HashMap<String, Vec<PromptIR>> = run
                .calls
                .iter()
                .filter(|call| call.kind == CallKind::Llm)
                .filter_map(|call| call.annotated_request.as_ref())
                .filter_map(|request| {
                    build_prompt_ir(request).ok().map(|prompt_ir| {
                        (derive_acg_learning_key(&self.agent_id, request), prompt_ir)
                    })
                })
                .fold(HashMap::new(), |mut grouped, (key, prompt_ir)| {
                    grouped.entry(key).or_default().push(prompt_ir);
                    grouped
                });

            if grouped_observations.is_empty() {
                return Ok(());
            }

            let mut profile_stability = HashMap::new();
            let mut profile_counts = HashMap::new();
            let mut best_profile_seed: Option<(
                Vec<PromptIR>,
                crate::acg::stability::StabilityAnalysisResult,
            )> = None;

            for (profile_key, new_observations) in grouped_observations.drain() {
                let existing = backend.load_observations(&profile_key).await?;
                let mut window: VecDeque<PromptIR> =
                    existing.unwrap_or_default().into_iter().collect();

                for observation in new_observations {
                    if window.len() >= self.observation_window {
                        window.pop_front();
                    }
                    window.push_back(observation);
                }

                let observations_vec: Vec<PromptIR> = window.into_iter().collect();
                backend
                    .store_observations(&profile_key, &observations_vec)
                    .await?;

                let stability_result = analyze_stability(&observations_vec, &self.thresholds);
                backend
                    .store_stability(&profile_key, &stability_result)
                    .await?;

                profile_counts.insert(profile_key.clone(), stability_result.total_observations);
                profile_stability.insert(profile_key, stability_result.clone());

                let replace_best = best_profile_seed
                    .as_ref()
                    .map(|(_, current)| {
                        (
                            stability_result.stable_prefix_length,
                            stability_result.total_observations,
                        ) > (current.stable_prefix_length, current.total_observations)
                    })
                    .unwrap_or(true);
                if replace_best {
                    best_profile_seed = Some((observations_vec.clone(), stability_result.clone()));
                }
            }

            if let Some((aggregate_observations, aggregate_stability)) = best_profile_seed.as_ref()
            {
                // Persist the runtime seed entry under plain agent_id so registration can
                // rehydrate HotCache without scanning profile-specific keys.
                backend
                    .store_observations(&self.agent_id, aggregate_observations)
                    .await?;
                backend
                    .store_stability(&self.agent_id, aggregate_stability)
                    .await?;
            }

            let mut guard = hot_cache.write().map_err(|error| {
                AdaptiveError::Internal(format!("hot cache lock poisoned: {error}"))
            })?;
            guard.acg_profiles.extend(profile_stability);
            guard.acg_profile_observation_counts.extend(profile_counts);
            if let Some((_, aggregate_stability)) = best_profile_seed {
                guard.acg_observation_count = aggregate_stability.total_observations;
                guard.acg_stability = Some(aggregate_stability);
            }

            Ok(())
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/acg_learner_tests.rs"]
mod tests;
