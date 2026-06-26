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
use crate::config::ConvergenceConfig;

use crate::acg_profile::derive_acg_learning_key;
use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::topology::{BettiNumbers, ConvergenceDetector};
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
    convergence: Option<ConvergenceConfig>,
    convergence_detectors: Arc<RwLock<HashMap<String, ConvergenceDetector>>>,
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
        Self::new_with_convergence(agent_id, observation_window, thresholds, None)
    }

    /// Create a new ACG learner with optional topological convergence
    /// detection.
    ///
    /// # Parameters
    /// - `agent_id`: Agent identifier whose observations should be updated.
    /// - `observation_window`: Maximum number of observations to retain per
    ///   profile.
    /// - `thresholds`: Stability thresholds used during analysis.
    /// - `convergence`: Optional convergence configuration; takes precedence
    ///   over any global settings when provided.
    ///
    /// # Returns
    /// A configured [`AcgLearner`].
    pub fn new_with_convergence(
        agent_id: impl Into<String>,
        observation_window: usize,
        thresholds: StabilityThresholds,
        convergence: Option<ConvergenceConfig>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            observation_window,
            thresholds,
            convergence,
            convergence_detectors: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Map a stability analysis result to the topological feature vector used
    /// by the convergence detector.
    ///
    /// The mapping treats the stable prefix of the observation sequence as the
    /// number of connected components (`beta_0`) and the remaining unstable
    /// spans as 1-dimensional holes (`beta_1`). Drift measures the unstable
    /// fraction, and error is the complement of the average stability score.
    fn stability_to_convergence_features(
        stability: &crate::acg::stability::StabilityAnalysisResult,
    ) -> (BettiNumbers, f64, f64) {
        let total_spans = stability.scores.len();
        let betti_0 = stability.stable_prefix_length as u32;
        let betti_1 = total_spans.saturating_sub(stability.stable_prefix_length) as u32;
        let drift = if stability.stable_prefix_length == 0 {
            1.0
        } else {
            0.0
        };
        let stable_prefix_len = stability.stable_prefix_length.min(stability.scores.len());
        let avg_score = if stable_prefix_len == 0 {
            0.0
        } else {
            stability
                .scores
                .iter()
                .take(stable_prefix_len)
                .map(|score| score.score)
                .sum::<f64>()
                / stable_prefix_len as f64
        };
        let error = 1.0 - avg_score;

        (BettiNumbers::new(betti_0, betti_1), drift, error)
    }

    fn prompt_topology_matches_stability(
        stability: &crate::acg::stability::StabilityAnalysisResult,
        observation: &PromptIR,
    ) -> bool {
        stability.scores.len() == observation.blocks.len()
            && stability
                .scores
                .iter()
                .zip(&observation.blocks)
                .all(|(score, block)| score.span_id == block.span_id)
    }

    /// Update the per-profile topological convergence detector and return
    /// whether the profile has converged.
    fn record_stability_epoch(
        &self,
        profile_key: &str,
        stability: &crate::acg::stability::StabilityAnalysisResult,
    ) -> Result<bool> {
        let Some(ref config) = self.convergence else {
            return Ok(false);
        };
        if !config.enabled {
            return Ok(false);
        }

        let mut detectors = self.convergence_detectors.write().map_err(|error| {
            AdaptiveError::Internal(format!("convergence detector lock poisoned: {error}"))
        })?;
        let stability_window = config.stability_window.max(3);
        let detector = detectors
            .entry(profile_key.to_string())
            .or_insert_with(|| ConvergenceDetector::new(config.epsilon, stability_window));

        let (betti, drift, error) = Self::stability_to_convergence_features(stability);
        detector.record_epoch(betti, drift, error);

        // Require at least `stability_window` epochs before allowing
        // convergence so that error-based convergence cannot fire on the very
        // first observation.
        let enough_epochs = detector.epoch() as usize >= stability_window;
        Ok(detector.is_converged() && enough_epochs)
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
            let mut best_aggregate_stability: Option<
                crate::acg::stability::StabilityAnalysisResult,
            > = None;

            for (profile_key, new_observations) in grouped_observations.drain() {
                let existing_stability = backend.load_stability(&profile_key).await?;
                let stability_window = self
                    .convergence
                    .as_ref()
                    .map(|config| config.stability_window.max(3))
                    .unwrap_or(3);

                // If the profile has already converged, reuse the cached
                // stability result and skip loading or adding observations.
                // Stale records below the stability window fall through to
                // the normal repair path. Requests whose span topology changed
                // under the same learning key also reopen learning.
                if let Some(cached) = existing_stability.as_ref().filter(|stability| {
                    stability.converged
                        && stability.total_observations as usize >= stability_window
                        && new_observations.iter().all(|observation| {
                            Self::prompt_topology_matches_stability(stability, observation)
                        })
                }) {
                    profile_counts.insert(profile_key.clone(), cached.total_observations);
                    profile_stability.insert(profile_key.clone(), cached.clone());

                    let replace_best = best_aggregate_stability
                        .as_ref()
                        .map(|current| {
                            (cached.stable_prefix_length, cached.total_observations)
                                > (current.stable_prefix_length, current.total_observations)
                        })
                        .unwrap_or(true);
                    if replace_best {
                        best_aggregate_stability = Some(cached.clone());
                    }
                    continue;
                }

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
                let mut stability_result = analyze_stability(&observations_vec, &self.thresholds);

                let converged_now = self.record_stability_epoch(&profile_key, &stability_result)?;

                // Store the observations that produced this stability result.
                // On the epoch that first declares convergence these
                // observations are preserved; on subsequent runs the cached
                // converged result is reused and this path is skipped.
                backend
                    .store_observations(&profile_key, &observations_vec)
                    .await?;

                if converged_now {
                    stability_result.converged = true;
                }

                backend
                    .store_stability(&profile_key, &stability_result)
                    .await?;

                profile_counts.insert(profile_key.clone(), stability_result.total_observations);
                profile_stability.insert(profile_key.clone(), stability_result.clone());

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
                    best_aggregate_stability = Some(stability_result.clone());
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
            if let Some(aggregate_stability) = best_aggregate_stability {
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
