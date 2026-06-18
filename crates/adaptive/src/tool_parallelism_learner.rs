// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learner that derives tool parallelism plans from observed runs.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use crate::acg::canonicalize::sha256_hex;
use chrono::{DateTime, Utc};
use nemo_relay_adaptive_topology::DriftDetector;
use serde_json::json;
use uuid::Uuid;

use crate::config::DriftConfig;
use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::metadata::{MetadataEnvelope, ParallelHint};
use crate::types::plan::{ExecutionPlan, ParallelGroup};
use crate::types::records::{CallKind, RunRecord};

/// Learner that discovers tool fan-out groups from run telemetry.
pub struct ToolParallelismLearner {
    agent_id: String,
    drift: Option<DriftConfig>,
    drift_detector: Arc<RwLock<DriftDetector<4>>>,
}

impl ToolParallelismLearner {
    /// Create a new tool-parallelism learner.
    ///
    /// # Parameters
    /// - `agent_id`: Agent identifier whose execution plan should be updated.
    ///
    /// # Returns
    /// A configured [`ToolParallelismLearner`].
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self::new_with_drift(agent_id, None)
    }

    /// Create a new tool-parallelism learner with optional drift detection.
    ///
    /// # Parameters
    /// - `agent_id`: Agent identifier whose execution plan should be updated.
    /// - `drift`: Optional topology-aware drift detection settings.
    ///
    /// # Returns
    /// A configured [`ToolParallelismLearner`].
    pub fn new_with_drift(agent_id: impl Into<String>, drift: Option<DriftConfig>) -> Self {
        Self {
            agent_id: agent_id.into(),
            drift,
            drift_detector: Arc::new(RwLock::new(DriftDetector::new())),
        }
    }

    fn record_cohort_drift(&self, observed_cohorts: &[Vec<String>]) -> Result<bool> {
        let Some(config) = &self.drift else {
            return Ok(false);
        };
        if !config.enabled {
            return Ok(false);
        }

        let centroid = cohort_feature_vector(observed_cohorts);
        let mut detector = self.drift_detector.write().map_err(|error| {
            AdaptiveError::Internal(format!("tool drift detector lock poisoned: {error}"))
        })?;
        let drift = detector.update(&centroid);
        Ok(drift > config.threshold)
    }
}

impl Learner for ToolParallelismLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let observed_cohorts = derive_observed_cohorts(run);
            if observed_cohorts.is_empty() {
                return Ok(());
            }

            let drifted = self.record_cohort_drift(&observed_cohorts)?;
            let mut plan = if drifted {
                empty_execution_plan(&self.agent_id, run.id)
            } else {
                backend
                    .load_plan_dyn(&self.agent_id)
                    .await?
                    .unwrap_or_else(|| empty_execution_plan(&self.agent_id, run.id))
            };
            plan.agent_id = self.agent_id.clone();

            merge_observed_cohorts(&mut plan, &observed_cohorts, run.id);
            backend.store_plan(&plan)?;

            let mut guard = hot_cache.write().map_err(|error| {
                AdaptiveError::Internal(format!("hot cache lock poisoned: {error}"))
            })?;
            guard.plan = Some(plan.clone());
            Ok(())
        })
    }
}

#[derive(Clone)]
struct ObservedToolCall {
    name: String,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
}

fn derive_observed_cohorts(run: &RunRecord) -> Vec<Vec<String>> {
    let mut calls: Vec<ObservedToolCall> = run
        .calls
        .iter()
        .filter(|call| call.kind == CallKind::Tool)
        .filter_map(|call| {
            call.ended_at.map(|ended_at| ObservedToolCall {
                name: call.name.clone(),
                started_at: call.started_at,
                ended_at,
            })
        })
        .collect();
    calls.sort_by_key(|call| call.started_at);

    let mut active: Vec<ObservedToolCall> = Vec::new();
    let mut cohorts: HashSet<Vec<String>> = HashSet::new();

    for current in calls {
        active.retain(|call| call.ended_at > current.started_at);
        if active.len() + 1 > 1 {
            let mut tool_names: Vec<String> = active.iter().map(|call| call.name.clone()).collect();
            tool_names.push(current.name.clone());
            tool_names.sort();
            cohorts.insert(tool_names);
        }
        active.push(current);
    }

    let mut observed: Vec<Vec<String>> = cohorts.into_iter().collect();
    observed.sort();
    observed
}

fn cohort_feature_vector(observed_cohorts: &[Vec<String>]) -> [f64; 4] {
    if observed_cohorts.is_empty() {
        return [0.0; 4];
    }

    let mut unique_tools = BTreeSet::new();
    let mut total_tool_refs = 0usize;
    let mut max_cohort_size = 0usize;

    for cohort in observed_cohorts {
        total_tool_refs += cohort.len();
        max_cohort_size = max_cohort_size.max(cohort.len());
        for tool in cohort {
            unique_tools.insert(tool);
        }
    }

    let duplicate_refs = total_tool_refs.saturating_sub(unique_tools.len());
    [
        observed_cohorts.len() as f64,
        unique_tools.len() as f64,
        duplicate_refs as f64 / total_tool_refs.max(1) as f64,
        max_cohort_size as f64,
    ]
}

fn merge_observed_cohorts(
    plan: &mut ExecutionPlan,
    observed_cohorts: &[Vec<String>],
    run_id: Uuid,
) {
    let mut groups_by_id: BTreeMap<String, ParallelGroup> = plan
        .parallel_groups
        .iter()
        .cloned()
        .map(|group| (group.group_id.clone(), group))
        .collect();
    let mut hints_by_key: BTreeMap<(String, String), ParallelHint> = plan
        .metadata_template
        .parallel_hints
        .iter()
        .cloned()
        .map(|hint| ((hint.tool_name.clone(), hint.group_id.clone()), hint))
        .collect();

    for cohort in observed_cohorts {
        let group = build_parallel_group(cohort);
        let group_id = group.group_id.clone();
        let mut unique_tool_names: BTreeSet<String> = BTreeSet::new();
        for tool_name in &group.tool_names {
            if unique_tool_names.insert(tool_name.clone()) {
                hints_by_key.insert(
                    (tool_name.clone(), group_id.clone()),
                    ParallelHint {
                        tool_name: tool_name.clone(),
                        group_id: group_id.clone(),
                        explicit: false,
                    },
                );
            }
        }
        groups_by_id.insert(group_id, group);
    }

    plan.parallel_groups = groups_by_id.into_values().collect();
    plan.metadata_template.agent_id = plan.agent_id.clone();
    plan.metadata_template.run_id = run_id;
    plan.metadata_template.parallel_hints = hints_by_key.into_values().collect();
}

fn build_parallel_group(tool_names: &[String]) -> ParallelGroup {
    let joined = tool_names.join("|");
    let group_hash = sha256_hex(&joined);
    ParallelGroup {
        group_id: format!("fanout:{}", &group_hash[..12]),
        tool_names: tool_names.to_vec(),
    }
}

fn empty_execution_plan(agent_id: &str, run_id: Uuid) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![],
        metadata_template: MetadataEnvelope {
            run_id,
            agent_id: agent_id.to_string(),
            parallel_hints: vec![],
            extensions: json!({}),
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/tool_parallelism_learner_tests.rs"]
mod tests;
