// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::{Arc, RwLock};

use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::trie::builder::{PredictionTrieBuilder, SensitivityConfig};
use crate::trie::data_models::PredictionTrieNode;
use crate::trie::serialization::TrieEnvelope;
use crate::types::cache::HotCache;
use crate::types::metadata::AgentHints;
use crate::types::records::RunRecord;

pub struct LatencySensitivityLearner {
    config: SensitivityConfig,
    agent_id: String,
}

impl LatencySensitivityLearner {
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
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let existing = backend.load_accumulators(&self.agent_id).await?;
            let mut builder = PredictionTrieBuilder::with_accumulators(
                existing.unwrap_or_default(),
                Some(self.config.clone()),
            );
            builder.add_run(run);
            let trie_root = builder.build();

            backend
                .store_accumulators(&self.agent_id, builder.accumulators())
                .await?;
            let envelope = TrieEnvelope::new(trie_root.clone(), &self.agent_id);
            backend.store_trie(&self.agent_id, &envelope).await?;

            {
                let mut guard = hot_cache.write().map_err(|error| {
                    AdaptiveError::Internal(format!("hot cache lock poisoned: {error}"))
                })?;
                guard.agent_hints_default =
                    compute_default_hints(&trie_root, self.config.sensitivity_scale);
                guard.trie = Some(trie_root);
            }

            Ok(())
        })
    }
}

pub fn compute_default_hints(
    trie_root: &PredictionTrieNode,
    sensitivity_scale: u32,
) -> Option<AgentHints> {
    let prediction = trie_root.predictions_any_index.as_ref()?;

    let latency_sensitivity = prediction.latency_sensitivity.unwrap_or(1);
    let priority = (sensitivity_scale as i32 - latency_sensitivity as i32).max(0);

    Some(AgentHints {
        osl: prediction.output_tokens.p90.round() as u32,
        iat: prediction.interarrival_ms.mean.round() as u32,
        priority,
        latency_sensitivity: if prediction.latency_sensitivity.is_some() {
            latency_sensitivity as f64
        } else {
            0.0
        },
        prefix_id: "default".to_string(),
        total_requests: prediction.remaining_calls.mean.round() as u32 + 1,
    })
}
