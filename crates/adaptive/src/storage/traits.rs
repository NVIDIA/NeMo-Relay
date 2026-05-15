// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage traits used by adaptive backends.

use std::future::Future;
use std::pin::Pin;

use crate::error::Result;
use crate::trie::accumulator::AccumulatorState;
use crate::trie::serialization::TrieEnvelope;
use crate::types::plan::ExecutionPlan;
use crate::types::records::RunRecord;

type BoxStorageFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;
type PromptIrList = Vec<crate::acg::prompt_ir::PromptIR>;
type StabilityResult = crate::acg::stability::StabilityAnalysisResult;

/// Minimal async storage interface required by the adaptive runtime.
pub trait StorageBackend: Send + Sync + 'static {
    /// Persist one observed run.
    fn store_run(&self, record: &RunRecord) -> impl Future<Output = Result<()>> + Send;
    /// Load the current execution plan for an agent.
    fn load_plan(
        &self,
        agent_id: &str,
    ) -> impl Future<Output = Result<Option<ExecutionPlan>>> + Send;
    /// List stored runs for an agent.
    fn list_runs(&self, agent_id: &str) -> impl Future<Output = Result<Vec<RunRecord>>> + Send;
}

/// Object-safe storage interface used by the adaptive runtime.
///
/// Backends implement this trait when they need to be stored behind trait
/// objects and accessed through dynamic dispatch.
pub trait StorageBackendDyn: Send + Sync + 'static {
    /// Persist one observed run.
    fn store_run_dyn<'a>(&'a self, record: &'a RunRecord) -> BoxStorageFuture<'a, ()>;

    /// Load the current execution plan for an agent.
    fn load_plan_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> BoxStorageFuture<'a, Option<ExecutionPlan>>;

    /// List stored runs for an agent.
    fn list_runs_dyn<'a>(&'a self, agent_id: &'a str) -> BoxStorageFuture<'a, Vec<RunRecord>>;

    /// Persist a serialized prediction trie for an agent.
    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> BoxStorageFuture<'a, ()>;

    /// Load the serialized prediction trie for an agent.
    fn load_trie<'a>(&'a self, agent_id: &'a str) -> BoxStorageFuture<'a, Option<TrieEnvelope>>;

    /// Persist trie accumulator state for an agent.
    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> BoxStorageFuture<'a, ()>;

    /// Load trie accumulator state for an agent.
    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> BoxStorageFuture<'a, Option<AccumulatorState>>;

    /// Persist an execution plan for an agent.
    ///
    /// # Notes
    /// The default implementation is a no-op for backends that do not persist
    /// plans.
    fn store_plan(&self, _plan: &ExecutionPlan) -> Result<()> {
        Ok(())
    }

    /// Persist prompt IR observations for an agent or derived Adaptive Cache
    /// Governor (ACG) profile.
    ///
    /// # Notes
    /// The default implementation is a no-op.
    fn store_observations<'a>(
        &'a self,
        _agent_id: &'a str,
        _observations: &'a [crate::acg::prompt_ir::PromptIR],
    ) -> BoxStorageFuture<'a, ()> {
        Box::pin(async move { Ok(()) })
    }

    /// Load prompt IR observations for an agent or derived Adaptive Cache
    /// Governor (ACG) profile.
    ///
    /// # Notes
    /// The default implementation returns `Ok(None)`.
    fn load_observations<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> BoxStorageFuture<'a, Option<PromptIrList>> {
        Box::pin(async move { Ok(None) })
    }

    /// Persist an ACG stability result for an agent or derived profile.
    ///
    /// # Notes
    /// The default implementation is a no-op.
    fn store_stability<'a>(
        &'a self,
        _agent_id: &'a str,
        _result: &'a StabilityResult,
    ) -> BoxStorageFuture<'a, ()> {
        Box::pin(async move { Ok(()) })
    }

    /// Load an ACG stability result for an agent or derived profile.
    ///
    /// # Notes
    /// The default implementation returns `Ok(None)`.
    fn load_stability<'a>(
        &'a self,
        _agent_id: &'a str,
    ) -> BoxStorageFuture<'a, Option<StabilityResult>> {
        Box::pin(async move { Ok(None) })
    }
}
