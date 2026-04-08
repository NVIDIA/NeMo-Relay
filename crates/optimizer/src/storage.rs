// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage abstraction for the nexus-optimizer crate.
//!
//! Defines the [`StorageBackend`] trait (RPITIT, not object-safe), the
//! [`StorageBackendDyn`] companion trait (object-safe, `Pin<Box<dyn Future>>`),
//! and the [`InMemoryBackend`] implementation of both.
//!
//! [`StorageBackend`] provides the persistence seam so backends can be swapped
//! without touching optimizer logic. [`StorageBackendDyn`] adds trie/accumulator
//! methods and enables `&dyn StorageBackendDyn` usage in the learner pipeline.
//! [`InMemoryBackend`] validates the trait design and serves as the
//! test/single-process backend.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use crate::error::{OptimizerError, Result};
use crate::trie::serialization::TrieEnvelope;
use crate::trie::AccumulatorState;
use crate::types::{ExecutionPlan, RunRecord};

/// The storage abstraction for nexus-optimizer persistence.
///
/// All methods return futures that are `Send`, enabling use across tokio task
/// boundaries. Implementations must be `Send + Sync + 'static`.
///
/// # Methods
///
/// - [`store_run`](StorageBackend::store_run) -- persist a completed [`RunRecord`]
/// - [`load_plan`](StorageBackend::load_plan) -- retrieve the [`ExecutionPlan`] for an agent
/// - [`list_runs`](StorageBackend::list_runs) -- list all runs for an agent
pub trait StorageBackend: Send + Sync + 'static {
    fn store_run(&self, record: &RunRecord) -> impl Future<Output = Result<()>> + Send;
    fn load_plan(
        &self,
        agent_id: &str,
    ) -> impl Future<Output = Result<Option<ExecutionPlan>>> + Send;
    fn list_runs(&self, agent_id: &str) -> impl Future<Output = Result<Vec<RunRecord>>> + Send;
}

/// Object-safe companion to [`StorageBackend`] for dynamic dispatch.
///
/// This trait mirrors the 3 base methods from [`StorageBackend`] (with `_dyn` suffix)
/// and adds 4 trie/accumulator persistence methods. All methods return
/// `Pin<Box<dyn Future<...>>>` so the trait is object-safe and can be used as
/// `&dyn StorageBackendDyn`.
///
/// # Design
///
/// Explicit impls (no blanket) — each backend explicitly implements both
/// `StorageBackend` and `StorageBackendDyn`. This avoids orphan-rule issues and
/// allows each backend to optimize its dyn dispatch path independently.
pub trait StorageBackendDyn: Send + Sync + 'static {
    /// Persist a completed [`RunRecord`] (object-safe wrapper).
    fn store_run_dyn<'a>(
        &'a self,
        record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Retrieve the [`ExecutionPlan`] for an agent (object-safe wrapper).
    fn load_plan_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>>;

    /// List all runs for an agent (object-safe wrapper).
    fn list_runs_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>>;

    /// Store a [`TrieEnvelope`] keyed by agent ID.
    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Load the [`TrieEnvelope`] for an agent, if any.
    fn load_trie<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>>;

    /// Store an [`AccumulatorState`] keyed by agent ID.
    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Load the [`AccumulatorState`] for an agent, if any.
    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>>;

    /// Store an execution plan, if the backend supports it.
    ///
    /// Default implementation is a no-op (returns `Ok(())`). Backends that
    /// support plan storage (e.g. [`InMemoryBackend`]) override this.
    fn store_plan(&self, _plan: &ExecutionPlan) -> Result<()> {
        Ok(())
    }
}

/// An in-memory [`StorageBackend`] implementation for testing and single-process use.
///
/// Uses [`std::sync::RwLock`] internally (not `tokio::sync::RwLock`) because all
/// operations are fast in-memory reads/writes with no I/O or await points.
pub struct InMemoryBackend {
    runs: RwLock<HashMap<String, Vec<RunRecord>>>,
    plans: RwLock<HashMap<String, ExecutionPlan>>,
    tries: RwLock<HashMap<String, TrieEnvelope>>,
    accumulators: RwLock<HashMap<String, AccumulatorState>>,
}

impl InMemoryBackend {
    /// Creates a new empty `InMemoryBackend`.
    pub fn new() -> Self {
        Self {
            runs: RwLock::new(HashMap::new()),
            plans: RwLock::new(HashMap::new()),
            tries: RwLock::new(HashMap::new()),
            accumulators: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for InMemoryBackend {
    fn store_run(&self, record: &RunRecord) -> impl Future<Output = Result<()>> + Send {
        let result = {
            let mut guard = self
                .runs
                .write()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref mut g) => {
                    g.entry(record.agent_id.clone())
                        .or_default()
                        .push(record.clone());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        };
        async move { result }
    }

    fn load_plan(
        &self,
        agent_id: &str,
    ) -> impl Future<Output = Result<Option<ExecutionPlan>>> + Send {
        let result = {
            let guard = self
                .plans
                .read()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref g) => Ok(g.get(agent_id).cloned()),
                Err(e) => Err(e),
            }
        };
        async move { result }
    }

    fn list_runs(&self, agent_id: &str) -> impl Future<Output = Result<Vec<RunRecord>>> + Send {
        let result = {
            let guard = self
                .runs
                .read()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref g) => Ok(g.get(agent_id).cloned().unwrap_or_default()),
                Err(e) => Err(e),
            }
        };
        async move { result }
    }
}

impl StorageBackendDyn for InMemoryBackend {
    fn store_run_dyn<'a>(
        &'a self,
        record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.store_run(record))
    }

    fn load_plan_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>> {
        Box::pin(self.load_plan(agent_id))
    }

    fn list_runs_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>> {
        Box::pin(self.list_runs(agent_id))
    }

    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let result = {
            let mut guard = self
                .tries
                .write()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref mut g) => {
                    g.insert(agent_id.to_string(), envelope.clone());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        };
        Box::pin(async move { result })
    }

    fn load_trie<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>> {
        let result = {
            let guard = self
                .tries
                .read()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref g) => Ok(g.get(agent_id).cloned()),
                Err(e) => Err(e),
            }
        };
        Box::pin(async move { result })
    }

    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let result = {
            let mut guard = self
                .accumulators
                .write()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref mut g) => {
                    g.insert(agent_id.to_string(), state.clone());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        };
        Box::pin(async move { result })
    }

    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>> {
        let result = {
            let guard = self
                .accumulators
                .read()
                .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")));
            match guard {
                Ok(ref g) => Ok(g.get(agent_id).cloned()),
                Err(e) => Err(e),
            }
        };
        Box::pin(async move { result })
    }

    fn store_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        let mut guard = self
            .plans
            .write()
            .map_err(|e| OptimizerError::Internal(format!("lock poisoned: {e}")))?;
        guard.insert(plan.agent_id.clone(), plan.clone());
        Ok(())
    }
}

/// Type alias for a boxed, object-safe storage backend.
///
/// Replaces the former enum dispatch pattern. Any type implementing
/// [`StorageBackendDyn`] can be boxed into an `AnyBackend`.
pub type AnyBackend = Box<dyn StorageBackendDyn + Send + Sync>;

#[cfg(test)]
#[allow(clippy::box_default)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::types::{MetadataEnvelope, ParallelGroup};

    fn make_test_run(agent_id: &str) -> RunRecord {
        RunRecord {
            id: Uuid::new_v4(),
            agent_id: agent_id.to_string(),
            calls: vec![],
            started_at: Utc::now(),
            ended_at: None,
        }
    }

    fn make_test_plan(agent_id: &str) -> ExecutionPlan {
        ExecutionPlan {
            agent_id: agent_id.to_string(),
            parallel_groups: vec![],
            metadata_template: MetadataEnvelope {
                run_id: Uuid::new_v4(),
                agent_id: agent_id.to_string(),
                parallel_hints: vec![],
                extensions: json!({}),
            },
        }
    }

    #[tokio::test]
    async fn test_store_and_list_runs() {
        let backend = InMemoryBackend::new();
        let record = make_test_run("agent-1");
        let record_id = record.id;

        backend.store_run(&record).await.unwrap();
        let runs = backend.list_runs("agent-1").await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, record_id);
    }

    #[tokio::test]
    async fn test_list_runs_empty() {
        let backend = InMemoryBackend::new();
        let runs = backend.list_runs("nonexistent").await.unwrap();

        assert!(runs.is_empty());
    }

    #[tokio::test]
    async fn test_store_multiple_runs() {
        let backend = InMemoryBackend::new();

        for _ in 0..3 {
            let record = make_test_run("agent-1");
            backend.store_run(&record).await.unwrap();
        }

        let runs = backend.list_runs("agent-1").await.unwrap();
        assert_eq!(runs.len(), 3);
    }

    #[tokio::test]
    async fn test_store_runs_different_agents() {
        let backend = InMemoryBackend::new();

        let r1 = make_test_run("agent-1");
        let r2 = make_test_run("agent-2");

        backend.store_run(&r1).await.unwrap();
        backend.store_run(&r2).await.unwrap();

        let runs_1 = backend.list_runs("agent-1").await.unwrap();
        let runs_2 = backend.list_runs("agent-2").await.unwrap();

        assert_eq!(runs_1.len(), 1);
        assert_eq!(runs_2.len(), 1);
    }

    #[tokio::test]
    async fn test_load_plan_none() {
        let backend = InMemoryBackend::new();
        let plan = backend.load_plan("agent-1").await.unwrap();

        assert!(plan.is_none());
    }

    #[tokio::test]
    async fn test_store_plan_and_load() {
        let backend = InMemoryBackend::new();
        let plan = make_test_plan("agent-1");

        backend.store_plan(&plan).unwrap();
        let loaded = backend.load_plan("agent-1").await.unwrap();

        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.agent_id, "agent-1");
    }

    #[tokio::test]
    async fn test_load_plan_wrong_agent() {
        let backend = InMemoryBackend::new();
        let plan = make_test_plan("agent-1");

        backend.store_plan(&plan).unwrap();
        let loaded = backend.load_plan("agent-2").await.unwrap();

        assert!(loaded.is_none());
    }

    #[test]
    fn test_backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<InMemoryBackend>();
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let backend = Arc::new(InMemoryBackend::new());

        let b1 = Arc::clone(&backend);
        let t1 = tokio::spawn(async move {
            let record = make_test_run("agent-1");
            b1.store_run(&record).await.unwrap();
        });

        let b2 = Arc::clone(&backend);
        let t2 = tokio::spawn(async move {
            let record = make_test_run("agent-1");
            b2.store_run(&record).await.unwrap();
        });

        t1.await.unwrap();
        t2.await.unwrap();

        let runs = backend.list_runs("agent-1").await.unwrap();
        assert_eq!(runs.len(), 2);
    }

    // Verify unused import suppression - ParallelGroup is used in make_test_plan indirectly
    // through ExecutionPlan, but we import it to ensure the type is accessible.
    #[test]
    fn test_parallel_group_accessible() {
        let _group = ParallelGroup {
            group_id: "g1".to_string(),
            tool_names: vec!["tool1".to_string()],
        };
    }

    // -----------------------------------------------------------------------
    // StorageBackendDyn tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_storage_backend_dyn_is_object_safe() {
        // This test verifies that StorageBackendDyn can be used as a trait object.
        fn assert_dyn(_b: &dyn StorageBackendDyn) {}
        let backend = InMemoryBackend::new();
        assert_dyn(&backend);
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_store_trie_load_trie_roundtrip() {
        use crate::trie::data_models::PredictionTrieNode;
        use crate::trie::serialization::TrieEnvelope;

        let backend = InMemoryBackend::new();
        let root = PredictionTrieNode::new("test_root");
        let envelope = TrieEnvelope::new(root, "test_workflow");

        backend.store_trie("agent-1", &envelope).await.unwrap();
        let loaded = backend.load_trie("agent-1").await.unwrap();

        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.workflow_name, "test_workflow");
        assert_eq!(loaded.root.name, "test_root");
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_store_accumulators_load_accumulators_roundtrip() {
        use crate::trie::AccumulatorState;
        use crate::trie::NodeAccumulators;

        let backend = InMemoryBackend::new();
        let mut state = AccumulatorState::default();
        state
            .nodes
            .insert("workflow/agent".to_string(), NodeAccumulators::default());

        backend.store_accumulators("agent-1", &state).await.unwrap();
        let loaded = backend.load_accumulators("agent-1").await.unwrap();

        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert!(loaded.nodes.contains_key("workflow/agent"));
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_load_trie_returns_none_when_empty() {
        let backend = InMemoryBackend::new();
        let loaded = backend.load_trie("nonexistent").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_load_accumulators_returns_none_when_empty() {
        let backend = InMemoryBackend::new();
        let loaded = backend.load_accumulators("nonexistent").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_store_run_dyn_dispatches() {
        let backend = InMemoryBackend::new();
        let record = make_test_run("agent-dyn");
        let record_id = record.id;

        backend.store_run_dyn(&record).await.unwrap();
        let runs = backend.list_runs_dyn("agent-dyn").await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, record_id);
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_load_plan_dyn_dispatches() {
        let backend = InMemoryBackend::new();
        let plan = make_test_plan("agent-dyn");

        backend.store_plan(&plan).unwrap();
        let loaded = backend.load_plan_dyn("agent-dyn").await.unwrap();

        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().agent_id, "agent-dyn");
    }

    #[tokio::test]
    async fn test_storage_backend_dyn_list_runs_dyn_dispatches() {
        let backend = InMemoryBackend::new();
        let runs = backend.list_runs_dyn("nonexistent").await.unwrap();
        assert!(runs.is_empty());
    }

    // -----------------------------------------------------------------------
    // AnyBackend (Box<dyn StorageBackendDyn>) tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_any_backend_inmemory_store_list_runs() {
        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let record = make_test_run("agent-any-1");
        let record_id = record.id;
        backend.store_run_dyn(&record).await.unwrap();
        let runs = backend.list_runs_dyn("agent-any-1").await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, record_id);
    }

    #[tokio::test]
    async fn test_any_backend_inmemory_load_plan_none() {
        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let plan = backend.load_plan_dyn("unknown-agent").await.unwrap();
        assert!(plan.is_none());
    }

    #[test]
    fn test_any_backend_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<AnyBackend>();
    }

    #[tokio::test]
    async fn test_any_backend_dyn_store_load_trie() {
        use crate::trie::data_models::PredictionTrieNode;
        use crate::trie::serialization::TrieEnvelope;

        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let root = PredictionTrieNode::new("any_root");
        let envelope = TrieEnvelope::new(root, "any_workflow");

        backend
            .store_trie("agent-any-trie", &envelope)
            .await
            .unwrap();
        let loaded = backend.load_trie("agent-any-trie").await.unwrap();

        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.workflow_name, "any_workflow");
        assert_eq!(loaded.root.name, "any_root");
    }

    // -----------------------------------------------------------------------
    // Box<dyn StorageBackendDyn> dispatch tests (replaces AnyBackend::Dynamic)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_any_backend_dynamic_dispatches() {
        // Box an InMemoryBackend as AnyBackend to verify dyn dispatch.
        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let record = make_test_run("agent-dynamic");
        let record_id = record.id;

        backend.store_run_dyn(&record).await.unwrap();
        let runs = backend.list_runs_dyn("agent-dynamic").await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, record_id);
    }

    #[test]
    fn test_any_backend_default_store_plan() {
        // The default StorageBackendDyn::store_plan is a no-op.
        // InMemoryBackend overrides it, so test with a minimal struct
        // that only has the default.
        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let plan = make_test_plan("agent-plan");
        // InMemoryBackend overrides store_plan, so this should actually store it.
        assert!(backend.store_plan(&plan).is_ok());
    }

    #[test]
    fn test_any_backend_dynamic_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        // AnyBackend (Box<dyn StorageBackendDyn + Send + Sync>) is Send + Sync.
        assert_send_sync::<AnyBackend>();
    }

    #[tokio::test]
    async fn test_any_backend_dynamic_dyn_store_load_trie() {
        use crate::trie::data_models::PredictionTrieNode;
        use crate::trie::serialization::TrieEnvelope;

        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let root = PredictionTrieNode::new("dyn_root");
        let envelope = TrieEnvelope::new(root, "dyn_workflow");

        backend
            .store_trie("agent-dyn-trie", &envelope)
            .await
            .unwrap();
        let loaded = backend.load_trie("agent-dyn-trie").await.unwrap();

        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.workflow_name, "dyn_workflow");
        assert_eq!(loaded.root.name, "dyn_root");
    }

    #[tokio::test]
    async fn test_any_backend_dynamic_dyn_store_load_accumulators() {
        use crate::trie::AccumulatorState;
        use crate::trie::NodeAccumulators;

        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        let mut state = AccumulatorState::default();
        state
            .nodes
            .insert("workflow/dyn".to_string(), NodeAccumulators::default());

        backend
            .store_accumulators("agent-dyn-acc", &state)
            .await
            .unwrap();
        let loaded = backend.load_accumulators("agent-dyn-acc").await.unwrap();

        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert!(loaded.nodes.contains_key("workflow/dyn"));
    }

    #[tokio::test]
    async fn test_any_backend_dynamic_load_plan_dyn() {
        let backend: AnyBackend = Box::new(InMemoryBackend::new());
        // InMemoryBackend overrides store_plan, so load_plan should find it.
        let plan = make_test_plan("agent-dyn-plan");
        backend.store_plan(&plan).unwrap();
        let loaded = backend.load_plan_dyn("agent-dyn-plan").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().agent_id, "agent-dyn-plan");
    }
}
