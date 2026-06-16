// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # NeMo Relay Adaptive
//!
//! Adaptive config helpers and core-plugin integration for NeMo Relay.
//! Adaptive behavior is enabled through the generic core plugin system.
//!
//! This crate provides the adaptive runtime, persistence abstractions, learner
//! implementations, and Adaptive Cache Governor (ACG) analysis types used to
//! derive and apply runtime hints from observed NeMo Relay executions.
pub mod acg;
pub mod acg_component;
pub mod acg_learner;
pub mod acg_profile;
pub mod adaptive_hints_intercept;
pub mod agent_context_intercept;
pub mod cache_diagnostics;
pub mod config;
pub mod context_helpers;
pub mod drain;
pub mod error;
pub mod intercepts;
/// Learning primitives and built-in learner implementations.
pub mod learner;
pub mod plugin_component;
#[cfg(feature = "redis-backend")]
pub mod redis;
mod runtime;
/// Storage backends and backend traits for adaptive state persistence.
pub mod storage;
pub mod subscriber;
/// Learner that derives tool fan-out plans from observed runs.
pub mod tool_parallelism_learner;
pub mod trie;
/// Serializable adaptive data models shared across runtime components.
pub mod types;

pub use config::{
    AcgComponentConfig, AdaptiveConfig, AdaptiveHintsComponentConfig, AgentContextComponentConfig,
    BackendSpec, StateConfig, TelemetryComponentConfig, ToolParallelismComponentConfig,
};
pub use context_helpers::{
    AGENT_CONTEXT_POINTER, LATENCY_SENSITIVITY_POINTER, extract_scope_path,
    read_manual_latency_sensitivity, resolve_agent_context, resolve_agent_id,
    resolve_shared_parent_scope_identity, set_latency_sensitivity,
};
pub use error::{AdaptiveError, Result};
#[cfg(feature = "redis-backend")]
pub use redis::RedisBackend;
pub use runtime::features::AdaptiveRuntime;
pub use storage::erased::AnyBackend;
pub use storage::memory::InMemoryBackend;
pub use storage::traits::{StorageBackend, StorageBackendDyn};

#[cfg(test)]
pub(crate) mod test_support {
    use tokio::sync::{Mutex, MutexGuard};

    static GLOBAL_RUNTIME_MUTEX: Mutex<()> = Mutex::const_new(());

    pub(crate) async fn lock_global_runtime() -> MutexGuard<'static, ()> {
        GLOBAL_RUNTIME_MUTEX.lock().await
    }

    pub(crate) fn blocking_lock_global_runtime() -> MutexGuard<'static, ()> {
        GLOBAL_RUNTIME_MUTEX.blocking_lock()
    }
}
