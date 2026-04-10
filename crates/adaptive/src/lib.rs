// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # NeMo Flow Adaptive
//!
//! Adaptive config helpers and core-plugin integration for NeMo Flow.
//! Adaptive behavior is enabled through the generic core plugin host.

pub mod adaptive_hints_intercept;
pub mod config;
pub mod context_helpers;
pub mod drain;
pub mod error;
pub mod intercepts;
pub mod learner;
pub mod plugin_component;
#[cfg(feature = "redis-backend")]
pub mod redis;
mod runtime;
pub mod storage;
pub mod subscriber;
pub mod trie;
pub mod types;

pub use adaptive_hints_intercept::AdaptiveHintsIntercept;
pub use config::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, BackendSpec, StateConfig,
    TelemetryComponentConfig, ToolParallelismComponentConfig,
};
pub use context_helpers::{
    LATENCY_SENSITIVITY_POINTER, extract_scope_path, read_manual_latency_sensitivity,
    resolve_agent_id, set_latency_sensitivity,
};
pub use error::{AdaptiveError, Result};
pub use intercepts::AGENT_HINTS_HEADER_KEY;
pub use learner::{LatencySensitivityLearner, Learner, compute_default_hints};
pub use nemo_flow::{
    ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel, UnsupportedBehavior,
};
pub use plugin_component::{
    ADAPTIVE_PLUGIN_KIND, ComponentSpec, deregister_adaptive_component, register_adaptive_component,
};
#[cfg(feature = "redis-backend")]
pub use redis::RedisBackend;
pub use storage::{AnyBackend, InMemoryBackend, StorageBackend, StorageBackendDyn};
pub use types::*;
