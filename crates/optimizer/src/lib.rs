// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # NeMo Agent Toolkit Nexus Optimizer
//!
//! Dynamic optimizer runtime for NeMo Agent Toolkit Nexus. The canonical public
//! interface is a config document plus a runtime handle. Built-in optimizer
//! features are implemented as internal components selected by config.

pub mod config;
pub mod context_helpers;
pub mod drain;
pub mod dynamo_intercept;
pub mod error;
pub mod intercepts;
pub mod learner;
#[cfg(feature = "redis-backend")]
pub mod redis;
pub mod runtime;
pub mod storage;
pub mod subscriber;
pub mod trie;
pub mod types;

pub use config::{
    BackendSpec, ComponentSpec, ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel,
    DynamoHintsComponentConfig, OptimizerConfig, StateConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig, UnsupportedBehavior,
};
pub use context_helpers::{
    extract_scope_path, read_manual_latency_sensitivity, resolve_agent_id, set_latency_sensitivity,
    LATENCY_SENSITIVITY_POINTER,
};
pub use dynamo_intercept::DynamoIntercept;
pub use error::{OptimizerError, Result};
pub use intercepts::AGENT_HINTS_HEADER_KEY;
pub use learner::{compute_default_hints, LatencySensitivityLearner, Learner};
#[cfg(feature = "redis-backend")]
pub use redis::RedisBackend;
pub use runtime::{
    deregister_component_factory, deregister_hosted_plugin_handler, list_component_kinds,
    list_hosted_plugin_kinds, register_component_factory, register_hosted_plugin_handler,
    BuildContext, ComponentRegistration, HostedPluginHandler, HostedRegistrationContext,
    OptimizerComponent, OptimizerComponentFactory, OptimizerRuntime, RegistrationContext,
    ValidationContext,
};
pub use storage::{AnyBackend, InMemoryBackend, StorageBackend, StorageBackendDyn};
pub use types::*;
