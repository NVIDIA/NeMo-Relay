// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # NeMo Agent Toolkit Nexus Proxy
//!
//! Telemetry and metadata wiring for NeMo Agent Toolkit Nexus. This crate provides
//! the data types, storage abstraction, and intercepts that turn Nexus lifecycle
//! events into executable intelligence.

pub mod context_helpers;
pub mod drain;
pub mod dynamo_intercept;
pub mod error;
pub mod intercepts;
pub mod learner;
pub mod manager;
pub mod proxy;
#[cfg(feature = "redis-backend")]
pub mod redis;
pub mod storage;
pub mod subscriber;
pub mod trie;
pub mod types;

pub use context_helpers::{
    extract_scope_path, read_manual_latency_sensitivity, resolve_agent_id, set_latency_sensitivity,
    LATENCY_SENSITIVITY_POINTER,
};
pub use dynamo_intercept::DynamoIntercept;
pub use error::{ProxyError, Result};
pub use intercepts::AGENT_HINTS_HEADER_KEY;
pub use learner::{compute_default_hints, LatencySensitivityLearner, Learner};
pub use manager::{
    ensure_proxy, proxy_active, set_dynamo_intercept, set_proxy_backend, set_proxy_sensitivity,
    set_use_proxy, teardown_proxy, ProxyConfig, ProxyManager, PROXY_EXTENSION_KEY,
};
pub use proxy::{NexusProxy, NexusProxyBuilder};
#[cfg(feature = "redis-backend")]
pub use redis::RedisBackend;
pub use storage::{AnyBackend, InMemoryBackend, StorageBackend, StorageBackendDyn};
pub use types::*;
