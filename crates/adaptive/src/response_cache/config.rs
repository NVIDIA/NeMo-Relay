// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend selection for the adaptive plugin's `response_cache` feature.
//!
//! The `response_cache` section struct ([`crate::config::ResponseCacheConfig`])
//! lives in [`crate::config`] alongside the other `AdaptiveConfig` sections
//! (`acg`, `adaptive_hints`, `tool_parallelism`). This module keeps the
//! response-cache-specific backend config and the key-strategy constant next to
//! the key/store code that consumes them.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

/// Exact-request key strategy identifier.
pub const KEY_STRATEGY_EXACT_REQUEST: &str = "exact_request";

/// Default in-memory byte budget: 256 MiB.
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Backend selection mirroring the adaptive [`crate::config::BackendSpec`] shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendConfig {
    /// Backend kind: `"in_memory"` or `"redis"` (needs the `redis-backend` feature).
    pub kind: String,
    /// Backend-specific options (in_memory: `max_bytes`; redis: `url`/`key_prefix`).
    pub config: Map<String, Json>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            kind: "in_memory".to_string(),
            config: Map::new(),
        }
    }
}

impl BackendConfig {
    /// In-memory total-bytes budget before oldest-first eviction.
    pub fn max_bytes(&self) -> usize {
        self.config
            .get("max_bytes")
            .and_then(Json::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_MAX_BYTES)
    }
}

#[cfg(not(feature = "redis-backend"))]
nemo_relay::editor_config! {
    impl BackendConfig {
        kind => { label: "kind", kind: Enum, values: ["in_memory"] },
        config => { label: "config", kind: Json },
    }
}

#[cfg(feature = "redis-backend")]
nemo_relay::editor_config! {
    impl BackendConfig {
        kind => { label: "kind", kind: Enum, values: ["in_memory", "redis"] },
        config => { label: "config", kind: Json },
    }
}
