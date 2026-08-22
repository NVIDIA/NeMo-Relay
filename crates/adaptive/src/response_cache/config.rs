// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend selection for the adaptive plugin's `response_cache` feature.
//!
//! The `response_cache` section struct ([`crate::config::ResponseCacheConfig`])
//! lives in [`crate::config`] alongside the other `AdaptiveConfig` sections
//! (`acg`, `adaptive_hints`, `tool_parallelism`). This module keeps the
//! response-cache-specific backend config and the key-strategy constant next to
//! the key/store code that consumes them.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value as Json};

/// Strategy for deriving an LLM response-cache key.
///
/// The `Unknown` variant preserves an unsupported JSON/TOML value long enough
/// for configuration validation to report it with a field-specific diagnostic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ResponseCacheKeyStrategy {
    /// Key on the normalized request exactly.
    #[default]
    ExactRequest,
    /// Normalize tool schemas structurally while preserving their interface.
    Logical,
    /// A wire value not supported by this Relay build.
    Unknown(String),
}

impl ResponseCacheKeyStrategy {
    /// Stable JSON/TOML representation of this strategy.
    pub fn as_str(&self) -> &str {
        match self {
            Self::ExactRequest => "exact_request",
            Self::Logical => "logical",
            Self::Unknown(value) => value,
        }
    }
}

impl From<&str> for ResponseCacheKeyStrategy {
    fn from(value: &str) -> Self {
        match value {
            "exact_request" => Self::ExactRequest,
            "logical" => Self::Logical,
            _ => Self::Unknown(value.to_string()),
        }
    }
}

impl From<String> for ResponseCacheKeyStrategy {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl Serialize for ResponseCacheKeyStrategy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ResponseCacheKeyStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value))
    }
}

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

/// Opt-in tool-result cache configuration.
///
/// Cache only tools that are read-only and stable for their TTL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCacheConfig {
    /// Master switch; off by default.
    pub enabled: bool,
    /// Tool execution-intercept priority. Lower values run earlier and outermost.
    pub priority: i32,
    /// Whether conventional in-band tool error results may be stored.
    pub cache_errors: bool,
    /// Policy for unclassified tools; not cacheable by default.
    pub default: ToolClass,
    /// Named tool classes.
    pub classes: BTreeMap<String, ToolClass>,
    /// Per-tool refinements keyed by exact name, `prefix*`, `*suffix`, or `*contains*`.
    pub overrides: BTreeMap<String, ToolOverride>,
}

impl Default for ToolCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: 100,
            cache_errors: false,
            default: ToolClass::default(),
            classes: BTreeMap::new(),
            overrides: BTreeMap::new(),
        }
    }
}

nemo_relay::editor_config! {
    impl ToolCacheConfig {
        enabled => { label: "enabled", kind: Boolean },
        priority => { label: "priority", kind: Integer },
        cache_errors => { label: "cache_errors", kind: Boolean },
        default => {
            label: "default",
            kind: Section,
            nested: ToolClass,
            default: ToolClass,
        },
        classes => { label: "classes", kind: Json },
        overrides => { label: "overrides", kind: Json },
    }
}

/// Policy shared by a class of tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolClass {
    /// Whether class members may be served from cache.
    pub cacheable: bool,
    /// TTL in seconds; inherits the response-cache TTL when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    /// Live-rerun probability; inherits the response-cache rate when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bypass_rate: Option<f64>,
    /// Version string folded into cache keys for members of this class.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    /// Top-level argument keys dropped before keying.
    pub arg_skip: Vec<String>,
    /// Exact tool names, `prefix*`, `*suffix`, or `*contains*` patterns in this class.
    pub members: Vec<String>,
}

/// Per-tool refinement applied after class resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolOverride {
    /// Overrides the class cacheability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cacheable: Option<bool>,
    /// Overrides the class TTL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    /// Overrides the class bypass rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bypass_rate: Option<f64>,
    /// Version string folded into the cache key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    /// Replaces the class argument skip list when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arg_skip: Option<Vec<String>>,
}

nemo_relay::editor_config! {
    impl ToolClass {
        cacheable => { label: "cacheable", kind: Boolean },
        ttl_seconds => { label: "ttl_seconds", kind: Integer, optional: true },
        bypass_rate => { label: "bypass_rate", kind: Float, optional: true },
        tool_version => { label: "tool_version", kind: String, optional: true },
        arg_skip => {
            label: "arg_skip",
            kind: List,
            list: &nemo_relay::config_editor::STRING_LIST_ITEM,
        },
        members => {
            label: "members",
            kind: List,
            list: &nemo_relay::config_editor::STRING_LIST_ITEM,
        },
    }
}

nemo_relay::editor_config! {
    impl ToolOverride {
        cacheable => { label: "cacheable", kind: Boolean, optional: true },
        ttl_seconds => { label: "ttl_seconds", kind: Integer, optional: true },
        bypass_rate => { label: "bypass_rate", kind: Float, optional: true },
        tool_version => { label: "tool_version", kind: String, optional: true },
        arg_skip => {
            label: "arg_skip",
            kind: List,
            optional: true,
            list: &nemo_relay::config_editor::STRING_LIST_ITEM,
        },
    }
}
