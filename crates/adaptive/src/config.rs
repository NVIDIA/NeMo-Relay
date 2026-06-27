// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical adaptive config and diagnostics types.

use nemo_relay::plugin::ConfigPolicy;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

/// Canonical config document for the adaptive plugin component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Adaptive config schema version.
    #[serde(default = "default_adaptive_config_version")]
    pub version: u32,
    /// Fallback agent identifier used when no Agent scope is active.
    /// Scoped runtime calls use the active Agent scope name instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Shared state backend configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<StateConfig>,
    /// Built-in adaptive telemetry settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<TelemetryComponentConfig>,
    /// Built-in LLM hint injection settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_hints: Option<AdaptiveHintsComponentConfig>,
    /// Built-in tool scheduling settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_parallelism: Option<ToolParallelismComponentConfig>,
    /// Adaptive Cache Governor settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acg: Option<AcgComponentConfig>,
    /// Global topological convergence settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence: Option<ConvergenceConfig>,
    /// Adaptive-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            version: default_adaptive_config_version(),
            agent_id: None,
            state: None,
            telemetry: None,
            adaptive_hints: None,
            tool_parallelism: None,
            acg: None,
            convergence: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// Shared state configuration consumed by adaptive features that need persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateConfig {
    /// Backend selection for adaptive state.
    pub backend: BackendSpec,
}

/// Dynamic backend selection. `config` is backend-specific.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSpec {
    /// Backend kind such as `in_memory` or `redis`.
    pub kind: String,
    /// Backend-specific JSON object.
    #[serde(default)]
    pub config: Map<String, Json>,
}

impl Default for BackendSpec {
    fn default() -> Self {
        Self::in_memory()
    }
}

impl BackendSpec {
    /// Creates an in-memory backend spec.
    pub fn in_memory() -> Self {
        Self {
            kind: "in_memory".to_string(),
            config: Map::new(),
        }
    }

    #[cfg(feature = "redis-backend")]
    /// Creates a Redis backend spec.
    pub fn redis(url: impl Into<String>, key_prefix: impl Into<String>) -> Self {
        let mut config = Map::new();
        config.insert("url".to_string(), Json::String(url.into()));
        config.insert("key_prefix".to_string(), Json::String(key_prefix.into()));
        Self {
            kind: "redis".to_string(),
            config,
        }
    }
}

/// Typed helper for telemetry settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetryComponentConfig {
    /// Optional subscriber registration name override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscriber_name: Option<String>,
    /// Enabled learner identifiers.
    #[serde(default)]
    pub learners: Vec<String>,
}

/// Typed helper for adaptive hints settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveHintsComponentConfig {
    /// Intercept priority. Lower values run first.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Whether later request intercepts should be skipped after this one runs.
    #[serde(default)]
    pub break_chain: bool,
    /// Whether to inject the adaptive hints header.
    #[serde(default = "default_true")]
    pub inject_header: bool,
    /// JSON path used when injecting request-body hints.
    #[serde(default = "default_adaptive_hints_path")]
    pub inject_body_path: String,
    /// Optional topology-aware load-shedding governor for hint injection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub governor: Option<GovernorConfig>,
}

impl Default for AdaptiveHintsComponentConfig {
    fn default() -> Self {
        Self {
            priority: default_priority(),
            break_chain: false,
            inject_header: true,
            inject_body_path: default_adaptive_hints_path(),
            governor: None,
        }
    }
}

/// Typed helper for tool parallelism settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParallelismComponentConfig {
    /// Intercept priority. Lower values run first.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Scheduling mode such as `observe_only`, `inject_hints`, or `schedule`.
    #[serde(default = "default_tool_parallelism_mode")]
    pub mode: String,
    /// Optional topology-aware drift detector for stale plan invalidation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift: Option<DriftConfig>,
}

impl Default for ToolParallelismComponentConfig {
    fn default() -> Self {
        Self {
            priority: default_priority(),
            mode: default_tool_parallelism_mode(),
            drift: None,
        }
    }
}

/// Typed helper for topology-aware hint load shedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorConfig {
    /// Whether the governor is active.
    #[serde(default)]
    pub enabled: bool,
    /// Initial sensitivity threshold used by the governor.
    #[serde(default = "default_governor_epsilon")]
    pub epsilon: f64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            epsilon: default_governor_epsilon(),
        }
    }
}

/// Typed helper for topology-aware tool plan drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Whether drift detection is active.
    #[serde(default)]
    pub enabled: bool,
    /// Drift distance above which the existing execution plan is invalidated.
    #[serde(default = "default_drift_threshold")]
    pub threshold: f64,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: default_drift_threshold(),
        }
    }
}

/// Typed helper for topological convergence detection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Whether convergence detection is active.
    #[serde(default)]
    pub enabled: bool,
    /// Error threshold below which the detector is considered converged.
    #[serde(default = "default_convergence_epsilon")]
    pub epsilon: f64,
    /// Minimum number of epochs required to judge Betti-number stability.
    #[serde(default = "default_convergence_stability_window")]
    pub stability_window: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            epsilon: default_convergence_epsilon(),
            stability_window: default_convergence_stability_window(),
        }
    }
}

/// Typed helper for the built-in Adaptive Cache Governor (ACG) component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcgComponentConfig {
    /// Which provider plugin to activate (e.g. "anthropic", "openai", "passthrough").
    #[serde(default = "default_acg_provider")]
    pub provider: String,
    /// Rolling observation window size. Default: 100.
    #[serde(default = "default_acg_observation_window")]
    pub observation_window: usize,
    /// LLM execution intercept priority. Default: 50.
    #[serde(default = "default_acg_priority")]
    pub priority: i32,
    /// Stability classification thresholds used by the learner.
    #[serde(default)]
    pub stability_thresholds: crate::acg::stability::StabilityThresholds,
    /// Optional component-scoped topological convergence settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence: Option<ConvergenceConfig>,
}

impl Default for AcgComponentConfig {
    fn default() -> Self {
        Self {
            provider: default_acg_provider(),
            observation_window: default_acg_observation_window(),
            priority: default_acg_priority(),
            stability_thresholds: crate::acg::stability::StabilityThresholds::default(),
            convergence: None,
        }
    }
}

fn default_adaptive_config_version() -> u32 {
    1
}

fn default_priority() -> i32 {
    100
}

fn default_true() -> bool {
    true
}

fn default_adaptive_hints_path() -> String {
    "nvext.agent_hints".to_string()
}

fn default_tool_parallelism_mode() -> String {
    "observe_only".to_string()
}

fn default_acg_provider() -> String {
    "passthrough".to_string()
}

fn default_acg_observation_window() -> usize {
    100
}

fn default_acg_priority() -> i32 {
    50
}

fn default_convergence_epsilon() -> f64 {
    0.001
}

fn default_convergence_stability_window() -> usize {
    3
}

fn default_governor_epsilon() -> f64 {
    1.0
}

fn default_drift_threshold() -> f64 {
    0.75
}

nemo_relay::editor_config! {
    impl AdaptiveConfig {
        agent_id => { label: "fallback_agent_id", kind: String, optional: true },
        state => {
            label: "state",
            kind: Section,
            optional: true,
            nested: StateConfig,
            default: StateConfig,
        },
        telemetry => {
            label: "telemetry",
            kind: Section,
            optional: true,
            nested: TelemetryComponentConfig,
            default: TelemetryComponentConfig,
        },
        adaptive_hints => {
            label: "adaptive_hints",
            kind: Section,
            optional: true,
            nested: AdaptiveHintsComponentConfig,
            default: AdaptiveHintsComponentConfig,
        },
        tool_parallelism => {
            label: "tool_parallelism",
            kind: Section,
            optional: true,
            nested: ToolParallelismComponentConfig,
            default: ToolParallelismComponentConfig,
        },
        acg => {
            label: "acg",
            kind: Section,
            optional: true,
            nested: AcgComponentConfig,
            default: AcgComponentConfig,
        },
        convergence => {
            label: "convergence",
            kind: Section,
            optional: true,
            nested: ConvergenceConfig,
            default: ConvergenceConfig,
        },
        policy => {
            label: "policy",
            kind: Section,
            nested: ConfigPolicy,
            default: ConfigPolicy,
        },
    }
}

nemo_relay::editor_config! {
    impl StateConfig {
        backend => {
            label: "backend",
            kind: Section,
            nested: BackendSpec,
            default: BackendSpec,
        },
    }
}

nemo_relay::editor_config! {
    impl BackendSpec {
        kind => { label: "kind", kind: Enum, values: ["in_memory", "redis"] },
        config => { label: "config", kind: Json },
    }
}

nemo_relay::editor_config! {
    impl TelemetryComponentConfig {
        subscriber_name => { label: "subscriber_name", kind: String, optional: true },
        learners => { label: "learners", kind: Json },
    }
}

nemo_relay::editor_config! {
    impl AdaptiveHintsComponentConfig {
        priority => { label: "priority", kind: Integer },
        break_chain => { label: "break_chain", kind: Boolean },
        inject_header => { label: "inject_header", kind: Boolean },
        inject_body_path => { label: "inject_body_path", kind: String },
        governor => {
            label: "governor",
            kind: Section,
            optional: true,
            nested: GovernorConfig,
            default: GovernorConfig,
        },
    }
}

nemo_relay::editor_config! {
    impl ToolParallelismComponentConfig {
        priority => { label: "priority", kind: Integer },
        mode => {
            label: "mode",
            kind: Enum,
            values: ["observe_only", "inject_hints", "schedule"],
        },
        drift => {
            label: "drift",
            kind: Section,
            optional: true,
            nested: DriftConfig,
            default: DriftConfig,
        },
    }
}

nemo_relay::editor_config! {
    impl GovernorConfig {
        enabled => { label: "enabled", kind: Boolean },
        epsilon => { label: "epsilon", kind: Float },
    }
}

nemo_relay::editor_config! {
    impl DriftConfig {
        enabled => { label: "enabled", kind: Boolean },
        threshold => { label: "threshold", kind: Float },
    }
}

nemo_relay::editor_config! {
    impl AcgComponentConfig {
        provider => {
            label: "provider",
            kind: Enum,
            values: ["passthrough", "anthropic", "openai"],
        },
        observation_window => { label: "observation_window", kind: Integer },
        priority => { label: "priority", kind: Integer },
        stability_thresholds => {
            label: "stability_thresholds",
            kind: Section,
            nested: crate::acg::stability::StabilityThresholds,
            default: crate::acg::stability::StabilityThresholds,
        },
        convergence => {
            label: "convergence",
            kind: Section,
            optional: true,
            nested: ConvergenceConfig,
            default: ConvergenceConfig,
        },
    }
}

nemo_relay::editor_config! {
    impl ConvergenceConfig {
        enabled => { label: "enabled", kind: Boolean },
        epsilon => { label: "epsilon", kind: Float },
        stability_window => { label: "stability_window", kind: Integer },
    }
}

nemo_relay::editor_config! {
    impl crate::acg::stability::StabilityThresholds {
        stable_threshold => { label: "stable_threshold", kind: Float },
        semi_stable_threshold => { label: "semi_stable_threshold", kind: Float },
        min_observations_for_full_confidence => {
            label: "min_observations_for_full_confidence",
            kind: Integer,
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/config_tests.rs"]
mod tests;
