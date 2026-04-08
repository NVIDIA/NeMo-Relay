// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical optimizer config and diagnostics types.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

/// Canonical config document for the optimizer runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    #[serde(default = "default_optimizer_config_version")]
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<StateConfig>,
    #[serde(default)]
    pub components: Vec<ComponentSpec>,
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            version: default_optimizer_config_version(),
            agent_id: None,
            state: None,
            components: vec![],
            policy: ConfigPolicy::default(),
        }
    }
}

/// Shared state configuration consumed by components that need persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateConfig {
    pub backend: BackendSpec,
}

/// Dynamic backend selection. `config` is backend-specific.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSpec {
    pub kind: String,
    #[serde(default)]
    pub config: Map<String, Json>,
}

impl BackendSpec {
    pub fn in_memory() -> Self {
        Self {
            kind: "in_memory".to_string(),
            config: Map::new(),
        }
    }

    #[cfg(feature = "redis-backend")]
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

/// Dynamic component selection. `config` is component-specific.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSpec {
    pub kind: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub config: Map<String, Json>,
}

impl ComponentSpec {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            enabled: true,
            config: Map::new(),
        }
    }
}

/// Policy for how unsupported config is handled.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ConfigPolicy {
    #[serde(default = "default_warn")]
    pub unknown_component: UnsupportedBehavior,
    #[serde(default = "default_warn")]
    pub unknown_field: UnsupportedBehavior,
    #[serde(default = "default_error")]
    pub unsupported_value: UnsupportedBehavior,
}

/// Per-policy behavior for unsupported configuration.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum UnsupportedBehavior {
    Ignore,
    #[default]
    Warn,
    Error,
}

/// Structured validation report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigReport {
    #[serde(default)]
    pub diagnostics: Vec<ConfigDiagnostic>,
}

impl ConfigReport {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error)
    }
}

/// One validation or compatibility diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDiagnostic {
    pub level: DiagnosticLevel,
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    pub message: String,
}

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    Warning,
    Error,
}

/// Typed helper for the built-in telemetry component.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetryComponentConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscriber_name: Option<String>,
    #[serde(default)]
    pub learners: Vec<String>,
}

impl From<TelemetryComponentConfig> for ComponentSpec {
    fn from(value: TelemetryComponentConfig) -> Self {
        let Json::Object(config) =
            serde_json::to_value(value).expect("telemetry config should serialize to object")
        else {
            unreachable!("telemetry config must serialize to object");
        };
        ComponentSpec {
            kind: "telemetry".to_string(),
            enabled: true,
            config,
        }
    }
}

/// Typed helper for the built-in dynamo hints component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamoHintsComponentConfig {
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default)]
    pub break_chain: bool,
    #[serde(default = "default_true")]
    pub inject_header: bool,
    #[serde(default = "default_dynamo_path")]
    pub inject_body_path: String,
}

impl Default for DynamoHintsComponentConfig {
    fn default() -> Self {
        Self {
            priority: default_priority(),
            break_chain: false,
            inject_header: true,
            inject_body_path: default_dynamo_path(),
        }
    }
}

impl From<DynamoHintsComponentConfig> for ComponentSpec {
    fn from(value: DynamoHintsComponentConfig) -> Self {
        let Json::Object(config) =
            serde_json::to_value(value).expect("dynamo config should serialize to object")
        else {
            unreachable!("dynamo config must serialize to object");
        };
        ComponentSpec {
            kind: "dynamo_hints".to_string(),
            enabled: true,
            config,
        }
    }
}

/// Typed helper for the built-in tool parallelism component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParallelismComponentConfig {
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default = "default_tool_parallelism_mode")]
    pub mode: String,
}

impl Default for ToolParallelismComponentConfig {
    fn default() -> Self {
        Self {
            priority: default_priority(),
            mode: default_tool_parallelism_mode(),
        }
    }
}

impl From<ToolParallelismComponentConfig> for ComponentSpec {
    fn from(value: ToolParallelismComponentConfig) -> Self {
        let Json::Object(config) = serde_json::to_value(value)
            .expect("tool parallelism config should serialize to object")
        else {
            unreachable!("tool parallelism config must serialize to object");
        };
        ComponentSpec {
            kind: "tool_parallelism".to_string(),
            enabled: true,
            config,
        }
    }
}

fn default_optimizer_config_version() -> u32 {
    1
}

fn default_enabled() -> bool {
    true
}

fn default_warn() -> UnsupportedBehavior {
    UnsupportedBehavior::Warn
}

fn default_error() -> UnsupportedBehavior {
    UnsupportedBehavior::Error
}

fn default_priority() -> i32 {
    100
}

fn default_true() -> bool {
    true
}

fn default_dynamo_path() -> String {
    "nvext.agent_hints".to_string()
}

fn default_tool_parallelism_mode() -> String {
    "observe_only".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_component_helpers_round_trip() {
        let telemetry: ComponentSpec = TelemetryComponentConfig::default().into();
        assert_eq!(telemetry.kind, "telemetry");

        let dynamo: ComponentSpec = DynamoHintsComponentConfig::default().into();
        assert_eq!(dynamo.kind, "dynamo_hints");

        let tool: ComponentSpec = ToolParallelismComponentConfig::default().into();
        assert_eq!(tool.kind, "tool_parallelism");
    }

    #[test]
    fn test_optimizer_config_defaults() {
        let config = OptimizerConfig::default();
        assert_eq!(config.version, 1);
        assert_eq!(config.policy.unknown_component, UnsupportedBehavior::Warn);
    }
}
