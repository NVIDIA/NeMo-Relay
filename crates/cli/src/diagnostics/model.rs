// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable diagnostic report data model.

use std::path::PathBuf;

use serde::Serialize;

use crate::configuration::{DynamicPluginHostConfigStatus, GatewayConfig};

/// Outcome of one check inside the doctor report.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct Check {
    pub name: &'static str,
    pub status: Status,
    pub details: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Status {
    Pass,
    Warn,
    Fail,
    /// The check ran but no relevant state was detected.
    Info,
}

/// Snapshot of the running system rendered by `doctor`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DoctorReport {
    pub schema_version: u32,
    pub binary_version: &'static str,
    pub target_agent: Option<String>,
    pub environment: EnvironmentInfo,
    pub configuration: ConfigurationInfo,
    pub agents: Vec<AgentInfo>,
    pub host_plugins: Vec<crate::installation::marketplace::HostPluginReadiness>,
    pub observability: Vec<Check>,
    pub completions: Vec<Check>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EnvironmentInfo {
    pub os: String,
    pub arch: &'static str,
    pub shell: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConfigurationInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit: Option<ConfigLayer>,
    pub global: ConfigLayer,
    pub system: ConfigLayer,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unsupported_project_files: Vec<ConfigLayer>,
    pub upstream_auth: UpstreamAuthInfo,
    pub plugin_configs: Vec<ConfigLayer>,
    pub plugin_resolution: Check,
    pub resolution: Check,
    pub default_agent: Option<String>,
    pub configured_agents: Vec<String>,
    pub dynamic_plugins: Vec<DynamicPluginReferenceInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UpstreamAuthInfo {
    pub openai: SecretPresence,
    pub anthropic: SecretPresence,
}

impl UpstreamAuthInfo {
    pub(crate) fn from_effective_gateway_auth(gateway: &GatewayConfig) -> Self {
        Self {
            openai: SecretPresence::from_effective_provider_auth(
                gateway.openai_auth_header.as_deref(),
                "OPENAI_API_KEY",
            ),
            anthropic: SecretPresence::from_effective_provider_auth(
                gateway.anthropic_auth_header.as_deref(),
                "ANTHROPIC_API_KEY",
            ),
        }
    }

    pub(crate) fn unknown() -> Self {
        Self {
            openai: SecretPresence::Unknown,
            anthropic: SecretPresence::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SecretPresence {
    Configured,
    Unset,
    Unknown,
}

impl SecretPresence {
    pub(crate) fn from_effective_provider_auth(
        configured_auth_header: Option<&str>,
        fallback_env_var: &str,
    ) -> Self {
        if configured_auth_header.is_some() || env_var_is_nonempty(fallback_env_var) {
            Self::Configured
        } else {
            Self::Unset
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Configured => "configured",
            Self::Unset => "unset",
            Self::Unknown => "unknown",
        }
    }
}

fn env_var_is_nonempty(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some()
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct DynamicPluginReferenceInfo {
    pub plugin_id: String,
    pub manifest_ref: String,
    pub source: PathBuf,
    pub host_config_status: DynamicPluginHostConfigStatus,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConfigLayer {
    pub path: PathBuf,
    pub status: Status,
    pub active: bool,
    pub details: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AgentInfo {
    pub name: &'static str,
    pub status: Status,
    pub configured: bool,
    pub command: String,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    pub annotation: String,
    /// Agent-specific preflight checks, folded into `status` above.
    ///
    /// Empty for every agent whose setup NeMo Relay writes itself, because
    /// `annotation` already says everything there is to say about it. pi is the
    /// exception: its hooks live in an extension the *user* installs, so where
    /// that extension sits decides whether pi loads it at all, and that is a
    /// finding with its own status rather than a sentence in a summary.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<Check>,
}
