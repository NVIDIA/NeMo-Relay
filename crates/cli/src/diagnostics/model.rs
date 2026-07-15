// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable diagnostic report data model.

use std::path::PathBuf;

use serde::Serialize;

use crate::configuration::DynamicPluginHostConfigStatus;

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
    pub workspace: ConfigLayer,
    pub global: ConfigLayer,
    pub system: ConfigLayer,
    pub plugin_configs: Vec<ConfigLayer>,
    pub plugin_resolution: Check,
    pub resolution: Check,
    pub default_agent: Option<String>,
    pub configured_agents: Vec<String>,
    pub dynamic_plugins: Vec<DynamicPluginReferenceInfo>,
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
}
