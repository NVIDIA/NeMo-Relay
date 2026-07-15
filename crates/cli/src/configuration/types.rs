// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolved runtime configuration model.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::http::HeaderMap;
use serde::Serialize;
use serde_json::{Map, Value};
use strum::{Display, IntoStaticStr};

use crate::plugins::policy::DynamicPluginHostPolicy;

use super::{
    DEFAULT_MAX_HOOK_PAYLOAD_BYTES, DEFAULT_MAX_PASSTHROUGH_BODY_BYTES, LoggingConfig, header_json,
    header_string,
};

#[derive(Debug, Clone)]
pub(crate) struct GatewayConfig {
    pub(crate) bind: SocketAddr,
    pub(crate) openai_base_url: String,
    pub(crate) anthropic_base_url: String,
    pub(crate) metadata: Option<Value>,
    pub(crate) plugin_config: Option<Value>,
    pub(crate) max_hook_payload_bytes: usize,
    pub(crate) max_passthrough_body_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SessionConfig {
    pub(crate) metadata: Option<Value>,
    pub(crate) plugin_config: Option<Value>,
    pub(crate) profile: Option<String>,
    pub(crate) gateway_mode: Option<String>,
}

impl GatewayConfig {
    pub(crate) fn session_config_from_headers(&self, headers: &HeaderMap) -> SessionConfig {
        let metadata =
            header_json(headers, "x-nemo-relay-session-metadata").or_else(|| self.metadata.clone());
        let plugin_config = header_json(headers, "x-nemo-relay-plugin-config")
            .or_else(|| self.plugin_config.clone());
        let profile = header_string(headers, "x-nemo-relay-config-profile");
        let gateway_mode = header_string(headers, "x-nemo-relay-gateway-mode");
        SessionConfig {
            metadata,
            plugin_config,
            profile,
            gateway_mode,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ResolvedConfig {
    pub(crate) gateway: GatewayConfig,
    pub(crate) agents: AgentConfigs,
    pub(crate) logging: LoggingConfig,
    pub(crate) dynamic_plugins: Vec<ResolvedDynamicPluginConfig>,
    pub(crate) dynamic_plugin_policy: DynamicPluginHostPolicy,
    pub(crate) bootstrap_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedDynamicPluginConfig {
    pub(crate) plugin_id: String,
    pub(crate) manifest_ref: String,
    pub(crate) config: Map<String, Value>,
    pub(crate) has_explicit_config: bool,
    pub(crate) source: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Display, IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub(crate) enum DynamicPluginHostConfigStatus {
    Absent,
    Present,
}

impl ResolvedDynamicPluginConfig {
    pub(crate) fn host_config_status(&self) -> DynamicPluginHostConfigStatus {
        if self.has_explicit_config {
            DynamicPluginHostConfigStatus::Present
        } else {
            DynamicPluginHostConfigStatus::Absent
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentConfigs {
    pub(crate) claude: AgentCommandConfig,
    pub(crate) codex: AgentCommandConfig,
    pub(crate) hermes: AgentCommandConfig,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentCommandConfig {
    pub(crate) command: Option<String>,
    /// Legacy Hermes config-path override retained for existing Relay configuration files.
    pub(crate) hooks_path: Option<PathBuf>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:4040"
                .parse()
                .expect("valid default bind address"),
            openai_base_url: "https://api.openai.com/v1".into(),
            anthropic_base_url: "https://api.anthropic.com".into(),
            metadata: None,
            plugin_config: None,
            max_hook_payload_bytes: DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
            max_passthrough_body_bytes: DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
        }
    }
}
