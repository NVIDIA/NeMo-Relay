// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Args;

#[derive(Debug, Clone, Default, Args)]
pub(crate) struct ServerArgs {
    /// Path to an explicit config file (disables auto-discovery of workspace/global/system)
    #[arg(long)]
    pub(super) config: Option<PathBuf>,
    /// Address for the gateway to listen on in daemon mode (default 127.0.0.1:4040)
    #[arg(long, env = "NEMO_RELAY_GATEWAY_BIND")]
    pub(super) bind: Option<SocketAddr>,
    /// Upstream OpenAI-compatible base URL (e.g. https://api.openai.com/v1, NVIDIA inference)
    #[arg(long, env = "NEMO_RELAY_OPENAI_BASE_URL")]
    pub(super) openai_base_url: Option<String>,
    /// Upstream Anthropic base URL (e.g. https://api.anthropic.com)
    #[arg(long, env = "NEMO_RELAY_ANTHROPIC_BASE_URL")]
    pub(super) anthropic_base_url: Option<String>,
    /// Internal override for the plugin configuration file.
    #[arg(long, env = "NEMO_RELAY_PLUGIN_CONFIG_PATH", hide = true)]
    pub(super) plugin_config_path: Option<PathBuf>,
    /// Internal readiness file used by plugin sidecar bootstrap.
    #[arg(long, hide = true)]
    pub(super) ready_file: Option<PathBuf>,
    /// Maximum accepted coding-agent hook payload size, in bytes.
    #[arg(long, env = "NEMO_RELAY_MAX_HOOK_PAYLOAD_BYTES")]
    pub(super) max_hook_payload_bytes: Option<usize>,
    /// Maximum accepted provider passthrough request body size, in bytes.
    #[arg(long, env = "NEMO_RELAY_MAX_PASSTHROUGH_BODY_BYTES")]
    pub(super) max_passthrough_body_bytes: Option<usize>,
}

impl ServerArgs {
    pub(super) fn to_runtime(&self) -> crate::server::GatewayOverrides {
        crate::server::GatewayOverrides {
            config: self.config.clone(),
            bind: self.bind,
            openai_base_url: self.openai_base_url.clone(),
            anthropic_base_url: self.anthropic_base_url.clone(),
            plugin_config_path: self.plugin_config_path.clone(),
            ready_file: self.ready_file.clone(),
            max_hook_payload_bytes: self.max_hook_payload_bytes,
            max_passthrough_body_bytes: self.max_passthrough_body_bytes,
        }
    }
}
