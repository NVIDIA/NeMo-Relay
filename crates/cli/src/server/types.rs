// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub(crate) struct GatewayOverrides {
    pub(crate) config: Option<PathBuf>,
    pub(crate) bind: Option<SocketAddr>,
    pub(crate) openai_base_url: Option<String>,
    pub(crate) anthropic_base_url: Option<String>,
    pub(crate) plugin_config_path: Option<PathBuf>,
    pub(crate) ready_file: Option<PathBuf>,
    pub(crate) max_hook_payload_bytes: Option<usize>,
    pub(crate) max_passthrough_body_bytes: Option<usize>,
}

impl GatewayOverrides {
    pub(crate) fn requested_daemon_mode(&self) -> bool {
        self.bind.is_some()
            || self.openai_base_url.is_some()
            || self.anthropic_base_url.is_some()
            || self.plugin_config_path.is_some()
            || self.ready_file.is_some()
            || self.max_hook_payload_bytes.is_some()
            || self.max_passthrough_body_bytes.is_some()
            || self.config.is_some()
    }
}
