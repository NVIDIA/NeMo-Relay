// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use crate::agents::CodingAgent;

#[derive(Debug, Clone)]
pub(crate) struct HookForwardRequest {
    pub(crate) agent: CodingAgent,
    pub(crate) hook_config: Option<PathBuf>,
    pub(crate) gateway_url: Option<String>,
    pub(crate) generation_file: Option<PathBuf>,
    pub(crate) generation_token: Option<String>,
    pub(crate) forward_only: bool,
    pub(crate) transparent_run: bool,
    pub(crate) profile: Option<String>,
    pub(crate) session_metadata: Option<String>,
    pub(crate) gateway_mode: Option<GatewayMode>,
    pub(crate) failure_policy: HookFailurePolicy,
}

impl HookForwardRequest {
    pub(crate) fn has_inline_configuration(&self) -> bool {
        self.gateway_url.is_some()
            || self.generation_file.is_some()
            || self.generation_token.is_some()
            || self.forward_only
            || self.transparent_run
            || self.profile.is_some()
            || self.session_metadata.is_some()
            || self.gateway_mode.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HookFailurePolicy {
    Default,
    FailOpen,
    FailClosed,
}

impl HookFailurePolicy {
    pub(crate) fn fail_closed(self) -> bool {
        match self {
            Self::Default => std::env::var("NEMO_RELAY_FAIL_CLOSED").ok().as_deref() == Some("1"),
            Self::FailOpen => false,
            Self::FailClosed => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum GatewayMode {
    HookOnly,
    Passthrough,
    Required,
}

impl GatewayMode {
    pub(crate) const fn as_arg(self) -> &'static str {
        match self {
            Self::HookOnly => "hook-only",
            Self::Passthrough => "passthrough",
            Self::Required => "required",
        }
    }
}
