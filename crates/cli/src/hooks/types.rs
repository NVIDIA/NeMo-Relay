// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use crate::agents::CodingAgent;

#[derive(Debug, Clone)]
pub(crate) struct HookForwardRequest {
    pub(crate) agent: CodingAgent,
    pub(crate) gateway_url: Option<String>,
    pub(crate) generation_file: Option<PathBuf>,
    pub(crate) generation_token: Option<String>,
    pub(crate) forward_only: bool,
    pub(crate) transparent_run: bool,
    pub(crate) profile: Option<String>,
    pub(crate) session_metadata: Option<String>,
    pub(crate) gateway_mode: Option<GatewayMode>,
    pub(crate) fail_closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
