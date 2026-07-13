// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::{Args, ValueEnum};

use super::root::AgentArg;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct HookForwardCommand {
    #[arg(value_enum)]
    pub(crate) agent: AgentArg,
    #[arg(long)]
    pub(crate) gateway_url: Option<String>,
    #[arg(long, hide = true)]
    pub(crate) generation_file: Option<PathBuf>,
    #[arg(long, hide = true)]
    pub(crate) generation_token: Option<String>,
    #[arg(long, conflicts_with_all = ["generation_file", "generation_token"])]
    pub(crate) forward_only: bool,
    #[arg(
        long,
        hide = true,
        conflicts_with_all = ["generation_file", "generation_token", "forward_only"]
    )]
    pub(crate) transparent_run: bool,
    #[arg(long)]
    pub(crate) profile: Option<String>,
    #[arg(long)]
    pub(crate) session_metadata: Option<String>,
    #[arg(long, value_enum)]
    pub(crate) gateway_mode: Option<GatewayModeArg>,
    #[arg(long)]
    pub(crate) fail_closed: bool,
}

impl HookForwardCommand {
    fn into_runtime(self) -> crate::hooks::HookForwardRequest {
        crate::hooks::HookForwardRequest {
            agent: self.agent.into(),
            gateway_url: self.gateway_url,
            generation_file: self.generation_file,
            generation_token: self.generation_token,
            forward_only: self.forward_only,
            transparent_run: self.transparent_run,
            profile: self.profile,
            session_metadata: self.session_metadata,
            gateway_mode: self.gateway_mode.map(Into::into),
            fail_closed: self.fail_closed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum GatewayModeArg {
    HookOnly,
    Passthrough,
    Required,
}

impl From<GatewayModeArg> for crate::hooks::GatewayMode {
    fn from(value: GatewayModeArg) -> Self {
        match value {
            GatewayModeArg::HookOnly => Self::HookOnly,
            GatewayModeArg::Passthrough => Self::Passthrough,
            GatewayModeArg::Required => Self::Required,
        }
    }
}

pub(super) async fn execute(command: HookForwardCommand) -> Result<(), CliError> {
    crate::hooks::hook_forward(command.into_runtime()).await
}
