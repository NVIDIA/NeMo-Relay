// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;

use super::install::InstallTarget;
use super::root::AgentArg;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct DoctorCommand {
    #[arg(value_enum)]
    pub(crate) agent: Option<AgentArg>,
    #[arg(long, value_enum)]
    pub(crate) plugin: Option<InstallTarget>,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct AgentsCommand {
    #[arg(long)]
    pub(crate) json: bool,
}

pub(super) async fn execute(command: DoctorCommand) -> Result<ExitCode, CliError> {
    if let Some(plugin) = command.plugin {
        let candidates = plugin.agents();
        let agents = if plugin.is_all() {
            crate::agents::installed_integrations(&candidates, command.install_dir.as_deref())
        } else {
            candidates
        };
        crate::installation::marketplace::doctor(&agents, command.install_dir, command.json)
    } else {
        crate::diagnostics::run_doctor(command.agent.map(Into::into), command.json).await
    }
}
