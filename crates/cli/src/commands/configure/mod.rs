// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use clap::Args;

use super::root::AgentArg;
use crate::error::CliError;

mod model;
mod wizard;

pub(super) use wizard::run;

#[derive(Debug, Clone, Args)]
pub(crate) struct ConfigCommand {
    #[arg(value_enum)]
    pub(crate) agent: Option<AgentArg>,
    /// Reset Relay configuration for the selected scope. Persistent Hermes integration state is
    /// managed separately with `nemo-relay uninstall hermes`.
    #[arg(long)]
    pub(crate) reset: bool,
}

pub(super) async fn execute(command: ConfigCommand) -> Result<ExitCode, CliError> {
    let agent = command.agent.map(Into::into);
    if command.reset {
        model::reset(agent)?;
    } else {
        wizard::run(agent).await?;
    }
    Ok(ExitCode::SUCCESS)
}
