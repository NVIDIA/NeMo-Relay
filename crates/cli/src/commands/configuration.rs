// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use clap::Args;

use super::root::AgentArg;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct ConfigCommand {
    #[arg(value_enum)]
    pub(crate) agent: Option<AgentArg>,
    #[arg(long)]
    pub(crate) reset: bool,
}

pub(super) async fn execute(command: ConfigCommand) -> Result<ExitCode, CliError> {
    let agent = command.agent.map(Into::into);
    if command.reset {
        crate::configuration::wizard::reset(agent)?;
    } else {
        crate::configuration::wizard::run(agent).await?;
    }
    Ok(ExitCode::SUCCESS)
}
