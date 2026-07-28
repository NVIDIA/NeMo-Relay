// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use clap::{ArgGroup, Args, Subcommand};

use super::root::AgentArg;
use crate::error::CliError;

mod editor;
mod model;
mod wizard;

pub(super) use wizard::run;

#[derive(Debug, Clone, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub(crate) struct ConfigCommand {
    #[command(subcommand)]
    pub(crate) command: Option<ConfigSubcommand>,
    #[arg(value_enum)]
    pub(crate) agent: Option<AgentArg>,
    /// Reset Relay configuration for the selected scope. Persistent Hermes integration state is
    /// managed separately with `nemo-relay uninstall hermes`.
    #[arg(long)]
    pub(crate) reset: bool,
    /// Configuration scope to reset. Defaults to the project configuration.
    #[arg(long, value_enum, requires = "reset")]
    pub(crate) scope: Option<model::ConfigScope>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum ConfigSubcommand {
    /// Interactively edit gateway, upstream, and operational logging configuration.
    Edit(ConfigEditCommand),
}

#[derive(Debug, Clone, Default, Args)]
#[command(group(
    ArgGroup::new("scope")
        .args(["user", "project", "global"])
        .multiple(false)
))]
pub(crate) struct ConfigEditCommand {
    /// Edit the user config at `$XDG_CONFIG_HOME/nemo-relay/config.toml`.
    #[arg(long)]
    pub(crate) user: bool,
    /// Edit the nearest project config at `.nemo-relay/config.toml`.
    #[arg(long)]
    pub(crate) project: bool,
    /// Edit the system config at `/etc/nemo-relay/config.toml`.
    #[arg(long)]
    pub(crate) global: bool,
}

pub(super) async fn execute(command: ConfigCommand) -> Result<ExitCode, CliError> {
    if let Some(ConfigSubcommand::Edit(edit)) = command.command.as_ref() {
        editor::edit(edit.clone())?;
        return Ok(ExitCode::SUCCESS);
    }
    let agent = command.agent.map(Into::into);
    if command.reset {
        model::reset(command.scope.unwrap_or(model::ConfigScope::Project), agent)?;
    } else {
        wizard::run(agent).await?;
    }
    Ok(ExitCode::SUCCESS)
}
