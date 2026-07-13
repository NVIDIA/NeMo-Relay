// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, ValueEnum};

use crate::agents::CodingAgent;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct InstallCommand {
    #[arg(value_enum)]
    pub(crate) host: InstallTarget,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) force: bool,
    #[arg(long)]
    pub(crate) dry_run: bool,
    #[arg(long)]
    pub(crate) skip_doctor: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct UninstallCommand {
    #[arg(value_enum)]
    pub(crate) host: InstallTarget,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum InstallTarget {
    Codex,
    #[value(name = "claude-code", alias = "claude")]
    ClaudeCode,
    Hermes,
    All,
}

impl InstallTarget {
    pub(crate) fn agents(self) -> Vec<CodingAgent> {
        match self {
            Self::Codex => vec![CodingAgent::Codex],
            Self::ClaudeCode => vec![CodingAgent::ClaudeCode],
            Self::Hermes => vec![CodingAgent::Hermes],
            Self::All => vec![
                CodingAgent::Codex,
                CodingAgent::ClaudeCode,
                CodingAgent::Hermes,
            ],
        }
    }

    pub(crate) const fn is_all(self) -> bool {
        matches!(self, Self::All)
    }
}

impl InstallCommand {
    pub(crate) fn into_runtime(self) -> crate::installation::InstallRequest {
        crate::installation::InstallRequest {
            install_dir: self.install_dir,
            force: self.force,
            dry_run: self.dry_run,
            skip_doctor: self.skip_doctor,
        }
    }
}

impl UninstallCommand {
    pub(crate) fn into_runtime(self) -> crate::installation::UninstallRequest {
        crate::installation::UninstallRequest {
            install_dir: self.install_dir,
            dry_run: self.dry_run,
        }
    }
}

pub(super) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    let target = command.host;
    let request = command.into_runtime();
    let candidates = target.agents();
    let agents = if target.is_all() {
        crate::installation::marketplace::detected_install_agents(&candidates)?
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no supported Claude Code, Codex, or Hermes host CLI was detected".into(),
        ));
    }
    for agent in agents {
        crate::agents::install_integration(agent, request.clone())?;
    }
    Ok(ExitCode::SUCCESS)
}

pub(super) fn uninstall(command: UninstallCommand) -> Result<ExitCode, CliError> {
    let target = command.host;
    let request = command.into_runtime();
    let candidates = target.agents();
    let agents = if target.is_all() {
        crate::installation::marketplace::installed_agents(
            &candidates,
            request.install_dir.as_deref(),
        )?
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no installed Claude Code, Codex, or Hermes integration state was found".into(),
        ));
    }
    for agent in agents {
        crate::agents::uninstall_integration(agent, request.clone())?;
    }
    Ok(ExitCode::SUCCESS)
}
