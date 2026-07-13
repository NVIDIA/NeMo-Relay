// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, ValueEnum};

use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct InstallCommand {
    #[arg(value_enum)]
    pub(crate) host: IntegrationHost,
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
    pub(crate) host: IntegrationHost,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum IntegrationHost {
    Codex,
    #[value(name = "claude-code", alias = "claude")]
    ClaudeCode,
    Hermes,
    All,
}

impl From<IntegrationHost> for crate::installation::IntegrationHost {
    fn from(value: IntegrationHost) -> Self {
        match value {
            IntegrationHost::Codex => Self::Codex,
            IntegrationHost::ClaudeCode => Self::ClaudeCode,
            IntegrationHost::Hermes => Self::Hermes,
            IntegrationHost::All => Self::All,
        }
    }
}

impl InstallCommand {
    pub(crate) fn into_runtime(self) -> crate::installation::InstallRequest {
        crate::installation::InstallRequest {
            host: self.host.into(),
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
            host: self.host.into(),
            install_dir: self.install_dir,
            dry_run: self.dry_run,
        }
    }
}

pub(super) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    crate::agents::install::install(command.into_runtime())
}

pub(super) fn uninstall(command: UninstallCommand) -> Result<ExitCode, CliError> {
    crate::agents::install::uninstall(command.into_runtime())
}
