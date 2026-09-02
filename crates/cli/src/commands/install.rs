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
    /// Experimental: move existing Codex thread history onto the Relay provider so it stays
    /// visible in the Codex resume picker. `nemo-relay uninstall codex` reverses this
    /// automatically.
    #[arg(long)]
    pub(crate) migrate_history: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct UninstallCommand {
    #[arg(value_enum)]
    pub(crate) host: InstallTarget,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    /// Attempt all Relay-owned cleanup steps even when normal uninstall safety checks fail.
    #[arg(long)]
    pub(crate) force: bool,
    #[arg(long)]
    pub(crate) dry_run: bool,
    /// Leave migrated Codex thread history on the Relay provider instead of restoring it.
    #[arg(long)]
    pub(crate) skip_history_migration: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum InstallTarget {
    Codex,
    #[value(name = "claude-code", alias = "claude")]
    ClaudeCode,
    All,
}

impl InstallTarget {
    pub(crate) fn agents(self) -> Vec<CodingAgent> {
        match self {
            Self::Codex => vec![CodingAgent::Codex],
            Self::ClaudeCode => vec![CodingAgent::ClaudeCode],
            Self::All => vec![CodingAgent::Codex, CodingAgent::ClaudeCode],
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
            migrate_history: self.migrate_history,
        }
    }
}

impl UninstallCommand {
    pub(crate) fn into_runtime(self) -> crate::installation::UninstallRequest {
        crate::installation::UninstallRequest {
            install_dir: self.install_dir,
            force: self.force,
            dry_run: self.dry_run,
            skip_history_migration: self.skip_history_migration,
        }
    }
}

pub(super) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    let target = command.host;
    if command.migrate_history && matches!(target, InstallTarget::ClaudeCode) {
        return Err(CliError::Install(
            "--migrate-history applies to the Codex integration only".into(),
        ));
    }
    let request = command.into_runtime();
    let candidates = target.agents();
    let agents = if target.is_all() {
        crate::agents::detected_install_integrations(&candidates)
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no supported Claude Code or Codex host CLI was detected".into(),
        ));
    }
    if !request.dry_run {
        crate::configuration::BootstrapChallengeKey::load()?;
    }
    run_agent_operations(agents, "install", |agent| {
        crate::agents::install_integration(agent, request.clone())
    })
}

pub(super) fn uninstall(command: UninstallCommand) -> Result<ExitCode, CliError> {
    let target = command.host;
    if command.skip_history_migration && matches!(target, InstallTarget::ClaudeCode) {
        return Err(CliError::Install(
            "--skip-history-migration applies to the Codex integration only".into(),
        ));
    }
    let request = command.into_runtime();
    let candidates = target.agents();
    let agents = if target.is_all() {
        crate::agents::installed_integrations(
            &candidates,
            request.install_dir.as_deref(),
            request.force,
        )
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no installed Claude Code or Codex integration state was found".into(),
        ));
    }
    run_agent_operations(agents, "uninstall", |agent| {
        crate::agents::uninstall_integration(agent, request.clone())
    })
}

pub(super) fn run_agent_operations(
    agents: Vec<CodingAgent>,
    operation: &str,
    mut run: impl FnMut(CodingAgent) -> Result<ExitCode, CliError>,
) -> Result<ExitCode, CliError> {
    let mut result = ExitCode::SUCCESS;
    let mut errors = Vec::new();
    for agent in agents {
        match run(agent) {
            Ok(status) if status != ExitCode::SUCCESS => result = status,
            Ok(_) => {}
            Err(error) => errors.push(format!("{}: {error}", agent.as_arg())),
        }
    }
    if errors.is_empty() {
        Ok(result)
    } else {
        Err(CliError::Install(format!(
            "failed to {operation} one or more integrations after attempting every target: {}",
            errors.join("; ")
        )))
    }
}
