// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
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
    Pi,
    All,
}

impl InstallTarget {
    pub(crate) fn agents(self) -> Vec<CodingAgent> {
        match self {
            Self::Codex => vec![CodingAgent::Codex],
            Self::ClaudeCode => vec![CodingAgent::ClaudeCode],
            Self::Pi => vec![CodingAgent::Pi],
            Self::All => vec![CodingAgent::Codex, CodingAgent::ClaudeCode, CodingAgent::Pi],
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

/// Drop pi from an `all` run that carries `--install-dir`.
///
/// `--install-dir` names the marketplace root, which pi does not have -- pi rejects the
/// flag rather than silently ignoring it, and inside `all` that rejection failed the whole
/// command. Codex and Claude Code would install correctly and `nemo-relay install all
/// --install-dir X` still exited 1, which is a regression for users who never asked about
/// pi. An explicit `install pi --install-dir` still gets the error, because there the flag
/// is the user's stated intent rather than a default carried along.
fn marketplace_scoped(target: InstallTarget, install_dir: Option<&Path>) -> Vec<CodingAgent> {
    let candidates = target.agents();
    if !target.is_all() || install_dir.is_none() {
        return candidates;
    }
    candidates
        .into_iter()
        .filter(|agent| !matches!(agent, CodingAgent::Pi))
        .collect()
}

pub(super) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    let target = command.host;
    let request = command.into_runtime();
    let candidates = marketplace_scoped(target, request.install_dir.as_deref());
    let agents = if target.is_all() {
        crate::agents::detected_install_integrations(&candidates)
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no supported Claude Code, Codex, or pi host CLI was detected".into(),
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
    let request = command.into_runtime();
    let candidates = marketplace_scoped(target, request.install_dir.as_deref());
    let agents = if target.is_all() {
        crate::agents::uninstallable_integrations(&candidates, request.install_dir.as_deref())
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no installed Claude Code, Codex, or pi integration state was found".into(),
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
