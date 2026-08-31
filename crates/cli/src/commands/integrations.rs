// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Refreshes persistent coding-agent integrations after a Relay upgrade.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::installation::InstallRequest;

#[derive(Debug, Clone, Args)]
pub(crate) struct IntegrationsCommand {
    #[command(subcommand)]
    command: IntegrationsSubcommand,
}

#[derive(Debug, Clone, Subcommand)]
enum IntegrationsSubcommand {
    /// Replace Relay-managed Codex and Claude Code integrations with the current Relay binary.
    Refresh(RefreshCommand),
}

#[derive(Debug, Clone, Args)]
struct RefreshCommand {
    /// Refresh integrations in one install directory and record it for future automatic refreshes.
    #[arg(long)]
    install_dir: Option<PathBuf>,
    /// Show the refresh operations without changing integrations.
    #[arg(long)]
    dry_run: bool,
}

pub(crate) fn execute(command: IntegrationsCommand) -> Result<ExitCode, CliError> {
    match command.command {
        IntegrationsSubcommand::Refresh(command) => refresh(command),
    }
}

fn refresh(command: RefreshCommand) -> Result<ExitCode, CliError> {
    let targets = refresh_targets(command.install_dir.as_deref())?;
    let managed_targets = targets
        .iter()
        .filter(|(agent, install_dir)| {
            crate::installation::marketplace::persisted_state_exists(*agent, install_dir)
        })
        .cloned()
        .collect::<Vec<_>>();
    if !command.dry_run {
        crate::installation::marketplace::prepare_integrations_for_refresh(&managed_targets)?;
    }
    let mut attempted = 0;
    let mut errors = Vec::new();

    for (agent, install_dir) in targets {
        if !crate::installation::marketplace::persisted_state_exists(agent, &install_dir) {
            if command.install_dir.is_some()
                && crate::installation::marketplace::local_install_exists(agent, &install_dir)
            {
                println!(
                    "{} at {}: unmanaged/manual — not changed; reinstall with `nemo-relay install {} --force`",
                    agent.label(),
                    install_dir.display(),
                    agent.install_arg()
                );
            }
            continue;
        }

        attempted += 1;
        let request = InstallRequest {
            install_dir: Some(install_dir.clone()),
            force: true,
            dry_run: command.dry_run,
            skip_doctor: false,
        };
        match crate::agents::install_integration(agent, request) {
            Ok(status) if status == ExitCode::SUCCESS => println!(
                "{} at {}: {}",
                agent.label(),
                install_dir.display(),
                if command.dry_run {
                    "would refresh — no reconnect required"
                } else {
                    "refreshed — reconnect required"
                }
            ),
            Ok(_) => errors.push(format!(
                "{} at {} returned a nonzero status",
                agent.label(),
                install_dir.display()
            )),
            Err(error) => errors.push(format!(
                "{} at {}: {error}",
                agent.label(),
                install_dir.display()
            )),
        }
    }

    if attempted == 0 {
        println!("No Relay-managed Codex or Claude Code integrations found.");
    }
    if errors.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        Err(CliError::Install(format!(
            "failed to refresh one or more integrations after attempting every target: {}",
            errors.join("; ")
        )))
    }
}

fn refresh_targets(
    install_dir: Option<&std::path::Path>,
) -> Result<Vec<(CodingAgent, PathBuf)>, CliError> {
    let mut targets = Vec::new();
    if let Some(install_dir) = install_dir {
        let install_dir = install_dir
            .canonicalize()
            .unwrap_or_else(|_| install_dir.to_path_buf());
        for agent in CodingAgent::ALL {
            targets.push((agent, install_dir.clone()));
        }
    } else {
        let default = crate::installation::marketplace::default_marketplace_install_dir();
        for agent in CodingAgent::ALL {
            targets.push((agent, default.clone()));
            for install_dir in crate::installation::marketplace::registered_install_dirs(agent)
                .map_err(CliError::Install)?
            {
                if !targets
                    .iter()
                    .any(|(existing, directory)| *existing == agent && *directory == install_dir)
                {
                    targets.push((agent, install_dir));
                }
            }
        }
    }

    Ok(targets)
}
