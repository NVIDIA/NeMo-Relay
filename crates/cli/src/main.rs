// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Relay coding-agent gateway CLI.

mod adapters;
mod alignment;
mod banner;
mod completions_install;
mod config;
mod doctor;
mod error;
mod gateway;
mod installer;
mod launcher;
mod model;
mod plugins;
mod pricing;
mod server;
mod session;
mod setup;
mod tls;

use std::process::ExitCode;

use clap::Parser;

use crate::config::{Cli, CodingAgent, Command, PluginsSubcommand, PricingSubcommand};

#[tokio::main]
// Runs the async CLI entrypoint and converts any surfaced gateway error into a non-zero process
// exit. Errors are printed once here so subcommands can return structured errors without also
// owning process-level reporting.
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(error) => {
            let exit_code = if error.guardrail_rejection_reason().is_some() {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            };
            eprintln!("{error}");
            exit_code
        }
    }
}

// Dispatches CLI subcommands while keeping the no-subcommand path as server mode. `run` inherits
// top-level server flags so transparent launch can share config parsing with daemon startup.
async fn run() -> Result<ExitCode, error::CliError> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::HookForward(command)) => {
            installer::hook_forward(command).await?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Run(command)) => launcher::run(command, Some(&cli.server)).await,
        Some(Command::Claude(command)) => {
            launcher::easy_path(CodingAgent::ClaudeCode, command, Some(&cli.server)).await
        }
        Some(Command::Codex(command)) => {
            launcher::easy_path(CodingAgent::Codex, command, Some(&cli.server)).await
        }
        Some(Command::Cursor(command)) => {
            launcher::easy_path(CodingAgent::Cursor, command, Some(&cli.server)).await
        }
        Some(Command::Hermes(command)) => {
            launcher::easy_path(CodingAgent::Hermes, command, Some(&cli.server)).await
        }
        Some(Command::Config(command)) => {
            if command.reset {
                setup::reset(command.agent)?;
            } else {
                setup::run(command.agent).await?;
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Plugins(command)) => {
            match command.command {
                PluginsSubcommand::Edit(command) => plugins::edit(command)?,
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Pricing(command)) => {
            match command.command {
                PricingSubcommand::Validate(command) => pricing::validate(command)?,
                PricingSubcommand::Init(command) => pricing::init(command)?,
                PricingSubcommand::AddSource(command) => pricing::add_source(command)?,
                PricingSubcommand::Resolve(command) => pricing::resolve(command)?,
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Doctor(command)) => doctor::run_doctor(command.agent, command.json).await,
        Some(Command::Agents(command)) => doctor::run_agents(command.json).await,
        Some(Command::Completions(command)) => {
            if command.install {
                let path = completions_install::install(command.shell)?;
                println!("✓ Installed completions: {}", path.display());
            } else {
                let shell = command.shell.ok_or_else(|| {
                    error::CliError::Config(
                        "missing shell argument; pass a shell name (bash, zsh, fish, ...) or \
                         use `--install` to auto-detect from $SHELL"
                            .into(),
                    )
                })?;
                let mut clap_command = <Cli as clap::CommandFactory>::command();
                clap_complete::generate(
                    shell,
                    &mut clap_command,
                    "nemo-relay",
                    &mut std::io::stdout(),
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        None => {
            // Bare `nemo-relay` with no subcommand:
            // - If the user passed any daemon-specific flag (`--bind`, upstream URLs, ATIF dir,
            //   OpenInference endpoint), they obviously want the long-running gateway daemon —
            //   keep that path so existing scripts that explicitly invoke daemon mode stay
            //   compatible.
            // - Otherwise — no flags, no subcommand — use the first-run path only when no config
            //   exists. Once configured, bare `nemo-relay` becomes a quick health check; explicit
            //   `nemo-relay config` remains the reconfiguration path.
            if cli.server.requested_daemon_mode() {
                let config = config::resolve_server_config(&cli.server)?;
                server::serve(config.gateway).await?;
                Ok(ExitCode::SUCCESS)
            } else if config::any_config_file_exists() {
                doctor::run_doctor(None, false).await
            } else {
                setup::run(None).await?;
                Ok(ExitCode::SUCCESS)
            }
        }
    }
}

#[cfg(test)]
mod test_support {
    pub(crate) static PLUGIN_CONFIG_TEST_LOCK: tokio::sync::Mutex<()> =
        tokio::sync::Mutex::const_new(());
}
