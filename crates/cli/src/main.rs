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
mod plugin_install;
mod plugin_shim;
mod plugins;
mod pricing;
mod server;
mod session;
mod setup;
mod tls;

use std::process::ExitCode;

use clap::Parser;

use crate::config::{
    Cli, CodingAgent, Command, CompletionsCommand, ConfigCommand, DoctorCommand, PluginsCommand,
    PluginsSubcommand, PricingCommand, PricingSubcommand, ServerArgs,
};

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
        Some(command) => run_command(command, &cli.server).await,
        None => run_default(&cli.server).await,
    }
}

async fn run_command(command: Command, server: &ServerArgs) -> Result<ExitCode, error::CliError> {
    match command {
        Command::HookForward(command) => {
            installer::hook_forward(command).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::PluginShim(command) => plugin_shim::run(command),
        Command::Install(command) => plugin_install::install(command),
        Command::Uninstall(command) => plugin_install::uninstall(command),
        Command::Run(command) => launcher::run(command, Some(server)).await,
        Command::Claude(command) => {
            launcher::easy_path(CodingAgent::ClaudeCode, command, Some(server)).await
        }
        Command::Codex(command) => {
            launcher::easy_path(CodingAgent::Codex, command, Some(server)).await
        }
        Command::Cursor(command) => {
            launcher::easy_path(CodingAgent::Cursor, command, Some(server)).await
        }
        Command::Hermes(command) => {
            launcher::easy_path(CodingAgent::Hermes, command, Some(server)).await
        }
        Command::Config(command) => run_config(command).await,
        Command::Plugins(command) => run_plugins(command, server),
        Command::Pricing(command) => run_pricing(command),
        Command::Doctor(command) => run_doctor(command).await,
        Command::Agents(command) => doctor::run_agents(command.json).await,
        Command::Completions(command) => run_completions(command),
    }
}

async fn run_config(command: ConfigCommand) -> Result<ExitCode, error::CliError> {
    if command.reset {
        setup::reset(command.agent)?;
    } else {
        setup::run(command.agent).await?;
    }
    Ok(ExitCode::SUCCESS)
}

fn run_plugins(command: PluginsCommand, server: &ServerArgs) -> Result<ExitCode, error::CliError> {
    match command.command {
        PluginsSubcommand::Edit(command) => plugins::edit(command)?,
        PluginsSubcommand::Add(command) => plugins::lifecycle::add(command, server)?,
        PluginsSubcommand::Validate(command) => plugins::lifecycle::validate(command, server)?,
        PluginsSubcommand::List(command) => plugins::lifecycle::list(command, server)?,
        PluginsSubcommand::Inspect(command) => plugins::lifecycle::inspect(command, server)?,
        PluginsSubcommand::Enable(command) => plugins::lifecycle::enable(command, server)?,
        PluginsSubcommand::Disable(command) => plugins::lifecycle::disable(command, server)?,
        PluginsSubcommand::Remove(command) => plugins::lifecycle::remove(command, server)?,
    }
    Ok(ExitCode::SUCCESS)
}

fn run_pricing(command: PricingCommand) -> Result<ExitCode, error::CliError> {
    match command.command {
        PricingSubcommand::Validate(command) => pricing::validate(command)?,
        PricingSubcommand::Init(command) => pricing::init(command)?,
        PricingSubcommand::AddSource(command) => pricing::add_source(command)?,
        PricingSubcommand::Resolve(command) => pricing::resolve(command)?,
    }
    Ok(ExitCode::SUCCESS)
}

async fn run_doctor(command: DoctorCommand) -> Result<ExitCode, error::CliError> {
    if let Some(plugin) = command.plugin {
        plugin_install::doctor(plugin, command.install_dir, command.json)
    } else {
        doctor::run_doctor(command.agent, command.json).await
    }
}

fn run_completions(command: CompletionsCommand) -> Result<ExitCode, error::CliError> {
    if command.install {
        let path = completions_install::install(command.shell)?;
        println!("✓ Installed completions: {}", path.display());
    } else {
        generate_completions(command.shell)?;
    }
    Ok(ExitCode::SUCCESS)
}

fn generate_completions(shell: Option<clap_complete::Shell>) -> Result<(), error::CliError> {
    generate_completions_to(shell, &mut std::io::stdout())
}

fn generate_completions_to(
    shell: Option<clap_complete::Shell>,
    writer: &mut dyn std::io::Write,
) -> Result<(), error::CliError> {
    let shell = shell.ok_or_else(|| {
        error::CliError::Config(
            "missing shell argument; pass a shell name (bash, zsh, fish, ...) or \
             use `--install` to auto-detect from $SHELL"
                .into(),
        )
    })?;
    let mut clap_command = <Cli as clap::CommandFactory>::command();
    clap_complete::generate(shell, &mut clap_command, "nemo-relay", writer);
    Ok(())
}

async fn run_default(server_args: &ServerArgs) -> Result<ExitCode, error::CliError> {
    // Bare `nemo-relay` with no subcommand:
    // - If the user passed any daemon-specific flag (`--bind`, upstream URLs, ATIF dir,
    //   OpenInference endpoint), they obviously want the long-running gateway daemon —
    //   keep that path so existing scripts that explicitly invoke daemon mode stay
    //   compatible.
    // - Otherwise — no flags, no subcommand — use the first-run path only when no config
    //   exists. Once configured, bare `nemo-relay` becomes a quick health check; explicit
    //   `nemo-relay config` remains the reconfiguration path.
    if server_args.requested_daemon_mode() {
        let config = config::resolve_server_config(server_args)?;
        server::serve(config.gateway).await?;
        Ok(ExitCode::SUCCESS)
    } else if config::any_config_file_exists() {
        doctor::run_doctor(None, false).await
    } else {
        setup::run(None).await?;
        Ok(ExitCode::SUCCESS)
    }
}

#[cfg(test)]
mod test_support {
    pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    pub(crate) static PLUGIN_CONFIG_TEST_LOCK: tokio::sync::Mutex<()> =
        tokio::sync::Mutex::const_new(());
}

#[cfg(test)]
#[path = "../tests/coverage/main_tests.rs"]
mod tests;
