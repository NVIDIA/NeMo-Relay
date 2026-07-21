// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command parsing, dispatch, rendering, and exit-code ownership.

mod completions;
mod configure;
mod diagnostics;
mod hook_forward;
mod install;
mod logging;
mod mcp;
mod model_pricing;
mod plugins;
pub(crate) mod root;
mod run;
mod serve;

use std::process::ExitCode;

use clap::Parser;

#[cfg(test)]
use self::completions::CompletionsCommand;
#[cfg(test)]
use self::model_pricing::PricingCommand;
#[cfg(test)]
use self::plugins::PluginsCommand;
use self::root::{Cli, Command};
use self::serve::ServerArgs;
use crate::agents::CodingAgent;
use crate::{
    configuration as runtime_configuration, diagnostics as runtime_diagnostics, error, server,
};

// Runs the async CLI entrypoint and converts any surfaced gateway error into a non-zero process
// exit. Errors are printed once here so subcommands can return structured errors without also
// owning process-level reporting.
pub(crate) async fn run(bootstrap_shutdown_token: Option<String>) -> ExitCode {
    match dispatch(bootstrap_shutdown_token).await {
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
async fn dispatch(bootstrap_shutdown_token: Option<String>) -> Result<ExitCode, error::CliError> {
    let cli = Cli::parse();
    let command_name = cli
        .command
        .as_ref()
        .map(Command::log_name)
        .unwrap_or("default");

    let initialize_logging = match cli.command.as_ref() {
        Some(command) => !command.skips_logging(),
        None => {
            cli.server.to_runtime().requested_daemon_mode()
                || runtime_configuration::any_config_file_exists()
        }
    };
    let _logging = if initialize_logging {
        let user_only = matches!(cli.command.as_ref(), Some(Command::Mcp));
        let explicit_config = if user_only {
            None
        } else {
            match cli.command.as_ref() {
                Some(Command::Run(command)) => {
                    command.config.as_deref().or(cli.server.config.as_deref())
                }
                _ => cli.server.config.as_deref(),
            }
        };
        let config = cli.logging.resolve(explicit_config, user_only)?;
        let runtime = nemo_relay::logging::LoggingRuntime::configure(config)?;
        Some(runtime)
    } else {
        None
    };

    log::info!(
        target: "nemo_relay.cli",
        event = "command_started",
        command = command_name;
        "CLI command started"
    );

    let result = match cli.command {
        Some(command) => run_command(command, &cli.server).await,
        None => run_default(&cli.server, bootstrap_shutdown_token).await,
    };
    match &result {
        Ok(code) if *code == ExitCode::SUCCESS => log::info!(
            target: "nemo_relay.cli",
            event = "command_completed",
            command = command_name,
            outcome = "success";
            "CLI command completed"
        ),
        Ok(_) => log::warn!(
            target: "nemo_relay.cli",
            event = "command_completed",
            command = command_name,
            outcome = "nonzero_exit";
            "CLI command completed with a non-zero exit status"
        ),
        Err(error) if error.guardrail_rejection_reason().is_some() => log::warn!(
            target: "nemo_relay.cli",
            event = "command_rejected",
            command = command_name,
            error_kind = error.log_kind();
            "CLI command was rejected by policy"
        ),
        Err(error) => log::error!(
            target: "nemo_relay.cli",
            event = "command_failed",
            command = command_name,
            error_kind = error.log_kind();
            "CLI command failed"
        ),
    }
    result
}

async fn run_command(command: Command, server: &ServerArgs) -> Result<ExitCode, error::CliError> {
    match command {
        Command::HookForward(command) => {
            hook_forward::execute(command).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Install(command) => install::install(command),
        Command::Uninstall(command) => install::uninstall(command),
        Command::Run(command) => run::execute(command, server).await,
        Command::Claude(command) => run::easy_path(CodingAgent::ClaudeCode, command, server).await,
        Command::Codex(command) => run::easy_path(CodingAgent::Codex, command, server).await,
        Command::Hermes(command) => run::easy_path(CodingAgent::Hermes, command, server).await,
        Command::Mcp => mcp::execute(server).await,
        Command::Config(command) => configure::execute(command).await,
        Command::Plugins(command) => plugins::execute(command, server),
        Command::ModelPricing(command) => model_pricing::execute(command),
        Command::Doctor(command) => diagnostics::execute(command).await,
        Command::Agents(command) => runtime_diagnostics::run_agents(command.json).await,
        Command::Completions(command) => completions::execute(command),
    }
}

#[cfg(test)]
fn generate_completions_to(
    shell: Option<clap_complete::Shell>,
    writer: &mut dyn std::io::Write,
) -> Result<(), error::CliError> {
    completions::generate_to(shell, writer)
}

async fn run_default(
    server_args: &ServerArgs,
    bootstrap_shutdown_token: Option<String>,
) -> Result<ExitCode, error::CliError> {
    let runtime_args = server_args.to_runtime();
    // Bare `nemo-relay` with no subcommand:
    // - If the user passed any daemon-specific flag (`--bind`, upstream URLs, ATIF dir,
    //   OpenInference endpoint), they obviously want the long-running gateway daemon —
    //   keep that path so existing scripts that explicitly invoke daemon mode stay
    //   compatible.
    // - Otherwise — no flags, no subcommand — use the first-run path only when no config
    //   exists. Once configured, bare `nemo-relay` becomes a quick health check; explicit
    //   `nemo-relay config` remains the reconfiguration path.
    if runtime_args.requested_daemon_mode() {
        let resolved = runtime_configuration::resolve_server_config(&runtime_args)?;
        let dynamic_plugins = crate::plugins::lifecycle::active_dynamic_plugin_components(
            runtime_args.config.as_ref(),
            &resolved,
        )?;
        let managed_bootstrap = runtime_configuration::managed_bootstrap_identity(
            &runtime_args,
            &resolved,
            &dynamic_plugins,
        )?;
        server::serve_with_dynamic(
            resolved.gateway,
            dynamic_plugins,
            managed_bootstrap,
            runtime_args.ready_file.as_deref(),
            bootstrap_shutdown_token,
        )
        .await?;
        Ok(ExitCode::SUCCESS)
    } else if runtime_configuration::any_config_file_exists() {
        runtime_diagnostics::run_doctor(None, false).await
    } else {
        configure::run(None).await?;
        Ok(ExitCode::SUCCESS)
    }
}

#[cfg(test)]
fn run_completions(command: CompletionsCommand) -> Result<ExitCode, error::CliError> {
    completions::execute(command)
}

#[cfg(test)]
fn run_plugins(command: PluginsCommand, server: &ServerArgs) -> Result<ExitCode, error::CliError> {
    plugins::execute(command, server)
}

#[cfg(test)]
fn run_pricing(command: PricingCommand) -> Result<ExitCode, error::CliError> {
    model_pricing::execute(command)
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/main_tests.rs"]
mod tests;
