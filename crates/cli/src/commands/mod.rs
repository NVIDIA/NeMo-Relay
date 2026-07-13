// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command parsing, dispatch, rendering, and exit-code ownership.

mod completions;
mod configuration;
mod diagnostics;
mod hook_forward;
mod install;
mod mcp;
mod model_pricing;
mod plugins;
mod run;

use std::process::ExitCode;

use clap::Parser;

use crate::configuration::{Cli, CodingAgent, Command, ServerArgs};
#[cfg(test)]
use crate::configuration::{CompletionsCommand, PluginsCommand, PricingCommand};
use crate::{
    configuration as runtime_configuration, diagnostics as runtime_diagnostics, error, server,
};

// Runs the async CLI entrypoint and converts any surfaced gateway error into a non-zero process
// exit. Errors are printed once here so subcommands can return structured errors without also
// owning process-level reporting.
pub(crate) async fn run() -> ExitCode {
    match dispatch().await {
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
async fn dispatch() -> Result<ExitCode, error::CliError> {
    let cli = Cli::parse();
    match cli.command {
        Some(command) => run_command(command, &cli.server).await,
        None => run_default(&cli.server).await,
    }
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
        Command::Config(command) => configuration::execute(command).await,
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
        let resolved = runtime_configuration::resolve_server_config(server_args)?;
        let dynamic_plugins = crate::plugins::lifecycle::active_dynamic_plugin_components(
            server_args.config.as_ref(),
            &resolved,
        )?;
        let managed_bootstrap = runtime_configuration::managed_bootstrap_identity(
            server_args,
            &resolved,
            &dynamic_plugins,
        )?;
        server::serve_with_dynamic(
            resolved.gateway,
            dynamic_plugins,
            managed_bootstrap,
            server_args.ready_file.as_deref(),
        )
        .await?;
        Ok(ExitCode::SUCCESS)
    } else if runtime_configuration::any_config_file_exists() {
        runtime_diagnostics::run_doctor(None, false).await
    } else {
        runtime_configuration::wizard::run(None).await?;
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
pub(crate) mod test_support {
    #[must_use]
    pub(crate) struct CwdTestScope {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Option<std::path::PathBuf>,
    }

    impl CwdTestScope {
        pub(crate) fn locked() -> Self {
            Self {
                _guard: lock_cwd(),
                prev: None,
            }
        }

        pub(crate) fn enter(path: &std::path::Path) -> Self {
            let guard = lock_cwd();
            let prev = std::env::current_dir().unwrap();
            std::env::set_current_dir(path).unwrap();
            Self {
                _guard: guard,
                prev: Some(prev),
            }
        }
    }

    impl Drop for CwdTestScope {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev
                && let Err(error) = std::env::set_current_dir(prev)
            {
                CWD_RESTORE_FAILED.store(true, std::sync::atomic::Ordering::SeqCst);
                if std::thread::panicking() {
                    eprintln!("failed to restore current_dir to {prev:?}: {error}");
                } else {
                    panic!("failed to restore current_dir to {prev:?}: {error}");
                }
            }
        }
    }

    pub(crate) static CWD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static CWD_RESTORE_FAILED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    pub(crate) static PLUGIN_CONFIG_TEST_LOCK: tokio::sync::Mutex<()> =
        tokio::sync::Mutex::const_new(());

    fn lock_cwd() -> std::sync::MutexGuard<'static, ()> {
        let guard = CWD_TEST_LOCK.lock().expect("CWD_TEST_LOCK poisoned");
        assert!(
            !CWD_RESTORE_FAILED.load(std::sync::atomic::Ordering::SeqCst),
            "current_dir restore failed in a previous test; aborting to prevent cross-test contamination",
        );
        guard
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/main_tests.rs"]
mod tests;
