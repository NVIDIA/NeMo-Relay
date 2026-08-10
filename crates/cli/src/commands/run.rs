// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;

use super::root::AgentArg;
use super::serve::ServerArgs;
use crate::agents::CodingAgent;
use crate::error::CliError;

/// Args for an easy-path agent shortcut.
#[derive(Debug, Clone, Args)]
pub(crate) struct EasyPathCommand {
    /// Print the resolved launch plan, including forwarded arguments, without executing it.
    #[arg(long)]
    pub(super) dry_run: bool,
    #[arg(last = true)]
    pub(super) command: Vec<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct RunCommand {
    #[arg(long, value_enum)]
    pub(super) agent: Option<AgentArg>,
    #[arg(long)]
    pub(super) config: Option<PathBuf>,
    #[arg(long)]
    pub(super) openai_base_url: Option<String>,
    #[arg(long)]
    pub(super) anthropic_base_url: Option<String>,
    #[arg(long)]
    pub(super) session_metadata: Option<String>,
    #[arg(long, env = "NEMO_RELAY_PLUGIN_CONFIG_PATH", hide = true)]
    pub(super) plugin_config_path: Option<PathBuf>,
    #[arg(long)]
    pub(super) dry_run: bool,
    #[arg(long)]
    pub(super) print: bool,
    #[arg(last = true)]
    pub(super) command: Vec<String>,
}

impl RunCommand {
    fn into_runtime(self) -> crate::process::RunOverrides {
        crate::process::RunOverrides {
            agent: self.agent.map(Into::into),
            config: self.config,
            openai_base_url: self.openai_base_url,
            anthropic_base_url: self.anthropic_base_url,
            session_metadata: self.session_metadata,
            plugin_config_path: self.plugin_config_path,
            dry_run: self.dry_run,
            print: self.print,
            command: self.command,
        }
    }
}

pub(super) async fn execute(
    command: RunCommand,
    server: &ServerArgs,
) -> Result<ExitCode, CliError> {
    if command.dry_run
        && let Some(agent) = command.agent.map(Into::into)
    {
        warn_for_possible_duplicate(agent, &command.command);
    }
    let inherited = server.to_runtime();
    // The launcher prints the plan and returns before gateway or child execution for dry runs.
    crate::process::launcher::run(command.into_runtime(), Some(&inherited)).await
}

/// Resolves the plugin document that easy-path setup must preserve.
pub(super) fn easy_path_plugin_config_path(
    inherited: &crate::server::GatewayOverrides,
) -> Option<PathBuf> {
    crate::configuration::explicit_plugin_config_path(
        inherited.config.as_ref(),
        inherited.plugin_config_path.as_ref(),
    )
}

pub(super) async fn easy_path(
    agent: CodingAgent,
    command: EasyPathCommand,
    server: &ServerArgs,
) -> Result<ExitCode, CliError> {
    if command.dry_run {
        warn_for_possible_duplicate(agent, &command.command);
    }
    let inherited = server.to_runtime();
    // An explicit config path is the user's contract. Without one, setup is required only when
    // none of the normal discovery layers exists. Keep this interactive decision in the command
    // layer so process supervision receives a complete, agent-neutral run request.
    let explicit_config = inherited.config.as_deref();
    let needs_setup = explicit_config.is_none() && !crate::configuration::any_config_file_exists();
    if needs_setup && !command.dry_run {
        let explicit_plugin_path = easy_path_plugin_config_path(&inherited);
        super::configure::run(Some(agent), explicit_plugin_path).await?;
    }
    let runtime = crate::process::RunOverrides {
        agent: Some(agent),
        config: explicit_config.map(PathBuf::from),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: command.dry_run,
        print: false,
        command: command.command,
    };
    // The launcher prints the plan and returns before gateway or child execution for dry runs.
    crate::process::launcher::run(runtime, Some(&inherited)).await
}

fn warn_for_possible_duplicate(agent: CodingAgent, command: &[String]) {
    if !command
        .first()
        .is_some_and(|executable| CodingAgent::infer(executable) == Some(agent))
    {
        return;
    }
    let agent = agent.as_arg();
    log::warn!(
        target: "nemo_relay.cli",
        event = "agent_invocation_warning",
        diagnostic_code = "possible_duplicate_agent_executable",
        agent = agent,
        duplicate_executable = agent,
        confidence = "high",
        action = "remove_duplicate_executable",
        command_modified = false,
        arguments_redacted = true;
        "Possible duplicate agent executable after `--`; remove the repeated executable"
    );
}
