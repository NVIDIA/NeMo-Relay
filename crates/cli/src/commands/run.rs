// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;

use super::root::AgentArg;
use super::serve::ServerArgs;
use crate::agents::CodingAgent;
use crate::error::CliError;

const POSSIBLE_DUPLICATE_AGENT_EXECUTABLE: &str = "possible_duplicate_agent_executable";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InvocationForm {
    Run,
    Shortcut,
}

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
        warn_for_possible_duplicate(agent, &command.command, InvocationForm::Run);
    }
    let inherited = server.to_runtime();
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
        warn_for_possible_duplicate(agent, &command.command, InvocationForm::Shortcut);
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
    crate::process::launcher::run(runtime, Some(&inherited)).await
}

fn warn_for_possible_duplicate(agent: CodingAgent, command: &[String], form: InvocationForm) {
    let Some(warning) = possible_duplicate_agent_warning(agent, command, form) else {
        return;
    };
    let agent = agent.as_arg();
    log::warn!(
        target: "nemo_relay.cli",
        event = "agent_invocation_warning",
        diagnostic_code = POSSIBLE_DUPLICATE_AGENT_EXECUTABLE,
        agent = agent,
        duplicate_executable = agent,
        confidence = "high",
        action = "dry_run_warning",
        command_modified = false,
        arguments_redacted = true;
        "Possible duplicate agent executable during dry-run validation"
    );
    super::print_invocation_warning(&warning);
}

pub(super) fn possible_duplicate_agent_warning(
    agent: CodingAgent,
    command: &[String],
    form: InvocationForm,
) -> Option<String> {
    let executable = command.first()?;
    if CodingAgent::infer(executable) != Some(agent) {
        return None;
    }

    let mut observed = relay_prefix(agent, form);
    observed.extend(["--dry-run".into(), "--".into(), agent.as_arg().into()]);
    if command.len() > 1 {
        observed.push("<arguments redacted>".into());
    }

    let mut recommended = relay_prefix(agent, form);
    recommended.extend(["--dry-run".into(), "--".into()]);
    if command.len() > 1 {
        recommended.push("<arguments redacted>".into());
    }

    Some(format!(
        "WARNING: Possible duplicate agent executable after `--`.\n\
         Diagnostic: {POSSIBLE_DUPLICATE_AGENT_EXECUTABLE}\n\
         Duplicate executable: {}\n\
         Observed: {}\n\
         Recommended: {}\n\
         Dry-run validation will continue without launching the agent.",
        agent.as_arg(),
        render_command(&observed),
        render_command(&recommended),
    ))
}

fn relay_prefix(agent: CodingAgent, form: InvocationForm) -> Vec<String> {
    match form {
        InvocationForm::Run => vec![
            "nemo-relay".into(),
            "run".into(),
            "--agent".into(),
            agent.as_arg().into(),
        ],
        InvocationForm::Shortcut => vec!["nemo-relay".into(), agent.as_arg().into()],
    }
}

fn render_command(command: &[String]) -> String {
    command
        .iter()
        .map(|argument| crate::process::shell_quote_arg_for_platform(argument, cfg!(windows)))
        .collect::<Vec<_>>()
        .join(" ")
}
