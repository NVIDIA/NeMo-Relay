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
    let inherited = server.to_runtime();
    crate::process::launcher::run(command.into_runtime(), Some(&inherited)).await
}

pub(super) async fn easy_path(
    agent: CodingAgent,
    command: EasyPathCommand,
    server: &ServerArgs,
) -> Result<ExitCode, CliError> {
    let inherited = server.to_runtime();
    // An explicit config path is the user's contract. Without one, setup is required only when
    // none of the normal discovery layers exists. Keep this interactive decision in the command
    // layer so process supervision receives a complete, agent-neutral run request.
    let explicit_config = inherited.config.as_deref();
    let needs_setup = explicit_config.is_none() && !crate::configuration::any_config_file_exists();
    if needs_setup {
        super::configure::run(Some(agent)).await?;
    }
    let runtime = crate::process::RunOverrides {
        agent: Some(agent),
        config: explicit_config.map(PathBuf::from),
        openai_base_url: None,
        anthropic_base_url: None,
        session_metadata: None,
        plugin_config_path: None,
        dry_run: false,
        print: false,
        command: command.command,
    };
    crate::process::launcher::run(runtime, Some(&inherited)).await
}
