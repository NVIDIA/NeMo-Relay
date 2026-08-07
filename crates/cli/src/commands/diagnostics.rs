// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use serde_json::{Value, json};

use super::install::InstallTarget;
use super::root::AgentArg;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub(crate) struct DoctorCommand {
    #[command(subcommand)]
    pub(crate) command: Option<DoctorSubcommand>,
    #[arg(value_enum, conflicts_with = "plugin")]
    pub(crate) agent: Option<AgentArg>,
    #[arg(long, value_enum)]
    pub(crate) plugin: Option<InstallTarget>,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) json: bool,
    #[arg(
        long,
        help = "Validate configuration and endpoint syntax without running live network probes"
    )]
    pub(crate) offline: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum DoctorSubcommand {
    /// Inspect an agent invocation without launching it.
    Invocation(InvocationDoctorCommand),
}

#[derive(Debug, Clone, Args)]
pub(crate) struct InvocationDoctorCommand {
    #[arg(long, value_enum)]
    agent: AgentArg,
    #[arg(long)]
    shortcut: bool,
    #[arg(
        long,
        help = "Display the complete invocation; arguments may contain sensitive data"
    )]
    show_full_command: bool,
    #[arg(last = true, required = true)]
    command: Vec<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct AgentsCommand {
    #[arg(long)]
    pub(crate) json: bool,
}

pub(super) async fn execute(
    command: DoctorCommand,
    server: &super::serve::ServerArgs,
    logging_fallback_error: Option<&CliError>,
) -> Result<ExitCode, CliError> {
    if let Some(DoctorSubcommand::Invocation(invocation)) = command.command {
        return execute_invocation_doctor(invocation);
    }
    if let Some(plugin) = command.plugin {
        return execute_plugin_doctor(plugin, command.install_dir, command.json);
    }
    let gateway_overrides = server.to_runtime();
    crate::diagnostics::run_doctor(
        command.agent.map(Into::into),
        command.json,
        crate::diagnostics::DoctorProbeMode::from_offline_flag(command.offline),
        &gateway_overrides,
        logging_fallback_error,
    )
    .await
}

fn execute_invocation_doctor(command: InvocationDoctorCommand) -> Result<ExitCode, CliError> {
    let agent = command.agent.into();
    let form = if command.shortcut {
        crate::diagnostics::invocation::InvocationForm::Shortcut
    } else {
        crate::diagnostics::invocation::InvocationForm::Run
    };
    match crate::diagnostics::invocation::DuplicateAgentExecutable::detect(
        agent,
        &command.command,
        form,
    ) {
        Some(diagnostic) => {
            println!("{}", diagnostic.format_doctor(command.show_full_command));
        }
        None => {
            println!(
                "INVOCATION DIAGNOSTIC\ncode = none\nselected_agent = {}\nresult = no duplicate agent executable detected",
                agent.as_arg()
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn execute_plugin_doctor(
    plugin: InstallTarget,
    install_dir: Option<PathBuf>,
    json: bool,
) -> Result<ExitCode, CliError> {
    let candidates = plugin.agents();
    let agents = if plugin.is_all() {
        crate::agents::installed_integrations(&candidates, install_dir.as_deref())
    } else {
        candidates
    };
    if agents.is_empty() {
        return Err(CliError::Install(
            "no installed Claude Code, Codex, or Hermes integration state was found".into(),
        ));
    }
    let options = crate::installation::marketplace::plugin_doctor_options(install_dir);
    if !json {
        for agent in agents {
            crate::agents::doctor_integration(agent, &options)?;
        }
        return Ok(ExitCode::SUCCESS);
    }
    print_plugin_doctor_json(&agents, &options)
}

fn print_plugin_doctor_json(
    agents: &[crate::agents::CodingAgent],
    options: &crate::installation::marketplace::state::PluginInstallOptions,
) -> Result<ExitCode, CliError> {
    let reports = agents
        .iter()
        .copied()
        .map(|agent| crate::agents::doctor_integration_report(agent, options))
        .collect::<Result<Vec<_>, _>>()?;
    let ready = reports
        .iter()
        .all(|report| report.get("ok").and_then(Value::as_bool) == Some(true));
    let output = if reports.len() > 1 {
        json!({ "schema_version": 1, "plugins": reports })
    } else {
        with_schema(reports.into_iter().next().expect("reports is not empty"))
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| CliError::Install(error.to_string()))?
    );
    Ok(if ready {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn with_schema(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.insert("schema_version".into(), json!(1));
    }
    value
}
