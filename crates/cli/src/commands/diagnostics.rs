// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use serde_json::{Value, json};

use super::install::InstallTarget;
use super::root::AgentArg;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct DoctorCommand {
    #[arg(value_enum, conflicts_with = "plugin")]
    pub(crate) agent: Option<AgentArg>,
    #[arg(long, value_enum)]
    pub(crate) plugin: Option<InstallTarget>,
    #[arg(long)]
    pub(crate) install_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct AgentsCommand {
    #[arg(long)]
    pub(crate) json: bool,
}

pub(super) async fn execute(command: DoctorCommand) -> Result<ExitCode, CliError> {
    if let Some(plugin) = command.plugin {
        return execute_plugin_doctor(plugin, command.install_dir, command.json);
    }
    crate::diagnostics::run_doctor(command.agent.map(Into::into), command.json).await
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
