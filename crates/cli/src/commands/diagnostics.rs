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
    #[arg(value_enum)]
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
        let candidates = plugin.agents();
        let agents = if plugin.is_all() {
            crate::agents::installed_integrations(&candidates, command.install_dir.as_deref())
        } else {
            candidates
        };
        if agents.is_empty() {
            return Err(CliError::Install(
                "no installed Claude Code, Codex, or Hermes integration state was found".into(),
            ));
        }
        let options = crate::installation::marketplace::plugin_doctor_options(command.install_dir);
        if command.json {
            let reports = agents
                .iter()
                .copied()
                .map(|agent| crate::agents::doctor_integration_report(agent, &options))
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
        } else {
            for agent in agents {
                crate::agents::doctor_integration(agent, &options)?;
            }
            Ok(ExitCode::SUCCESS)
        }
    } else {
        crate::diagnostics::run_doctor(command.agent.map(Into::into), command.json).await
    }
}

fn with_schema(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.insert("schema_version".into(), json!(1));
    }
    value
}
