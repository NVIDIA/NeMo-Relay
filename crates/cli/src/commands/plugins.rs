// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use super::arguments::{PluginsCommand, PluginsSubcommand};
use super::serve::ServerArgs;
use crate::error::CliError;

pub(super) fn execute(command: PluginsCommand, server: &ServerArgs) -> Result<ExitCode, CliError> {
    let server = server.to_runtime();
    let json_context = command
        .command
        .json_context()
        .map(|context| (context.command, context.target.map(str::to_owned)));
    let json = json_context.is_some();
    let result = match command.command {
        PluginsSubcommand::Edit(command) => crate::plugins::edit(command.into_runtime()),
        PluginsSubcommand::Add(command) => {
            crate::plugins::lifecycle::add(command.into_runtime(), &server)
        }
        PluginsSubcommand::Validate(command) => {
            crate::plugins::lifecycle::validate(command.into_runtime(), &server)
        }
        PluginsSubcommand::List(command) => {
            crate::plugins::lifecycle::list(command.into_runtime(), &server)
        }
        PluginsSubcommand::Inspect(command) => {
            crate::plugins::lifecycle::inspect(command.into_runtime(), &server)
        }
        PluginsSubcommand::Enable(command) => {
            crate::plugins::lifecycle::enable(command.into_runtime(), &server)
        }
        PluginsSubcommand::Disable(command) => {
            crate::plugins::lifecycle::disable(command.into_runtime(), &server)
        }
        PluginsSubcommand::Remove(command) => {
            crate::plugins::lifecycle::remove(command.into_runtime(), &server)
        }
    };
    match result {
        Ok(()) => Ok(ExitCode::SUCCESS),
        Err(error) => {
            if let Some(exit_code) = crate::plugins::lifecycle::render_plugin_error(&error, json)? {
                Ok(exit_code)
            } else if json {
                let (command, target) = json_context
                    .as_ref()
                    .expect("json plugin command context should exist when enabled");
                crate::plugins::lifecycle::render_generic_plugin_json_error(
                    command,
                    target.as_deref(),
                    &error.to_string(),
                )
            } else {
                Err(error)
            }
        }
    }
}
