// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::config::{PluginsCommand, PluginsSubcommand, ServerArgs};
use crate::error::CliError;

pub(super) fn execute(command: PluginsCommand, server: &ServerArgs) -> Result<ExitCode, CliError> {
    let json_context = command
        .command
        .json_context()
        .map(|context| (context.command, context.target.map(str::to_owned)));
    let json = json_context.is_some();
    let result = match command.command {
        PluginsSubcommand::Edit(command) => crate::plugins::edit(command),
        PluginsSubcommand::Add(command) => crate::plugins::lifecycle::add(command, server),
        PluginsSubcommand::Validate(command) => {
            crate::plugins::lifecycle::validate(command, server)
        }
        PluginsSubcommand::List(command) => crate::plugins::lifecycle::list(command, server),
        PluginsSubcommand::Inspect(command) => crate::plugins::lifecycle::inspect(command, server),
        PluginsSubcommand::Enable(command) => crate::plugins::lifecycle::enable(command, server),
        PluginsSubcommand::Disable(command) => crate::plugins::lifecycle::disable(command, server),
        PluginsSubcommand::Remove(command) => crate::plugins::lifecycle::remove(command, server),
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
