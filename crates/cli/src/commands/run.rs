// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use super::arguments::{EasyPathCommand, RunCommand, ServerArgs};
use crate::agents::CodingAgent;
use crate::error::CliError;

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
    crate::process::launcher::easy_path(agent, command.command, Some(&inherited)).await
}
