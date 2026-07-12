// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::agents::CodingAgent;
use crate::config::{EasyPathCommand, RunCommand, ServerArgs};
use crate::error::CliError;

pub(super) async fn execute(
    command: RunCommand,
    server: &ServerArgs,
) -> Result<ExitCode, CliError> {
    crate::launcher::run(command, Some(server)).await
}

pub(super) async fn easy_path(
    agent: CodingAgent,
    command: EasyPathCommand,
    server: &ServerArgs,
) -> Result<ExitCode, CliError> {
    crate::launcher::easy_path(agent, command, Some(server)).await
}
