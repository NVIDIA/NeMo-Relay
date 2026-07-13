// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::configuration::ConfigCommand;
use crate::error::CliError;

pub(super) async fn execute(command: ConfigCommand) -> Result<ExitCode, CliError> {
    if command.reset {
        crate::setup::reset(command.agent)?;
    } else {
        crate::setup::run(command.agent).await?;
    }
    Ok(ExitCode::SUCCESS)
}
