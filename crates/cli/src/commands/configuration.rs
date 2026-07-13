// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use super::arguments::ConfigCommand;
use crate::error::CliError;

pub(super) async fn execute(command: ConfigCommand) -> Result<ExitCode, CliError> {
    let agent = command.agent.map(Into::into);
    if command.reset {
        crate::configuration::wizard::reset(agent)?;
    } else {
        crate::configuration::wizard::run(agent).await?;
    }
    Ok(ExitCode::SUCCESS)
}
