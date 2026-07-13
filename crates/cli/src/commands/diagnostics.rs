// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::configuration::DoctorCommand;
use crate::error::CliError;

pub(super) async fn execute(command: DoctorCommand) -> Result<ExitCode, CliError> {
    if let Some(plugin) = command.plugin {
        crate::agents::install::doctor(plugin, command.install_dir, command.json)
    } else {
        crate::diagnostics::run_doctor(command.agent, command.json).await
    }
}
