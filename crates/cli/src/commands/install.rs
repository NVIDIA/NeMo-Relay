// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::config::{InstallCommand, UninstallCommand};
use crate::error::CliError;

pub(super) fn install(command: InstallCommand) -> Result<ExitCode, CliError> {
    crate::agents::install::install(command)
}

pub(super) fn uninstall(command: UninstallCommand) -> Result<ExitCode, CliError> {
    crate::agents::install::uninstall(command)
}
