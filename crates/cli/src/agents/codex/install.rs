// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::installation::{InstallRequest, UninstallRequest};

pub(crate) fn install(command: InstallRequest) -> Result<ExitCode, CliError> {
    crate::installation::marketplace::install(CodingAgent::Codex, command)
}

pub(crate) fn uninstall(command: UninstallRequest) -> Result<ExitCode, CliError> {
    crate::installation::marketplace::uninstall(CodingAgent::Codex, command)
}
