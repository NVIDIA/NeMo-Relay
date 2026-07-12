// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::config::ServerArgs;
use crate::error::CliError;

pub(super) async fn execute(server: &ServerArgs) -> Result<ExitCode, CliError> {
    crate::mcp::run(server).await
}
