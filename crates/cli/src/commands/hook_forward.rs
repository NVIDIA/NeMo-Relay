// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::HookForwardCommand;
use crate::error::CliError;

pub(super) async fn execute(command: HookForwardCommand) -> Result<(), CliError> {
    crate::installer::hook_forward(command).await
}
