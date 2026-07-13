// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::arguments::HookForwardCommand;
use crate::error::CliError;

pub(super) async fn execute(command: HookForwardCommand) -> Result<(), CliError> {
    crate::hooks::hook_forward(command.into_runtime()).await
}
