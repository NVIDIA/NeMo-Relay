// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use super::arguments::{PricingCommand, PricingSubcommand};
use crate::error::CliError;

pub(super) fn execute(command: PricingCommand) -> Result<ExitCode, CliError> {
    match command.command {
        PricingSubcommand::Validate(command) => {
            crate::plugins::pricing::validate(command.into_runtime())?
        }
        PricingSubcommand::Init(command) => crate::plugins::pricing::init(command.into_runtime())?,
        PricingSubcommand::AddSource(command) => {
            crate::plugins::pricing::add_source(command.into_runtime())?
        }
        PricingSubcommand::Resolve(command) => {
            crate::plugins::pricing::resolve(command.into_runtime())?
        }
    }
    Ok(ExitCode::SUCCESS)
}
