// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::configuration::{PricingCommand, PricingSubcommand};
use crate::error::CliError;

pub(super) fn execute(command: PricingCommand) -> Result<ExitCode, CliError> {
    match command.command {
        PricingSubcommand::Validate(command) => crate::plugins::pricing::validate(command)?,
        PricingSubcommand::Init(command) => crate::plugins::pricing::init(command)?,
        PricingSubcommand::AddSource(command) => crate::plugins::pricing::add_source(command)?,
        PricingSubcommand::Resolve(command) => crate::plugins::pricing::resolve(command)?,
    }
    Ok(ExitCode::SUCCESS)
}
