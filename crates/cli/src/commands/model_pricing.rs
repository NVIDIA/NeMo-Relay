// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::config::{PricingCommand, PricingSubcommand};
use crate::error::CliError;

pub(super) fn execute(command: PricingCommand) -> Result<ExitCode, CliError> {
    match command.command {
        PricingSubcommand::Validate(command) => crate::model_pricing::validate(command)?,
        PricingSubcommand::Init(command) => crate::model_pricing::init(command)?,
        PricingSubcommand::AddSource(command) => crate::model_pricing::add_source(command)?,
        PricingSubcommand::Resolve(command) => crate::model_pricing::resolve(command)?,
    }
    Ok(ExitCode::SUCCESS)
}
