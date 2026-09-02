// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use crate::agents::CodingAgent;
use crate::error::CliError;
use crate::installation::{InstallRequest, UninstallRequest};

use super::history;

pub(crate) fn install(command: InstallRequest) -> Result<ExitCode, CliError> {
    let migrate_history = command.migrate_history;
    let dry_run = command.dry_run;
    let history_database = command.history_database.clone();
    let status = crate::installation::marketplace::install(CodingAgent::Codex, command)?;
    if status != ExitCode::SUCCESS || !migrate_history {
        return Ok(status);
    }
    // The provider must exist in config.toml before threads are pointed at it,
    // so migrate only after the install itself has succeeded.
    history::migrate_to_relay(dry_run, history_database.as_deref()).map_err(CliError::Install)?;
    Ok(status)
}

pub(crate) fn uninstall(command: UninstallRequest) -> Result<ExitCode, CliError> {
    let skip_history_migration = command.skip_history_migration;
    let dry_run = command.dry_run;
    let history_database = command.history_database.clone();
    let status = crate::installation::marketplace::uninstall(CodingAgent::Codex, command)?;
    if status != ExitCode::SUCCESS || skip_history_migration {
        return Ok(status);
    }
    // Reversal is inferred from the migration journal rather than a flag, so a
    // user who migrated at install time does not have to remember to ask for it
    // again here.
    history::restore_from_relay(dry_run, history_database.as_deref()).map_err(CliError::Install)?;
    Ok(status)
}
