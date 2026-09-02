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
    //
    // A failure here is reported as a nonzero status rather than an error: the
    // integration is installed and working, and calling it an install failure
    // would send the caller to the wrong remedy.
    match history::migrate_to_relay(dry_run, history_database.as_deref()) {
        Ok(_) => Ok(status),
        Err(error) => {
            log::error!(
                target: "nemo_relay.installation",
                event = "codex_history_migration_failed",
                host = "codex",
                error_kind = "history_migration";
                "Codex integration installed but thread-history migration failed"
            );
            println!("the Codex integration is installed, but history migration failed: {error}");
            println!(
                "retry the migration with `nemo-relay install codex --force --migrate-history`."
            );
            Ok(ExitCode::FAILURE)
        }
    }
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
    //
    // As with install, a failure here is a nonzero status rather than an error.
    // The integration is already removed; the journal survives so a later
    // uninstall can still reverse the migration.
    match history::restore_from_relay(dry_run, history_database.as_deref()) {
        Ok(_) => Ok(status),
        Err(error) => {
            log::error!(
                target: "nemo_relay.installation",
                event = "codex_history_restore_failed",
                host = "codex",
                error_kind = "history_migration";
                "Codex integration uninstalled but thread-history restore failed"
            );
            println!(
                "the Codex integration is uninstalled, but restoring thread history failed: {error}"
            );
            println!(
                "thread history is still recorded under the Relay provider; resuming those threads \
                 fails until it is restored."
            );
            Ok(ExitCode::FAILURE)
        }
    }
}
