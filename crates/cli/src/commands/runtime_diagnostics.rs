// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::CliError;
use clap::Args;
use std::{io::Write, process::ExitCode};

const DEFAULT_INSPECTOR_RESPONSE_BYTES: u32 = crate::daemon::broker::collection::PAGE_BYTES as u32;
const DEFAULT_WORKER_RESPONSE_BYTES: u32 = crate::daemon::broker::collection::RESPONSE_BYTES as u32;
const MIN_INSPECTOR_RESPONSE_BYTES: u32 = 4 * 1024;
const MAX_INSPECTOR_RESPONSE_BYTES: u32 = 32 * 1024 * 1024;

#[derive(Debug, Clone, Args)]
pub(crate) struct InspectCommand {
    /// Existing local gateway address. Missing listeners are reported as unavailable.
    #[arg(long, default_value = crate::bootstrap::DEFAULT_URL)]
    gateway_url: String,
    /// Inspect an existing local daemon using its pinned TLS identity and operator credential.
    #[arg(long, conflicts_with = "gateway_url")]
    daemon_url: Option<String>,
    /// Read one cached managed Relay worker page from the daemon, without polling on demand.
    #[arg(long, requires = "daemon_url", conflicts_with_all = ["query", "capabilities"])]
    worker: Option<uuid::Uuid>,
    /// Read effective settings and supported producer coverage without a snapshot.
    #[arg(long)]
    capabilities: bool,
    /// Bounded JSON selection, including any opaque continuation from the previous page.
    #[arg(long, conflicts_with = "capabilities")]
    query: Option<String>,
    /// Maximum response bytes accepted by this inspector, independent of the server.
    #[arg(long, value_parser = clap::value_parser!(u32).range(
        (MIN_INSPECTOR_RESPONSE_BYTES as i64)..=(MAX_INSPECTOR_RESPONSE_BYTES as i64)
    ))]
    max_response_bytes: Option<u32>,
}
pub(crate) async fn inspect(command: InspectCommand) -> Result<ExitCode, CliError> {
    let max_bytes = command
        .max_response_bytes
        .unwrap_or(if command.worker.is_some() {
            DEFAULT_WORKER_RESPONSE_BYTES
        } else {
            DEFAULT_INSPECTOR_RESPONSE_BYTES
        });
    let daemon = command.daemon_url.is_some();
    let url = command.daemon_url.unwrap_or(command.gateway_url);
    let mut path = command.worker.map_or_else(
        || crate::diagnostics_http::PREFIX.to_owned(),
        |worker| format!("/_nemo_relay/v1/workers/{worker}/diagnostics"),
    );
    if command.capabilities {
        path.push_str("/capabilities");
    }
    if let Some(query) = command.query {
        if query.len() > 1024 {
            return Err(CliError::Config(
                "diagnostic query exceeds inspector limit".into(),
            ));
        }
        path.push_str("?query=");
        path.push_str(
            &percent_encoding::utf8_percent_encode(&query, percent_encoding::NON_ALPHANUMERIC)
                .to_string(),
        );
    }
    let response = tokio::task::spawn_blocking(move || crate::gateway::client::get_diagnostics(
        &url, &path, max_bytes as usize, daemon)).await
        .map_err(|_| CliError::Launch("diagnostic inspection task failed".into()))?
        .map_err(|error| {
            CliError::Launch(format!(
                "diagnostic inspection unavailable ({error}); an existing listener, private operator credentials, HTTP read policy, and a sufficient --max-response-bytes are required"
            ))
        })?;
    if response.status == 404 {
        return Err(CliError::Launch(
            "listener diagnostics unsupported or not exposed".into(),
        ));
    }
    // Explicit inspection alone may print restricted evidence. No logging of the body.
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&response.body)?;
    stdout.write_all(b"\n")?;
    Ok(if response.status == 200 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/runtime_inspector_tests.rs"]
mod tests;
