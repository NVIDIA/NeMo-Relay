// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::CliError;
use clap::Args;
use std::{io::Write, process::ExitCode};

#[derive(Debug, Clone, Args)]
pub(crate) struct InspectCommand {
    /// Existing local gateway address. Missing listeners are reported as unavailable.
    #[arg(long, default_value = "http://127.0.0.1:47632")]
    gateway_url: String,
    /// Read effective settings and supported producer coverage without a snapshot.
    #[arg(long)]
    capabilities: bool,
    /// Bounded JSON selection, including any opaque continuation from the previous page.
    #[arg(long, conflicts_with = "capabilities")]
    query: Option<String>,
    /// Maximum response bytes accepted by this inspector, independent of the server.
    #[arg(long, default_value_t = 131072, value_parser = clap::value_parser!(u32).range(4096..=33554432))]
    max_response_bytes: u32,
}
pub(crate) async fn inspect(command: InspectCommand) -> Result<ExitCode, CliError> {
    let mut path = crate::diagnostics_http::PREFIX.to_owned();
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
        &command.gateway_url, &path, command.max_response_bytes as usize)).await
        .map_err(|_| CliError::Launch("diagnostic inspection task failed".into()))?
        .map_err(|_| CliError::Launch("diagnostic inspection unavailable; existing gateway, private credentials, and HTTP read policy are required".into()))?;
    if response.status == 404 {
        return Err(CliError::Launch(
            "gateway diagnostics unsupported or not exposed".into(),
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
