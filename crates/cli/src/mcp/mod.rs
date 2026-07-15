// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle-bound MCP stdio client for the shared native Relay gateway.

mod gateway;
mod protocol;
mod session;
mod transport;

use std::net::SocketAddr;
use std::path::Path;
use std::process::ExitCode;

use serde_json::{Value, json};

use crate::error::CliError;
use crate::installation::generation::{GENERATION_FILE_ENV, GENERATION_TOKEN_ENV};
use crate::server::GatewayOverrides;

pub(crate) const SERVER_NAME: &str = "nemo-relay";
const LAUNCH_ARGS: &[&str] = &["mcp"];

pub(crate) async fn run(server_args: &GatewayOverrides) -> Result<ExitCode, CliError> {
    if transparent_run_active() {
        // An installed plugin can still be enabled inside `nemo-relay run`. In that process the
        // wrapper already owns a healthy dynamic gateway, so this MCP instance authenticates and
        // monitors it instead of launching the fixed persistent sidecar.
        let gateway_url = std::env::var(crate::configuration::GATEWAY_URL_ENV).map_err(|_| {
            CliError::Launch(format!(
                "{} is required when {}=1",
                crate::configuration::GATEWAY_URL_ENV,
                crate::configuration::TRANSPARENT_RUN_ENV
            ))
        })?;
        let bootstrap_fingerprint =
            crate::configuration::transparent_gateway_fingerprint(&gateway_url);
        let lease = gateway::GatewayLease::borrow(gateway_url, bootstrap_fingerprint).await?;
        let frames = transport::spawn_stdin_reader()?;
        session::run(lease, frames, tokio::io::stdout()).await?;
        return Ok(ExitCode::SUCCESS);
    }
    // Starting the MCP process is the lifecycle boundary. Acquire the shared gateway before
    // reading protocol frames so hosts can rely on process startup rather than their individual
    // initialize and hook ordering.
    let lease = gateway::GatewayPlan::resolve(server_args)
        .await?
        .acquire()
        .await?;
    let frames = transport::spawn_stdin_reader()?;
    session::run(lease, frames, tokio::io::stdout()).await?;
    Ok(ExitCode::SUCCESS)
}

/// Builds the host-independent persistent MCP launch contract.
///
/// Host adapters add only schema-specific activation and environment-forwarding fields. Keeping
/// the command, arguments, fixed gateway bind, and generation fence here ensures Codex, Claude
/// Code, and Hermes launch the same process.
pub(crate) fn persistent_server(
    relay: &Path,
    generation_file: &Path,
    generation_token: &str,
) -> Value {
    json!({
        "command": relay,
        "args": LAUNCH_ARGS,
        "env": {
            "NEMO_RELAY_GATEWAY_BIND": crate::bootstrap::DEFAULT_BIND,
            (GENERATION_FILE_ENV): generation_file,
            (GENERATION_TOKEN_ENV): generation_token
        }
    })
}

fn transparent_run_active() -> bool {
    std::env::var(crate::configuration::TRANSPARENT_RUN_ENV)
        .ok()
        .as_deref()
        == Some("1")
}

fn default_mcp_bind() -> SocketAddr {
    crate::bootstrap::DEFAULT_BIND
        .parse()
        .expect("default MCP gateway bind is valid")
}

#[cfg(test)]
async fn run_session<R, W>(reader: R, writer: W) -> Result<(), CliError>
where
    R: tokio::io::AsyncBufRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let lease = gateway::GatewayLease::test_pending();
    session::serve_with_lease(lease, reader, writer).await
}

#[cfg(test)]
use gateway::{maintain_gateway_with, maintain_gateway_with_generation};
#[cfg(test)]
use protocol::{MCP_PROTOCOL_VERSION, jsonrpc_error, response_for};
#[cfg(test)]
use session::serve_stdio;
#[cfg(test)]
use transport::{MAX_MCP_FRAME_BYTES, read_bounded_frame};

#[cfg(test)]
#[path = "../../tests/coverage/shared/mcp_tests.rs"]
mod tests;
