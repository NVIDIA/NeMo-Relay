// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle-bound MCP stdio client for the shared native Relay gateway.

mod gateway;
mod protocol;
mod session;
mod transport;

use std::net::SocketAddr;
use std::process::ExitCode;

use crate::config::{CodingAgent, ServerArgs};
use crate::error::CliError;

pub(crate) async fn run(
    agent: CodingAgent,
    server_args: &ServerArgs,
) -> Result<ExitCode, CliError> {
    // Starting the MCP process is the lifecycle boundary. Acquire the shared gateway before
    // reading protocol frames so hosts can rely on process startup rather than their individual
    // initialize and hook ordering.
    let lease = gateway::GatewayPlan::resolve(agent, server_args)
        .await?
        .acquire()
        .await?;
    let frames = transport::spawn_stdin_reader()?;
    session::run(lease, frames, tokio::io::stdout()).await?;
    Ok(ExitCode::SUCCESS)
}

fn default_mcp_bind() -> SocketAddr {
    crate::sidecar::DEFAULT_BIND
        .parse()
        .expect("default MCP gateway bind is valid")
}

#[cfg(test)]
async fn run_session<R, W>(
    bind: SocketAddr,
    gateway_url: String,
    sidecar_args: Vec<std::ffi::OsString>,
    bootstrap_fingerprint: String,
    heartbeat_interval: std::time::Duration,
    reader: R,
    writer: W,
) -> Result<(), CliError>
where
    R: tokio::io::AsyncBufRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let lease = gateway::GatewayPlan::test_lease(
        bind,
        gateway_url,
        sidecar_args,
        bootstrap_fingerprint,
        heartbeat_interval,
    );
    session::serve_with_lease(lease, reader, writer).await
}

#[cfg(test)]
use gateway::{
    maintain_gateway_with, maintain_gateway_with_generation, verify_bootstrap_generation,
};
#[cfg(test)]
use protocol::{MCP_PROTOCOL_VERSION, jsonrpc_error, response_for};
#[cfg(test)]
use session::serve_stdio;
#[cfg(test)]
use transport::{MAX_MCP_FRAME_BYTES, read_bounded_frame};

#[cfg(test)]
#[path = "../tests/coverage/mcp_tests.rs"]
mod tests;
