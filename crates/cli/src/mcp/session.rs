// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MCP stdio session coordinated with a shared-gateway liveness lease.

use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::gateway::GatewayLease;
use super::protocol::{FrameAction, evaluate_frame};
use super::transport::FrameReceiver;
use crate::error::{CliError, McpFailureReason};

pub(super) async fn run<W>(
    mut lease: GatewayLease,
    frames: FrameReceiver,
    writer: W,
) -> Result<(), CliError>
where
    W: AsyncWrite + Unpin,
{
    tokio::select! {
        result = run_without_gateway(frames, writer) => result,
        result = lease.wait() => result,
    }
}

/// Serves the no-tools MCP protocol when lifecycle is owned by the brokered daemon client.
pub(super) async fn run_without_gateway<W>(
    mut frames: super::transport::FrameReceiver,
    mut writer: W,
) -> Result<(), CliError>
where
    W: AsyncWrite + Unpin,
{
    while let Some(frame) = frames.recv().await {
        let frame = frame.map_err(CliError::Io).map_err(|error| {
            error.with_mcp_failure_reason(McpFailureReason::StdinFrameReadFailed)
        })?;
        write_response(evaluate_frame(&frame), &mut writer).await?;
    }
    Ok(())
}

async fn write_response<W>(action: FrameAction, writer: &mut W) -> Result<(), CliError>
where
    W: AsyncWrite + Unpin,
{
    let Some(response) = action.response else {
        return Ok(());
    };
    let mut encoded = serde_json::to_vec(&response).map_err(|error| {
        CliError::Launch(format!("failed to encode MCP response: {error}"))
            .with_mcp_failure_reason(McpFailureReason::McpResponseSerializationFailed)
    })?;
    encoded.push(b'\n');
    writer
        .write_all(&encoded)
        .await
        .map_err(CliError::Io)
        .map_err(|error| error.with_mcp_failure_reason(McpFailureReason::StdoutWriteFailed))?;
    writer
        .flush()
        .await
        .map_err(CliError::Io)
        .map_err(|error| error.with_mcp_failure_reason(McpFailureReason::StdoutFlushFailed))?;
    Ok(())
}

#[cfg(test)]
pub(super) async fn serve_stdio<R, W>(mut reader: R, mut writer: W) -> Result<(), CliError>
where
    R: tokio::io::AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    use tokio::io::AsyncBufReadExt;

    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        write_response(evaluate_frame(&line), &mut writer).await?;
    }
}

#[cfg(test)]
pub(super) async fn serve_with_lease<R, W>(
    mut lease: GatewayLease,
    reader: R,
    writer: W,
) -> Result<(), CliError>
where
    R: tokio::io::AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    tokio::select! {
        result = serve_stdio(reader, writer) => result,
        result = lease.wait() => result,
    }
}
