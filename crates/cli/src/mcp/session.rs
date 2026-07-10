// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MCP stdio session coordinated with a shared-gateway liveness lease.

use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::gateway::{GatewayLease, GatewayPlan};
use super::protocol::{FrameAction, evaluate_frame};
use super::transport::FrameReceiver;
use crate::error::CliError;

pub(super) async fn run<W>(
    gateway: GatewayPlan,
    mut frames: FrameReceiver,
    mut writer: W,
) -> Result<(), CliError>
where
    W: AsyncWrite + Unpin,
{
    let mut lease: Option<GatewayLease> = None;
    loop {
        let received = match lease.as_mut() {
            Some(active) => tokio::select! {
                frame = frames.recv() => frame,
                result = active.wait() => return result,
            },
            None => frames.recv().await,
        };
        let Some(frame) = received else {
            return Ok(());
        };
        let frame = frame?;
        let action = evaluate_frame(&frame);
        if action.requires_gateway && lease.is_none() {
            lease = Some(gateway.acquire().await?);
        }
        write_response(action, &mut writer).await?;
    }
}

async fn write_response<W>(action: FrameAction, writer: &mut W) -> Result<(), CliError>
where
    W: AsyncWrite + Unpin,
{
    let Some(response) = action.response else {
        return Ok(());
    };
    let mut encoded = serde_json::to_vec(&response)
        .map_err(|error| CliError::Launch(format!("failed to encode MCP response: {error}")))?;
    encoded.push(b'\n');
    writer.write_all(&encoded).await?;
    writer.flush().await?;
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
