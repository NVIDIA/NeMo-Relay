// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded newline framing for MCP stdio.

use crate::error::CliError;

pub(super) const MAX_MCP_FRAME_BYTES: usize = 1024 * 1024;
pub(super) type FrameReceiver = tokio::sync::mpsc::Receiver<Result<String, std::io::Error>>;

/// Read stdin on a plain thread so EOF remains dependable across Tokio platforms.
pub(super) fn spawn_stdin_reader() -> Result<FrameReceiver, CliError> {
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    std::thread::Builder::new()
        .name("nemo-relay-mcp-stdin".into())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut stdin = stdin.lock();
            loop {
                let mut frame = Vec::new();
                match read_bounded_frame(&mut stdin, &mut frame, MAX_MCP_FRAME_BYTES) {
                    Ok(0) => return,
                    Ok(_) => {
                        let line = String::from_utf8(frame).map_err(|error| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData, error)
                        });
                        if sender.blocking_send(line).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = sender.blocking_send(Err(error));
                        return;
                    }
                }
            }
        })
        .map_err(|error| CliError::Launch(format!("failed to start MCP stdin reader: {error}")))?;
    Ok(receiver)
}

pub(super) fn read_bounded_frame<R: std::io::BufRead>(
    reader: &mut R,
    frame: &mut Vec<u8>,
    limit: usize,
) -> std::io::Result<usize> {
    loop {
        let (consumed, complete) = {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                return Ok(frame.len());
            }
            let consumed = available
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(available.len(), |index| index + 1);
            if frame.len().saturating_add(consumed) > limit {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("MCP frame exceeds the {limit}-byte limit"),
                ));
            }
            frame.extend_from_slice(&available[..consumed]);
            (consumed, available[consumed - 1] == b'\n')
        };
        reader.consume(consumed);
        if complete {
            return Ok(frame.len());
        }
    }
}
