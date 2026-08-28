// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

use clap::{Args, Subcommand};

use crate::error::CliError;

use super::serve::ServerArgs;

#[derive(Debug, Clone, Args)]
pub(crate) struct GatewayCommand {
    #[command(subcommand)]
    command: GatewaySubcommand,
}

#[derive(Debug, Clone, Subcommand)]
enum GatewaySubcommand {
    /// Start the gateway with the same server configuration as a bare daemon invocation.
    Start,
    /// Stop the managed shared gateway at the configured loopback endpoint.
    Stop,
}

impl GatewayCommand {
    /// Returns whether this command only stops an existing managed gateway.
    pub(crate) fn is_stop(&self) -> bool {
        matches!(self.command, GatewaySubcommand::Stop)
    }
}

/// Executes a gateway lifecycle command.
pub(crate) async fn execute(
    command: GatewayCommand,
    server: &ServerArgs,
    bootstrap_shutdown_token: Option<String>,
) -> Result<ExitCode, CliError> {
    match command.command {
        GatewaySubcommand::Start => super::serve_gateway(server, bootstrap_shutdown_token).await,
        GatewaySubcommand::Stop => crate::mcp::stop(&server.to_runtime()),
    }
}
