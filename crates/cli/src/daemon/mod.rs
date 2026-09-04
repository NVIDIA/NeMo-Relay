// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Brokered daemon runtime for managed coding-agent integrations.

use std::net::Ipv4Addr;
use std::path::PathBuf;

use crate::error::CliError;

pub(crate) mod broker;
pub(crate) mod common;
pub(crate) mod hook;
pub(crate) mod managed;
pub(crate) mod mcp;
pub(crate) mod worker;

/// Runtime options for the public daemon listener.
#[derive(Debug, Clone)]
pub(crate) struct ServerOptions {
    pub(crate) bind: Ipv4Addr,
    pub(crate) port: u16,
    pub(crate) advertise_address: Option<String>,
    pub(crate) pass_through: bool,
    pub(crate) gateway: crate::server::GatewayOverrides,
    pub(crate) tls_cert: Option<PathBuf>,
    pub(crate) tls_key: Option<PathBuf>,
    pub(crate) client_token_file: Option<PathBuf>,
}

pub(crate) async fn serve(options: ServerOptions) -> Result<(), CliError> {
    broker::server::serve(options).await
}
