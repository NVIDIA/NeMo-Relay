// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Flow coding-agent gateway sidecar.

mod adapters;
mod config;
mod error;
mod gateway;
mod installer;
mod launcher;
mod model;
mod server;
mod session;

use std::process::ExitCode;

use clap::Parser;

use crate::config::{Cli, Command};

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<ExitCode, error::SidecarError> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Install(command)) => {
            installer::install(command)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::HookForward(command)) => {
            installer::hook_forward(command).await?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Run(command)) => launcher::run(command, Some(&cli.server)).await,
        None => {
            let config = config::resolve_server_config(&cli.server)?;
            server::serve(config.sidecar).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
