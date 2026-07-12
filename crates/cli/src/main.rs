// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Relay coding-agent gateway CLI.

mod agent_process;
mod agents;
mod banner;
mod commands;
mod completions_install;
mod config;
mod doctor;
mod error;
mod file_io;
mod gateway;
mod install_generation;
mod installer;
mod json_path;
mod launcher;
mod mcp;
mod mcp_environment;
mod model;
mod model_pricing;
mod plugins;
mod server;
mod session;
mod setup;
mod sidecar;

#[cfg(test)]
#[path = "../tests/coverage/shared/hook_assertions.rs"]
mod hook_assertions;

#[cfg(test)]
pub(crate) use commands::test_support;

use std::process::ExitCode;

fn main() -> ExitCode {
    mcp_environment::remove_unresolved_mcp_placeholders();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("failed to initialize async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(commands::run())
}
