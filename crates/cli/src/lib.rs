// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal application library for the `nemo-relay` executable.

mod agents;
mod banner;
mod bootstrap;
mod commands;
mod configuration;
mod diagnostics;
mod error;
mod events;
mod filesystem;
mod gateway;
mod hooks;
mod installation;
mod mcp;
mod mcp_environment;
mod plugins;
mod process;
mod server;
mod sessions;

#[cfg(test)]
#[path = "../tests/coverage/shared/hook_assertions.rs"]
mod hook_assertions;

#[cfg(test)]
#[path = "../tests/coverage/shared/test_support.rs"]
pub(crate) mod test_support;

use std::process::ExitCode;

/// Runs the `nemo-relay` process.
///
/// This is an executable entrypoint, not a supported library API.
#[doc(hidden)]
pub fn run_cli() -> ExitCode {
    mcp_environment::remove_unresolved_mcp_placeholders();
    let bootstrap_shutdown_token = take_bootstrap_shutdown_token();
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
    runtime.block_on(commands::run(bootstrap_shutdown_token))
}

fn take_bootstrap_shutdown_token() -> Option<String> {
    let token = std::env::var(bootstrap::state::BOOTSTRAP_SHUTDOWN_TOKEN_ENV)
        .ok()
        .filter(|token| !token.is_empty());
    // SAFETY: this runs before the Tokio runtime and application threads are created.
    unsafe {
        std::env::remove_var(bootstrap::state::BOOTSTRAP_SHUTDOWN_TOKEN_ENV);
    }
    token
}
