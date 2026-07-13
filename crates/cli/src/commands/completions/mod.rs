// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod install;

use std::process::ExitCode;

use clap::CommandFactory;

use crate::configuration::{Cli, CompletionsCommand};
use crate::error::CliError;

pub(super) fn execute(command: CompletionsCommand) -> Result<ExitCode, CliError> {
    if command.install {
        let path = install::install(command.shell)?;
        println!("✓ Installed completions: {}", path.display());
    } else {
        generate_to(command.shell, &mut std::io::stdout())?;
    }
    Ok(ExitCode::SUCCESS)
}

pub(super) fn generate_to(
    shell: Option<clap_complete::Shell>,
    writer: &mut dyn std::io::Write,
) -> Result<(), CliError> {
    let shell = shell.ok_or_else(|| {
        CliError::Config(
            "missing shell argument; pass a shell name (bash, zsh, fish, ...) or use `--install` to auto-detect from $SHELL".into(),
        )
    })?;
    let mut command = Cli::command();
    clap_complete::generate(shell, &mut command, "nemo-relay", writer);
    Ok(())
}
