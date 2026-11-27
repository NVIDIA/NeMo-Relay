// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod install;

use std::io::{ErrorKind, Write};
use std::process::ExitCode;

use clap::Args;

use clap::CommandFactory;

use crate::commands::root::Cli;
use crate::error::CliError;

#[derive(Debug, Clone, Args)]
pub(crate) struct CompletionsCommand {
    #[arg(value_enum)]
    pub(crate) shell: Option<clap_complete::Shell>,
    #[arg(long)]
    pub(crate) install: bool,
}

pub(super) fn execute(command: CompletionsCommand) -> Result<ExitCode, CliError> {
    if command.install {
        let path = install::install(command.shell)?;
        println!("✓ Installed completions: {}", path.display());
    } else {
        let mut stdout = std::io::stdout().lock();
        match generate_to(command.shell, &mut stdout)
            .and_then(|()| stdout.flush().map_err(CliError::from))
        {
            Err(CliError::Io(error)) if error.kind() == ErrorKind::BrokenPipe => {}
            result => result?,
        }
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
    // clap_complete panics on write errors, so generate in memory before writing to the output.
    let mut buffer = Vec::new();
    clap_complete::generate(shell, &mut command, "nemo-relay", &mut buffer);
    writer.write_all(&buffer)?;
    Ok(())
}
