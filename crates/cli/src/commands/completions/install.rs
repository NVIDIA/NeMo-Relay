// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `nemo-relay completions install` — write a shell completion script to the standard fpath /
//! completions directory for the user's current `$SHELL`. Mirrors the file layout used by
//! `scripts/install.sh` so curl-pipe installs and `cargo install` installs land in the same
//! place.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap_complete::Shell;

use crate::commands::arguments::Cli;
use crate::error::CliError;

/// Generates the completion script for `$SHELL` and writes it to the matching shell-specific
/// path under `$HOME`. Returns the path written so the CLI can echo it back to the user. The
/// shell argument is optional; when omitted, the function infers the shell from the `$SHELL`
/// environment variable. Unsupported or undetectable shells produce a `Config` error so the
/// caller can surface a clear message instead of writing to an unrelated path.
pub(crate) fn install(shell: Option<Shell>) -> Result<PathBuf, CliError> {
    let shell = match shell {
        Some(shell) => shell,
        None => detect_shell(std::env::var_os("SHELL"))?,
    };
    let target = completion_path(shell, std::env::var_os("HOME"), std::env::var_os("ZDOTDIR"))?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut clap_command = <Cli as clap::CommandFactory>::command();
    let mut buffer = Vec::new();
    clap_complete::generate(shell, &mut clap_command, "nemo-relay", &mut buffer);
    write_atomic(&target, &buffer)?;
    Ok(target)
}

/// Returns the file path where the completion script for `shell` is installed. Pure function so
/// tests can exercise path selection without touching the filesystem; the live call passes the
/// process environment in.
fn completion_path(
    shell: Shell,
    home: Option<OsString>,
    zdotdir: Option<OsString>,
) -> Result<PathBuf, CliError> {
    match shell {
        Shell::Zsh => {
            let base = zdotdir.or(home).ok_or_else(|| {
                CliError::Config("cannot resolve $ZDOTDIR or $HOME for zsh completion".into())
            })?;
            Ok(PathBuf::from(base).join(".zfunc/_nemo-relay"))
        }
        Shell::Bash => {
            let home = home.ok_or_else(|| {
                CliError::Config("cannot resolve $HOME for bash completion".into())
            })?;
            Ok(PathBuf::from(home).join(".bash_completion.d/nemo-relay"))
        }
        Shell::Fish => {
            let home = home.ok_or_else(|| {
                CliError::Config("cannot resolve $HOME for fish completion".into())
            })?;
            Ok(PathBuf::from(home).join(".config/fish/completions/nemo-relay.fish"))
        }
        other => Err(CliError::Config(format!(
            "`nemo-relay completions install` does not support {other} — \
             run `nemo-relay completions {other}` and redirect manually"
        ))),
    }
}

/// Infers a `clap_complete::Shell` from `$SHELL`. Looks only at the basename of the path and
/// matches the four shells the install path supports. Anything else produces a `Config` error
/// pointing the user at the explicit-shell form.
fn detect_shell(shell_env: Option<OsString>) -> Result<Shell, CliError> {
    let raw = shell_env.ok_or_else(|| {
        CliError::Config(
            "$SHELL is not set; pass an explicit shell, e.g. `nemo-relay completions install zsh`"
                .into(),
        )
    })?;
    let name = Path::new(&raw)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    match name {
        "zsh" => Ok(Shell::Zsh),
        "bash" => Ok(Shell::Bash),
        "fish" => Ok(Shell::Fish),
        _ => Err(CliError::Config(format!(
            "unsupported $SHELL `{name}` — \
             run `nemo-relay completions <bash|zsh|fish>` and redirect manually"
        ))),
    }
}

// Writes `bytes` to `target` via a same-directory temp file + rename so a half-finished install
// never leaves the user with a partially-written completion script.
fn write_atomic(target: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let file_name = target
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("nemo-relay");
    let temp = parent.join(format!(".{file_name}.tmp"));
    let mut handle = std::fs::File::create(&temp)?;
    handle.write_all(bytes)?;
    handle.sync_all()?;
    std::fs::rename(&temp, target)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/coverage/shared/completions_install_tests.rs"]
mod tests;
