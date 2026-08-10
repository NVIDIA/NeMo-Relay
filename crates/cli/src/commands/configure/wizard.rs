// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! First-run setup for `nemo-relay` configuration.
//!
//! Coordinates first-run setup while terminal-only interaction lives in `wizard/prompt.rs`.

use std::path::PathBuf;

#[cfg(test)]
use toml_edit::DocumentMut;

#[cfg(test)]
use self::model::{
    build_config, plugins_edit_command, plugins_resume_command, preview_paths, save_config,
};
use super::model;
use crate::agents::CodingAgent;
use crate::error::CliError;

#[cfg(test)]
use self::model::{
    Defaults, SetupAnswers, read_agents_from_doc, read_existing_defaults, reset, user_config_dir,
    write_or_merge,
};

#[cfg(test)]
use self::model::detect_installed_agents_in;

mod prompt;

/// Top-level setup entry point used by `nemo-relay config` and the easy-path fallback.
/// Detects agents, prompts the user, writes the config, prints a final summary.
///
/// `agent_hint` carries the agent the user typed on the easy path (`nemo-relay claude`); when
/// `Some`, the agent multi-select is skipped because intent is already declared. `None` from
/// `nemo-relay config` asks the full set so users can configure multiple agents at once.
pub(crate) async fn run(
    agent_hint: Option<CodingAgent>,
    explicit_plugin_path: Option<PathBuf>,
) -> Result<(), CliError> {
    prompt::run(agent_hint, explicit_plugin_path).await
}

fn plugin_prompt_was_interrupted(error: &dialoguer::Error) -> bool {
    matches!(
        error,
        dialoguer::Error::IO(io_error)
            if matches!(
                io_error.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
            )
    )
}

#[cfg(test)]
#[path = "../../../tests/coverage/shared/setup_tests.rs"]
mod tests;
