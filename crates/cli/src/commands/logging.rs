// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide operational logging arguments and source selection.

use std::path::{Path, PathBuf};

use clap::Args;
use nemo_relay::error::FlowError;
use nemo_relay::logging::{LogFormat, LogLevel, LoggingConfig};

use crate::error::CliError;

#[derive(Debug, Clone, Default, Args)]
pub(super) struct LoggingArgs {
    /// Minimum operational log level.
    #[arg(
        long = "log-level",
        value_parser = ["error", "warn", "info", "debug", "trace"],
        conflicts_with = "config_path"
    )]
    level: Option<String>,
    /// Format for the mandatory stderr logging sink.
    #[arg(
        long = "log-stderr-format",
        value_parser = ["human", "jsonl"],
        conflicts_with = "config_path"
    )]
    stderr_format: Option<String>,
    /// Absolute path to a TOML document containing a `[logging]` section.
    #[arg(
        long = "log-config-path",
        conflicts_with_all = ["level", "stderr_format"]
    )]
    config_path: Option<PathBuf>,
}

impl LoggingArgs {
    /// Selects one logging source: direct CLI settings, environment, file configuration, or
    /// built-in defaults. Sources are not merged with one another.
    pub(super) fn resolve(
        &self,
        explicit_config: Option<&Path>,
        user_only: bool,
    ) -> Result<LoggingConfig, CliError> {
        if let Some(path) = &self.config_path {
            return LoggingConfig::from_file_path(path).map_err(logging_config_error);
        }

        if self.level.is_some() || self.stderr_format.is_some() {
            let mut config = LoggingConfig::default();
            if let Some(level) = self.level.as_deref() {
                config.level = LogLevel::parse(level).map_err(logging_config_error)?;
            }
            if let Some(stderr_format) = self.stderr_format.as_deref() {
                config.stderr_format =
                    LogFormat::parse(stderr_format).map_err(logging_config_error)?;
            }
            return Ok(config);
        }

        if let Some(config) = LoggingConfig::from_environment().map_err(logging_config_error)? {
            return Ok(config);
        }

        crate::configuration::resolve_logging_config(explicit_config, user_only)
    }
}

fn logging_config_error(error: FlowError) -> CliError {
    match error {
        FlowError::InvalidArgument(message) => CliError::Config(message),
        other => CliError::Flow(other),
    }
}
