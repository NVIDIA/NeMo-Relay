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
    /// Returns whether direct command-line, log-file, or environment settings are present.
    pub(super) fn has_explicit_configuration(&self) -> Result<bool, CliError> {
        Ok(self.resolve_explicit()?.is_some())
    }

    /// Selects one logging source: direct CLI settings, environment, file configuration, or
    /// built-in defaults. Sources are not merged with one another.
    pub(super) fn resolve(
        &self,
        explicit_config: Option<&Path>,
    ) -> Result<LoggingConfig, CliError> {
        if let Some(config) = self.resolve_explicit()? {
            return Ok(config);
        }

        crate::configuration::resolve_logging_config(explicit_config)
    }

    /// Resolves direct logging settings without consulting Relay configuration files discovered
    /// from the environment. Commands that repair or remove Relay state use this so malformed
    /// ambient configuration cannot block the operation.
    pub(super) fn resolve_without_ambient_config(&self) -> Result<LoggingConfig, CliError> {
        Ok(self.resolve_explicit()?.unwrap_or_default())
    }

    /// Resolves only command-line, log-file, and environment logging sources.
    fn resolve_explicit(&self) -> Result<Option<LoggingConfig>, CliError> {
        if let Some(path) = &self.config_path {
            return LoggingConfig::from_file_path(path)
                .map(Some)
                .map_err(logging_config_error);
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
            return Ok(Some(config));
        }

        LoggingConfig::from_environment().map_err(logging_config_error)
    }
}

fn logging_config_error(error: FlowError) -> CliError {
    match error {
        FlowError::InvalidArgument(message) => CliError::Config(message),
        other => CliError::Flow(other),
    }
}
