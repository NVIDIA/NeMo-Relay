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
        if self.config_path.is_some() || self.level.is_some() || self.stderr_format.is_some() {
            return Ok(true);
        }
        Ok(LoggingConfig::from_environment()
            .map_err(logging_config_error)?
            .is_some())
    }

    /// Loads logging settings from files, then applies environment and CLI overrides.
    pub(super) fn resolve(
        &self,
        explicit_config: Option<&Path>,
    ) -> Result<LoggingConfig, CliError> {
        self.resolve_with_base(|| crate::configuration::resolve_logging_config(explicit_config))
    }

    /// Loads `[daemon.logging]`, then applies environment and CLI overrides.
    pub(super) fn resolve_daemon(
        &self,
        explicit_config: Option<&Path>,
    ) -> Result<LoggingConfig, CliError> {
        self.resolve_with_base(|| {
            crate::configuration::resolve_daemon_logging_config(explicit_config)
        })
    }

    /// Uses default logging settings plus explicit overrides. Skips config files found
    /// automatically so repair and removal commands still work if those files are invalid.
    pub(super) fn resolve_without_ambient_config(&self) -> Result<LoggingConfig, CliError> {
        self.resolve_with_base(|| Ok(LoggingConfig::default()))
    }

    fn resolve_with_base(
        &self,
        load_base: impl FnOnce() -> Result<LoggingConfig, CliError>,
    ) -> Result<LoggingConfig, CliError> {
        if let Some(path) = &self.config_path {
            return LoggingConfig::from_file_path(path).map_err(logging_config_error);
        }

        let environment = LoggingConfig::from_environment().map_err(logging_config_error)?;
        let mut config = match environment {
            // The named logging file replaces the usual config search, including for the daemon.
            Some(environment) if std::env::var_os("NEMO_RELAY_LOG_CONFIG_PATH").is_some() => {
                environment
            }
            Some(environment) => {
                let mut config = load_base()?;
                if std::env::var_os("NEMO_RELAY_LOG").is_some() {
                    config.level = environment.level;
                }
                if std::env::var_os("NEMO_RELAY_LOG_STDERR").is_some() {
                    config.stderr_enabled = environment.stderr_enabled;
                }
                if std::env::var_os("NEMO_RELAY_LOG_STDERR_FORMAT").is_some() {
                    config.stderr_format = environment.stderr_format;
                }
                config
            }
            None => load_base()?,
        };
        if let Some(level) = self.level.as_deref() {
            config.level = LogLevel::parse(level).map_err(logging_config_error)?;
        }
        if let Some(stderr_format) = self.stderr_format.as_deref() {
            config.stderr_format = LogFormat::parse(stderr_format).map_err(logging_config_error)?;
        }
        Ok(config)
    }
}

fn logging_config_error(error: FlowError) -> CliError {
    match error {
        FlowError::InvalidArgument(message) => CliError::Config(message),
        other => CliError::Flow(other),
    }
}
