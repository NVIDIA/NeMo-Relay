// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide operational logging arguments and source selection.

use std::path::{Path, PathBuf};

use clap::Args;
use nemo_relay::error::FlowError;
use nemo_relay::logging::{LogFormat, LogLevel, LoggingConfig};

use crate::error::CliError;

// TODO EE: Temporary live smoke-test switch for .vscode/launch.json. Remove with the
// temporary logging launch configurations.
const LOG_SMOKE_TEST_ENV: &str = "NEMO_RELAY_LOG_SMOKE_TEST";

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

pub(super) fn emit_smoke_events() {
    if !matches!(std::env::var(LOG_SMOKE_TEST_ENV).as_deref(), Ok("1")) {
        return;
    }

    log::error!(
        target: "nemo_relay.smoke",
        event = "smoke_error",
        smoke = true;
        "Temporary error-level logging smoke event"
    );
    log::warn!(
        target: "nemo_relay.smoke",
        event = "smoke_warn",
        smoke = true;
        "Temporary warn-level logging smoke event"
    );
    log::info!(
        target: "nemo_relay.smoke",
        event = "smoke_info",
        smoke = true;
        "Temporary info-level logging smoke event"
    );
    log::debug!(
        target: "nemo_relay.smoke",
        event = "smoke_debug",
        smoke = true;
        "Temporary debug-level logging smoke event"
    );
    log::trace!(
        target: "nemo_relay.smoke",
        event = "smoke_trace",
        smoke = true;
        "Temporary trace-level logging smoke event"
    );
}

fn logging_config_error(error: FlowError) -> CliError {
    match error {
        FlowError::InvalidArgument(message) => CliError::Config(message),
        other => CliError::Flow(other),
    }
}
