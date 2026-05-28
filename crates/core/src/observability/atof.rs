// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent Trajectory Observability Format (ATOF) JSONL exporter support for NeMo
//! Flow.
//!
//! The [`AtofExporter`] registers as an event subscriber and writes each
//! canonical NeMo Relay Agent Trajectory Observability Format (ATOF) event as
//! one JSON object per JSONL line.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::Utc;

use crate::api::event::Event;
use crate::api::runtime::EventSubscriberFn;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;

/// Result type for the ATOF JSONL exporter.
pub type Result<T> = std::result::Result<T, AtofExporterError>;

/// Errors produced while configuring or operating the ATOF JSONL exporter.
#[derive(Debug, thiserror::Error)]
pub enum AtofExporterError {
    /// Failed to resolve the current working directory for default config.
    #[error("failed to resolve current working directory: {0}")]
    CurrentDirectory(std::io::Error),
    /// Failed to open the output file.
    #[error("failed to open ATOF output file {path:?}: {source}")]
    OpenFile {
        /// Output path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Failed while flushing the output file.
    #[error("failed to flush ATOF output file {path:?}: {source}")]
    Flush {
        /// Output path that failed to flush.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The exporter recorded an earlier write or serialization error.
    #[error("previous ATOF export failed for {path:?}: {message}")]
    StoredFailure {
        /// Output path associated with the failure.
        path: PathBuf,
        /// Stored failure message.
        message: String,
    },
    /// The internal exporter state lock was poisoned.
    #[error("the ATOF exporter state lock was poisoned")]
    LockPoisoned,
    /// Runtime subscriber registration failed.
    #[error(transparent)]
    Runtime(#[from] FlowError),
}

/// File write behavior for [`AtofExporter`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AtofExporterMode {
    /// Append events to an existing file or create it if missing.
    #[default]
    Append,
    /// Truncate an existing file when the exporter is created.
    Overwrite,
}

impl AtofExporterMode {
    /// Parse a string mode used by language bindings.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "append" => Some(Self::Append),
            "overwrite" => Some(Self::Overwrite),
            _ => None,
        }
    }

    /// Return the stable string representation used by language bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Overwrite => "overwrite",
        }
    }
}

/// Configuration for [`AtofExporter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtofExporterConfig {
    /// Directory that contains the JSONL output file.
    pub output_directory: PathBuf,
    /// Append or overwrite behavior used when opening the file.
    pub mode: AtofExporterMode,
    /// Output filename.
    pub filename: String,
}

impl Default for AtofExporterConfig {
    fn default() -> Self {
        Self {
            output_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            mode: AtofExporterMode::Append,
            filename: default_filename(),
        }
    }
}

impl AtofExporterConfig {
    /// Create a config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the output directory.
    pub fn with_output_directory(mut self, output_directory: impl Into<PathBuf>) -> Self {
        self.output_directory = output_directory.into();
        self
    }

    /// Override the output mode.
    pub fn with_mode(mut self, mode: AtofExporterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Override the output filename.
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = filename.into();
        self
    }

    /// Return the full output path for this config.
    pub fn path(&self) -> PathBuf {
        self.output_directory.join(&self.filename)
    }
}

struct AtofExporterState {
    writer: BufWriter<File>,
    last_error: Option<String>,
}

/// Filesystem-backed Agent Trajectory Observability Format (ATOF) JSONL event exporter.
pub struct AtofExporter {
    path: PathBuf,
    state: Arc<Mutex<AtofExporterState>>,
}

impl AtofExporter {
    /// Create a new exporter from config and open its output file.
    pub fn new(config: AtofExporterConfig) -> Result<Self> {
        let path = config.path();
        let file = open_file(&path, config.mode)?;
        Ok(Self {
            path,
            state: Arc::new(Mutex::new(AtofExporterState {
                writer: BufWriter::new(file),
                last_error: None,
            })),
        })
    }

    /// Return the output JSONL path.
    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    /// Return an event subscriber that writes one JSONL record per observed event.
    pub fn subscriber(&self) -> EventSubscriberFn {
        let state = Arc::clone(&self.state);
        Arc::new(move |event: &Event| {
            let Ok(mut state) = state.lock() else {
                return;
            };
            if state.last_error.is_some() {
                return;
            }
            if let Err(error) = write_event(&mut state.writer, event) {
                state.last_error = Some(error);
            }
        })
    }

    /// Register this exporter globally under the given subscriber name.
    pub fn register(&self, name: &str) -> Result<()> {
        register_subscriber(name, self.subscriber()).map_err(Into::into)
    }

    /// Deregister a global subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        deregister_subscriber(name).map_err(Into::into)
    }

    /// Flush the underlying file and report any stored write error.
    pub fn force_flush(&self) -> Result<()> {
        flush_subscribers()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| AtofExporterError::LockPoisoned)?;
        state
            .writer
            .flush()
            .map_err(|source| AtofExporterError::Flush {
                path: self.path.clone(),
                source,
            })?;
        if let Some(message) = &state.last_error {
            return Err(AtofExporterError::StoredFailure {
                path: self.path.clone(),
                message: message.clone(),
            });
        }
        Ok(())
    }

    /// Shut down the exporter by flushing any buffered data.
    pub fn shutdown(&self) -> Result<()> {
        self.force_flush()
    }
}

fn default_filename() -> String {
    format!(
        "nemo-relay-events-{}.jsonl",
        Utc::now().format("%Y-%m-%d-%H.%M.%S")
    )
}

fn open_file(path: &Path, mode: AtofExporterMode) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true);
    match mode {
        AtofExporterMode::Append => {
            options.append(true);
        }
        AtofExporterMode::Overwrite => {
            options.write(true).truncate(true);
        }
    }
    options
        .open(path)
        .map_err(|source| AtofExporterError::OpenFile {
            path: path.to_path_buf(),
            source,
        })
}

fn write_event(writer: &mut BufWriter<File>, event: &Event) -> std::result::Result<(), String> {
    let value = event
        .try_to_json_value()
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut *writer, &value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/observability/atof_tests.rs"]
mod tests;
