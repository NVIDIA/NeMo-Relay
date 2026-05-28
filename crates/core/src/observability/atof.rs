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
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use chrono::Utc;

use crate::api::event::Event;
use crate::api::runtime::EventSubscriberFn;
use crate::api::subscriber::{deregister_subscriber, register_subscriber};
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
    /// Failed to connect to an ATOF stream receiver.
    #[error("failed to connect to ATOF stream receiver {address}: {source}")]
    ConnectStream {
        /// Address that failed to connect.
        address: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Failed to configure the ATOF stream connection.
    #[error(
        "failed to configure ATOF stream receiver {address} with {operation} (ATOF_STREAM_WRITE_TIMEOUT={timeout:?}): {source}"
    )]
    ConfigureStream {
        /// Address associated with the stream.
        address: String,
        /// Stream option that failed.
        operation: &'static str,
        /// Write timeout used when configuring the stream.
        timeout: Option<Duration>,
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
    /// The streaming exporter recorded an earlier write or serialization error.
    #[error("previous ATOF stream export failed for {address}: {message}")]
    StoredStreamFailure {
        /// Address associated with the stream.
        address: String,
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

/// Configuration for [`AtofStreamingExporter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtofStreamingExporterConfig {
    /// TCP address for a separate local process that receives ATOF JSONL events.
    pub address: String,
}

impl AtofStreamingExporterConfig {
    /// Create a streaming exporter config for the given TCP address.
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
        }
    }
}

const ATOF_STREAM_QUEUE_BOUND: usize = 1024;
const ATOF_STREAM_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

enum AtofStreamMessage {
    Event(String),
    Flush(mpsc::Sender<std::result::Result<(), String>>),
    Shutdown(mpsc::Sender<std::result::Result<(), String>>),
}

struct AtofStreamingExporterState {
    sender: Option<mpsc::SyncSender<AtofStreamMessage>>,
    writer_thread: Option<JoinHandle<()>>,
    events_sent: u64,
    events_dropped: u64,
    last_error: Arc<Mutex<Option<String>>>,
}

/// Snapshot of [`AtofStreamingExporter`] delivery state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AtofStreamingExporterStats {
    /// Number of ATOF events observed by the streaming exporter.
    pub events_sent: u64,
    /// Number of ATOF events dropped because the bounded streaming queue was full.
    pub events_dropped: u64,
    /// Most recent serialization or exporter state error, if one was recorded.
    pub last_error: Option<String>,
}

/// TCP-backed Agent Trajectory Observability Format (ATOF) event stream exporter.
///
/// The exporter exposes a regular NeMo Relay event subscriber and writes each
/// canonical ATOF JSON value as one JSONL line to a separate local process over
/// a TCP connection. A local UI, CLI, or bridge process can own the receiving
/// socket and fan events out over HTTP, SSE, WebSocket, stdout, or another
/// transport without redefining the ATOF event contract.
#[derive(Clone)]
pub struct AtofStreamingExporter {
    address: String,
    state: Arc<Mutex<AtofStreamingExporterState>>,
}

impl AtofStreamingExporter {
    /// Connect to a separate local ATOF stream receiver.
    pub fn new(config: AtofStreamingExporterConfig) -> Result<Self> {
        let address = config.address;
        let stream =
            TcpStream::connect(&address).map_err(|source| AtofExporterError::ConnectStream {
                address: address.clone(),
                source,
            })?;
        stream
            .set_nodelay(true)
            .map_err(|source| AtofExporterError::ConfigureStream {
                address: address.clone(),
                operation: "set_nodelay",
                timeout: None,
                source,
            })?;
        stream
            .set_write_timeout(Some(ATOF_STREAM_WRITE_TIMEOUT))
            .map_err(|source| AtofExporterError::ConfigureStream {
                address: address.clone(),
                operation: "set_write_timeout",
                timeout: Some(ATOF_STREAM_WRITE_TIMEOUT),
                source,
            })?;
        let (sender, receiver) = mpsc::sync_channel(ATOF_STREAM_QUEUE_BOUND);
        let last_error = Arc::new(Mutex::new(None));
        let writer_error = Arc::clone(&last_error);
        let writer_thread = std::thread::spawn(move || {
            let mut writer = BufWriter::new(stream);
            while let Ok(message) = receiver.recv() {
                match message {
                    AtofStreamMessage::Event(value) => {
                        if let Err(error) = write_serialized_event(&mut writer, &value) {
                            store_stream_error(&writer_error, error);
                        }
                    }
                    AtofStreamMessage::Flush(reply) => {
                        let result = writer.flush().map_err(|error| error.to_string());
                        if let Err(error) = &result {
                            store_stream_error(&writer_error, error.clone());
                        }
                        let _ = reply.send(result);
                    }
                    AtofStreamMessage::Shutdown(reply) => {
                        let result = writer.flush().map_err(|error| error.to_string());
                        if let Err(error) = &result {
                            store_stream_error(&writer_error, error.clone());
                        }
                        let _ = writer.get_ref().shutdown(Shutdown::Both);
                        let _ = reply.send(result);
                        break;
                    }
                }
            }
        });
        Ok(Self {
            address,
            state: Arc::new(Mutex::new(AtofStreamingExporterState {
                sender: Some(sender),
                writer_thread: Some(writer_thread),
                events_sent: 0,
                events_dropped: 0,
                last_error,
            })),
        })
    }

    /// Connect to a separate local ATOF stream receiver at the given TCP address.
    pub fn connect(address: impl Into<String>) -> Result<Self> {
        Self::new(AtofStreamingExporterConfig::new(address))
    }

    /// Return the connected stream receiver address.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Return an event subscriber that writes one canonical JSONL record per event.
    pub fn subscriber(&self) -> EventSubscriberFn {
        let state = Arc::clone(&self.state);
        Arc::new(move |event: &Event| {
            let value = match serialize_event(event) {
                Ok(value) => value,
                Err(error) => {
                    if let Ok(state) = state.lock() {
                        store_stream_error(&state.last_error, error);
                    }
                    return;
                }
            };
            let Ok(mut state) = state.lock() else {
                return;
            };
            if stream_last_error(&state.last_error).is_some() {
                return;
            }
            let Some(sender) = state.sender.as_ref() else {
                store_stream_error(&state.last_error, "stream receiver is closed".to_string());
                return;
            };
            match sender.try_send(AtofStreamMessage::Event(value)) {
                Ok(()) => {
                    state.events_sent += 1;
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    state.events_dropped += 1;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    store_stream_error(
                        &state.last_error,
                        "ATOF stream writer is disconnected".to_string(),
                    );
                }
            }
        })
    }

    /// Register this streaming exporter globally under the given subscriber name.
    pub fn register(&self, name: &str) -> Result<()> {
        register_subscriber(name, self.subscriber()).map_err(Into::into)
    }

    /// Deregister a global subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        deregister_subscriber(name).map_err(Into::into)
    }

    /// Flush the stream and report any stored write error.
    pub fn force_flush(&self) -> Result<()> {
        let (sender, last_error) = {
            let state = self
                .state
                .lock()
                .map_err(|_| AtofExporterError::LockPoisoned)?;
            if let Some(message) = stream_last_error(&state.last_error) {
                return Err(AtofExporterError::StoredStreamFailure {
                    address: self.address.clone(),
                    message,
                });
            }
            (state.sender.clone(), Arc::clone(&state.last_error))
        };
        let Some(sender) = sender else {
            return Ok(());
        };
        let (reply_sender, reply_receiver) = mpsc::channel();
        if sender.send(AtofStreamMessage::Flush(reply_sender)).is_err() {
            return Err(AtofExporterError::StoredStreamFailure {
                address: self.address.clone(),
                message: "ATOF stream writer is disconnected".to_string(),
            });
        }
        match reply_receiver.recv() {
            Ok(Ok(())) => {
                if let Some(message) = stream_last_error(&last_error) {
                    return Err(AtofExporterError::StoredStreamFailure {
                        address: self.address.clone(),
                        message,
                    });
                }
                Ok(())
            }
            Ok(Err(message)) => Err(AtofExporterError::StoredStreamFailure {
                address: self.address.clone(),
                message,
            }),
            Err(error) => Err(AtofExporterError::StoredStreamFailure {
                address: self.address.clone(),
                message: error.to_string(),
            }),
        }
    }

    /// Shut down the stream by flushing and closing the TCP connection.
    pub fn shutdown(&self) -> Result<()> {
        let flush_result = self.force_flush();
        let (sender, writer_thread, last_error) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| AtofExporterError::LockPoisoned)?;
            (
                state.sender.take(),
                state.writer_thread.take(),
                Arc::clone(&state.last_error),
            )
        };
        let shutdown_result = if let Some(sender) = sender {
            let (reply_sender, reply_receiver) = mpsc::channel();
            let send_result = sender
                .send(AtofStreamMessage::Shutdown(reply_sender))
                .map_err(|_| AtofExporterError::StoredStreamFailure {
                    address: self.address.clone(),
                    message: "ATOF stream writer is disconnected".to_string(),
                });
            match send_result {
                Ok(()) => match reply_receiver.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(message)) => Err(AtofExporterError::StoredStreamFailure {
                        address: self.address.clone(),
                        message,
                    }),
                    Err(error) => Err(AtofExporterError::StoredStreamFailure {
                        address: self.address.clone(),
                        message: error.to_string(),
                    }),
                },
                Err(error) => Err(error),
            }
        } else {
            Ok(())
        };
        if let Some(writer_thread) = writer_thread {
            let _ = writer_thread.join();
        }
        let stored_result =
            stream_last_error(&last_error).map(|message| AtofExporterError::StoredStreamFailure {
                address: self.address.clone(),
                message,
            });
        match (flush_result, shutdown_result) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => stored_result.map_or(Ok(()), Err),
        }
    }

    /// Return a point-in-time delivery snapshot for diagnostics and tests.
    pub fn stats(&self) -> AtofStreamingExporterStats {
        let Ok(state) = self.state.lock() else {
            return AtofStreamingExporterStats {
                last_error: Some("the ATOF streaming exporter state lock was poisoned".to_string()),
                ..AtofStreamingExporterStats::default()
            };
        };
        AtofStreamingExporterStats {
            events_sent: state.events_sent,
            events_dropped: state.events_dropped,
            last_error: stream_last_error(&state.last_error),
        }
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

fn write_event(writer: &mut impl Write, event: &Event) -> std::result::Result<(), String> {
    write_serialized_event(writer, &serialize_event(event)?)
}

fn serialize_event(event: &Event) -> std::result::Result<String, String> {
    let value = event
        .try_to_json_value()
        .map_err(|error| error.to_string())?;
    serde_json::to_string(&value).map_err(|error| error.to_string())
}

fn write_serialized_event(writer: &mut impl Write, value: &str) -> std::result::Result<(), String> {
    writer
        .write_all(value.as_bytes())
        .map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

fn store_stream_error(last_error: &Arc<Mutex<Option<String>>>, error: String) {
    if let Ok(mut last_error) = last_error.lock() {
        last_error.get_or_insert(error);
    }
}

fn stream_last_error(last_error: &Arc<Mutex<Option<String>>>) -> Option<String> {
    last_error.lock().ok().and_then(|error| error.clone())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/observability/atof_tests.rs"]
mod tests;
