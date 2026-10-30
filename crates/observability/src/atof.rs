// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(missing_docs)]

//! Agent Trajectory Observability Format (ATOF) JSONL exporter support for NeMo
//! Flow.
//!
//! The [`AtofExporter`] registers as an event subscriber and writes each
//! canonical NeMo Relay Agent Trajectory Observability Format (ATOF) event as
//! one JSON object per JSONL line.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::time::Duration;

use chrono::Utc;
use futures_util::{FutureExt, SinkExt, future::FusedFuture, stream};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

use super::header_file::{HeaderFiles, PlaintextPolicy};
use super::header_file::{
    has_configured_headers, resolve_header_files, validate_header_files,
    validate_header_http_endpoint, validate_header_websocket_endpoint,
};
use super::private_file::{create_private_dir_all, open_private};
use super::runtime_diagnostics::{ExporterRuntimeDiagnostics, ExporterRuntimeDiagnosticsRecorder};
use crate::host::EventSubscriberFn;
use nemo_relay_types::api::event::Event;

use nemo_relay_types::error::FlowError;

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
    /// A streaming endpoint configuration is invalid.
    #[error("invalid ATOF streaming endpoint: {0}")]
    InvalidEndpoint(String),
    /// The internal exporter state lock was poisoned.
    #[error("the ATOF exporter state lock was poisoned")]
    LockPoisoned,
    /// Runtime subscriber registration failed.
    #[error(transparent)]
    Runtime(#[from] FlowError),
}

/// File write behavior for [`AtofExporter`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Streaming transport used by an ATOF endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtofEndpointTransport {
    /// POST each event as one JSONL record.
    #[default]
    HttpPost,
    /// Send each event as one WebSocket JSON text message.
    Websocket,
    /// Stream events over one long-lived HTTP NDJSON upload.
    Ndjson,
}

/// Field name transformation policy used before sending events to an endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtofEndpointFieldNamePolicy {
    /// Preserve canonical ATOF field names exactly.
    #[default]
    Preserve,
    /// Replace dots in JSON object keys with underscores, recursively.
    ReplaceDots,
}

impl AtofEndpointFieldNamePolicy {
    /// Parse a string policy used by configuration and bindings.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "preserve" => Some(Self::Preserve),
            "replace_dots" => Some(Self::ReplaceDots),
            _ => None,
        }
    }

    /// Return the stable string representation used by configuration and bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preserve => "preserve",
            Self::ReplaceDots => "replace_dots",
        }
    }
}

impl AtofEndpointTransport {
    /// Parse a string transport used by configuration and bindings.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "http_post" => Some(Self::HttpPost),
            "websocket" => Some(Self::Websocket),
            "ndjson" => Some(Self::Ndjson),
            _ => None,
        }
    }

    /// Return the stable string representation used by configuration and bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HttpPost => "http_post",
            Self::Websocket => "websocket",
            Self::Ndjson => "ndjson",
        }
    }
}

/// Streaming sink for raw ATOF events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtofStreamSinkConfig {
    /// Endpoint URL.
    pub url: String,
    /// Endpoint transport.
    #[serde(default)]
    pub transport: AtofEndpointTransport,
    /// Headers applied to endpoint requests or handshakes.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    /// Header names mapped to environment variables containing their values.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub header_env: HashMap<String, String>,
    /// Header names mapped to files whose values are read for each request or connection.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub header_file: HeaderFiles,
    /// Per-endpoint timeout in milliseconds.
    #[serde(default = "default_endpoint_timeout_millis")]
    pub timeout_millis: u64,
    /// Field name transformation policy applied before sending events.
    #[serde(default)]
    pub field_name_policy: AtofEndpointFieldNamePolicy,
    /// Allow `http://` or `ws://` to any host; events and configured headers travel unencrypted.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_insecure: bool,
}

impl AtofStreamSinkConfig {
    /// Create a streaming endpoint with defaults.
    pub fn new(url: impl Into<String>, transport: AtofEndpointTransport) -> Self {
        Self {
            url: url.into(),
            transport,
            headers: HashMap::new(),
            header_env: HashMap::new(),
            header_file: HashMap::new(),
            timeout_millis: default_endpoint_timeout_millis(),
            field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
            allow_insecure: false,
        }
    }

    /// Allow `http://` or `ws://` to any host; events and configured headers travel unencrypted.
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }

    /// Add a header to this endpoint config.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Add a header whose value is resolved from an environment variable at activation.
    pub fn with_header_env(
        mut self,
        key: impl Into<String>,
        environment_variable: impl Into<String>,
    ) -> Self {
        self.header_env
            .insert(key.into(), environment_variable.into());
        self
    }

    /// Add a header whose value is read from a file when the destination exports.
    pub fn with_header_file(mut self, key: impl Into<String>, path: impl Into<String>) -> Self {
        self.header_file.insert(key.into(), path.into());
        self
    }

    /// Override the endpoint timeout.
    pub fn with_timeout_millis(mut self, timeout_millis: u64) -> Self {
        self.timeout_millis = timeout_millis;
        self
    }

    /// Override the endpoint field name policy.
    pub fn with_field_name_policy(
        mut self,
        field_name_policy: AtofEndpointFieldNamePolicy,
    ) -> Self {
        self.field_name_policy = field_name_policy;
        self
    }
}

/// Backward-compatible name for an ATOF stream sink.
pub type AtofEndpointConfig = AtofStreamSinkConfig;

/// Filesystem sink for raw ATOF JSONL events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtofFileSinkConfig {
    /// Directory that contains the JSONL output file.
    #[serde(default = "default_output_directory")]
    pub output_directory: PathBuf,
    /// Append or overwrite behavior used when opening the file.
    #[serde(default)]
    pub mode: AtofExporterMode,
    /// Output filename.
    #[serde(default = "default_filename")]
    pub filename: String,
}

impl Default for AtofFileSinkConfig {
    fn default() -> Self {
        Self {
            output_directory: default_output_directory(),
            mode: AtofExporterMode::Append,
            filename: default_filename(),
        }
    }
}

impl AtofFileSinkConfig {
    /// Create a file sink with native defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the full output path for this sink.
    pub fn path(&self) -> PathBuf {
        self.output_directory.join(&self.filename)
    }
}

/// One destination for raw ATOF events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AtofSinkConfig {
    /// Write canonical ATOF records to one JSONL file.
    File(AtofFileSinkConfig),
    /// Send canonical ATOF records to one remote stream.
    Stream(AtofStreamSinkConfig),
}

/// Configuration for [`AtofExporter`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtofExporterConfig {
    /// The one output sink owned by this exporter.
    #[serde(flatten)]
    pub sink: AtofSinkConfig,
}

impl Default for AtofExporterConfig {
    fn default() -> Self {
        Self {
            sink: AtofSinkConfig::File(AtofFileSinkConfig::default()),
        }
    }
}

impl AtofExporterConfig {
    /// Create a config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the file sink output directory.
    ///
    /// This has no effect after selecting a stream sink with
    /// [`Self::with_stream_sink`] or [`Self::with_endpoint`].
    pub fn with_output_directory(mut self, output_directory: impl Into<PathBuf>) -> Self {
        if let AtofSinkConfig::File(file) = &mut self.sink {
            file.output_directory = output_directory.into();
        }
        self
    }

    /// Override the file sink output mode.
    ///
    /// This has no effect after selecting a stream sink with
    /// [`Self::with_stream_sink`] or [`Self::with_endpoint`].
    pub fn with_mode(mut self, mode: AtofExporterMode) -> Self {
        if let AtofSinkConfig::File(file) = &mut self.sink {
            file.mode = mode;
        }
        self
    }

    /// Override the file sink output filename.
    ///
    /// This has no effect after selecting a stream sink with
    /// [`Self::with_stream_sink`] or [`Self::with_endpoint`].
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        if let AtofSinkConfig::File(file) = &mut self.sink {
            file.filename = filename.into();
        }
        self
    }

    /// Select one stream sink.
    pub fn with_stream_sink(mut self, sink: AtofStreamSinkConfig) -> Self {
        self.sink = AtofSinkConfig::Stream(sink);
        self
    }

    /// Select one stream sink.
    ///
    /// This compatibility spelling replaces the configured file sink; use the
    /// observability plugin's `sinks` array for file-and-stream fan-out.
    pub fn with_endpoint(self, endpoint: AtofEndpointConfig) -> Self {
        self.with_stream_sink(endpoint)
    }

    /// Return the full output path for this config.
    pub fn path(&self) -> Option<PathBuf> {
        match &self.sink {
            AtofSinkConfig::File(file) => Some(file.path()),
            AtofSinkConfig::Stream(_) => None,
        }
    }
}

#[doc(hidden)]
pub(crate) const ATOF_STREAM_ENDPOINT_DISABLED: &str = "atof.stream_endpoint_disabled";
#[doc(hidden)]
pub(crate) const ATOF_STREAM_CONNECTION_FAILED: &str = "atof.stream_connection_failed";
#[doc(hidden)]
pub(crate) const ATOF_STREAM_DELIVERY_FAILED: &str = "atof.stream_delivery_failed";
#[doc(hidden)]
pub(crate) const ATOF_STREAM_EVENTS_DROPPED: &str = "atof.stream_events_dropped";
#[doc(hidden)]
pub(crate) const ATOF_STREAM_FLUSH_FAILED: &str = "atof.stream_flush_failed";
#[doc(hidden)]
pub(crate) const ATOF_STREAM_CLOSE_FAILED: &str = "atof.stream_close_failed";

#[doc(hidden)]
pub(crate) struct AtofExporterState {
    pub writer: Option<BufWriter<File>>,
    pub last_error: Option<String>,
    pub endpoints: Vec<AtofEndpointWorker>,
    pub closed: bool,
}

/// Filesystem-backed Agent Trajectory Observability Format (ATOF) JSONL event exporter.
pub struct AtofExporter {
    host: crate::host::HostCell,
    path: Option<PathBuf>,
    state: Arc<Mutex<AtofExporterState>>,
    diagnostics: ExporterRuntimeDiagnosticsRecorder,
}

impl AtofExporter {
    /// Create a new exporter from config and open its output file.
    pub fn new(config: AtofExporterConfig) -> Result<Self> {
        Self::new_with_plaintext_policy(config, &PlaintextPolicy::LoopbackOnly, None)
    }

    #[doc(hidden)]
    pub(crate) fn new_with_plaintext_policy(
        config: AtofExporterConfig,
        plaintext: &PlaintextPolicy,
        diagnostic_field: Option<String>,
    ) -> Result<Self> {
        let diagnostics = ExporterRuntimeDiagnosticsRecorder::new(diagnostic_field);
        let (path, writer, endpoints) = match config.sink {
            AtofSinkConfig::File(file_sink) => {
                let path = file_sink.path();
                create_private_dir_all(&file_sink.output_directory).map_err(|source| {
                    AtofExporterError::OpenFile {
                        path: path.clone(),
                        source,
                    }
                })?;
                let file = open_file(&file_sink.output_directory, &path, file_sink.mode)?;
                log::info!(
                    target: "nemo_relay.observability",
                    event = "storage_access_validated",
                    plugin_kind = "observability",
                    exporter = "atof",
                    resource_kind = "local_file",
                    permission = "write";
                    "ATOF storage access validated"
                );
                (Some(path), Some(BufWriter::new(file)), Vec::new())
            }
            AtofSinkConfig::Stream(stream_sink) => {
                let workers = start_endpoint_workers(&[stream_sink], plaintext, &diagnostics)?;
                (None, None, workers)
            }
        };
        Ok(Self {
            host: crate::host::HostCell::default(),
            path,
            state: Arc::new(Mutex::new(AtofExporterState {
                writer,
                last_error: None,
                endpoints,
                closed: false,
            })),
            diagnostics,
        })
    }

    /// Return the output JSONL path.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Return delivery diagnostics recorded by a stream sink.
    pub fn runtime_diagnostics(&self) -> ExporterRuntimeDiagnostics {
        self.diagnostics.snapshot()
    }

    /// Return an event subscriber that writes one JSONL record per observed event.
    pub fn subscriber(&self) -> EventSubscriberFn {
        let state = Arc::clone(&self.state);
        Arc::new(move |event: &Event| {
            let Ok(mut state) = state.lock() else {
                return;
            };
            if state.closed || state.last_error.is_some() {
                return;
            }
            let Ok(value) = event.try_to_json_value() else {
                state.last_error = Some("failed to serialize ATOF event".to_string());
                return;
            };
            if let Some(writer) = &mut state.writer
                && let Err(error) = write_json_value(writer, &value)
            {
                state.last_error = Some(error);
                return;
            }
            if state.endpoints.is_empty() {
                return;
            }
            let Ok(raw_json) = serde_json::to_string(&value) else {
                state.last_error = Some("failed to serialize ATOF event".to_string());
                return;
            };
            for endpoint in &state.endpoints {
                endpoint.enqueue(raw_json.clone());
            }
        })
    }

    /// Register this exporter globally under the given subscriber name.
    pub fn register(&self, name: &str) -> Result<()> {
        self.host_ref()
            .register_subscriber(name, self.subscriber())
            .map_err(AtofExporterError::from)?;
        log::info!(
            target: "nemo_relay.observability",
            event = "exporter_registered",
            exporter = "atof",
            subscriber = name;
            "ATOF exporter registered"
        );
        Ok(())
    }

    /// Deregister a global subscriber by name.
    pub fn deregister(&self, name: &str) -> Result<bool> {
        let removed = self
            .host_ref()
            .deregister_subscriber(name)
            .map_err(AtofExporterError::from)?;
        if removed {
            log::info!(
                target: "nemo_relay.observability",
                event = "exporter_deregistered",
                exporter = "atof",
                subscriber = name;
                "ATOF exporter deregistered"
            );
        }
        Ok(removed)
    }

    /// Outside a native subscriber callback, wait for queued subscriber delivery, then flush the
    /// configured file sink or ask the configured stream sink to drain for up to its timeout.
    ///
    /// A re-entrant call does not establish the delivery barrier. A stream timeout is logged and
    /// does not by itself return an error.
    pub fn force_flush(&self) -> Result<()> {
        self.host_ref().flush_subscribers()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| AtofExporterError::LockPoisoned)?;
        if state.closed {
            return stored_failure_result(
                self.path
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<stream>")),
                &state,
            );
        }
        state
            .writer
            .as_mut()
            .map(|writer| writer.flush())
            .transpose()
            .map_err(|source| AtofExporterError::Flush {
                path: self
                    .path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("<stream>")),
                source,
            })?;
        for endpoint in &state.endpoints {
            endpoint.flush();
        }
        stored_failure_result(
            self.path
                .as_deref()
                .unwrap_or_else(|| Path::new("<stream>")),
            &state,
        )
    }

    /// Outside a native subscriber callback, wait for queued subscriber delivery, then flush the
    /// configured file sink or ask the configured stream sink to drain and close up to its timeout.
    ///
    /// A re-entrant call does not establish the delivery barrier. A stream timeout is logged and
    /// does not by itself return an error.
    pub fn shutdown(&self) -> Result<()> {
        self.host_ref().flush_subscribers()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| AtofExporterError::LockPoisoned)?;
        if state.closed {
            return stored_failure_result(
                self.path
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<stream>")),
                &state,
            );
        }
        state.closed = true;
        let flush_result = state
            .writer
            .as_mut()
            .map(|writer| writer.flush())
            .transpose()
            .map_err(|source| AtofExporterError::Flush {
                path: self
                    .path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("<stream>")),
                source,
            });
        for endpoint in &state.endpoints {
            endpoint.close();
        }
        flush_result?;
        let result = stored_failure_result(
            self.path
                .as_deref()
                .unwrap_or_else(|| Path::new("<stream>")),
            &state,
        );
        if result.is_ok() {
            log::info!(
                target: "nemo_relay.observability",
                event = "exporter_shutdown",
                exporter = "atof";
                "ATOF exporter shut down"
            );
        }
        result
    }
}

#[doc(hidden)]
pub(crate) fn default_filename() -> String {
    format!(
        "nemo-relay-events-{}.jsonl",
        Utc::now().format("%Y-%m-%d-%H.%M.%S")
    )
}

#[doc(hidden)]
pub(crate) fn default_output_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[doc(hidden)]
pub(crate) fn default_endpoint_timeout_millis() -> u64 {
    3_000
}

#[doc(hidden)]
pub(crate) fn open_file(root: &Path, path: &Path, mode: AtofExporterMode) -> Result<File> {
    open_private(root, path, matches!(mode, AtofExporterMode::Append)).map_err(|source| {
        AtofExporterError::OpenFile {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[doc(hidden)]
pub(crate) fn write_json_value(
    writer: &mut BufWriter<File>,
    value: &Json,
) -> std::result::Result<(), String> {
    serde_json::to_writer(&mut *writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

#[doc(hidden)]
pub(crate) fn stored_failure_result(path: &Path, state: &AtofExporterState) -> Result<()> {
    if let Some(message) = &state.last_error {
        return Err(AtofExporterError::StoredFailure {
            path: path.to_path_buf(),
            message: message.clone(),
        });
    }
    Ok(())
}

#[doc(hidden)]
pub(crate) enum EndpointMessage {
    Event(String),
    Flush(std_mpsc::Sender<()>),
    Close(std_mpsc::Sender<()>),
}

#[doc(hidden)]
pub(crate) enum NdjsonBodyMessage {
    Event(Vec<u8>),
    Flush(std_mpsc::Sender<()>),
}

impl NdjsonBodyMessage {
    #[doc(hidden)]
    pub(crate) fn acknowledge_if_flush(self) {
        if let Self::Flush(done) = self {
            let _ = done.send(());
        }
    }
}

#[doc(hidden)]
pub(crate) struct AtofEndpointWorker {
    pub sender: tokio::sync::mpsc::UnboundedSender<EndpointMessage>,
    pub timeout: Duration,
    pub index: usize,
    pub diagnostics: EndpointDiagnostics,
}

/// Validated endpoint settings with headers resolved at activation.
#[derive(Debug)]
#[doc(hidden)]
pub(crate) struct ActivatedAtofEndpoint {
    pub config: AtofEndpointConfig,
    pub headers: reqwest::header::HeaderMap,
    pub origin: String,
}

#[derive(Clone)]
#[doc(hidden)]
pub(crate) struct EndpointDiagnostics {
    pub recorder: ExporterRuntimeDiagnosticsRecorder,
    pub transport: AtofEndpointTransport,
    pub origin: String,
}

impl EndpointDiagnostics {
    #[doc(hidden)]
    pub(crate) fn record(&self, code: &str, detail: &str, count: u64) {
        self.recorder.record(
            code,
            format!(
                "ATOF {} stream to {} {detail}",
                self.transport.as_str(),
                self.origin
            ),
            count,
        );
    }

    #[doc(hidden)]
    pub(crate) fn endpoint_disabled(&self, detail: &str) {
        self.record(ATOF_STREAM_ENDPOINT_DISABLED, detail, 1);
    }

    #[doc(hidden)]
    pub(crate) fn connection_failed(&self, detail: &str) {
        self.record(ATOF_STREAM_CONNECTION_FAILED, detail, 1);
    }

    #[doc(hidden)]
    pub(crate) fn delivery_failed(&self, detail: &str, count: u64) {
        self.record(ATOF_STREAM_DELIVERY_FAILED, detail, count);
    }

    #[doc(hidden)]
    pub(crate) fn events_dropped(&self, detail: &str, count: u64) {
        self.record(ATOF_STREAM_EVENTS_DROPPED, detail, count);
    }

    #[doc(hidden)]
    pub(crate) fn flush_failed(&self, detail: &str) {
        self.record(ATOF_STREAM_FLUSH_FAILED, detail, 1);
    }

    #[doc(hidden)]
    pub(crate) fn close_failed(&self, detail: &str) {
        self.record(ATOF_STREAM_CLOSE_FAILED, detail, 1);
    }
}

impl AtofEndpointWorker {
    #[doc(hidden)]
    pub(crate) fn enqueue(&self, raw_json: String) {
        if self.sender.send(EndpointMessage::Event(raw_json)).is_err() {
            self.diagnostics
                .events_dropped("dropped an event because its worker stopped", 1);
        }
    }

    #[doc(hidden)]
    pub(crate) fn flush(&self) {
        let (tx, rx) = std_mpsc::channel();
        let Err(failure) = self.send_and_wait(EndpointMessage::Flush(tx), &rx) else {
            return;
        };
        log::warn!(
            target: "nemo_relay.observability",
            event = "endpoint_flush_failed",
            exporter = "atof",
            endpoint_index = self.index,
            transport = self.diagnostics.transport.as_str(),
            reason = failure.reason();
            "ATOF endpoint flush failed"
        );
        self.diagnostics
            .flush_failed(&failure.detail("a flush", self.timeout));
    }

    #[doc(hidden)]
    pub(crate) fn close(&self) {
        let (tx, rx) = std_mpsc::channel();
        let Err(failure) = self.send_and_wait(EndpointMessage::Close(tx), &rx) else {
            return;
        };
        log::warn!(
            target: "nemo_relay.observability",
            event = "endpoint_close_failed",
            exporter = "atof",
            endpoint_index = self.index,
            transport = self.diagnostics.transport.as_str(),
            reason = failure.reason();
            "ATOF endpoint close failed"
        );
        self.diagnostics
            .close_failed(&failure.detail("close", self.timeout));
    }

    #[doc(hidden)]
    pub(crate) fn send_and_wait(
        &self,
        message: EndpointMessage,
        done: &std_mpsc::Receiver<()>,
    ) -> std::result::Result<(), Unacknowledged> {
        if self.sender.send(message).is_err() {
            return Err(Unacknowledged::WorkerStopped);
        }
        done.recv_timeout(self.timeout)
            .map_err(|error| match error {
                std_mpsc::RecvTimeoutError::Timeout => Unacknowledged::Timeout,
                std_mpsc::RecvTimeoutError::Disconnected => Unacknowledged::WorkerStopped,
            })
    }
}

/// Why the endpoint worker did not acknowledge a flush or close request.
#[doc(hidden)]
pub(crate) enum Unacknowledged {
    WorkerStopped,
    Timeout,
}

impl Unacknowledged {
    #[doc(hidden)]
    pub(crate) fn reason(&self) -> &'static str {
        match self {
            Self::WorkerStopped => "worker_stopped",
            Self::Timeout => "timeout",
        }
    }

    #[doc(hidden)]
    pub(crate) fn detail(&self, operation: &str, timeout: Duration) -> String {
        match self {
            Self::WorkerStopped => {
                format!("did not acknowledge {operation} because its worker stopped")
            }
            Self::Timeout => format!(
                "did not acknowledge {operation} within {} ms",
                timeout.as_millis()
            ),
        }
    }
}

#[doc(hidden)]
pub(crate) fn start_endpoint_workers(
    configs: &[AtofStreamSinkConfig],
    plaintext: &PlaintextPolicy,
    recorder: &ExporterRuntimeDiagnosticsRecorder,
) -> Result<Vec<AtofEndpointWorker>> {
    let mut workers = Vec::with_capacity(configs.len());
    for (index, config) in configs.iter().enumerate() {
        match start_endpoint_worker(index, config.clone(), plaintext, recorder) {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                return Err(AtofExporterError::InvalidEndpoint(format!(
                    "endpoints[{index}]: {error}"
                )));
            }
        }
    }
    Ok(workers)
}

#[doc(hidden)]
pub(crate) fn start_endpoint_worker(
    index: usize,
    config: AtofEndpointConfig,
    plaintext: &PlaintextPolicy,
    recorder: &ExporterRuntimeDiagnosticsRecorder,
) -> Result<AtofEndpointWorker> {
    let endpoint = validate_endpoint_config(config, plaintext)?;
    let timeout = Duration::from_millis(endpoint.config.timeout_millis);
    let transport = endpoint.config.transport;
    let diagnostics = EndpointDiagnostics {
        recorder: recorder.clone(),
        transport,
        origin: endpoint.origin.clone(),
    };
    let worker_diagnostics = diagnostics.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name(format!("nemo-relay-atof-endpoint-{index}"))
        .spawn(move || run_endpoint_worker(index, endpoint, rx, worker_diagnostics))
        .map_err(|error| AtofExporterError::InvalidEndpoint(error.to_string()))?;
    log::info!(
        target: "nemo_relay.observability",
        event = "endpoint_started",
        exporter = "atof",
        endpoint_index = index,
        transport = transport.as_str();
        "ATOF endpoint worker started"
    );
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_resource_access_pending",
        plugin_kind = "observability",
        resource_kind = "stream_endpoint",
        resource_index = index,
        permission = "write";
        "Plugin resource access will be validated on first use"
    );
    Ok(AtofEndpointWorker {
        sender: tx,
        timeout,
        index,
        diagnostics,
    })
}

#[doc(hidden)]
pub(crate) fn validate_endpoint_config(
    config: AtofEndpointConfig,
    plaintext: &PlaintextPolicy,
) -> Result<ActivatedAtofEndpoint> {
    if config.url.trim().is_empty() {
        return Err(AtofExporterError::InvalidEndpoint(
            "endpoint url must be non-empty".to_string(),
        ));
    }
    if config.timeout_millis == 0 {
        return Err(AtofExporterError::InvalidEndpoint(
            "endpoint timeout_millis must be greater than 0".to_string(),
        ));
    }
    let url = reqwest::Url::parse(&config.url)
        .map_err(|error| AtofExporterError::InvalidEndpoint(error.to_string()))?;
    let valid_scheme = match config.transport {
        AtofEndpointTransport::HttpPost | AtofEndpointTransport::Ndjson => {
            matches!(url.scheme(), "http" | "https")
        }
        AtofEndpointTransport::Websocket => matches!(url.scheme(), "ws" | "wss"),
    };
    if !valid_scheme {
        return Err(AtofExporterError::InvalidEndpoint(format!(
            "endpoint {} transport does not support URL scheme {:?}",
            config.transport.as_str(),
            url.scheme()
        )));
    }
    validate_header_files(&config.headers, &config.header_env, &config.header_file)
        .map_err(AtofExporterError::InvalidEndpoint)?;
    let plaintext = if config.allow_insecure {
        &PlaintextPolicy::AnyHost
    } else {
        plaintext
    };
    if has_configured_headers(&config.headers, &config.header_env, &config.header_file) {
        let validation = match config.transport {
            AtofEndpointTransport::HttpPost | AtofEndpointTransport::Ndjson => {
                validate_header_http_endpoint(&config.url, plaintext)
            }
            AtofEndpointTransport::Websocket => {
                validate_header_websocket_endpoint(&config.url, plaintext)
            }
        };
        validation.map_err(AtofExporterError::InvalidEndpoint)?;
    }
    let headers = resolved_header_map(&config.headers, &config.header_env)?;
    let origin = url.origin().ascii_serialization();
    Ok(ActivatedAtofEndpoint {
        config,
        headers,
        origin,
    })
}

#[doc(hidden)]
pub(crate) fn resolved_header_map(
    headers: &HashMap<String, String>,
    header_env: &HashMap<String, String>,
) -> Result<reqwest::header::HeaderMap> {
    let mut out = reqwest::header::HeaderMap::new();
    for (key, value) in headers {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|error| AtofExporterError::InvalidEndpoint(error.to_string()))?;
        let value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|error| AtofExporterError::InvalidEndpoint(error.to_string()))?;
        out.insert(name, value);
    }
    for (key, variable) in header_env {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|error| AtofExporterError::InvalidEndpoint(error.to_string()))?;
        if out.contains_key(&name) {
            return Err(AtofExporterError::InvalidEndpoint(format!(
                "header {key:?} cannot be configured in both headers and header_env"
            )));
        }
        let value = std::env::var(variable).map_err(|_| {
            AtofExporterError::InvalidEndpoint(format!(
                "environment variable {variable:?} for header {key:?} is not set"
            ))
        })?;
        if value.trim().is_empty() {
            return Err(AtofExporterError::InvalidEndpoint(format!(
                "environment variable {variable:?} for header {key:?} is blank"
            )));
        }
        let value = reqwest::header::HeaderValue::from_str(&value)
            .map_err(|error| AtofExporterError::InvalidEndpoint(error.to_string()))?;
        out.insert(name, value);
    }
    Ok(out)
}

#[doc(hidden)]
pub(crate) fn run_endpoint_worker(
    index: usize,
    endpoint: ActivatedAtofEndpoint,
    rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
    diagnostics: EndpointDiagnostics,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_failed",
                exporter = "atof",
                endpoint_index = index,
                reason = "runtime_initialization";
                "ATOF endpoint runtime failed"
            );
            diagnostics.endpoint_disabled(&format!("could not start its runtime: {error}"));
            return;
        }
    };
    runtime.block_on(async move {
        match endpoint.config.transport {
            AtofEndpointTransport::HttpPost => {
                run_http_post_endpoint(index, endpoint, rx, &diagnostics).await
            }
            AtofEndpointTransport::Websocket => {
                run_websocket_endpoint(index, endpoint, rx, &diagnostics).await
            }
            AtofEndpointTransport::Ndjson => {
                run_ndjson_endpoint(index, endpoint, rx, &diagnostics).await
            }
        }
    });
}

#[doc(hidden)]
pub(crate) async fn run_http_post_endpoint(
    index: usize,
    endpoint: ActivatedAtofEndpoint,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
    diagnostics: &EndpointDiagnostics,
) {
    let has_headers = !endpoint.headers.is_empty() || !endpoint.config.header_file.is_empty();
    let mut client_builder = reqwest::Client::builder()
        .timeout(Duration::from_millis(endpoint.config.timeout_millis))
        .default_headers(endpoint.headers);
    if has_headers {
        client_builder = client_builder.redirect(reqwest::redirect::Policy::none());
    }
    let client = match client_builder.build() {
        Ok(client) => client,
        Err(error) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_disabled",
                exporter = "atof",
                endpoint_index = index,
                transport = "http_post",
                reason = "client_initialization";
                "ATOF endpoint disabled"
            );
            diagnostics.endpoint_disabled(&format!(
                "could not build its HTTP client: {}",
                error.without_url()
            ));
            drain_closed(rx, diagnostics).await;
            return;
        }
    };
    let mut access_validated = false;
    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Event(raw_json) => {
                let body = format!("{}\n", endpoint_event_json(&endpoint.config, raw_json));
                let mut request = client.post(&endpoint.config.url);
                let headers = match resolve_header_files(&endpoint.config.header_file) {
                    Ok(headers) => headers,
                    Err(error) => {
                        log::warn!(
                            target: "nemo_relay.observability",
                            event = "endpoint_delivery_failed",
                            exporter = "atof",
                            endpoint_index = index,
                            transport = "http_post",
                            reason = "header_file_resolution";
                            "ATOF endpoint delivery failed"
                        );
                        diagnostics.delivery_failed(
                            &format!("could not resolve a header file: {error}"),
                            1,
                        );
                        continue;
                    }
                };
                for (header, value) in headers {
                    request = request.header(header, value);
                }
                let result = request
                    .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
                    .body(body)
                    .send()
                    .await;
                match result {
                    Ok(response) if response.status().is_success() => {
                        if !access_validated {
                            log::info!(
                                target: "nemo_relay.observability",
                                event = "endpoint_access_validated",
                                plugin_kind = "observability",
                                exporter = "atof",
                                endpoint_index = index,
                                transport = "http_post",
                                permission = "write";
                                "ATOF endpoint access validated"
                            );
                            access_validated = true;
                        }
                    }
                    Ok(response) => {
                        let status = response.status();
                        log_http_error(index, "http_post", status);
                        diagnostics.delivery_failed(&format!("returned status {status}"), 1);
                    }
                    Err(error) => {
                        log::warn!(
                            target: "nemo_relay.observability",
                            event = "endpoint_delivery_failed",
                            exporter = "atof",
                            endpoint_index = index,
                            transport = "http_post",
                            reason = "request_failed";
                            "ATOF endpoint delivery failed"
                        );
                        diagnostics.delivery_failed(
                            &format!("could not complete a request: {}", error.without_url()),
                            1,
                        );
                    }
                }
            }
            EndpointMessage::Flush(done) => {
                let _ = done.send(());
            }
            EndpointMessage::Close(done) => {
                let _ = done.send(());
                return;
            }
        }
    }
}

#[doc(hidden)]
pub(crate) async fn run_websocket_endpoint(
    index: usize,
    endpoint: ActivatedAtofEndpoint,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
    diagnostics: &EndpointDiagnostics,
) {
    let mut pending = std::collections::VecDeque::new();
    let mut retry = WebSocketRetryState::default();
    let mut socket = match connect_websocket(&endpoint).await {
        Ok(socket) => {
            retry.record_recovered(index);
            Some(socket)
        }
        Err(error) => {
            retry.record_failure(index, diagnostics, &format!("could not connect: {error}"));
            None
        }
    };
    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Event(raw_json) => {
                pending.push_back(endpoint_event_json(&endpoint.config, raw_json));
                let _ = drain_websocket_pending(
                    index,
                    &endpoint,
                    &mut socket,
                    &mut pending,
                    &mut retry,
                    diagnostics,
                )
                .await;
            }
            EndpointMessage::Flush(done) => {
                let _ = drain_websocket_pending(
                    index,
                    &endpoint,
                    &mut socket,
                    &mut pending,
                    &mut retry,
                    diagnostics,
                )
                .await;
                let _ = done.send(());
            }
            EndpointMessage::Close(done) => {
                let _ = drain_websocket_pending(
                    index,
                    &endpoint,
                    &mut socket,
                    &mut pending,
                    &mut retry,
                    diagnostics,
                )
                .await;
                if !pending.is_empty() {
                    diagnostics
                        .events_dropped("closed with undelivered events", pending.len() as u64);
                }
                if let Some(mut ws) = socket.take() {
                    let _ = ws.close(None).await;
                }
                let _ = done.send(());
                return;
            }
        }
    }
}

#[doc(hidden)]
pub(crate) type AtofWebSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Default)]
#[doc(hidden)]
pub(crate) struct WebSocketRetryState {
    pub attempts: u64,
    pub warning_emitted: bool,
    pub access_validated: bool,
}

impl WebSocketRetryState {
    #[doc(hidden)]
    pub(crate) fn record_failure(
        &mut self,
        index: usize,
        diagnostics: &EndpointDiagnostics,
        detail: &str,
    ) {
        self.attempts = self.attempts.saturating_add(1);
        diagnostics.connection_failed(detail);
        if !self.warning_emitted {
            log::warn!(
                target: "nemo_relay.observability",
                event = "endpoint_reconnecting",
                exporter = "atof",
                endpoint_index = index,
                transport = "websocket";
                "ATOF endpoint connection failed; reconnecting"
            );
            self.warning_emitted = true;
        }
    }

    #[doc(hidden)]
    pub(crate) fn record_recovered(&mut self, index: usize) {
        if self.attempts > 0 {
            log::info!(
                target: "nemo_relay.observability",
                event = "endpoint_reconnected",
                exporter = "atof",
                endpoint_index = index,
                transport = "websocket",
                attempt_count = self.attempts + 1;
                "ATOF endpoint reconnected"
            );
        }
        if !self.access_validated {
            log::info!(
                target: "nemo_relay.observability",
                event = "endpoint_access_validated",
                plugin_kind = "observability",
                exporter = "atof",
                endpoint_index = index,
                transport = "websocket",
                permission = "connect";
                "ATOF endpoint access validated"
            );
            self.access_validated = true;
        }
        self.attempts = 0;
        self.warning_emitted = false;
    }
}

#[doc(hidden)]
pub(crate) async fn drain_websocket_pending(
    index: usize,
    endpoint: &ActivatedAtofEndpoint,
    socket: &mut Option<AtofWebSocket>,
    pending: &mut std::collections::VecDeque<String>,
    retry: &mut WebSocketRetryState,
    diagnostics: &EndpointDiagnostics,
) -> bool {
    let timeout = Duration::from_millis(endpoint.config.timeout_millis);
    let attempts_before = retry.attempts;
    match tokio::time::timeout(
        timeout,
        drain_websocket_pending_inner(index, endpoint, socket, pending, retry, diagnostics),
    )
    .await
    {
        Ok(drained) => drained,
        Err(_) => {
            if retry.attempts == attempts_before {
                retry.record_failure(
                    index,
                    diagnostics,
                    &format!("did not drain within {} ms", timeout.as_millis()),
                );
            }
            false
        }
    }
}

#[doc(hidden)]
pub(crate) async fn drain_websocket_pending_inner(
    index: usize,
    endpoint: &ActivatedAtofEndpoint,
    socket: &mut Option<AtofWebSocket>,
    pending: &mut std::collections::VecDeque<String>,
    retry: &mut WebSocketRetryState,
    diagnostics: &EndpointDiagnostics,
) -> bool {
    while let Some(raw_json) = pending.front().cloned() {
        if socket.is_none() {
            match connect_websocket(endpoint).await {
                Ok(ws) => {
                    *socket = Some(ws);
                    retry.record_recovered(index);
                }
                Err(error) => {
                    retry.record_failure(
                        index,
                        diagnostics,
                        &format!("could not connect: {error}"),
                    );
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            }
        }

        let Some(ws) = socket.as_mut() else {
            continue;
        };
        match ws
            .send(tokio_tungstenite::tungstenite::Message::Text(
                raw_json.into(),
            ))
            .await
        {
            Ok(()) => {
                pending.pop_front();
            }
            Err(error) => {
                retry.record_failure(
                    index,
                    diagnostics,
                    &format!("could not send: {}", websocket_error_detail(&error)),
                );
                *socket = None;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    true
}

#[doc(hidden)]
pub(crate) async fn connect_websocket(
    endpoint: &ActivatedAtofEndpoint,
) -> std::result::Result<AtofWebSocket, String> {
    let mut request = endpoint
        .config
        .url
        .as_str()
        .into_client_request()
        .map_err(|error| websocket_error_detail(&error))?;
    for (name, value) in endpoint.headers.clone() {
        let Some(name) = name else {
            continue;
        };
        request.headers_mut().insert(name, value);
    }
    for (name, value) in resolve_header_files(&endpoint.config.header_file)
        .map_err(|_| "header_file resolution failed".to_string())?
    {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| "header_file contains an invalid header name".to_string())?;
        let value = reqwest::header::HeaderValue::from_str(&value)
            .map_err(|_| "header_file contains an invalid header value".to_string())?;
        request.headers_mut().insert(name, value);
    }
    tokio::time::timeout(
        Duration::from_millis(endpoint.config.timeout_millis),
        tokio_tungstenite::connect_async(request),
    )
    .await
    .map_err(|_| {
        format!(
            "did not connect within {} ms",
            endpoint.config.timeout_millis
        )
    })?
    .map(|(socket, _)| socket)
    .map_err(|error| websocket_error_detail(&error))
}

/// Describe a WebSocket error without the endpoint URL.
#[doc(hidden)]
pub(crate) fn websocket_error_detail(error: &tokio_tungstenite::tungstenite::Error) -> String {
    use tokio_tungstenite::tungstenite::Error;
    use tokio_tungstenite::tungstenite::error::UrlError;
    match error {
        Error::Url(UrlError::UnableToConnect(_)) => "unable to connect".to_string(),
        Error::ConnectionClosed
        | Error::AlreadyClosed
        | Error::Io(_)
        | Error::Tls(_)
        | Error::Capacity(_)
        | Error::Protocol(_)
        | Error::WriteBufferFull(_)
        | Error::Utf8(_)
        | Error::AttackAttempt
        | Error::Url(_)
        | Error::Http(_)
        | Error::HttpFormat(_) => error.to_string(),
    }
}

#[doc(hidden)]
pub(crate) async fn run_ndjson_endpoint(
    index: usize,
    endpoint: ActivatedAtofEndpoint,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
    diagnostics: &EndpointDiagnostics,
) {
    let client = match build_ndjson_client(&endpoint) {
        Ok(client) => client,
        Err(error) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_disabled",
                exporter = "atof",
                endpoint_index = index,
                transport = "ndjson",
                reason = "client_initialization";
                "ATOF endpoint disabled"
            );
            diagnostics.endpoint_disabled(&format!("could not build its HTTP client: {error}"));
            drain_closed(rx, diagnostics).await;
            return;
        }
    };

    let (body_tx, body, written) = ndjson_body_channel();
    let url = endpoint.config.url.clone();
    let mut upload = tokio::spawn(async move {
        client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
            .body(body)
            .send()
            .await
    })
    .fuse();
    let close_timeout = Duration::from_millis(endpoint.config.timeout_millis);
    let mut queued: u64 = 0;

    loop {
        tokio::select! {
            message = rx.recv() => {
                let Some(message) = message else {
                    return;
                };
                match message {
                    EndpointMessage::Event(raw_json) => {
                        if upload.is_terminated() {
                            reject_ndjson_event(index, "upload_ended", diagnostics);
                        } else if send_ndjson_event(
                            index,
                            &body_tx,
                            endpoint_event_json(&endpoint.config, raw_json),
                            diagnostics,
                        ) {
                            queued = queued.saturating_add(1);
                        }
                    }
                    EndpointMessage::Flush(done) => {
                        if upload.is_terminated() {
                            reject_ndjson_flush(index, "upload_ended", diagnostics);
                            let _ = done.send(());
                        } else {
                            send_ndjson_flush(index, &body_tx, done, diagnostics);
                        }
                    }
                    EndpointMessage::Close(done) => {
                        drop(body_tx);
                        if upload.is_terminated() {
                            let _ = done.send(());
                        } else {
                            finish_ndjson_upload(
                                index,
                                upload,
                                close_timeout,
                                done,
                                diagnostics,
                                queued,
                                &written,
                            )
                            .await;
                        }
                        return;
                    }
                }
            }
            outcome = &mut upload, if !upload.is_terminated() => {
                report_ndjson_upload(index, outcome, diagnostics, queued, &written);
            }
        }
    }
}

#[doc(hidden)]
pub(crate) fn build_ndjson_client(
    endpoint: &ActivatedAtofEndpoint,
) -> std::result::Result<reqwest::Client, String> {
    let mut headers = endpoint.headers.clone();
    for (name, value) in resolve_header_files(&endpoint.config.header_file)
        .map_err(|_| "header_file resolution failed".to_string())?
    {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| "header_file contains an invalid header name".to_string())?;
        let value = reqwest::header::HeaderValue::from_str(&value)
            .map_err(|_| "header_file contains an invalid header value".to_string())?;
        headers.insert(name, value);
    }
    let has_headers = !headers.is_empty();
    let mut client_builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(endpoint.config.timeout_millis))
        .default_headers(headers);
    if has_headers {
        client_builder = client_builder.redirect(reqwest::redirect::Policy::none());
    }
    client_builder
        .build()
        .map_err(|error| format!("client build failed: {error}"))
}

/// Return the upload body channel and a count of events read by the HTTP client.
#[doc(hidden)]
pub(crate) fn ndjson_body_channel() -> (
    tokio::sync::mpsc::UnboundedSender<NdjsonBodyMessage>,
    reqwest::Body,
    Arc<AtomicU64>,
) {
    let (body_tx, body_rx) = tokio::sync::mpsc::unbounded_channel::<NdjsonBodyMessage>();
    let written = Arc::new(AtomicU64::new(0));
    let body_stream = stream::unfold(
        (body_rx, Arc::clone(&written)),
        |(mut body_rx, written)| async move {
            loop {
                match body_rx.recv().await? {
                    NdjsonBodyMessage::Event(bytes) => {
                        written.fetch_add(1, Ordering::Relaxed);
                        return Some((Ok::<_, std::io::Error>(bytes), (body_rx, written)));
                    }
                    NdjsonBodyMessage::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        },
    );
    (body_tx, reqwest::Body::wrap_stream(body_stream), written)
}

#[doc(hidden)]
pub(crate) fn send_ndjson_event(
    index: usize,
    body_tx: &tokio::sync::mpsc::UnboundedSender<NdjsonBodyMessage>,
    raw_json: String,
    diagnostics: &EndpointDiagnostics,
) -> bool {
    if body_tx
        .send(NdjsonBodyMessage::Event(
            format!("{raw_json}\n").into_bytes(),
        ))
        .is_err()
    {
        reject_ndjson_event(index, "body_channel_closed", diagnostics);
        return false;
    }
    true
}

#[doc(hidden)]
pub(crate) fn reject_ndjson_event(
    index: usize,
    reason: &'static str,
    diagnostics: &EndpointDiagnostics,
) {
    log::warn!(
        target: "nemo_relay.observability",
        event = "endpoint_delivery_failed",
        exporter = "atof",
        endpoint_index = index,
        transport = "ndjson",
        reason = reason;
        "ATOF endpoint delivery failed"
    );
    diagnostics.delivery_failed("ended its upload before an event was written", 1);
}

#[doc(hidden)]
pub(crate) fn reject_ndjson_flush(
    index: usize,
    reason: &'static str,
    diagnostics: &EndpointDiagnostics,
) {
    log::warn!(
        target: "nemo_relay.observability",
        event = "endpoint_flush_failed",
        exporter = "atof",
        endpoint_index = index,
        transport = "ndjson",
        reason = reason;
        "ATOF endpoint flush failed"
    );
    diagnostics.flush_failed("ended its upload before a flush was acknowledged");
}

#[doc(hidden)]
pub(crate) fn send_ndjson_flush(
    index: usize,
    body_tx: &tokio::sync::mpsc::UnboundedSender<NdjsonBodyMessage>,
    done: std_mpsc::Sender<()>,
    diagnostics: &EndpointDiagnostics,
) {
    if let Err(error) = body_tx.send(NdjsonBodyMessage::Flush(done)) {
        reject_ndjson_flush(index, "body_channel_closed", diagnostics);
        error.0.acknowledge_if_flush();
    }
}

#[doc(hidden)]
pub(crate) type NdjsonUploadOutcome =
    std::result::Result<reqwest::Result<reqwest::Response>, tokio::task::JoinError>;

#[doc(hidden)]
pub(crate) async fn finish_ndjson_upload(
    index: usize,
    upload: impl std::future::Future<Output = NdjsonUploadOutcome>,
    close_timeout: Duration,
    done: std_mpsc::Sender<()>,
    diagnostics: &EndpointDiagnostics,
    queued: u64,
    written: &AtomicU64,
) {
    match tokio::time::timeout(close_timeout, upload).await {
        Ok(outcome) => report_ndjson_upload(index, outcome, diagnostics, queued, written),
        Err(_) => {
            log::warn!(
                target: "nemo_relay.observability",
                event = "endpoint_close_failed",
                exporter = "atof",
                endpoint_index = index,
                transport = "ndjson",
                reason = "timeout";
                "ATOF endpoint close timed out"
            );
            diagnostics.close_failed(&format!(
                "did not finish its upload within {} ms",
                close_timeout.as_millis()
            ));
        }
    }
    let _ = done.send(());
}

#[doc(hidden)]
pub(crate) fn report_ndjson_upload(
    index: usize,
    outcome: NdjsonUploadOutcome,
    diagnostics: &EndpointDiagnostics,
    queued: u64,
    written: &AtomicU64,
) {
    let failed_upload = match outcome {
        Ok(Ok(response)) if response.status().is_success() => {
            log::info!(
                target: "nemo_relay.observability",
                event = "endpoint_access_validated",
                plugin_kind = "observability",
                exporter = "atof",
                endpoint_index = index,
                transport = "ndjson",
                permission = "write";
                "ATOF endpoint access validated"
            );
            let unwritten = queued.saturating_sub(written.load(Ordering::Relaxed));
            if unwritten > 0 {
                diagnostics.events_dropped("ended upload with events queued", unwritten);
            }
            None
        }
        Ok(Ok(response)) => {
            let status = response.status();
            log_http_error(index, "ndjson", status);
            Some(format!("returned status {status}"))
        }
        Ok(Err(error)) => {
            log::warn!(
                target: "nemo_relay.observability",
                event = "endpoint_delivery_failed",
                exporter = "atof",
                endpoint_index = index,
                transport = "ndjson",
                reason = "upload_failed";
                "ATOF endpoint upload failed"
            );
            Some(format!(
                "could not complete its upload: {}",
                error.without_url()
            ))
        }
        Err(error) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_failed",
                exporter = "atof",
                endpoint_index = index,
                transport = "ndjson",
                reason = "task_failed";
                "ATOF endpoint task failed"
            );
            Some(format!("lost its upload task: {error}"))
        }
    };
    // Queued events are lost when upload fails.
    if let Some(detail) = failed_upload
        && queued > 0
    {
        diagnostics.delivery_failed(&detail, queued);
    }
}

#[doc(hidden)]
pub(crate) async fn drain_closed(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
    diagnostics: &EndpointDiagnostics,
) {
    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Flush(done) => {
                let _ = done.send(());
            }
            EndpointMessage::Close(done) => {
                let _ = done.send(());
                return;
            }
            EndpointMessage::Event(_) => {
                diagnostics.events_dropped("dropped an event because the sink is disabled", 1);
            }
        }
    }
}

#[doc(hidden)]
pub(crate) fn endpoint_event_json(config: &AtofEndpointConfig, raw_json: String) -> String {
    match config.field_name_policy {
        AtofEndpointFieldNamePolicy::Preserve => raw_json,
        AtofEndpointFieldNamePolicy::ReplaceDots => replace_dotted_field_names(&raw_json),
    }
}

#[doc(hidden)]
pub(crate) fn replace_dotted_field_names(raw_json: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<Json>(raw_json) else {
        return raw_json.to_string();
    };
    replace_dotted_value_keys(&mut value);
    serde_json::to_string(&value).unwrap_or_else(|_| raw_json.to_string())
}

#[doc(hidden)]
pub(crate) fn replace_dotted_value_keys(value: &mut Json) {
    match value {
        Json::Object(object) => replace_dotted_object_keys(object),
        Json::Array(items) => {
            for item in items {
                replace_dotted_value_keys(item);
            }
        }
        _ => {}
    }
}

#[doc(hidden)]
pub(crate) fn replace_dotted_object_keys(object: &mut serde_json::Map<String, Json>) {
    let mut old = std::mem::take(object)
        .into_iter()
        .map(|(key, mut value)| {
            replace_dotted_value_keys(&mut value);
            (key, value)
        })
        .collect::<Vec<_>>();
    old.sort_by_key(|(key, _)| !key.contains('.'));

    for (key, value) in old {
        let sanitized_key = key.replace('.', "_");
        let final_key = collision_free_key(object, sanitized_key);
        object.insert(final_key, value);
    }
}

#[doc(hidden)]
pub(crate) fn collision_free_key(object: &serde_json::Map<String, Json>, key: String) -> String {
    if !object.contains_key(&key) {
        return key;
    }
    for suffix in 2.. {
        let candidate = format!("{key}_{suffix}");
        if !object.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search must find a key")
}

#[doc(hidden)]
pub(crate) fn log_http_error(index: usize, transport: &str, status: reqwest::StatusCode) {
    log::warn!(
        target: "nemo_relay.observability",
        event = "endpoint_delivery_failed",
        exporter = "atof",
        endpoint_index = index,
        transport = transport,
        status = status.as_u16();
        "ATOF endpoint returned an unsuccessful status"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

impl AtofExporter {
    pub fn with_host(self, host: Arc<dyn crate::host::ObservabilityHost>) -> Self {
        self.diagnostics.bind_host(host.clone());
        self.host.bind(host);
        self
    }
    pub fn host_ref(&self) -> Arc<dyn crate::host::ObservabilityHost> {
        self.host.get()
    }
}

#[cfg(test)]
#[path = "../tests/unit/atof_tests.rs"]
mod atof_tests;

#[cfg(test)]
#[path = "../tests/unit/atof_detached_tests.rs"]
mod atof_detached_tests;

#[cfg(test)]
#[path = "../tests/unit/atof_streaming_tests.rs"]
mod atof_streaming_tests;
