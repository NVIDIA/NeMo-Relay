// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent Trajectory Observability Format (ATOF) JSONL exporter support for NeMo
//! Flow.
//!
//! The [`AtofExporter`] registers as an event subscriber and writes each
//! canonical NeMo Relay Agent Trajectory Observability Format (ATOF) event as
//! one JSON object per JSONL line.

use std::collections::HashMap;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::time::Duration;

use chrono::Utc;
#[cfg(feature = "atof-streaming")]
use futures_util::{SinkExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
#[cfg(feature = "atof-streaming")]
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

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
    /// Per-endpoint timeout in milliseconds.
    #[serde(default = "default_endpoint_timeout_millis")]
    pub timeout_millis: u64,
    /// Field name transformation policy applied before sending events.
    #[serde(default)]
    pub field_name_policy: AtofEndpointFieldNamePolicy,
}

impl AtofStreamSinkConfig {
    /// Create a streaming endpoint with defaults.
    pub fn new(url: impl Into<String>, transport: AtofEndpointTransport) -> Self {
        Self {
            url: url.into(),
            transport,
            headers: HashMap::new(),
            header_env: HashMap::new(),
            timeout_millis: default_endpoint_timeout_millis(),
            field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        }
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

struct AtofExporterState {
    writer: Option<BufWriter<File>>,
    last_error: Option<String>,
    endpoints: Vec<AtofEndpointWorker>,
    closed: bool,
}

/// Filesystem-backed Agent Trajectory Observability Format (ATOF) JSONL event exporter.
pub struct AtofExporter {
    path: Option<PathBuf>,
    state: Arc<Mutex<AtofExporterState>>,
}

impl AtofExporter {
    /// Create a new exporter from config and open its output file.
    pub fn new(config: AtofExporterConfig) -> Result<Self> {
        let (path, writer, endpoints) = match config.sink {
            AtofSinkConfig::File(file_sink) => {
                let path = file_sink.path();
                create_dir_all(&file_sink.output_directory).map_err(|source| {
                    AtofExporterError::OpenFile {
                        path: path.clone(),
                        source,
                    }
                })?;
                let file = open_file(&path, file_sink.mode)?;
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
                let workers = start_endpoint_workers(&[stream_sink])?;
                (None, None, workers)
            }
        };
        Ok(Self {
            path,
            state: Arc::new(Mutex::new(AtofExporterState {
                writer,
                last_error: None,
                endpoints,
                closed: false,
            })),
        })
    }

    /// Return the output JSONL path.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
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
        register_subscriber(name, self.subscriber()).map_err(AtofExporterError::from)?;
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
        let removed = deregister_subscriber(name).map_err(AtofExporterError::from)?;
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

    /// Flush the underlying file and drain queued endpoint events.
    pub fn force_flush(&self) -> Result<()> {
        flush_subscribers()?;
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

    /// Shut down the exporter by flushing buffered data and closing endpoints.
    pub fn shutdown(&self) -> Result<()> {
        flush_subscribers()?;
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
        stored_failure_result(
            self.path
                .as_deref()
                .unwrap_or_else(|| Path::new("<stream>")),
            &state,
        )
    }
}

fn default_filename() -> String {
    format!(
        "nemo-relay-events-{}.jsonl",
        Utc::now().format("%Y-%m-%d-%H.%M.%S")
    )
}

fn default_output_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn default_endpoint_timeout_millis() -> u64 {
    3_000
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

fn write_json_value(writer: &mut BufWriter<File>, value: &Json) -> std::result::Result<(), String> {
    serde_json::to_writer(&mut *writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

fn stored_failure_result(path: &Path, state: &AtofExporterState) -> Result<()> {
    if let Some(message) = &state.last_error {
        return Err(AtofExporterError::StoredFailure {
            path: path.to_path_buf(),
            message: message.clone(),
        });
    }
    Ok(())
}

#[cfg_attr(not(feature = "atof-streaming"), allow(dead_code))]
enum EndpointMessage {
    Event(String),
    Flush(std_mpsc::Sender<()>),
    Close(std_mpsc::Sender<()>),
}

#[cfg(feature = "atof-streaming")]
enum NdjsonBodyMessage {
    Event(Vec<u8>),
    Flush(std_mpsc::Sender<()>),
}

#[cfg(feature = "atof-streaming")]
impl NdjsonBodyMessage {
    fn acknowledge_if_flush(self) {
        if let Self::Flush(done) = self {
            let _ = done.send(());
        }
    }
}

struct AtofEndpointWorker {
    sender: tokio::sync::mpsc::UnboundedSender<EndpointMessage>,
    timeout: Duration,
    index: usize,
    transport: AtofEndpointTransport,
}

impl AtofEndpointWorker {
    fn enqueue(&self, raw_json: String) {
        let _ = self.sender.send(EndpointMessage::Event(raw_json));
    }

    fn flush(&self) {
        let (tx, rx) = std_mpsc::channel();
        if self.sender.send(EndpointMessage::Flush(tx)).is_ok()
            && rx.recv_timeout(self.timeout).is_err()
        {
            log::warn!(
                target: "nemo_relay.observability",
                event = "endpoint_flush_failed",
                exporter = "atof",
                endpoint_index = self.index,
                transport = self.transport.as_str(),
                reason = "timeout";
                "ATOF endpoint flush timed out"
            );
        }
    }

    fn close(&self) {
        let (tx, rx) = std_mpsc::channel();
        if self.sender.send(EndpointMessage::Close(tx)).is_ok()
            && rx.recv_timeout(self.timeout).is_err()
        {
            log::warn!(
                target: "nemo_relay.observability",
                event = "endpoint_close_failed",
                exporter = "atof",
                endpoint_index = self.index,
                transport = self.transport.as_str(),
                reason = "timeout";
                "ATOF endpoint close timed out"
            );
        }
    }
}

#[cfg(feature = "atof-streaming")]
fn start_endpoint_workers(configs: &[AtofStreamSinkConfig]) -> Result<Vec<AtofEndpointWorker>> {
    let mut workers = Vec::with_capacity(configs.len());
    for (index, config) in configs.iter().enumerate() {
        match start_endpoint_worker(index, config.clone()) {
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

#[cfg(not(feature = "atof-streaming"))]
fn start_endpoint_workers(configs: &[AtofEndpointConfig]) -> Result<Vec<AtofEndpointWorker>> {
    if configs.is_empty() {
        return Ok(Vec::new());
    }
    let message = "ATOF streaming endpoints are not supported in this build".to_string();
    Err(AtofExporterError::InvalidEndpoint(message))
}

#[cfg(feature = "atof-streaming")]
fn start_endpoint_worker(index: usize, config: AtofEndpointConfig) -> Result<AtofEndpointWorker> {
    validate_endpoint_config(&config)?;
    let timeout = Duration::from_millis(config.timeout_millis);
    let transport = config.transport;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name(format!("nemo-relay-atof-endpoint-{index}"))
        .spawn(move || run_endpoint_worker(index, config, rx))
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
        transport,
    })
}

#[cfg(feature = "atof-streaming")]
fn validate_endpoint_config(config: &AtofEndpointConfig) -> Result<()> {
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
    resolved_header_map(&config.headers, &config.header_env)?;
    Ok(())
}

#[cfg(feature = "atof-streaming")]
fn resolved_header_map(
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

#[cfg(feature = "atof-streaming")]
fn run_endpoint_worker(
    index: usize,
    config: AtofEndpointConfig,
    rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_failed",
                exporter = "atof",
                endpoint_index = index,
                reason = "runtime_initialization";
                "ATOF endpoint runtime failed"
            );
            return;
        }
    };
    runtime.block_on(async move {
        match config.transport {
            AtofEndpointTransport::HttpPost => run_http_post_endpoint(index, config, rx).await,
            AtofEndpointTransport::Websocket => run_websocket_endpoint(index, config, rx).await,
            AtofEndpointTransport::Ndjson => run_ndjson_endpoint(index, config, rx).await,
        }
    });
}

#[cfg(feature = "atof-streaming")]
async fn run_http_post_endpoint(
    index: usize,
    config: AtofEndpointConfig,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(config.timeout_millis))
        .default_headers(
            match resolved_header_map(&config.headers, &config.header_env) {
                Ok(headers) => headers,
                Err(_) => {
                    log::error!(
                        target: "nemo_relay.observability",
                        event = "endpoint_disabled",
                        exporter = "atof",
                        endpoint_index = index,
                        transport = "http_post",
                        reason = "invalid_headers";
                        "ATOF endpoint disabled"
                    );
                    drain_closed(rx).await;
                    return;
                }
            },
        )
        .build()
    {
        Ok(client) => client,
        Err(_) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_disabled",
                exporter = "atof",
                endpoint_index = index,
                transport = "http_post",
                reason = "client_initialization";
                "ATOF endpoint disabled"
            );
            drain_closed(rx).await;
            return;
        }
    };
    let mut access_validated = false;
    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Event(raw_json) => {
                let body = format!("{}\n", endpoint_event_json(&config, raw_json));
                let result = client
                    .post(&config.url)
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
                    Ok(response) => log_http_error(index, "http_post", response),
                    Err(_) => log::warn!(
                        target: "nemo_relay.observability",
                        event = "endpoint_delivery_failed",
                        exporter = "atof",
                        endpoint_index = index,
                        transport = "http_post",
                        reason = "request_failed";
                        "ATOF endpoint delivery failed"
                    ),
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

#[cfg(feature = "atof-streaming")]
async fn run_websocket_endpoint(
    index: usize,
    config: AtofEndpointConfig,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
) {
    let mut pending = std::collections::VecDeque::new();
    let mut retry = WebSocketRetryState::default();
    let mut socket = match connect_websocket(&config).await {
        Ok(socket) => {
            retry.record_recovered(index);
            Some(socket)
        }
        Err(_) => {
            retry.record_failure(index);
            None
        }
    };
    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Event(raw_json) => {
                pending.push_back(endpoint_event_json(&config, raw_json));
                let _ =
                    drain_websocket_pending(index, &config, &mut socket, &mut pending, &mut retry)
                        .await;
            }
            EndpointMessage::Flush(done) => {
                let _ =
                    drain_websocket_pending(index, &config, &mut socket, &mut pending, &mut retry)
                        .await;
                let _ = done.send(());
            }
            EndpointMessage::Close(done) => {
                let _ =
                    drain_websocket_pending(index, &config, &mut socket, &mut pending, &mut retry)
                        .await;
                if let Some(mut ws) = socket.take() {
                    let _ = ws.close(None).await;
                }
                let _ = done.send(());
                return;
            }
        }
    }
}

#[cfg(feature = "atof-streaming")]
type AtofWebSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[cfg(feature = "atof-streaming")]
#[derive(Default)]
struct WebSocketRetryState {
    attempts: u64,
    warning_emitted: bool,
    access_validated: bool,
}

#[cfg(feature = "atof-streaming")]
impl WebSocketRetryState {
    fn record_failure(&mut self, index: usize) {
        self.attempts = self.attempts.saturating_add(1);
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

    fn record_recovered(&mut self, index: usize) {
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

#[cfg(feature = "atof-streaming")]
async fn drain_websocket_pending(
    index: usize,
    config: &AtofEndpointConfig,
    socket: &mut Option<AtofWebSocket>,
    pending: &mut std::collections::VecDeque<String>,
    retry: &mut WebSocketRetryState,
) -> bool {
    let timeout = Duration::from_millis(config.timeout_millis);
    let attempts_before = retry.attempts;
    match tokio::time::timeout(
        timeout,
        drain_websocket_pending_inner(index, config, socket, pending, retry),
    )
    .await
    {
        Ok(drained) => drained,
        Err(_) => {
            if retry.attempts == attempts_before {
                retry.record_failure(index);
            }
            false
        }
    }
}

#[cfg(feature = "atof-streaming")]
async fn drain_websocket_pending_inner(
    index: usize,
    config: &AtofEndpointConfig,
    socket: &mut Option<AtofWebSocket>,
    pending: &mut std::collections::VecDeque<String>,
    retry: &mut WebSocketRetryState,
) -> bool {
    while let Some(raw_json) = pending.front().cloned() {
        if socket.is_none() {
            match connect_websocket(config).await {
                Ok(ws) => {
                    *socket = Some(ws);
                    retry.record_recovered(index);
                }
                Err(_) => {
                    retry.record_failure(index);
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
            Err(_) => {
                retry.record_failure(index);
                *socket = None;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    true
}

#[cfg(feature = "atof-streaming")]
async fn connect_websocket(
    config: &AtofEndpointConfig,
) -> std::result::Result<AtofWebSocket, String> {
    let mut request = config
        .url
        .as_str()
        .into_client_request()
        .map_err(|error| error.to_string())?;
    let headers = resolved_header_map(&config.headers, &config.header_env)
        .map_err(|error| error.to_string())?;
    for (name, value) in headers {
        let Some(name) = name else {
            continue;
        };
        request.headers_mut().insert(name, value);
    }
    tokio::time::timeout(
        Duration::from_millis(config.timeout_millis),
        tokio_tungstenite::connect_async(request),
    )
    .await
    .map_err(|error| error.to_string())?
    .map(|(socket, _)| socket)
    .map_err(|error| error.to_string())
}

#[cfg(feature = "atof-streaming")]
async fn run_ndjson_endpoint(
    index: usize,
    config: AtofEndpointConfig,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>,
) {
    let client = match build_ndjson_client(&config) {
        Ok(client) => client,
        Err(_) => {
            log::error!(
                target: "nemo_relay.observability",
                event = "endpoint_disabled",
                exporter = "atof",
                endpoint_index = index,
                transport = "ndjson",
                reason = "client_initialization";
                "ATOF endpoint disabled"
            );
            drain_closed(rx).await;
            return;
        }
    };

    let (body_tx, body) = ndjson_body_channel();
    let url = config.url.clone();
    let request = tokio::spawn(async move {
        client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
            .body(body)
            .send()
            .await
    });
    let close_timeout = Duration::from_millis(config.timeout_millis);

    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Event(raw_json) => {
                send_ndjson_event(index, &body_tx, endpoint_event_json(&config, raw_json))
            }
            EndpointMessage::Flush(done) => send_ndjson_flush(index, &body_tx, done),
            EndpointMessage::Close(done) => {
                drop(body_tx);
                finish_ndjson_upload(index, request, close_timeout, done).await;
                return;
            }
        }
    }
}

#[cfg(feature = "atof-streaming")]
fn build_ndjson_client(
    config: &AtofEndpointConfig,
) -> std::result::Result<reqwest::Client, String> {
    let headers = resolved_header_map(&config.headers, &config.header_env)
        .map_err(|error| format!("disabled: {error}"))?;
    reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(config.timeout_millis))
        .default_headers(headers)
        .build()
        .map_err(|error| format!("client build failed: {error}"))
}

#[cfg(feature = "atof-streaming")]
fn ndjson_body_channel() -> (
    tokio::sync::mpsc::UnboundedSender<NdjsonBodyMessage>,
    reqwest::Body,
) {
    let (body_tx, body_rx) = tokio::sync::mpsc::unbounded_channel::<NdjsonBodyMessage>();
    let body_stream = stream::unfold(body_rx, |mut body_rx| async {
        loop {
            match body_rx.recv().await? {
                NdjsonBodyMessage::Event(bytes) => {
                    return Some((Ok::<_, std::io::Error>(bytes), body_rx));
                }
                NdjsonBodyMessage::Flush(done) => {
                    let _ = done.send(());
                }
            }
        }
    });
    (body_tx, reqwest::Body::wrap_stream(body_stream))
}

#[cfg(feature = "atof-streaming")]
fn send_ndjson_event(
    index: usize,
    body_tx: &tokio::sync::mpsc::UnboundedSender<NdjsonBodyMessage>,
    raw_json: String,
) {
    if body_tx
        .send(NdjsonBodyMessage::Event(
            format!("{raw_json}\n").into_bytes(),
        ))
        .is_err()
    {
        log::warn!(
            target: "nemo_relay.observability",
            event = "endpoint_delivery_failed",
            exporter = "atof",
            endpoint_index = index,
            transport = "ndjson",
            reason = "body_channel_closed";
            "ATOF endpoint delivery failed"
        );
    }
}

#[cfg(feature = "atof-streaming")]
fn send_ndjson_flush(
    index: usize,
    body_tx: &tokio::sync::mpsc::UnboundedSender<NdjsonBodyMessage>,
    done: std_mpsc::Sender<()>,
) {
    if let Err(error) = body_tx.send(NdjsonBodyMessage::Flush(done)) {
        log::warn!(
            target: "nemo_relay.observability",
            event = "endpoint_flush_failed",
            exporter = "atof",
            endpoint_index = index,
            transport = "ndjson",
            reason = "body_channel_closed";
            "ATOF endpoint flush failed"
        );
        error.0.acknowledge_if_flush();
    }
}

#[cfg(feature = "atof-streaming")]
async fn finish_ndjson_upload(
    index: usize,
    request: tokio::task::JoinHandle<reqwest::Result<reqwest::Response>>,
    close_timeout: Duration,
    done: std_mpsc::Sender<()>,
) {
    match tokio::time::timeout(close_timeout, request).await {
        Ok(Ok(Ok(response))) if response.status().is_success() => log::info!(
            target: "nemo_relay.observability",
            event = "endpoint_access_validated",
            plugin_kind = "observability",
            exporter = "atof",
            endpoint_index = index,
            transport = "ndjson",
            permission = "write";
            "ATOF endpoint access validated"
        ),
        Ok(Ok(Ok(response))) => log_http_error(index, "ndjson", response),
        Ok(Ok(Err(_))) => log::warn!(
            target: "nemo_relay.observability",
            event = "endpoint_delivery_failed",
            exporter = "atof",
            endpoint_index = index,
            transport = "ndjson",
            reason = "upload_failed";
            "ATOF endpoint upload failed"
        ),
        Ok(Err(_)) => log::error!(
            target: "nemo_relay.observability",
            event = "endpoint_failed",
            exporter = "atof",
            endpoint_index = index,
            transport = "ndjson",
            reason = "task_failed";
            "ATOF endpoint task failed"
        ),
        Err(_) => log::warn!(
            target: "nemo_relay.observability",
            event = "endpoint_close_failed",
            exporter = "atof",
            endpoint_index = index,
            transport = "ndjson",
            reason = "timeout";
            "ATOF endpoint close timed out"
        ),
    }
    let _ = done.send(());
}

#[cfg(feature = "atof-streaming")]
async fn drain_closed(mut rx: tokio::sync::mpsc::UnboundedReceiver<EndpointMessage>) {
    while let Some(message) = rx.recv().await {
        match message {
            EndpointMessage::Flush(done) => {
                let _ = done.send(());
            }
            EndpointMessage::Close(done) => {
                let _ = done.send(());
                return;
            }
            EndpointMessage::Event(_) => {}
        }
    }
}

#[cfg(feature = "atof-streaming")]
fn endpoint_event_json(config: &AtofEndpointConfig, raw_json: String) -> String {
    match config.field_name_policy {
        AtofEndpointFieldNamePolicy::Preserve => raw_json,
        AtofEndpointFieldNamePolicy::ReplaceDots => replace_dotted_field_names(&raw_json),
    }
}

#[cfg(feature = "atof-streaming")]
fn replace_dotted_field_names(raw_json: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<Json>(raw_json) else {
        return raw_json.to_string();
    };
    replace_dotted_value_keys(&mut value);
    serde_json::to_string(&value).unwrap_or_else(|_| raw_json.to_string())
}

#[cfg(feature = "atof-streaming")]
fn replace_dotted_value_keys(value: &mut Json) {
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

#[cfg(feature = "atof-streaming")]
fn replace_dotted_object_keys(object: &mut serde_json::Map<String, Json>) {
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

#[cfg(feature = "atof-streaming")]
fn collision_free_key(object: &serde_json::Map<String, Json>, key: String) -> String {
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

#[cfg(feature = "atof-streaming")]
fn log_http_error(index: usize, transport: &str, response: reqwest::Response) {
    let status = response.status();
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

#[cfg(test)]
#[path = "../../tests/unit/observability/atof_tests.rs"]
mod tests;
