// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP file export for NeMo Relay Core.
//!
//! [`OtlpFileSpanExporter`] is a [`SpanExporter`] that writes the same
//! `ExportTraceServiceRequest` the OTLP network exporters put on the wire to a
//! local file instead. It exists for consumers that treat a trajectory as an
//! artifact rather than as telemetry -- evaluation harnesses, offline replay,
//! and any environment with no collector to export to.
//!
//! The default [`OtlpFileFormat::JsonLines`] follows the OpenTelemetry Protocol
//! File Exporter specification: one OTLP/JSON-encoded request per line.
//! [`OtlpFileFormat::Proto`] writes each request length-delimited, matching the
//! OpenTelemetry Collector file exporter's `format: proto` layout, for
//! consumers that would rather not pay JSON's size and parse cost.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::transform::common::tonic::ResourceAttributesWithSchema;
use opentelemetry_proto::transform::trace::tonic::group_spans_by_resource_and_scope;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use prost::Message;
use serde::{Deserialize, Serialize};

use super::private_file::{create_private_dir_all, open_private};

/// Result type for the OTLP file exporter.
pub type Result<T> = std::result::Result<T, OtlpFileExporterError>;

/// Errors produced while configuring or operating the OTLP file exporter.
#[derive(Debug, thiserror::Error)]
pub enum OtlpFileExporterError {
    /// Failed to create the directory containing the output file.
    #[error("failed to create OTLP output directory {path:?}: {source}")]
    CreateDirectory {
        /// Directory that could not be created.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Failed to open the output file.
    #[error("failed to open OTLP output file {path:?}: {source}")]
    OpenFile {
        /// Output path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

/// On-disk encoding used by [`OtlpFileSpanExporter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum OtlpFileFormat {
    /// One OTLP/JSON-encoded `ExportTraceServiceRequest` per line.
    ///
    /// This is the serialization described by the OpenTelemetry Protocol File
    /// Exporter specification, and the default for that reason.
    #[default]
    JsonLines,
    /// Length-delimited OTLP protobuf: a big-endian `u32` byte count followed
    /// by that many bytes of encoded `ExportTraceServiceRequest`.
    Proto,
}

impl OtlpFileFormat {
    /// Returns the conventional file extension for the format.
    pub fn extension(self) -> &'static str {
        match self {
            Self::JsonLines => "jsonl",
            Self::Proto => "otlp.pb",
        }
    }
}

/// Writes exported spans to a local file as OTLP.
///
/// The exporter owns its writer and closes it on shutdown. Every export is
/// flushed before it returns: a batch that the SDK reports as delivered is
/// durable on disk, so a run that dies between batches still leaves a readable
/// prefix rather than an empty file.
#[derive(Debug)]
pub struct OtlpFileSpanExporter {
    path: PathBuf,
    format: OtlpFileFormat,
    writer: Mutex<Option<BufWriter<File>>>,
    resource: ResourceAttributesWithSchema,
}

impl OtlpFileSpanExporter {
    /// Creates an exporter writing `path` in `format`.
    ///
    /// `root` confines the output the same way the ATOF and ATIF file sinks are
    /// confined, and the file is created with owner-only permissions because a
    /// trajectory carries prompt and response content.
    pub fn new(root: &Path, path: &Path, format: OtlpFileFormat, append: bool) -> Result<Self> {
        if let Some(parent) = path.parent() {
            create_private_dir_all(parent).map_err(|source| {
                OtlpFileExporterError::CreateDirectory {
                    path: parent.to_path_buf(),
                    source,
                }
            })?;
        }
        let file =
            open_private(root, path, append).map_err(|source| OtlpFileExporterError::OpenFile {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(Self {
            path: path.to_path_buf(),
            format,
            writer: Mutex::new(Some(BufWriter::new(file))),
            resource: ResourceAttributesWithSchema::default(),
        })
    }

    /// Returns the path this exporter writes to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Encodes one export request in the configured format.
    ///
    /// Kept separate from the write so the framing is testable without a file,
    /// and so a serialization failure never leaves a partial record behind: the
    /// record is fully encoded before any of it reaches the writer.
    fn encode(&self, request: &ExportTraceServiceRequest) -> std::result::Result<Vec<u8>, String> {
        match self.format {
            OtlpFileFormat::JsonLines => {
                // Compact, never pretty: the specification's file layout is one
                // record per line, so an embedded newline would split a record.
                let mut line = serde_json::to_vec(request)
                    .map_err(|error| format!("failed to serialize OTLP/JSON: {error}"))?;
                line.push(b'\n');
                Ok(line)
            }
            OtlpFileFormat::Proto => {
                let payload = request.encode_to_vec();
                let length = u32::try_from(payload.len()).map_err(|_| {
                    format!(
                        "OTLP export of {} bytes exceeds the length-delimited frame maximum",
                        payload.len()
                    )
                })?;
                let mut frame = Vec::with_capacity(payload.len() + 4);
                frame.extend_from_slice(&length.to_be_bytes());
                frame.extend_from_slice(&payload);
                Ok(frame)
            }
        }
    }
}

impl SpanExporter for OtlpFileSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        // An empty batch would otherwise encode to a record with no spans,
        // which a reader cannot distinguish from a lost one.
        if batch.is_empty() {
            return Ok(());
        }
        let resource_spans = group_spans_by_resource_and_scope(batch, &self.resource);
        let record = self
            .encode(&ExportTraceServiceRequest { resource_spans })
            .map_err(OTelSdkError::InternalFailure)?;

        let mut guard = self
            .writer
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure(lock_poisoned(&self.path)))?;
        let writer = guard.as_mut().ok_or(OTelSdkError::AlreadyShutdown)?;
        writer
            .write_all(&record)
            .and_then(|()| writer.flush())
            .map_err(|error| {
                OTelSdkError::InternalFailure(format!(
                    "failed to write OTLP export to {:?}: {error}",
                    self.path
                ))
            })
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        let mut guard = self
            .writer
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure(lock_poisoned(&self.path)))?;
        let Some(mut writer) = guard.take() else {
            return Err(OTelSdkError::AlreadyShutdown);
        };
        writer.flush().map_err(|error| {
            OTelSdkError::InternalFailure(format!(
                "failed to flush OTLP output file {:?}: {error}",
                self.path
            ))
        })
    }

    fn force_flush(&self) -> OTelSdkResult {
        let mut guard = self
            .writer
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure(lock_poisoned(&self.path)))?;
        // A flush after shutdown is not an error: every record the exporter
        // accepted is already durable, so there is nothing left to fail on.
        let Some(writer) = guard.as_mut() else {
            return Ok(());
        };
        writer.flush().map_err(|error| {
            OTelSdkError::InternalFailure(format!(
                "failed to flush OTLP output file {:?}: {error}",
                self.path
            ))
        })
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.resource = resource.into();
    }
}

fn lock_poisoned(path: &Path) -> String {
    format!("the OTLP file exporter state lock for {path:?} was poisoned")
}

// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/observability/otel_file_tests.rs"]
mod tests;
