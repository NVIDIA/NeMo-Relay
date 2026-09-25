// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP file export for NeMo Relay Core.
//!
//! [`OtlpFileSpanExporter`] writes the same `ExportTraceServiceRequest` the
//! network exporters put on the wire to a local file. [`OtlpFileFormat`]
//! selects between the OpenTelemetry file-exporter specification's JSON lines
//! and the Collector's length-delimited protobuf.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};
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
    /// The path is already open with settings the new exporter disagrees with.
    #[error("OTLP output file {path:?} is already open with a different {setting}")]
    ConflictingOpen {
        /// Output path that is already open.
        path: PathBuf,
        /// Setting the two exporters disagree on.
        setting: &'static str,
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
    /// One OTLP/JSON-encoded `ExportTraceServiceRequest` per line: the
    /// serialization the OpenTelemetry file-exporter specification describes,
    /// and the default for that reason.
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
/// Every export is written, flushed and synced before it returns, so a batch
/// the SDK reports as delivered has reached the disk and a run that dies
/// mid-stream leaves a readable prefix.
#[derive(Debug)]
pub struct OtlpFileSpanExporter {
    format: OtlpFileFormat,
    writer: Arc<SharedFileWriter>,
    shutdown: AtomicBool,
    resource: ResourceAttributesWithSchema,
}

/// One open output file, shared by every exporter writing that path.
///
/// A resource-metadata pipeline builds a second exporter from the same
/// configuration. Opening the path twice would truncate the records the base
/// pipeline had already written and then interleave two independent file
/// offsets into the same stream, so the handle is shared instead.
#[derive(Debug)]
struct SharedFileWriter {
    path: PathBuf,
    format: OtlpFileFormat,
    append: bool,
    file: Mutex<BufWriter<File>>,
}

/// Live output files by path.
///
/// The entries are weak: once every exporter for a path is dropped the file is
/// closed, and the next exporter opens it afresh, honouring `append`.
static OPEN_WRITERS: LazyLock<Mutex<HashMap<PathBuf, Weak<SharedFileWriter>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Reports whether `path` names a file inside `root` reached without traversal.
///
/// `Path::starts_with` compares components lexically, so it accepts
/// `<root>/../elsewhere`; every component after `root` has to be a plain name.
/// `open_private` enforces this when it opens the file, but a reused handle
/// never reaches that call and the parent is created before it.
fn is_confined_output(root: &Path, path: &Path) -> bool {
    path.strip_prefix(root).is_ok_and(|relative| {
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
    })
}

fn shared_writer(
    root: &Path,
    path: &Path,
    format: OtlpFileFormat,
    append: bool,
) -> Result<Arc<SharedFileWriter>> {
    // Keyed absolute: two configurations that spell one file differently must
    // share its handle, not open it twice.
    let key = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut registry = OPEN_WRITERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.retain(|_, writer| writer.strong_count() > 0);
    if let Some(existing) = registry.get(&key).and_then(Weak::upgrade) {
        // Confinement is enforced by the open below, which a reused handle
        // skips: without this a sink whose own `output_directory` excludes the
        // path would still be handed the file another sink opened.
        if !is_confined_output(root, path) {
            return Err(OtlpFileExporterError::OpenFile {
                path: path.to_path_buf(),
                source: std::io::Error::other(format!(
                    "observability output '{}' is outside configured directory '{}'",
                    path.display(),
                    root.display()
                )),
            });
        }
        // Sharing a handle silently adopts the settings it was opened with, so
        // a caller that asked for something else is told rather than having its
        // records written in the other encoding.
        for (setting, agrees) in [
            ("format", existing.format == format),
            ("mode", existing.append == append),
        ] {
            if !agrees {
                return Err(OtlpFileExporterError::ConflictingOpen {
                    path: path.to_path_buf(),
                    setting,
                });
            }
        }
        return Ok(existing);
    }
    // `open_private` confines the output to `root`, but only once it is
    // reached: create the parent only when it already lies inside `root`, so a
    // path outside cannot create directories before it is rejected.
    if let Some(parent) = path.parent().filter(|_| is_confined_output(root, path)) {
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
    let shared = Arc::new(SharedFileWriter {
        path: path.to_path_buf(),
        format,
        append,
        file: Mutex::new(BufWriter::new(file)),
    });
    registry.insert(key, Arc::downgrade(&shared));
    Ok(shared)
}

impl OtlpFileSpanExporter {
    /// Creates an exporter writing `path` in `format`.
    ///
    /// `root` confines the output and the file is owner-only, as for the ATOF
    /// and ATIF sinks: a trajectory carries prompt and response content.
    pub fn new(root: &Path, path: &Path, format: OtlpFileFormat, append: bool) -> Result<Self> {
        Ok(Self {
            format,
            writer: shared_writer(root, path, format, append)?,
            shutdown: AtomicBool::new(false),
            resource: ResourceAttributesWithSchema::default(),
        })
    }

    /// Returns the path this exporter writes to.
    pub fn path(&self) -> &Path {
        &self.writer.path
    }

    /// Encodes one export request in the configured format.
    ///
    /// Fully encoded before anything reaches the writer, so a serialization
    /// failure cannot leave a partial record behind.
    fn encode(&self, request: &ExportTraceServiceRequest) -> std::result::Result<Vec<u8>, String> {
        match self.format {
            OtlpFileFormat::JsonLines => {
                // Compact, never pretty: one record per line, so an embedded
                // newline would split a record.
                let mut line = serde_json::to_vec(request)
                    .map_err(|error| format!("failed to serialize OTLP/JSON: {error}"))?;
                line.push(b'\n');
                Ok(line)
            }
            OtlpFileFormat::Proto => {
                let payload = request.encode_to_vec();
                let length =
                    u32::try_from(payload.len()).map_err(|_| frame_too_large(payload.len()))?;
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
        // An empty record is indistinguishable from a lost one.
        if batch.is_empty() {
            return Ok(());
        }
        if self.shutdown.load(Ordering::SeqCst) {
            return Err(OTelSdkError::AlreadyShutdown);
        }
        let resource_spans = group_spans_by_resource_and_scope(batch, &self.resource);
        let record = self
            .encode(&ExportTraceServiceRequest { resource_spans })
            .map_err(OTelSdkError::InternalFailure)?;

        let mut guard = self
            .writer
            .file
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure(lock_poisoned(&self.writer.path)))?;
        let writer = &mut *guard;
        writer
            .write_all(&record)
            .and_then(|()| writer.flush())
            // `flush` only moves the record into the page cache. The batch is
            // reported to the SDK as delivered once this returns, so it is
            // synced to disk first: a host that loses power after an
            // acknowledged export must not lose the record.
            .and_then(|()| writer.get_ref().sync_data())
            .map_err(|error| write_failure(&self.writer.path, &error))
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        let mut guard = self
            .writer
            .file
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure(lock_poisoned(&self.writer.path)))?;
        // The handle stays open for the exporters still sharing it; it is
        // closed when the last of them is dropped. Every accepted record is
        // already on disk, so nothing is lost in the meantime.
        if self.shutdown.swap(true, Ordering::SeqCst) {
            return Err(OTelSdkError::AlreadyShutdown);
        }
        let writer = &mut *guard;
        writer
            .flush()
            .and_then(|()| writer.get_ref().sync_data())
            .map_err(|error| flush_failure(&self.writer.path, &error))
    }

    fn force_flush(&self) -> OTelSdkResult {
        let mut guard = self
            .writer
            .file
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure(lock_poisoned(&self.writer.path)))?;
        let writer = &mut *guard;
        writer
            .flush()
            .and_then(|()| writer.get_ref().sync_data())
            .map_err(|error| flush_failure(&self.writer.path, &error))
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.resource = resource.into();
    }
}

/// Reports a frame the length prefix cannot describe.
fn frame_too_large(length: usize) -> String {
    format!("OTLP export of {length} bytes exceeds the length-delimited frame maximum")
}

/// Reports a failed export write.
fn write_failure(path: &Path, error: &std::io::Error) -> OTelSdkError {
    OTelSdkError::InternalFailure(format!("failed to write OTLP export to {path:?}: {error}"))
}

/// Reports a failed flush, shared by `shutdown_with_timeout` and `force_flush`.
fn flush_failure(path: &Path, error: &std::io::Error) -> OTelSdkError {
    OTelSdkError::InternalFailure(format!(
        "failed to flush OTLP output file {path:?}: {error}"
    ))
}

fn lock_poisoned(path: &Path) -> String {
    format!("the OTLP file exporter state lock for {path:?} was poisoned")
}

// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/observability/otel_file_tests.rs"]
mod tests;
