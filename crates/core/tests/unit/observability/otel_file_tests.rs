// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the OTLP file exporter.
//!
//! The round-trip assertions decode with `prost` and `serde_json` directly
//! rather than through this module's own encoder, so a writer and reader that
//! agree only with each other cannot pass.

use super::*;
use opentelemetry::KeyValue;
use opentelemetry::trace::{Tracer, TracerProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::Value as Json;
use std::fs;

fn sample_spans(names: &[&str]) -> Vec<SpanData> {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let tracer = provider.tracer("nemo-relay-otlp-file-tests");
    for name in names {
        tracer.in_span(name.to_string(), |_cx| {});
    }
    provider.force_flush().unwrap();
    exporter.get_finished_spans().unwrap()
}

fn exporter_at(
    directory: &Path,
    filename: &str,
    format: OtlpFileFormat,
    append: bool,
) -> OtlpFileSpanExporter {
    OtlpFileSpanExporter::new(directory, &directory.join(filename), format, append).unwrap()
}

/// Reads length-delimited protobuf frames without using the exporter's encoder.
fn read_proto_frames(path: &Path) -> Vec<ExportTraceServiceRequest> {
    let bytes = fs::read(path).unwrap();
    let mut requests = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let frame = &bytes[offset..offset + length];
        offset += length;
        requests.push(ExportTraceServiceRequest::decode(frame).unwrap());
    }
    requests
}

fn read_json_lines(path: &Path) -> Vec<Json> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn span_names(request: &ExportTraceServiceRequest) -> Vec<String> {
    request
        .resource_spans
        .iter()
        .flat_map(|resource| resource.scope_spans.iter())
        .flat_map(|scope| scope.spans.iter())
        .map(|span| span.name.clone())
        .collect()
}

#[tokio::test]
async fn proto_export_round_trips_through_an_independent_decoder() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.otlp.pb",
        OtlpFileFormat::Proto,
        false,
    );
    let spans = sample_spans(&["outer", "inner"]);

    exporter.export(spans).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let requests = read_proto_frames(&directory.path().join("trace.otlp.pb"));
    assert_eq!(requests.len(), 1, "one export is one frame");
    let mut names = span_names(&requests[0]);
    names.sort();
    assert_eq!(names, vec!["inner".to_string(), "outer".to_string()]);
}

#[tokio::test]
async fn proto_frames_are_length_prefixed_big_endian() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.otlp.pb",
        OtlpFileFormat::Proto,
        false,
    );
    exporter.export(sample_spans(&["only"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let bytes = fs::read(directory.path().join("trace.otlp.pb")).unwrap();
    let declared = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    assert_eq!(
        declared,
        bytes.len() - 4,
        "the prefix must describe exactly the bytes that follow it"
    );
}

#[tokio::test]
async fn each_proto_export_appends_one_frame() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.otlp.pb",
        OtlpFileFormat::Proto,
        false,
    );
    exporter.export(sample_spans(&["first"])).await.unwrap();
    exporter.export(sample_spans(&["second"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let requests = read_proto_frames(&directory.path().join("trace.otlp.pb"));
    assert_eq!(requests.len(), 2);
    assert_eq!(span_names(&requests[0]), vec!["first".to_string()]);
    assert_eq!(span_names(&requests[1]), vec!["second".to_string()]);
}

#[tokio::test]
async fn json_lines_export_is_one_json_object_per_export() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter.export(sample_spans(&["first"])).await.unwrap();
    exporter.export(sample_spans(&["second"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let lines = read_json_lines(&directory.path().join("trace.jsonl"));
    assert_eq!(
        lines.len(),
        2,
        "one line per export, per the file-exporter spec"
    );
    assert!(lines.iter().all(|line| line.is_object()));
}

#[tokio::test]
async fn json_lines_encode_span_identifiers_as_hex() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter.export(sample_spans(&["only"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let lines = read_json_lines(&directory.path().join("trace.jsonl"));
    let span = lines[0]
        .pointer("/resourceSpans/0/scopeSpans/0/spans/0")
        .expect("OTLP/JSON uses camelCase member names");
    let trace_id = span.pointer("/traceId").unwrap().as_str().unwrap();
    let span_id = span.pointer("/spanId").unwrap().as_str().unwrap();
    // OTLP/JSON encodes these two fields as hex rather than the base64 the
    // protobuf JSON mapping would otherwise give a bytes field.
    assert_eq!(trace_id.len(), 32);
    assert_eq!(span_id.len(), 16);
    assert!(trace_id.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(span_id.chars().all(|c| c.is_ascii_hexdigit()));
}

#[tokio::test]
async fn json_lines_records_never_contain_an_embedded_newline() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    // A span name carrying a newline would split the record if the encoder
    // ever pretty-printed.
    let exporter_spans = sample_spans(&["outer\nnewline"]);
    exporter.export(exporter_spans).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let contents = fs::read_to_string(directory.path().join("trace.jsonl")).unwrap();
    assert_eq!(contents.matches('\n').count(), 1);
    assert!(contents.ends_with('\n'));
}

#[tokio::test]
async fn an_empty_batch_writes_no_record() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter.export(Vec::new()).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let contents = fs::read_to_string(directory.path().join("trace.jsonl")).unwrap();
    assert!(
        contents.is_empty(),
        "an empty record would be indistinguishable from a lost one"
    );
}

#[tokio::test]
async fn resource_attributes_reach_the_written_record() {
    let directory = tempfile::tempdir().unwrap();
    let mut exporter = exporter_at(
        directory.path(),
        "trace.otlp.pb",
        OtlpFileFormat::Proto,
        false,
    );
    let resource = Resource::builder_empty()
        .with_attributes(vec![KeyValue::new("service.name", "relay-file-sink")])
        .build();
    exporter.set_resource(&resource);
    exporter.export(sample_spans(&["only"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let requests = read_proto_frames(&directory.path().join("trace.otlp.pb"));
    let attributes = &requests[0].resource_spans[0]
        .resource
        .as_ref()
        .unwrap()
        .attributes;
    assert!(
        attributes
            .iter()
            .any(|attribute| attribute.key == "service.name"),
        "set_resource must reach the encoded request"
    );
}

#[tokio::test]
async fn exporting_after_shutdown_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let error = exporter.export(sample_spans(&["late"])).await.unwrap_err();
    assert!(matches!(error, OTelSdkError::AlreadyShutdown));
}

#[test]
fn shutting_down_twice_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let error = exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap_err();
    assert!(matches!(error, OTelSdkError::AlreadyShutdown));
}

#[test]
fn flushing_after_shutdown_succeeds() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    // Everything the exporter accepted is already durable, so there is nothing
    // left for a flush to fail on.
    exporter.force_flush().unwrap();
}

#[tokio::test]
async fn each_export_is_durable_before_it_returns() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = exporter_at(
        directory.path(),
        "trace.otlp.pb",
        OtlpFileFormat::Proto,
        false,
    );
    exporter.export(sample_spans(&["only"])).await.unwrap();

    // Deliberately no shutdown: a run that dies between batches must still
    // leave the batches it already delivered on disk.
    let requests = read_proto_frames(&directory.path().join("trace.otlp.pb"));
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn overwrite_mode_replaces_an_existing_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");
    fs::write(&path, b"stale\n").unwrap();

    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    exporter.export(sample_spans(&["fresh"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    assert!(!contents.contains("stale"));
    assert_eq!(contents.lines().count(), 1);
}

#[tokio::test]
async fn append_mode_preserves_existing_records() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");
    fs::write(&path, b"{\"resourceSpans\":[]}\n").unwrap();

    let exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        true,
    );
    exporter.export(sample_spans(&["fresh"])).await.unwrap();
    exporter
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    let lines = read_json_lines(&path);
    assert_eq!(lines.len(), 2);
}

#[test]
fn a_missing_output_directory_is_created() {
    let directory = tempfile::tempdir().unwrap();
    let nested = directory.path().join("nested").join("deeper");
    let exporter = OtlpFileSpanExporter::new(
        directory.path(),
        &nested.join("trace.jsonl"),
        OtlpFileFormat::JsonLines,
        false,
    )
    .unwrap();

    assert!(nested.is_dir());
    assert_eq!(exporter.path(), nested.join("trace.jsonl"));
}

#[test]
fn an_unwritable_directory_reports_the_path() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");
    fs::create_dir(&path).unwrap();

    let error =
        OtlpFileSpanExporter::new(directory.path(), &path, OtlpFileFormat::JsonLines, false)
            .unwrap_err();
    assert!(
        error.to_string().contains("trace.jsonl"),
        "the error must name the path that failed: {error}"
    );
}

#[cfg(unix)]
#[test]
fn the_output_file_is_readable_only_by_its_owner() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().unwrap();
    let _exporter = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );

    let mode = fs::metadata(directory.path().join("trace.jsonl"))
        .unwrap()
        .permissions()
        .mode();
    // A trajectory carries prompt and response content, so it is created with
    // the same owner-only permissions as the ATOF and ATIF sinks.
    assert_eq!(
        mode & 0o077,
        0,
        "unexpected group or other permissions: {mode:o}"
    );
}

#[test]
fn each_format_names_its_conventional_extension() {
    assert_eq!(OtlpFileFormat::JsonLines.extension(), "jsonl");
    assert_eq!(OtlpFileFormat::Proto.extension(), "otlp.pb");
    assert_eq!(OtlpFileFormat::default(), OtlpFileFormat::JsonLines);
}
