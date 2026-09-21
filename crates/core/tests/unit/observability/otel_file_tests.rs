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

#[tokio::test]
async fn a_second_exporter_for_one_path_shares_the_open_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");

    // `overwrite` is the default: a second open of the same path would discard
    // what the first exporter had already written.
    let base = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    base.export(sample_spans(&["base"])).await.unwrap();
    let sibling = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    sibling.export(sample_spans(&["sibling"])).await.unwrap();
    base.export(sample_spans(&["base-again"])).await.unwrap();

    let records = read_json_lines(&path);
    assert_eq!(records.len(), 3);
    assert_eq!(
        records
            .iter()
            .map(
                |record| record["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"]
                    .as_str()
                    .unwrap()
                    .to_string()
            )
            .collect::<Vec<_>>(),
        ["base", "sibling", "base-again"],
    );
}

#[tokio::test]
async fn one_exporter_shutting_down_leaves_its_siblings_writing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");

    let base = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    let sibling = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    sibling
        .shutdown_with_timeout(Duration::from_secs(1))
        .unwrap();

    assert!(matches!(
        sibling.export(sample_spans(&["late"])).await,
        Err(OTelSdkError::AlreadyShutdown)
    ));
    base.export(sample_spans(&["still-open"])).await.unwrap();
    assert_eq!(read_json_lines(&path).len(), 1);
}

#[tokio::test]
async fn a_path_reopened_after_every_exporter_is_dropped_honours_overwrite() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");

    let first = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    first.export(sample_spans(&["stale"])).await.unwrap();
    drop(first);

    let second = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    second.export(sample_spans(&["fresh"])).await.unwrap();

    let records = read_json_lines(&path);
    assert_eq!(records.len(), 1);
    assert_eq!(
        span_names(&serde_json::from_value(records[0].clone()).unwrap()),
        ["fresh"],
    );
}

#[test]
fn a_path_outside_the_output_directory_creates_no_directories() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("configured");
    let escape = directory.path().join("elsewhere");

    // A sibling directory, and the same directory reached by traversing out of
    // the root: `Path::starts_with` compares components lexically, so the
    // second one looks confined until `..` is resolved.
    for path in [
        escape.join("trace.jsonl"),
        root.join("..").join("elsewhere").join("trace.jsonl"),
    ] {
        let error =
            OtlpFileSpanExporter::new(&root, &path, OtlpFileFormat::JsonLines, false).unwrap_err();

        assert!(
            matches!(error, OtlpFileExporterError::OpenFile { .. }),
            "expected the confinement rejection for {path:?}, got {error}"
        );
        assert!(
            !escape.exists(),
            "{path:?} created a directory outside the root"
        );
    }
}

#[tokio::test]
async fn a_reused_handle_is_still_confined_to_the_callers_own_directory() {
    let directory = tempfile::tempdir().unwrap();
    let opened = directory.path().join("configured");
    let elsewhere = directory.path().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let path = opened.join("trace.jsonl");

    let base = OtlpFileSpanExporter::new(&opened, &path, OtlpFileFormat::JsonLines, false).unwrap();
    base.export(sample_spans(&["base"])).await.unwrap();

    // The confinement check lives in the open below the registry lookup, so a
    // reused handle would hand this caller a file its own root excludes.
    let error =
        OtlpFileSpanExporter::new(&elsewhere, &path, OtlpFileFormat::JsonLines, false).unwrap_err();

    assert!(
        matches!(error, OtlpFileExporterError::OpenFile { .. }),
        "expected the confinement rejection, got {error}"
    );
    assert!(error.to_string().contains("outside configured directory"));
}

#[tokio::test]
async fn a_second_exporter_disagreeing_about_the_encoding_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.jsonl");
    let base = exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    );
    base.export(sample_spans(&["base"])).await.unwrap();

    // Sharing the handle would write protobuf frames into a JSON lines file,
    // leaving a stream neither reader can parse.
    for (format, append, setting) in [
        (OtlpFileFormat::Proto, false, "format"),
        (OtlpFileFormat::JsonLines, true, "mode"),
    ] {
        let error = OtlpFileSpanExporter::new(directory.path(), &path, format, append).unwrap_err();

        assert!(
            matches!(
                &error,
                OtlpFileExporterError::ConflictingOpen { setting: reported, .. }
                    if *reported == setting
            ),
            "expected a {setting} conflict, got {error}"
        );
    }

    assert_eq!(read_json_lines(&path).len(), 1);
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
fn from_parts_rejects_every_filename_that_is_not_one_plain_component() {
    // `.` and `..` would otherwise reach the confinement check, which reports
    // them as an internal failure rather than an invalid argument.
    for filename in [
        "",
        "   ",
        " trace.jsonl",
        ".",
        "..",
        "nested/trace.jsonl",
        "/trace.jsonl",
    ] {
        let error = crate::observability::otel::OtlpFileSinkSettings::from_parts(
            "/tmp/relay",
            Some(filename),
            None,
            None,
        )
        .expect_err("expected {filename:?} to be rejected");
        assert!(
            error.contains("filename must be"),
            "unexpected error for {filename:?}: {error}"
        );
    }

    let settings = crate::observability::otel::OtlpFileSinkSettings::from_parts(
        "/tmp/relay",
        Some("trace.jsonl"),
        None,
        None,
    )
    .unwrap();
    assert_eq!(settings.path, Path::new("/tmp/relay/trace.jsonl"));
}

#[test]
fn io_failures_name_the_path_and_the_underlying_error() {
    // The write and flush paths need a failing file to reach, so the messages
    // they build are asserted directly: a diagnostic that omits the path leaves
    // a multi-sink configuration with no way to tell which file failed.
    let path = Path::new("/var/log/nemo-relay/trace.jsonl");
    let source = std::io::Error::new(std::io::ErrorKind::StorageFull, "no space left on device");

    let write = write_failure(path, &source).to_string();
    assert!(write.contains("trace.jsonl"), "{write}");
    assert!(write.contains("no space left on device"), "{write}");

    let flush = flush_failure(path, &source).to_string();
    assert!(flush.contains("trace.jsonl"), "{flush}");
    assert!(flush.contains("no space left on device"), "{flush}");
    assert_ne!(write, flush, "the two stages should be distinguishable");

    // `u32::try_from` only fails past 4 GiB, which no test can allocate.
    let oversized = frame_too_large(u64::from(u32::MAX) as usize + 1);
    assert!(oversized.contains("4294967296"), "{oversized}");
    assert!(oversized.contains("frame maximum"), "{oversized}");
}

#[test]
fn each_format_names_its_conventional_extension() {
    assert_eq!(OtlpFileFormat::JsonLines.extension(), "jsonl");
    assert_eq!(OtlpFileFormat::Proto.extension(), "otlp.pb");
    assert_eq!(OtlpFileFormat::default(), OtlpFileFormat::JsonLines);
}

// --- File sink config -------------------------------------------------------
//
// Endpoint options are absent from `OpenTelemetryFileSinkConfig` rather than
// rejected, so the cases a runtime check used to cover are now compile errors
// and have no test.

fn file_sink_config(directory: &Path) -> crate::observability::otel::OpenTelemetryFileSinkConfig {
    crate::observability::otel::OpenTelemetryFileSinkConfig::new(
        crate::observability::OpenTelemetryType::Full,
        crate::observability::otel::OtlpFileSinkSettings {
            output_directory: directory.to_path_buf(),
            path: directory.join("trace.jsonl"),
            format: OtlpFileFormat::JsonLines,
            append: false,
        },
    )
}

#[test]
fn a_file_sink_config_keeps_the_sink_it_was_given() {
    let directory = tempfile::tempdir().unwrap();
    let config = file_sink_config(directory.path());

    assert_eq!(config.sink().path, directory.path().join("trace.jsonl"));
    assert_eq!(config.sink().output_directory, directory.path());
    assert!(!config.sink().append);
}

#[test]
fn a_file_sink_config_builds_a_subscriber_and_opens_its_file() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let subscriber = crate::observability::otel::OpenTelemetrySubscriber::new_file_sink(
        file_sink_config(directory.path()),
    )
    .expect("a file sink config builds");

    assert!(directory.path().join("trace.jsonl").is_file());
    subscriber.shutdown().unwrap();
}

#[test]
fn a_file_sink_carries_the_shared_trace_options() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().unwrap();
    let config = file_sink_config(directory.path())
        .with_service_name("file-sink-agent")
        .with_service_namespace("agents")
        .with_completed_span_context_ttl(Duration::from_secs(30));

    // The same builders an endpoint config offers, applied to a file sink.
    assert_eq!(config.completed_span_context_ttl(), Duration::from_secs(30));
    let subscriber = crate::observability::otel::OpenTelemetrySubscriber::new_file_sink(config)
        .expect("shared options apply to a file sink");
    subscriber.shutdown().unwrap();
}

#[test]
fn a_file_sink_ignores_process_global_otlp_headers() {
    const CHILD_MARKER: &str = "NEMO_RELAY_TEST_FILE_SINK_GLOBAL_HEADER_CHILD";
    if std::env::var(CHILD_MARKER).is_ok() {
        let directory = tempfile::tempdir().unwrap();
        // `OTEL_EXPORTER_OTLP_HEADERS` cannot reach a file, so a process that
        // sets it for some other exporter must not break this one.
        crate::observability::otel::OpenTelemetrySubscriber::new_file_sink(file_sink_config(
            directory.path(),
        ))
        .expect("global OTLP headers do not apply to a file sink")
        .shutdown()
        .unwrap();
        return;
    }

    // Set in a child so the variable cannot leak into sibling tests.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "--nocapture", "--test-threads", "1"])
        .arg("observability::otel_file::tests::a_file_sink_ignores_process_global_otlp_headers")
        .env(CHILD_MARKER, "1")
        .env("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer token")
        .output()
        .unwrap();
    let summary = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "child run failed: {}\n{summary}",
        output.status
    );
    // An exact filter that no longer matches leaves libtest exiting
    // successfully, so the child has to report that it ran the test.
    assert!(
        summary.contains("1 passed"),
        "child ran no test:\n{summary}"
    );
}

#[test]
fn a_directory_that_cannot_be_created_reports_the_directory() {
    let directory = tempfile::tempdir().unwrap();
    // A file where a parent directory is expected: `create_dir_all` fails
    // before the output file is ever opened.
    let blocker = directory.path().join("not-a-directory");
    fs::write(&blocker, b"").unwrap();

    let error = OtlpFileSpanExporter::new(
        directory.path(),
        &blocker.join("trace.jsonl"),
        OtlpFileFormat::JsonLines,
        false,
    )
    .unwrap_err();

    assert!(
        matches!(error, OtlpFileExporterError::CreateDirectory { .. }),
        "expected a directory-creation error, got {error}"
    );
    assert!(error.to_string().contains("not-a-directory"));
}

#[tokio::test]
async fn a_poisoned_writer_lock_is_reported_by_every_entry_point() {
    let directory = tempfile::tempdir().unwrap();
    let exporter = std::sync::Arc::new(exporter_at(
        directory.path(),
        "trace.jsonl",
        OtlpFileFormat::JsonLines,
        false,
    ));

    let poisoner = std::sync::Arc::clone(&exporter);
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.writer.file.lock().unwrap();
        panic!("poison the writer lock");
    })
    .join();

    // Every path that takes the lock reports the poisoning rather than
    // unwrapping into a second panic.
    for message in [
        exporter.export(sample_spans(&["late"])).await.unwrap_err(),
        exporter
            .shutdown_with_timeout(Duration::from_secs(1))
            .unwrap_err(),
        exporter.force_flush().unwrap_err(),
    ] {
        assert!(
            message.to_string().contains("poisoned"),
            "unexpected error: {message}"
        );
    }
}

// --- Endpoint exporter construction -----------------------------------------
//
// These exercise the endpoint branch of `otlp_span_exporter`, which the file
// sink work moved out of `build_tracer_provider_with_resource`.

#[test]
fn an_endpoint_with_a_header_file_builds_on_both_transports() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    use crate::observability::otel::{OpenTelemetryConfig, OpenTelemetrySubscriber, OtlpTransport};

    let directory = tempfile::tempdir().unwrap();
    let header_path = directory.path().join("token");
    fs::write(&header_path, b"Bearer file-token").unwrap();

    for transport in [OtlpTransport::HttpBinary, OtlpTransport::Grpc] {
        let config = OpenTelemetryConfig::new(
            crate::observability::OpenTelemetryType::Full,
            "https://collector.example/v1/traces",
        )
        .with_transport(transport)
        .with_header_file("authorization", header_path.display().to_string());

        let subscriber = OpenTelemetrySubscriber::new(config)
            .unwrap_or_else(|error| panic!("{transport:?} with a header file: {error}"));
        subscriber.shutdown().unwrap();
    }
}

#[test]
fn an_endpoint_without_a_header_file_builds_on_both_transports() {
    let _guard = crate::observability::test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    use crate::observability::otel::{OpenTelemetryConfig, OpenTelemetrySubscriber, OtlpTransport};

    for transport in [OtlpTransport::HttpBinary, OtlpTransport::Grpc] {
        let config = OpenTelemetryConfig::new(
            crate::observability::OpenTelemetryType::Full,
            "https://collector.example/v1/traces",
        )
        .with_transport(transport)
        .with_header("authorization", "Bearer inline");

        OpenTelemetrySubscriber::new(config)
            .unwrap()
            .shutdown()
            .unwrap();
    }
}
