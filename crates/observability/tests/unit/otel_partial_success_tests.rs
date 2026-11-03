// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP/HTTP partial success responses become runtime diagnostics.

use super::*;
use crate::runtime_diagnostics::ExporterRuntimeDiagnostic;
use opentelemetry_http::{Bytes, HttpClient, HttpError, Request, Response};
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsPartialSuccess, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsPartialSuccess, ExportMetricsServiceResponse,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTracePartialSuccess, ExportTraceServiceResponse,
};
use prost::Message;

const ENDPOINT: &str = "https://user:secret@collector.example:4318/v1/traces?token=abc";

#[derive(Debug)]
struct StaticResponseClient {
    status: u16,
    body: Vec<u8>,
}

#[async_trait::async_trait]
impl HttpClient for StaticResponseClient {
    async fn send_bytes(
        &self,
        _request: Request<Bytes>,
    ) -> std::result::Result<Response<Bytes>, HttpError> {
        Ok(Response::builder()
            .status(self.status)
            .body(Bytes::from(self.body.clone()))
            .unwrap())
    }
}

fn request(content_type: &str) -> Request<Bytes> {
    Request::builder()
        .uri(ENDPOINT)
        .header("content-type", content_type)
        .body(Bytes::new())
        .unwrap()
}

fn send(
    signal: OtlpSignal,
    status: u16,
    body: Vec<u8>,
    content_type: &str,
    diagnostics: &SignalRuntimeDiagnostics,
) -> Response<Bytes> {
    let client = PartialSuccessHttpClient {
        inner: StaticResponseClient { status, body },
        signal,
        diagnostics: diagnostics.clone(),
    };
    futures::executor::block_on(client.send_bytes(request(content_type))).unwrap()
}

fn trace_response(rejected: i64, message: &str) -> ExportTraceServiceResponse {
    ExportTraceServiceResponse {
        partial_success: Some(ExportTracePartialSuccess {
            rejected_spans: rejected,
            error_message: message.to_string(),
        }),
    }
}

fn only_diagnostic(diagnostics: &SignalRuntimeDiagnostics) -> ExporterRuntimeDiagnostic {
    let snapshot = diagnostics.snapshot();
    assert_eq!(snapshot.diagnostics.len(), 1, "{snapshot:?}");
    snapshot.diagnostics[0].clone()
}

#[test]
fn protobuf_trace_partial_success_records_rejected_spans() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let body = trace_response(3, "span name too long").encode_to_vec();
    let response = send(
        OtlpSignal::Traces,
        200,
        body.clone(),
        "application/x-protobuf",
        &diagnostics,
    );

    assert_eq!(response.body().as_ref(), body.as_slice());
    let diagnostic = only_diagnostic(&diagnostics);
    assert_eq!(diagnostic.code, "otel.partial_success");
    assert_eq!(diagnostic.count, 3);
    assert!(diagnostic.message.contains("collector.example:4318"));
    assert!(!diagnostic.message.contains("secret"));
    assert!(!diagnostic.message.contains("token=abc"));
    assert!(!diagnostic.message.contains("span name too long"));
}

#[test]
fn json_trace_partial_success_records_rejected_spans() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let body = serde_json::to_vec(&trace_response(2, "")).unwrap();
    send(
        OtlpSignal::Traces,
        200,
        body,
        "application/json",
        &diagnostics,
    );

    let diagnostic = only_diagnostic(&diagnostics);
    assert_eq!(diagnostic.code, "otel.partial_success");
    assert_eq!(diagnostic.count, 2);
}

#[test]
fn log_and_metric_partial_success_record_their_rejected_counts() {
    let logs = SignalRuntimeDiagnostics::new(None);
    let body = ExportLogsServiceResponse {
        partial_success: Some(ExportLogsPartialSuccess {
            rejected_log_records: 4,
            error_message: String::new(),
        }),
    }
    .encode_to_vec();
    send(OtlpSignal::Logs, 200, body, "application/x-protobuf", &logs);
    let diagnostic = only_diagnostic(&logs);
    assert_eq!(diagnostic.code, "otel.partial_success");
    assert_eq!(diagnostic.count, 4);

    let metrics = SignalRuntimeDiagnostics::new(None);
    let body = ExportMetricsServiceResponse {
        partial_success: Some(ExportMetricsPartialSuccess {
            rejected_data_points: 5,
            error_message: String::new(),
        }),
    }
    .encode_to_vec();
    send(
        OtlpSignal::Metrics,
        200,
        body,
        "application/x-protobuf",
        &metrics,
    );
    let diagnostic = only_diagnostic(&metrics);
    assert_eq!(diagnostic.code, "otel.partial_success");
    assert_eq!(diagnostic.count, 5);
}

#[test]
fn rejected_counts_accumulate_across_exports() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    for rejected in [3, 2] {
        send(
            OtlpSignal::Traces,
            200,
            trace_response(rejected, "").encode_to_vec(),
            "application/x-protobuf",
            &diagnostics,
        );
    }
    assert_eq!(only_diagnostic(&diagnostics).count, 5);
}

#[test]
fn responses_without_rejections_record_nothing() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    let cases: Vec<(u16, Vec<u8>, &str)> = vec![
        // Full success: empty body, no partial_success, and zero rejections.
        (200, Vec::new(), "application/x-protobuf"),
        (
            200,
            ExportTraceServiceResponse::default().encode_to_vec(),
            "application/x-protobuf",
        ),
        (
            200,
            trace_response(0, "").encode_to_vec(),
            "application/x-protobuf",
        ),
        // A warning message with zero rejections is not data loss.
        (
            200,
            trace_response(0, "deprecated attribute").encode_to_vec(),
            "application/x-protobuf",
        ),
        // Unreadable bodies and error statuses are left to the exporter.
        (
            200,
            b"not a protobuf message".to_vec(),
            "application/x-protobuf",
        ),
        (200, b"{not json".to_vec(), "application/json"),
        (
            400,
            trace_response(7, "").encode_to_vec(),
            "application/x-protobuf",
        ),
    ];
    for (status, body, content_type) in cases {
        let response = send(
            OtlpSignal::Traces,
            status,
            body.clone(),
            content_type,
            &diagnostics,
        );
        assert_eq!(response.status().as_u16(), status);
        assert_eq!(response.body().as_ref(), body.as_slice());
    }
    assert!(
        diagnostics.snapshot().diagnostics.is_empty(),
        "{:?}",
        diagnostics.snapshot()
    );
}

#[test]
fn json_counts_encoded_as_strings_are_read() {
    // OTLP/JSON encodes int64 as a decimal string; receivers accept both forms.
    for (signal, body, code, expected) in [
        (
            OtlpSignal::Traces,
            r#"{"partialSuccess":{"rejectedSpans":"3","errorMessage":"x"}}"#,
            "otel.partial_success",
            3,
        ),
        (
            OtlpSignal::Logs,
            r#"{"partialSuccess":{"rejectedLogRecords":"4"}}"#,
            "otel.partial_success",
            4,
        ),
        (
            OtlpSignal::Metrics,
            r#"{"partialSuccess":{"rejectedDataPoints":5}}"#,
            "otel.partial_success",
            5,
        ),
    ] {
        let diagnostics = SignalRuntimeDiagnostics::new(None);
        send(
            signal,
            200,
            body.as_bytes().to_vec(),
            "application/json",
            &diagnostics,
        );
        let diagnostic = only_diagnostic(&diagnostics);
        assert_eq!(diagnostic.code, code);
        assert_eq!(diagnostic.count, expected);
    }
}

#[test]
fn negative_or_malformed_counts_record_nothing() {
    for (signal, body, content_type) in [
        (
            OtlpSignal::Traces,
            trace_response(-5, "").encode_to_vec(),
            "application/x-protobuf",
        ),
        (
            OtlpSignal::Logs,
            br#"{"partialSuccess":{"rejectedLogRecords":"-2"}}"#.to_vec(),
            "application/json",
        ),
        (
            OtlpSignal::Metrics,
            br#"{"partialSuccess":{"rejectedDataPoints":"many"}}"#.to_vec(),
            "application/json",
        ),
        (
            OtlpSignal::Metrics,
            br#"{"partialSuccess":{"errorMessage":"warning only"}}"#.to_vec(),
            "application/json",
        ),
    ] {
        let diagnostics = SignalRuntimeDiagnostics::new(None);
        send(signal, 200, body, content_type, &diagnostics);
        assert!(
            diagnostics.snapshot().diagnostics.is_empty(),
            "{signal:?}: {:?}",
            diagnostics.snapshot()
        );
    }
}

#[test]
fn warning_is_logged_when_the_rejected_total_crosses_a_power_of_two() {
    // Totals 3, 6, 9, 12, 18 never land on a power of two, but each step that
    // passes one (4, 8, 16) still logs, so steady rejections are not silent.
    assert!(rejection_crosses_power_of_two(0, 3));
    assert!(rejection_crosses_power_of_two(3, 6));
    assert!(rejection_crosses_power_of_two(6, 9));
    assert!(!rejection_crosses_power_of_two(9, 12));
    assert!(rejection_crosses_power_of_two(12, 18));
    assert!(!rejection_crosses_power_of_two(4, 6));
    assert!(rejection_crosses_power_of_two(0, 1));
    assert!(rejection_crosses_power_of_two(1, 2));
    // `record` returns 0 when the diagnostics table is full: nothing to log.
    assert!(!rejection_crosses_power_of_two(0, 0));
}

#[test]
fn full_diagnostics_table_records_nothing_for_partial_success() {
    let diagnostics = SignalRuntimeDiagnostics::new(None);
    for index in 0..crate::runtime_diagnostics::MAX_RUNTIME_DIAGNOSTICS {
        diagnostics.record(format!("test.filler_{index}"), "filler".into(), 1);
    }
    send(
        OtlpSignal::Traces,
        200,
        trace_response(3, "").encode_to_vec(),
        "application/x-protobuf",
        &diagnostics,
    );
    let snapshot = diagnostics.snapshot();
    assert_eq!(
        snapshot.diagnostics.len(),
        crate::runtime_diagnostics::MAX_RUNTIME_DIAGNOSTICS
    );
    assert!(
        !snapshot
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "otel.partial_success")
    );
}
